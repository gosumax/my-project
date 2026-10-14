"""Literal service messages and an evidence-preserving roster observation view.

Recognition is not seat or stack authority. All source observations survive;
repeated state notifications are idempotent, while separate chip transfers are
never deduplicated merely because actor, text and amount happen to agree.
"""
import copy
import re

NUMBER = r'\d+(?:[ \u00a0]\d{3})*'
# A service actor must be one literal nickname token. A numbered header or an
# extra action clause cannot be swallowed as the actor of a harmless notice.
REBUY = re.compile(r'([^\s:#;]+) makes rebuy\(s\) and gets (' + NUMBER + r') chips[.!]?', re.I)
BOUNTY_HEAD = r'[^\s:#;]+ is awarded ' + NUMBER + r'(?:[.,]\d{1,2})?\s*(?:[pр]\.|₽|\$|€)?\s+for'
BUSTED_PLAYERS = r'busted player\(s\):'
PLAYER_LIST = r'[^\s,:;#]+(?:\s*,\s*[^\s,:;#]+)*'
BREAK_HEAD = r'Break has started and will last'
BREAK_TAIL = r'[1-9]\d* min after the end of active hands\.'


def describe_service(text):
    clean = re.sub(r'^\s*\[\d{1,2}:\d{2}\]\s*', '', str(text)).strip()
    clean = re.sub(r'^(?:Dealer|Дилер)\s*:\s*', '', clean, flags=re.I)
    if re.fullmatch(BREAK_HEAD + r'\s+' + BREAK_TAIL, clean, re.I):
        return dict(kind='BREAK', actor=None, amount=None, stack_transfer=False)
    bounty = re.fullmatch(BOUNTY_HEAD + r'\s+' + BUSTED_PLAYERS + r'(?:\s+(' + PLAYER_LIST + r'))?', clean, re.I)
    if bounty:
        return dict(kind='BOUNTY', actor=None, amount=None, stack_transfer=False,
                    continuation_pending=bounty[1] is None)
    if re.fullmatch(r'Please wait while players are reseating[.!●]?', clean, re.I):
        return dict(kind='RESEATING', actor=None, amount=None, stack_transfer=False)
    rebuy = REBUY.fullmatch(clean)
    if rebuy:
        amount = int(rebuy[2].replace(' ', '').replace('\u00a0', ''))
        if amount > 0:
            return dict(kind='REBUY', actor=rebuy[1], amount=amount, stack_transfer=True)
        return None
    patterns = [
        (r'(.+?) has joined the table[.!]?', 'JOIN'),
        (r'(.+?) has left the table[.!]?', 'LEAVE'),
        (r'(.+?) finished the tournament[.!]?', 'FINISHED'),
        (r'(.+?) sits in[.!]?', 'SIT_IN'),
        (r'(.+?) sits out[.!]?', 'SIT_OUT'),
        (r'(.+?) is reconnected[.!]?', 'RECONNECTED'),
        (r'(.+?) uses Time Bank[.!]?', 'TIME_BANK'),
    ]
    for pattern, kind in patterns:
        match = re.fullmatch(pattern, clean, re.I)
        if match:
            return dict(kind=kind, actor=match[1], amount=None, stack_transfer=False)
    waiting = re.fullmatch(r'Waiting for (.+?) to (reconnect|rebuy)\.{0,3}', clean, re.I)
    if waiting:
        return dict(kind='WAIT_' + waiting[2].upper(), actor=waiting[1], amount=None, stack_transfer=False)
    return None


def service_view(hands):
    """Project all observed messages without removing rows from hand validation.

    Only exact repeated source objects share an event identity. Repeated JOIN
    or LEAVE at distinct sources remains a separate observation confirming the
    same last observed state. This view cannot prove a complete table roster.
    """
    events, exact_seen, presence, repeated = [], {}, {}, 0
    for hand in hands:
        table, session = hand['table_id'], hand['session_id']
        for message in hand['messages']:
            service = describe_service(message['raw_text'])
            if service is None:
                continue
            rows = message['source_rows']
            identity = tuple((r.get('table_id'), r.get('session_id'), r.get('stable_row_id', r.get('row_id')),
                r.get('frame_id'), r.get('row_index'), r.get('row_pixel_sha256')) for r in rows)
            if not identity or any(any(v is None for v in part) for part in identity):
                raise ValueError('SERVICE_SOURCE_IDENTITY_INCOMPLETE')
            if any(r.get('table_id') != table or r.get('session_id') != session for r in rows):
                raise ValueError('SERVICE_SOURCE_CONTEXT_CONFLICT')
            observed = dict(table=table, session=session, hand_id=hand.get('hand_id'),
                **service, raw_text=message['raw_text'], source_rows=copy.deepcopy(rows))
            if identity in exact_seen:
                prior = exact_seen[identity]
                if prior['observation'] != observed:
                    raise ValueError('SERVICE_SAME_SOURCE_FIELDS_CONFLICT')
                prior['observations'].append(copy.deepcopy(observed))
                repeated += 1
                continue
            event = dict(observation=observed, observations=[copy.deepcopy(observed)],
                roster_effect='NONE', authority='LITERAL_SERVICE_OBSERVATION_ONLY')
            key = (table, session, service['actor'])
            if service['kind'] in {'JOIN', 'LEAVE'}:
                state = 'PRESENT' if service['kind'] == 'JOIN' else 'ABSENT'
                event['roster_effect'] = ('CONFIRMS_SAME_OBSERVED_STATE' if presence.get(key) == state
                                          else 'OBSERVED_STATE_TRANSITION')
                presence[key] = state
            events.append(event)
            exact_seen[identity] = event
    return dict(events=events, exact_source_reobservations=repeated,
        observed_presence=[dict(table=t, session=s, actor=a, state=state) for (t,s,a),state in presence.items()],
        complete_roster_proven=False, seat_authority=False, stack_authority=False,
        raw_observations_preserved=True)
