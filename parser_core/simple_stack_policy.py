"""Production STACK policy: first fixed-seat numeric with confidence >= 0.85."""
import re
from math import isfinite

POLICY = 'FIRST_PARSEABLE_FIXED_STACK_ROI_V8'
STACK_NUMERIC_CONFIDENCE_MIN = 0.85
NUMBER = re.compile(r'\d+(?:[ \u00a0]\d{3})*')


def normalize_stack_text(raw):
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if text.startswith('_') and NUMBER.fullmatch(text[1:]):
        return text[1:]
    return text


def parse_stack(raw):
    text = normalize_stack_text(raw)
    if not isinstance(text, str) or not NUMBER.fullmatch(text):
        return None
    return int(text.replace(' ', '').replace('\u00a0', ''))


def first_displayed_stack(observations, *, table_id, session_id, seat, hand_start, hand_end):
    for row in sorted(observations, key=lambda r: (r['frame'], r.get('source_order', 0))):
        if (row['table_id'], row['session_id'], row['seat']) != (table_id, session_id, seat):
            continue
        if not hand_start <= row['frame'] <= hand_end:
            continue
        value = parse_stack(row.get('raw_text'))
        if value is not None:
            if row.get('numeric_admission',{}).get('status') == 'UNKNOWN':
                continue
            confidence = row.get('confidence')
            if (not isinstance(confidence, (int, float)) or not isfinite(confidence)
                    or confidence < STACK_NUMERIC_CONFIDENCE_MIN):
                continue
            raw = row.get('raw_text')
            return dict(row, displayed_stack=value, normalized_stack_text=normalize_stack_text(raw),
                        underscore_normalized=isinstance(raw, str) and raw.strip().startswith('_'),
                        policy=POLICY)
    return None


def corrected_start_stack(displayed, forced):
    if displayed is None or any(forced.get(role) is None for role in ('ANTE', 'SB', 'BB')):
        return None
    return displayed + sum(forced[role] for role in ('ANTE', 'SB', 'BB'))


FORCED_ALL_IN_STACK_RULE = 'START_STACK_FROM_OBSERVED_FORCED_ALL_IN_V1'
FORCED_ROLES = {'ANTE':'ANTE','SMALL_BLIND':'SB','BIG_BLIND':'BB','SB':'SB','BB':'BB'}


def forced_allin_prefix_total(messages, actor):
    """Economic pattern only; native source and WHO must be proved separately."""
    endings=[i for i,m in enumerate(messages) if m.get('player')==actor and
             m.get('action') in FORCED_ROLES and m.get('all_in') is True]
    if len(endings)!=1:return None
    end=endings[0];prefix=messages[:end+1]
    if any(m.get('action') not in {*FORCED_ROLES,'HAND_HEADER','NEW_HAND','HAND_ID','TABLE_INFO','BLIND_LEVEL'}
           for m in prefix):return None
    own=[m for m in prefix if m.get('player')==actor and m.get('action') in FORCED_ROLES]
    roles=[FORCED_ROLES[m['action']] for m in own]
    if len(roles)!=len(set(roles)) or {'SB','BB'}<=set(roles):return None
    if (roles[-1]!='ANTE' and any(m.get('action')=='ANTE' for m in prefix) and 'ANTE' not in roles):return None
    if any(m.get('player')==actor and m.get('action') in {*FORCED_ROLES,'CALL','BET','RAISE','ALL_IN'}
           for m in messages[end+1:]):return None
    amounts=[parse_stack(m.get('amount')) for m in own]
    if any(v is None or v<=0 for v in amounts):return None
    return dict(rule=FORCED_ALL_IN_STACK_RULE,calculated_start_stack=sum(amounts),
                end_index=end,roles=roles,amounts=amounts)
