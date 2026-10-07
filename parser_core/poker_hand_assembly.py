"""Source-preserving hand assembly; production uses strict pixel/adjacency mode."""
from __future__ import annotations

import copy
import re
from collections import defaultdict
from parser_core.normalize_chat import normalize
from parser_core.poker_card_engine import described_class
from parser_core.canonical_semantics import bound_forced_all_in

VERSION = 'experimental-assembly-4-source-continuations'
MONEY = re.compile(r'(?<!\d)\d+(?:[ \u00a0]\d{3})*(?!\d)')
CARDS = re.compile(r'(?<![A-Za-z0-9])([2-9TJQKA]|10)([shdc♠♥♦♣])(?![A-Za-z0-9])', re.I)
SUITS = {'♠': 's', '♥': 'h', '♦': 'd', '♣': 'c'}
BETTING = {'FOLD', 'CHECK', 'CALL', 'BET', 'RAISE'}
MONEY_ACTIONS = {'ANTE', 'SMALL_BLIND', 'BIG_BLIND', 'CALL', 'BET', 'RAISE', 'WIN', 'UNCALLED_RETURN'}
CHIP_NUMBER = r'\d+(?:[ \u00a0]\d{3})*'
RANK_DESCRIPTION = r'(?:High card|Pair|Two pairs|Three of a kind|Straight|Flush|Full house|Four of a kind|Straight flush)[ ,].+'


def observed_wrapped_continuation(group,row):
    """Known literal wraps and a hash-bound card row in one physical source."""
    first,previous=group[0],group[-1]
    frame=previous.get('frame_id');index=previous.get('row_index')
    if (frame is None or frame!=row.get('frame_id') or not isinstance(index,int)
            or row.get('row_index')!=index+1 or any(previous.get(k)!=row.get(k)
                for k in ('table_id','session_id','epoch'))):
        return False
    text=lambda r:str(r.get('raw_ocr_text') or r.get('raw_text') or r.get('text') or '').strip()
    head=re.sub(r'^(?:Dealer|Дилер)\s*:\s*','',text(first),flags=re.I)
    tail=text(row)
    if len(group)==1 and (
            re.fullmatch(r'.+ gets uncalled',head,re.I) and re.fullmatch(r'bet\s*'+CHIP_NUMBER+r'\s+back[.!●]?',tail,re.I)
            or re.fullmatch(r".+ (?:doesn't show|does not show)",head,re.I) and re.fullmatch(r'cards[.!●]?',tail,re.I)
            or re.fullmatch(r'.+ (?:has joined|has left)',head,re.I) and re.fullmatch(r'the table[.!●]?',tail,re.I)
            or re.fullmatch(r'Waiting for .+ to',head,re.I) and re.fullmatch(r'(?:reconnect|rebuy)\.{0,3}',tail,re.I)
            or re.fullmatch(r'.+ posts (?:Small|Big)',head,re.I) and re.fullmatch(r'Blind '+CHIP_NUMBER,tail,re.I)
            or re.fullmatch(r'.+ shows',head,re.I) and re.fullmatch(RANK_DESCRIPTION,tail,re.I)):
        return True
    card_id=first.get('card_row_id')
    digest=row.get('row_pixel_sha256')
    card_group=(len(group)==1 or len(group)==2 and re.fullmatch(RANK_DESCRIPTION,text(group[1]),re.I))
    same_source=all(r.get('frame_id')==frame and all(r.get(k)==row.get(k)
        for k in ('table_id','session_id','epoch')) for r in group)
    return bool(card_group and same_source and card_id and card_id==row.get('stable_row_id',row.get('row_id'))
        and isinstance(digest,str) and re.fullmatch(r'[0-9a-f]{64}',digest)
        and first.get('identity_status')=='CHAT_ROW_SHA_ROUNDTRIP_VERIFIED'
        and first.get('card_crop_pixel_sha256')==digest
        and make_message(group+[row],require_pixel_cards=True)['field_sources'].get('cards')=='OBSERVED_PIXEL'
        and (re.fullmatch(r'(?:Dealing (?:the )?)?(?:Flop|Turn|River)',head,re.I)
             or re.fullmatch(r'.+ shows',head,re.I)))


def split_small_blind(first, second):
    """Only the proven Small/Blind layout in one adjacent physical snapshot."""
    def text(row):
        return str(row.get('raw_text') or row.get('raw_ocr_text') or row.get('text') or '').strip()
    left = re.sub(r'^\s*\[\d+:\d+\]\s*', '', text(first))
    left = re.sub(r'^Dealer\s*:\s*', '', left, flags=re.I)
    if not re.fullmatch(r'.+?\s+posts Small', left, re.I):
        return False
    if not re.fullmatch(r'Blind \d+(?:[ \u00a0]\d{3})*', text(second), re.I):
        return False
    frame = first.get('frame_id', first.get('source_frame'))
    index = first.get('row_index', first.get('index'))
    return (frame is not None and frame == second.get('frame_id', second.get('source_frame'))
            and isinstance(index, int)
            and second.get('row_index', second.get('index')) == index + 1
            and all(first.get(k) == second.get(k) for k in
                    ('table_id', 'session_id', 'epoch', 'hand_id')))


def card_tokens(text):
    return [('T' if r == '10' else r.upper()) + SUITS.get(s, s.lower()) for r, s in CARDS.findall(text)]


def parse_text(text):
    clean = re.sub(r'^\s*\[\d{1,2}:\d{2}\]\s*', '', text).strip()
    clean = re.sub(r'^(?:Dealer|Дилер)\s*:\s*', '', clean, flags=re.I)
    c = normalize([text])
    event = c.event_type
    actor, amount, hand_id = c.actor, c.amount_decimal, c.hand_number
    # English action/amount extraction is owned by the same normalizer that
    # annotates physical OCR rows. Retain the existing Russian return wording.
    for pattern, kind in [(r'возвращается|получает обратно|получает неуравненную ставку', 'UNCALLED_RETURN')]:
        # OCR may join a nickname's final underscore to the known verb. Keep
        # that underscore in the actor; never normalize player identity.
        match = re.search(r'(?<![^\W_])(?:'+pattern+r')(?!\w)', clean, re.I)
        if match:
            actor, event = clean[:match.start()].strip(' :'), kind
            tail = clean[match.end():].strip()
            number = MONEY.match(tail) if kind in MONEY_ACTIONS else None
            if number:
                suffix = tail[number.end():].strip()
                allowed = (not suffix or suffix in {'.', '!', '●'} or
                           bool(re.fullmatch(r'(?:and\s*(?:is All-in)?|is All-in|back)\s*[.!●]?', suffix, re.I)) or
                           (kind == 'WIN' and bool(re.match(r'from\b', suffix, re.I))))
                amount = number.group().replace(' ', '').replace('\u00a0', '') if allowed else None
            else:
                amount = None
            break
    header = re.search(r'(?:Hand\s*#|Рука\s*#|^#)\s*(\d+)', clean, re.I)
    if header:
        event, hand_id = 'HAND_HEADER', header.group(1)
    if re.search(r'Starting a new hand|Начинается новая рука', clean, re.I):
        event = 'NEW_HAND'
        n = re.search(r'#\s*(\d+)', clean)
        hand_id = n.group(1) if n else None
    for pattern, kind in [(r'^(?:Dealing (?:the )?|разда[её]тся )?flop\b', 'FLOP'),
                          (r'^(?:Dealing (?:the )?|разда[её]тся )?turn\b', 'TURN'),
                          (r'^(?:Dealing (?:the )?|разда[её]тся )?river\b', 'RIVER'),
                          (r'^showdown\b', 'SHOWDOWN'), (r'^hand end\b', 'HAND_END')]:
        if re.search(pattern, clean, re.I):
            event = kind
    if re.fullmatch(r'Dealing cards[.!●]?', clean, re.I):
        event = 'DEAL'
    if re.search(r'(?:uses Time Bank|sits out|sits in)$', clean, re.I):
        event = 'TABLE_INFO'
    if re.fullmatch(r'.+ (?:finished the tournament|has joined the table|has left the table)[.!]?', clean, re.I):
        event, actor, amount = 'TABLE_INFO', None, None
    # Explicit server notices carry no betting action. Anchor the whole line so
    # arbitrary unrecognized chat cannot be accepted as harmless telemetry.
    if re.fullmatch(r'(?:Waiting for .+ to(?: (?:reconnect|rebuy)\.{0,3})?|(?:reconnect|rebuy)\.{0,3}|.+ is reconnected[.!]?)', clean, re.I):
        event, actor, amount = 'TABLE_INFO', None, None
    if re.fullmatch(r'Limits going up: stakes(?:\s+\d+(?:[ \u00a0]\d{3})*/\d+(?:[ \u00a0]\d{3})*\s+ante\s+\d+(?:[ \u00a0]\d{3})*)?[.!]?', clean, re.I):
        event, actor, amount = 'BLIND_LEVEL', None, None
    return {'action': event, 'player': actor, 'amount': amount, 'hand_number': hand_id,
            'amount_role': c.amount_role if event in MONEY_ACTIONS else None,
            'all_in': bool(re.search(r'all[ -]?in|олл.?ин|ва.?банк', clean, re.I)),
            'cards': card_tokens(clean), 'showdown_class': described_class(clean) if event in {'SHOW', 'WIN'} else None,
            'normalized_text': clean}


def make_message(rows, *, require_pixel_cards=False, pixel_sources=None):
    text = ' '.join(str(r.get('raw_text') or r.get('raw_ocr_text') or r.get('text') or '') for r in rows)
    parsed = parse_text(text)
    message = dict(parsed, raw_text=text, source_rows=copy.deepcopy(rows),
                source_row_ids=[str(r.get('row_id', r.get('stable_row_id', r.get('physical_row_id', 'MISSING_ROW_ID')))) for r in rows],
                frame_id=rows[0].get('frame_id'), timestamp_ms=rows[0].get('timestamp_ms'),
                source_type='DERIVED', field_sources={'amount': 'OBSERVED', 'player': 'OBSERVED', 'cards': 'OBSERVED'},
                assembly='EXPLICIT_CONTINUATION' if len(rows) > 1 else 'SINGLE_ROW')
    if require_pixel_cards:
        message['cards'] = []
        message['field_sources']['cards'] = 'UNRESOLVED'
    trigger = rows[0]
    if trigger.get('canonical_provenance'):
        message['canonical_event_id'] = trigger.get('canonical_event_id')
        message['canonical_provenance'] = copy.deepcopy(trigger['canonical_provenance'])
        message['seat'] = trigger.get('seat')
        message['field_sources'].update(amount='OBSERVED_SOURCE_BOUND',player='OBSERVED_SOURCE_BOUND')
    if any(r.get('canonical_conflict') or r.get('canonical_duplicate_of') for r in rows):
        message['canonical_withheld'] = [r.get('canonical_conflict') or r.get('canonical_duplicate_of') for r in rows]
        message.update(action='UNKNOWN_EVENT', amount=None, player=None)
        message['field_sources'].update(amount='UNRESOLVED',player='UNRESOLVED')
    claimed,all_in_proof=bound_forced_all_in(rows,message['action'],message['player'],message['amount'])
    if all_in_proof:
        message['all_in']=True
        message['all_in_provenance']=all_in_proof
        message['field_sources']['all_in']='OBSERVED_SOURCE_BOUND'
    elif claimed:
        message.update(action='UNKNOWN_EVENT',amount=None,player=None,all_in=False)
        message.setdefault('canonical_withheld',[]).append('CANONICAL_ALL_IN_PROOF_CONFLICT')
        message['field_sources'].update(amount='UNRESOLVED',player='UNRESOLVED',all_in='UNRESOLVED')
    if trigger.get('cards') and parsed['action'] in {'FLOP', 'TURN', 'RIVER', 'SHOW'}:
        source = next((r for r in rows if str(r.get('row_id', r.get('stable_row_id'))) == str(trigger.get('card_row_id'))), None)
        if source is None and require_pixel_cards and pixel_sources is not None:
            source = pixel_sources.get(str(trigger.get('card_row_id')))
        response = trigger.get('raw_pixel_response') or {}
        hashes = [trigger.get('card_crop_pixel_sha256'), trigger.get('visual_crop_sha256'),
                  response.get('source_sha256'), source.get('row_pixel_sha256') if source else None]
        verified = (trigger.get('evidence_status') == 'RAW_UNREVIEWED' and
                    trigger.get('identity_status') == 'CHAT_ROW_SHA_ROUNDTRIP_VERIFIED' and
                    trigger.get('card_evidence_id') and source is not None and
                    all(isinstance(h, str) and re.fullmatch(r'[0-9a-f]{64}', h) for h in hashes) and
                    response.get('status') == 'OK' and response.get('request_id') and
                    response.get('recognizer') == 'FROZEN_NATIVE_PIXEL_CARD_READER' and
                    trigger.get('recognition_mode') == 'PIXEL_ONLY' and
                    trigger.get('development_only') is False and
                    source.get('row_pixel_sha256') == trigger.get('card_crop_pixel_sha256') and
                    response.get('source_sha256') == trigger.get('visual_crop_sha256') and
                    source.get('frame_id') == trigger.get('frame_id') and
                    source.get('table_id') == trigger.get('table_id') and
                    source.get('session_id') == trigger.get('session_id') and
                    source.get('epoch') == trigger.get('epoch') and
                    (not require_pixel_cards or
                     (isinstance(source.get('row_index'), int) and
                      isinstance(trigger.get('row_index'), int) and
                      1 <= source['row_index'] - trigger['row_index'] <= 2)))
        if verified:
            message['cards'] = list(trigger['cards'])
            message['field_sources']['cards'] = 'OBSERVED_PIXEL'
            message['card_provenance'] = {k: trigger.get(k) for k in ['card_evidence_id', 'card_row_id',
                                           'card_crop_pixel_sha256', 'visual_crop_sha256', 'identity_status',
                                           'evidence_status', 'hand_id', 'recognition_mode']}
            if source not in rows:
                message['card_provenance']['support_row'] = copy.deepcopy(source)
        else:
            message['card_binding_conflict'] = True
    return message


def assemble_messages(rows, *, require_pixel_cards=False, require_adjacency=False, pixel_sources=None,
                      join_show_description=False):
    groups = []
    for row in rows:
        text = str(row.get('raw_text') or row.get('raw_ocr_text') or row.get('text') or '')
        continuation = False
        if groups:
            prior = make_message(groups[-1], require_pixel_cards=require_pixel_cards, pixel_sources=pixel_sources)
            no_prefix = not re.match(r'^\s*(?:\[\d+:\d+\]|Dealer:|Дилер:)', text, re.I)
            visual_link = (str(row.get('row_id', row.get('stable_row_id'))) == str(groups[-1][0].get('card_row_id')) and
                           bool(groups[-1][0].get('card_evidence_id')))
            continuation = visual_link or no_prefix and (
                (observed_wrapped_continuation(groups[-1],row) and
                 (join_show_description or prior['action']!='SHOW' or
                  not re.fullmatch(RANK_DESCRIPTION,text.strip(),re.I))) or
                (len(groups[-1]) == 1 and split_small_blind(groups[-1][0], row)) or
                (prior['action'] == 'TABLE_INFO' and
                 bool(re.fullmatch(r'Waiting for .+ to', prior['normalized_text'], re.I)) and
                 bool(re.fullmatch(r'\s*reconnect\.{0,3}\s*', text, re.I))) or
                (prior['action'] == 'BLIND_LEVEL' and
                 bool(re.fullmatch(r'Limits going up: stakes', prior['normalized_text'], re.I)) and
                 bool(re.fullmatch(r'\s*\d+(?:[ \u00a0]\d{3})*/\d+(?:[ \u00a0]\d{3})*\s+ante\s+\d+(?:[ \u00a0]\d{3})*\s*', text, re.I))) or
                (prior['action'] in MONEY_ACTIONS and prior['amount'] is None and
                 bool(re.fullmatch(r'\s*\d+(?:[ \u00a0]\d{3})*(?:\s+(?:back|and is All-in|is All-in))?\s*[.!●]?\s*', text, re.I))) or
                (prior['action'] in {'ANTE', 'CALL', 'BET', 'RAISE', 'SMALL_BLIND', 'BIG_BLIND'} and
                 bool(re.fullmatch(r'\s*(?:and is All-in|is All-in|и (?:идет|идёт) ва-банк)\s*[.!]?\s*', text, re.I))) or
                (prior['action'] in MONEY_ACTIONS and
                 bool(re.search(r'\band is$', prior['normalized_text'], re.I)) and
                 bool(re.fullmatch(r'\s*All-in\s*[.!●]?\s*', text, re.I))) or
                (bool(re.fullmatch(r'.+ (?:finished the|has joined the|has left the)', prior['normalized_text'], re.I)) and
                 bool(re.fullmatch(r'\s*(?:tournament|table)[.!]?\s*', text, re.I))) or
                (prior['action'] == 'NEW_HAND' and prior['hand_number'] is None and
                 bool(re.fullmatch(r'\s*#\s*\d+\s*', text))) or
                (prior['action'] == 'UNCALLED_RETURN' and bool(re.fullmatch(r'\s*back\s*', text, re.I))) or
                (prior['action'] == 'WIN' and bool(re.fullmatch(r'\s*(?:from )?(?:the )?(?:main pot|side pot(?: #\d+)?)(?:\s+with .*)?\s*', text, re.I))) or
                (prior['action'] in {'FLOP', 'TURN', 'RIVER', 'SHOW'} and bool(card_tokens(text)) and
                 bool(re.fullmatch(r'[\s\[\],2-9TJQKA10shdc♠♥♦♣]+', text, re.I))))
            if continuation and (row.get('epoch') != groups[-1][-1].get('epoch')):
                continuation = False
            if continuation and require_adjacency:
                previous = groups[-1][-1]
                continuation = (row.get('table_id') == previous.get('table_id') and
                    row.get('session_id') == previous.get('session_id') and
                    row.get('frame_id') is not None and
                    row.get('frame_id') == previous.get('frame_id') and
                    isinstance(row.get('row_index'), int) and
                    isinstance(previous.get('row_index'), int) and
                    row['row_index'] == previous['row_index'] + 1)
        if continuation:
            groups[-1].append(row)
        else:
            groups.append([row])
    return [make_message(group, require_pixel_cards=require_pixel_cards, pixel_sources=pixel_sources) for group in groups]


def assemble_hands(rows, *, require_pixel_cards=False, require_adjacency=False, pixel_sources=None,
                   join_show_description=False):
    """Preserve source order per table/session. New numbered marker wins."""
    tables = defaultdict(list)
    for row in rows:
        tables[(str(row.get('table_id', 'UNKNOWN')), str(row.get('session_id', row.get('table_session_id', 'UNKNOWN'))))].append(row)
    hands = []
    for (table, session), source in tables.items():
        current = None
        for message in assemble_messages(source, require_pixel_cards=require_pixel_cards,
                                         require_adjacency=require_adjacency, pixel_sources=pixel_sources,
                                         join_show_description=join_show_description):
            number = message['hand_number']
            marker = message['action'] in {'HAND_HEADER', 'HAND_ID', 'NEW_HAND'}
            new = current is None or (marker and number and number != current['hand_id'])
            if marker and not number and current and current['end_observed']:
                new = True
            if new:
                if current:
                    if marker and number and not current['end_observed']:
                        current['boundary_conflicts'].append('NEW_HAND_NUMBER_BEFORE_COMPLETION')
                    current['next_boundary_observed'] = True
                    hands.append(current)
                current = {'table_id': table, 'session_id': session, 'hand_id': number,
                           'hand_epoch': len(hands), 'messages': [], 'start_observed': marker,
                           'end_observed': False, 'next_boundary_observed': False, 'provenance': 'OBSERVED_TIMELINE',
                           'boundary_conflicts': []}
            if marker and number and current['hand_id'] is None:
                current['hand_id'] = number
                current['start_observed'] = True
            current['messages'].append(message)
            if message['action'] in {'WIN', 'HAND_END'}:
                current['end_observed'] = True
        if current:
            hands.append(current)
    return hands
