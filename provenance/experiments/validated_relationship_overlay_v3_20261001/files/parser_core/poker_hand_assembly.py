"""Experimental, source-preserving hand assembly. Never used by production."""
from __future__ import annotations

import copy
import re
from collections import defaultdict
from parser_core.normalize_chat import normalize
from parser_core.poker_card_engine import described_class

VERSION = 'experimental-assembly-4-source-continuations'
MONEY = re.compile(r'(?<!\d)\d+(?:[ \u00a0]\d{3})*(?!\d)')
CARDS = re.compile(r'(?<![A-Za-z0-9])([2-9TJQKA]|10)([shdc♠♥♦♣])(?![A-Za-z0-9])', re.I)
SUITS = {'♠': 's', '♥': 'h', '♦': 'd', '♣': 'c'}
BETTING = {'FOLD', 'CHECK', 'CALL', 'BET', 'RAISE'}
MONEY_ACTIONS = {'ANTE', 'SMALL_BLIND', 'BIG_BLIND', 'CALL', 'BET', 'RAISE', 'WIN', 'UNCALLED_RETURN'}


def card_tokens(text):
    return [('T' if r == '10' else r.upper()) + SUITS.get(s, s.lower()) for r, s in CARDS.findall(text)]


def parse_text(text):
    clean = re.sub(r'^\s*\[\d{1,2}:\d{2}\]\s*', '', text).strip()
    clean = re.sub(r'^(?:Dealer|Дилер)\s*:\s*', '', clean, flags=re.I)
    c = normalize([text])
    event = c.event_type
    actor, amount, hand_id = c.actor, c.amount_decimal, c.hand_number
    english = [
        (r'posts (?:the )?ante', 'ANTE'), (r'posts (?:the )?small blind', 'SMALL_BLIND'),
        (r'posts (?:the )?big blind', 'BIG_BLIND'), (r'raises to', 'RAISE'),
        (r'calls', 'CALL'), (r'checks', 'CHECK'), (r'folds', 'FOLD'),
        (r'bets', 'BET'), (r'wins', 'WIN'), (r'gets uncalled bet', 'UNCALLED_RETURN'),
        (r'возвращается|получает обратно|получает неуравненную ставку', 'UNCALLED_RETURN'),
        (r'does not show|doesn.t show|не показывает', 'NO_SHOW'), (r'shows', 'SHOW')]
    for pattern, kind in english:
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
    if re.fullmatch(r'(?:Waiting for .+ to(?: reconnect\.{0,3})?|reconnect\.{0,3}|.+ is reconnected[.!]?)', clean, re.I):
        event, actor, amount = 'TABLE_INFO', None, None
    if re.fullmatch(r'Limits going up: stakes(?:\s+\d+(?:[ \u00a0]\d{3})*/\d+(?:[ \u00a0]\d{3})*\s+ante\s+\d+(?:[ \u00a0]\d{3})*)?[.!]?', clean, re.I):
        event, actor, amount = 'BLIND_LEVEL', None, None
    return {'action': event, 'player': actor, 'amount': amount, 'hand_number': hand_id,
            'all_in': bool(re.search(r'all[ -]?in|олл.?ин|ва.?банк', clean, re.I)),
            'cards': card_tokens(clean), 'showdown_class': described_class(clean) if event in {'SHOW', 'WIN'} else None,
            'normalized_text': clean}


def make_message(rows):
    text = ' '.join(str(r.get('raw_text') or r.get('raw_ocr_text') or r.get('text') or '') for r in rows)
    parsed = parse_text(text)
    message = dict(parsed, raw_text=text, source_rows=copy.deepcopy(rows),
                source_row_ids=[str(r.get('row_id', r.get('stable_row_id', r.get('physical_row_id', 'MISSING_ROW_ID')))) for r in rows],
                frame_id=rows[0].get('frame_id'), timestamp_ms=rows[0].get('timestamp_ms'),
                source_type='DERIVED', field_sources={'amount': 'OBSERVED', 'player': 'OBSERVED', 'cards': 'OBSERVED'},
                assembly='EXPLICIT_CONTINUATION' if len(rows) > 1 else 'SINGLE_ROW')
    trigger = rows[0]
    if trigger.get('cards') and parsed['action'] in {'FLOP', 'TURN', 'RIVER', 'SHOW'}:
        source = next((r for r in rows if str(r.get('row_id', r.get('stable_row_id'))) == str(trigger.get('card_row_id'))), None)
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
                    source.get('epoch') == trigger.get('epoch'))
        if verified:
            message['cards'] = list(trigger['cards'])
            message['card_provenance'] = {k: trigger.get(k) for k in ['card_evidence_id', 'card_row_id',
                                           'card_crop_pixel_sha256', 'visual_crop_sha256', 'identity_status',
                                           'evidence_status', 'hand_id', 'recognition_mode']}
        else:
            message['card_binding_conflict'] = True
    return message


def assemble_messages(rows):
    groups = []
    for row in rows:
        text = str(row.get('raw_text') or row.get('raw_ocr_text') or row.get('text') or '')
        continuation = False
        if groups:
            prior = make_message(groups[-1])
            no_prefix = not re.match(r'^\s*(?:\[\d+:\d+\]|Dealer:|Дилер:)', text, re.I)
            visual_link = (str(row.get('row_id', row.get('stable_row_id'))) == str(groups[-1][0].get('card_row_id')) and
                           bool(groups[-1][0].get('card_evidence_id')))
            continuation = visual_link or no_prefix and (
                (prior['action'] == 'TABLE_INFO' and
                 bool(re.fullmatch(r'Waiting for .+ to', prior['normalized_text'], re.I)) and
                 bool(re.fullmatch(r'\s*reconnect\.{0,3}\s*', text, re.I))) or
                (prior['action'] == 'BLIND_LEVEL' and
                 bool(re.fullmatch(r'Limits going up: stakes', prior['normalized_text'], re.I)) and
                 bool(re.fullmatch(r'\s*\d+(?:[ \u00a0]\d{3})*/\d+(?:[ \u00a0]\d{3})*\s+ante\s+\d+(?:[ \u00a0]\d{3})*\s*', text, re.I))) or
                (prior['action'] in MONEY_ACTIONS and prior['amount'] is None and
                 bool(re.fullmatch(r'\s*\d+(?:[ \u00a0]\d{3})*(?:\s+(?:back|and is All-in|is All-in))?\s*[.!]?\s*', text, re.I))) or
                (prior['action'] in {'CALL', 'BET', 'RAISE', 'SMALL_BLIND', 'BIG_BLIND'} and
                 bool(re.fullmatch(r'\s*(?:and is All-in|is All-in|и (?:идет|идёт) ва-банк)\s*[.!]?\s*', text, re.I))) or
                (prior['action'] in MONEY_ACTIONS and
                 bool(re.search(r'\band is$', prior['normalized_text'], re.I)) and
                 bool(re.fullmatch(r'\s*All-in\s*[.!●]?\s*', text, re.I))) or
                (bool(re.fullmatch(r'.+ (?:finished the|has joined the|has left the)', prior['normalized_text'], re.I)) and
                 bool(re.fullmatch(r'\s*(?:tournament|table)[.!]?\s*', text, re.I))) or
                (prior['action'] == 'NEW_HAND' and prior['hand_number'] is None and
                 bool(re.fullmatch(r'\s*#\s*\d+\s*', text))) or
                (prior['action'] == 'UNCALLED_RETURN' and bool(re.fullmatch(r'\s*back\s*', text, re.I))) or
                (prior['action'] == 'WIN' and bool(re.fullmatch(r'\s*(?:the main pot|main pot|the side pot)(?:\s+with .*)?\s*', text, re.I))) or
                (prior['action'] in {'FLOP', 'TURN', 'RIVER', 'SHOW'} and bool(card_tokens(text)) and
                 bool(re.fullmatch(r'[\s\[\],2-9TJQKA10shdc♠♥♦♣]+', text, re.I))))
            if continuation and (row.get('epoch') != groups[-1][-1].get('epoch')):
                continuation = False
        if continuation:
            groups[-1].append(row)
        else:
            groups.append([row])
    return [make_message(group) for group in groups]


def assemble_hands(rows):
    """Preserve source order per table/session. New numbered marker wins."""
    tables = defaultdict(list)
    for row in rows:
        tables[(str(row.get('table_id', 'UNKNOWN')), str(row.get('session_id', row.get('table_session_id', 'UNKNOWN'))))].append(row)
    hands = []
    for (table, session), source in tables.items():
        current = None
        for message in assemble_messages(source):
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
