"""Source-bound winning five-card displays; never board or hole-card authority."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from parser_core.poker_card_engine import CLASSES, described_class, five_rank
from parser_core.visual_cards import _verified_chat_identity, parse_cards
from parser_core.suit_backup import decide_suit

ROOT = Path(__file__).resolve().parents[1]
RULE = 'NATIVE_WIN_BEST_FIVE_ADJACENT_SOURCE_V1'
HAND_BINDING_RULE = 'OBSERVED_NUMBERED_HAND_MEMBERSHIP_V1'
NUMBER = r'\d+(?:[ \u00a0]\d{3})*'
HEAD = re.compile(r'(?:Dealer|Дилер)\s*:\s*(?P<actor>.+?) wins (?P<amount>'+NUMBER+
                  r')(?P<wrap> from(?: the| side)?)?', re.I)
PAYOUT = re.compile(r'(?:Dealer|Дилер)\s*:\s*(?P<actor>.+?) wins (?P<amount>'+NUMBER+
                  r') from (?:the )?(?P<pot>main pot|side pot(?: #\d+)?) with '
                  r'(?P<description>(?:High card|Pair|Two pairs|Three of a kind|Straight|Flush|'
                  r'Full house|Four of a kind|Straight flush)[ ,].+)', re.I)
SHA = re.compile(r'[0-9a-f]{64}')
HEADER = re.compile(r'(?:Hand\s*#|#)\s*(\d+)\s*', re.I)


def _text(row):
    return str(row.get('raw_ocr_text') or row.get('raw_text') or row.get('text') or '').strip()


def _id(row):
    return row.get('stable_row_id', row.get('row_id'))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _ref(row):
    keys = ('table_id','session_id','epoch','frame_id','row_index','video_time','row_pixel_sha256')
    return dict({k:row.get(k) for k in keys}, stable_row_id=_id(row), raw_ocr_text=_text(row))


def _binding_ref(row):
    """New membership receipts retain negative source evidence as well."""
    ref = _ref(row)
    for field in ('hand_id','hand_number','source_file','origin','status',
                  'canonical_conflict','canonical_duplicate_of','win_best_five_conflict',
                  'row_identity_status'):
        if row.get(field) is not None:
            ref[field] = copy.deepcopy(row[field])
    return ref


def _context_hand(context, rows, *, hand_bound=False):
    ref = _binding_ref if hand_bound else _ref
    if not context or len(context)<4 or context[-3:] != [ref(r) for r in rows]:
        return None
    if len({_id(r) for r in context}) != len(context) or not HEADER.fullmatch(_text(context[0])):
        return None
    number = None
    previous = None
    for row in context:
        fields = ('table_id','session_id') if hand_bound else ('table_id','session_id','epoch')
        if (any(row.get(k) != rows[0].get(k) for k in fields)
                or hand_bound and row.get('epoch') is None
                or not _id(row) or not SHA.fullmatch(row.get('row_pixel_sha256',''))):
            return None
        position = (row.get('frame_id'),row.get('row_index'))
        if (any(type(n) is not int for n in position) or
                previous is not None and position <= previous):
            return None
        previous = position
        header = HEADER.fullmatch(_text(row))
        if header:
            number = header[1]
        elif re.fullmatch(r'(?:Dealer|Дилер)\s*:\s*Starting a new hand[.!]?', _text(row), re.I):
            number = None
    return number


def _numbered_hand_binding(context, rows):
    """Reuse the existing boundary assembler; CHAT epoch is not a hand ID."""
    number = _context_hand(context, rows, hand_bound=True)
    if number is None:
        return None
    if (len({r['source_file'] for r in context if r.get('source_file') is not None}) > 1
            or any(r.get('origin') == 'UNRESOLVED_IDENTITY' or r.get('status') == 'UNRESOLVED'
                or r.get('canonical_conflict') or r.get('canonical_duplicate_of')
                or r.get('win_best_five_conflict')
                or r.get('row_identity_status') == 'UNRESOLVED'
                or any(r.get(field) is not None and str(r[field]) != number
                       for field in ('hand_id','hand_number')) for r in context)):
        return None
    from parser_core.poker_hand_assembly import assemble_hands
    hands = assemble_hands(context, require_adjacency=True)
    if (len(hands) != 1 or hands[0]['hand_id'] != number
            or not hands[0]['start_observed'] or hands[0]['boundary_conflicts']):
        return None
    members = [r for m in hands[0]['messages'] for r in m['source_rows']]
    if members != context:
        return None
    return dict(rule=HAND_BINDING_RULE, hand_id=number,
                header_refs=[copy.deepcopy(r) for r in context if HEADER.fullmatch(_text(r))],
                source_row_ids=[_id(r) for r in members])


def _description_matches(description, cards):
    # Literal rank summaries must agree with the native five, including kickers.
    counts = {'PAIR':2, 'TWO_PAIR':4, 'THREE_OF_A_KIND':3, 'FULL_HOUSE':5, 'FOUR_OF_A_KIND':4}
    kind = CLASSES[five_rank(cards)[0]]
    if described_class(description) != kind:
        return False
    compact = description.rsplit(',',1)
    if len(compact)!=2:
        return False
    summary=compact[1].strip()
    ranks_by_value='23456789TJQKA'
    def value(rank):return ranks_by_value.index('T' if rank=='10' else rank)+2
    best=five_rank(cards)
    if kind in {'STRAIGHT','STRAIGHT_FLUSH'}:
        span=re.fullmatch(r'(10|[2-9TJQKA])-(10|[2-9TJQKA])',summary)
        return bool(span and value(span[1])==best[1] and
                    value(span[2])==(14 if best[1]==5 else best[1]-4))
    if kind=='FLUSH':
        high=re.fullmatch(r'(10|[2-9TJQKA]) high',summary)
        return bool(high and value(high[1])==best[1])
    if kind=='HIGH_CARD':
        high=re.fullmatch(r'(10|[2-9TJQKA])',summary)
        return bool(high and value(high[1])==best[1])
    if kind not in counts:
        return False
    repeated_summary=re.fullmatch(r'((?:10|[2-9TJQKA])+)(?: with (10|[2-9TJQKA]) kicker)?',summary)
    if not repeated_summary:
        return False
    ranks = re.findall(r'10|[2-9TJQKA]', repeated_summary[1])
    if len(ranks)!=counts[kind]:
        return False
    values = ['T' if rank=='10' else rank for rank in ranks]
    repeated = [c[0] for c in cards if sum(x[0]==c[0] for x in cards)>1]
    kicker=repeated_summary[2]
    return (sorted(values)==sorted(repeated) and
            (kicker is None or kind=='PAIR' and value(kicker)==best[2]))


def _scores(record, prefix, config):
    best, second, margin = (record[prefix+k] for k in ('best_score','second_score','margin'))
    return (all(type(n) in (int,float) and math.isfinite(n) for n in (best,second,margin))
            and -1 <= second <= best <= 1 and math.isclose(best-second,margin,abs_tol=1e-7)
            and best>=config['best_score_threshold'] and margin>=config['margin_threshold'])


def _png(encoded, byte_sha, pixel_sha):
    data = base64.b64decode(encoded, validate=True)
    if _sha(data) != byte_sha:
        raise ValueError('PNG_HASH_CONFLICT')
    with Image.open(io.BytesIO(data)) as image:
        if image.format != 'PNG':
            raise ValueError('NOT_PNG')
        image.load()
        array = np.asarray(image)
    if _sha(array.tobytes()) != pixel_sha:
        raise ValueError('PIXEL_HASH_CONFLICT')
    return array


def _frozen(reading):
    """Verify saved recognizer provenance without recognizing the image again."""
    hybrid = reading['hybrid']
    response = reading['raw_pixel_response']
    specs = [('configs/card_hybrid_local.json', hybrid['template_config_sha256']),
             ('configs/rank_ncc_production.json', response['rank_config_sha256']),
             ('configs/suit_backup_local.json', response['suit_config_sha256'])]
    configs = []
    for filename, digest in specs:
        data = (ROOT / filename).read_bytes()
        if _sha(data) != digest:
            raise ValueError('FROZEN_CONFIG_CONFLICT')
        configs.append(json.loads(data))
    suit, rank, backup = configs
    if (rank.get('mode') != 'PIXEL_ONLY' or rank.get('development_only') is not False
            or rank.get('production_authorized') is not True or backup.get('mode') != 'BACKUP'
            or hybrid['rank_ncc_config_sha256'] != response['rank_config_sha256']):
        raise ValueError('FROZEN_MODE_CONFLICT')
    manifest = hybrid['template_manifest_sha256']
    if any(c['train_manifest_sha256'] != manifest for c in configs):
        raise ValueError('TRAIN_MANIFEST_CONFLICT')
    if (_sha(Path(suit['train_manifest']).read_bytes()) != manifest
            or hybrid['template_selection_sha256'] != suit['selection_sha256']
            or _sha(Path(suit['selection_file']).read_bytes()) != suit['selection_sha256']):
        raise ValueError('FROZEN_TEMPLATE_CONFLICT')
    for filename, digest in [('parser_core/rank_ncc.py',rank['implementation_sha256']),
                             ('experiments/suit_template_experiment.py',backup['implementation_sha256'])]:
        if _sha((ROOT/filename).read_bytes())!=digest:
            raise ValueError('FROZEN_IMPLEMENTATION_CONFLICT')
    train = [json.loads(line) for line in Path(suit['train_manifest']).read_text('utf-8').splitlines()]
    return response['suit_config_sha256'], _sha((ROOT / 'models/board_glyphs_v1.json').read_bytes()), rank, backup, train


def prove_win_tail(rows, reading, context, *, hand_context=False):
    """Reproduce a distinct best-five display bound to three physical rows."""
    try:
        if not isinstance(reading, dict) or len(rows) != 3:
            return None
        head, tail, icons = rows
        win = HEAD.fullmatch(_text(head))
        payout = PAYOUT.fullmatch(_text(head)+' '+_text(tail))
        if not win or not payout or head.get('event_type') != 'WIN':
            return None
        if (len({_id(r) for r in rows}) != 3 or any(not _id(r) for r in rows)
                or any(r.get('origin') == 'UNRESOLVED_IDENTITY' or r.get('status') == 'UNRESOLVED'
                       or r.get('canonical_conflict') or r.get('canonical_duplicate_of')
                       or r.get('win_best_five_conflict') for r in rows)
                or any(not isinstance(r.get('row_pixel_sha256'), str)
                       or not SHA.fullmatch(r['row_pixel_sha256']) for r in rows)):
            return None
        for row in rows:
            if any(row.get(k) != head.get(k) or row.get(k) is None
                   for k in ('table_id', 'session_id', 'epoch', 'frame_id', 'video_time')):
                return None
        if (not all(type(r.get('row_index')) is int for r in rows)
                or [r['row_index'] for r in rows] != list(range(head['row_index'], head['row_index']+3))):
            return None
        if (not _verified_chat_identity(reading) or reading.get('recognition_mode') != 'PIXEL_ONLY'
                or reading.get('hybrid_authoritative_rejection')
                or not hand_context and reading.get('hand_id') is None
                or reading.get('reason') != 'NO_NEARBY_CARD_EVENT' or reading.get('status') != 'UNRESOLVED'
                or reading.get('crop_type') != 'CHAT_ROW_CARDS'
                or reading.get('table_id') not in {head['table_id'], int(str(head['table_id']).lstrip('T'))}
                or reading.get('table_session_id') != head['session_id']
                or reading.get('frame_id') != head['frame_id'] or reading.get('epoch') != head['epoch']
                or reading.get('source_span_index') != icons['row_index']
                or reading.get('row_id') != _id(icons)
                or reading.get('row_pixel_sha256') != icons['row_pixel_sha256']):
            return None
        binding = _numbered_hand_binding(context, rows) if hand_context else None
        if hand_context and binding is None:
            return None
        hand_id = binding['hand_id'] if binding else _context_hand(context, rows)
        if (hand_id is None or (reading.get('hand_id') not in (None,hand_id) if binding
                               else hand_id != reading.get('hand_id'))
                or any(r.get('hand_id') not in (None, hand_id) for r in rows)):
            return None
        if (type(reading.get('source_pts')) is not int or reading['source_pts'] < 0 or
                any(type(reading.get(k)) is not int or reading[k] <= 0
                    for k in ('time_base_num','time_base_den')) or
                any(type(t) not in (int,float) or not math.isfinite(t)
                    for t in (reading['video_time'],head['video_time']))):
            return None
        source_time = Fraction(reading['source_pts']*reading['time_base_num'], reading['time_base_den'])
        # The native producer serializes video_time as a float. Compare its
        # exact round trip, while retaining rational PTS equality where present.
        # No epsilon may admit a different source instant.
        if (reading['video_time'] != float(source_time) or
                head['video_time'] != float(source_time)):
            return None
        for row in rows:
            if row.get('source_pts') is not None or row.get('time_base') is not None:
                if Fraction(row['source_pts']*row['time_base'][0], row['time_base'][1]) != source_time:
                    return None
        hybrid = reading['hybrid']
        cards = reading['cards']
        if (not isinstance(cards, list) or parse_cards(' '.join(cards), 5) != cards
                or hybrid.get('cards') != cards or hybrid.get('reason') is not None
                or hybrid.get('rank_mode') != 'PIXEL_ONLY' or hybrid.get('development_only') is not False
                or hybrid.get('primary_vl_rank_conflict')
                or hybrid.get('card_reader') != 'pixel_rank_ncc_template_suit_v1'
                or reading.get('card_reader') != hybrid['card_reader']
                or len(hybrid.get('tiles', [])) != 5):
            return None
        hand_class = CLASSES[five_rank(cards)[0]]
        if not _description_matches(payout['description'], cards):
            return None
        gray = _png(reading['crop_png_base64'], reading['crop_sha256'], reading['row_pixel_sha256'])
        if gray.ndim != 2 or gray.shape[0] != 16:
            return None
        from parser_core.board_tiles import verify_card_geometry_resolution
        if not verify_card_geometry_resolution(gray, hybrid):
            return None
        suit_config, glyph_config, rank_config, backup, train = _frozen(reading)
        boxes = []
        for index, (tile, card) in enumerate(zip(hybrid['tiles'], cards)):
            x0, y0, x1, y1 = tile['box_xyxy_exclusive']
            if (not all(type(n) is int for n in (x0,y0,x1,y1)) or y0 != 0 or y1 != 16
                    or not 0 <= x0 < x1 <= gray.shape[1] or not 18 <= x1-x0 <= 22
                    or boxes and x0 < boxes[-1][2] or tile.get('card_index') != index
                    or tile.get('card') != card or tile.get('final_rank') != card[0]
                    or tile.get('suit') != card[1] or tile.get('suit_source') not in {'agreement','template_backup'}
                    or tile.get('suit_backup_config_sha256') != suit_config
                    or tile.get('legacy_rank_template_sha256') != glyph_config
                    or tile.get('rank_source') not in {'EXACT_VERIFIED_TRAIN_RANK_GLYPH','FROZEN_RANK_NCC_PIXEL_FALLBACK'}
                    or tile.get('rank_status') not in {None,'PIXEL_RANK_FALLBACK'}
                    or tile.get('template_confident') is not True or not _scores(tile,'template_',backup)
                    or tile.get('suit_backup_mode') != 'BACKUP'
                    or tile.get('suit_status') not in {None,'PRIMARY_UNCALIBRATED_DISAGREEMENT'}):
                return None
            primary = tile['primary_suit'] if tile['primary_suit'] in 'shdc' else None
            decision = decide_suit(primary, tile.get('primary_confidence')=='BOARD_TILE_ACCEPTED', tile)
            if (any(tile.get(k) != decision.get(k) for k in ('suit','suit_source','suit_status'))
                    or tile.get('raw_suit2_suit') in 'shdc' and tile['raw_suit2_suit']!=tile['template_suit']):
                return None
            if tile['rank_source'] == 'EXACT_VERIFIED_TRAIN_RANK_GLYPH':
                indices = tile.get('rank_exact_train_indices')
                if (tile.get('rank_exact_candidate') != card[0] or not isinstance(indices,list) or not indices
                        or any(type(n) is not int or not 0<=n<len(train) for n in indices)
                        or {train[n]['ground_truth'][0] for n in indices}!={card[0]}):
                    return None
            else:
                ncc = tile['rank_ncc']
                if (ncc.get('rank_accepted') is not True or ncc.get('rank_result') != card[0]
                        or not _scores(ncc,'rank_',rank_config)
                        or ncc.get('rank_template_manifest_sha256')!=hybrid['template_manifest_sha256']):
                    return None
            pixels = _png(tile['tile_png_base64'], tile['tile_sha256'], tile['tile_pixel_sha256'])
            if not np.array_equal(pixels, np.repeat(gray[y0:y1,x0:x1,None],3,axis=2)):
                return None
            boxes.append([x0,y0,x1,y1])
        if hybrid['detected_spans_inclusive'] != [[b[0], b[2]-1] for b in boxes]:
            return None
        refs = [_ref(r) for r in rows]
        proof = dict(rule=RULE, evidence_kind='WIN_BEST_FIVE', actor=win['actor'],
                    amount=win['amount'].replace(' ','').replace('\u00a0',''),
                    pot=payout['pot'], hand_class=hand_class, cards=list(cards),
                    hand_id=hand_id, context_rows=copy.deepcopy(context), source_refs=refs, reading=copy.deepcopy(reading))
        if binding:
            proof['hand_binding'] = binding
        return proof
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError, OSError, AttributeError, OverflowError):
        return None


def current_win_context(rows, index, *, across_epochs=False):
    """Actual source window in the consumer, independent of the retained proof."""
    head=rows[index]
    binding=(head.get('win_best_five_proof') or {}).get('hand_binding')
    across_epochs = across_epochs or isinstance(binding,dict) and binding.get('rule')==HAND_BINDING_RULE
    fields=('table_id','session_id') if across_epochs else ('table_id','session_id','epoch')
    key=tuple(head.get(k) for k in fields)
    context=[]
    active=False
    for n,row in enumerate(rows[:index+3]):
        if tuple(row.get(k) for k in fields)!=key:
            continue
        if n<=index and HEADER.fullmatch(_text(row)):
            context=[]
            active=True
        if active:context.append((_binding_ref if across_epochs else _ref)(row))
    return context


def verified_win_tail(rows, *, context=None):
    proof = rows[0].get('win_best_five_proof') if rows else None
    if not isinstance(proof, dict) or proof.get('rule') != RULE:
        return None
    reproduced = prove_win_tail(rows, proof.get('reading', {}),
                               proof.get('context_rows') if context is None else context,
                               hand_context='hand_binding' in proof)
    return proof if reproduced == proof and reproduced is not None else None


def project_win_tails(timeline, readings):
    """Attach proof only; preserve every raw row and the monetary event type."""
    physical = {}
    for row in timeline:
        key = (row.get('table_id'),row.get('session_id'),row.get('epoch'),row.get('frame_id'),row.get('row_index'))
        physical.setdefault(key, []).append(row)
    count = 0
    contexts = {}
    for position,head in enumerate(timeline):
        head.pop('win_best_five_proof',None)
        source_key = (head.get('table_id'),head.get('session_id'),head.get('epoch'))
        if HEADER.fullmatch(_text(head)):
            contexts[source_key] = []
        if source_key in contexts:
            contexts[source_key].append(_ref(head))
        if head.get('event_type') != 'WIN' or not HEAD.fullmatch(_text(head)):
            continue
        prefix = (head.get('table_id'),head.get('session_id'),head.get('epoch'),head.get('frame_id'))
        index = head.get('row_index')
        if type(index) is not int:
            continue
        groups = [physical.get((*prefix,index+i), []) for i in range(3)]
        if any(len(group) != 1 for group in groups):
            continue
        rows = [group[0] for group in groups]
        key = f"{int(str(head['table_id']).lstrip('T'))}:{head['frame_id']}:{index+2}"
        context = contexts.get(source_key, []) + [_ref(r) for r in rows[1:]]
        proof = prove_win_tail(rows, readings.get(key, {}), context)
        if proof is None:
            # The current numbered source hand can span physical CHAT epochs.
            # This separate receipt changes binding only, never the raw reading.
            context = current_win_context(timeline,position,across_epochs=True)
            proof = prove_win_tail(rows, readings.get(key, {}), context, hand_context=True)
        if proof:
            head['win_best_five_proof'] = proof
            count += 1
    return count
