"""Source-bound finite card-set proof; never changes raw recognizer output."""
import copy
import itertools
import math
import re
from fractions import Fraction

import numpy as np

from parser_core.poker_card_engine import RANKS, CLASSES, best_rank, five_rank
from parser_core.suit_backup import decide_suit
from parser_core.board_tiles import verify_card_geometry_resolution
from parser_core.win_tail_evidence import (_frozen, _png, _scores, _text, _ref, _id,
    _verified_chat_identity, _description_matches, PAYOUT, HEADER, SHA, prove_win_tail)

RULE = 'NATIVE_COMPONENTS_AND_INDEPENDENT_CARD_SET_UNIQUE_V1'
SHOW = re.compile(r'(?:Dealer|Дилер)\s*:\s*(.+?) shows (.+)', re.I)
RIVER = re.compile(r'(?:Dealer|Дилер)\s*:\s*(?:Dealing (?:the )?)?River', re.I)


def components(reading, icons, count, *, complete=False):
    """Reproduce accepted component receipts; unaccepted component gets its full domain."""
    if (not _verified_chat_identity(reading) or reading.get('recognition_mode') != 'PIXEL_ONLY'
            or complete and reading.get('hybrid_authoritative_rejection') or reading.get('crop_type') != 'CHAT_ROW_CARDS'
            or reading.get('row_id') != _id(icons) or reading.get('row_pixel_sha256') != icons['row_pixel_sha256']
            or reading.get('table_id') != int(str(icons['table_id']).lstrip('T'))
            or reading.get('table_session_id') != icons['session_id']
            or reading.get('frame_id') != icons['frame_id'] or reading.get('epoch') != icons['epoch']
            or reading.get('source_span_index') != icons['row_index']):
        raise ValueError('CARD_SOURCE_IDENTITY_UNKNOWN')
    if (type(reading.get('source_pts')) is not int or reading['source_pts'] < 0
            or any(type(reading.get(k)) is not int or reading[k] <= 0 for k in ('time_base_num', 'time_base_den'))):
        raise ValueError('CARD_SOURCE_TIME_UNKNOWN')
    stamp = Fraction(reading['source_pts'] * reading['time_base_num'], reading['time_base_den'])
    if any(type(r.get('video_time')) not in (int, float) or not math.isfinite(r['video_time'])
           or r['video_time'] != float(stamp) for r in (reading, icons)):
        raise ValueError('CARD_SOURCE_TIME_CONFLICT')
    hybrid = reading['hybrid']
    response = reading['raw_pixel_response']
    if (response.get('status') != 'OK' or response.get('recognizer') != 'FROZEN_NATIVE_PIXEL_CARD_READER'
            or not response.get('request_id') or hybrid.get('rank_mode') != 'PIXEL_ONLY'
            or hybrid.get('development_only') is not False or hybrid.get('primary_vl_rank_conflict')
            or hybrid.get('card_reader') != 'pixel_rank_ncc_template_suit_v1'
            or reading.get('card_reader') not in ({hybrid['card_reader']} if complete else {None, hybrid['card_reader']})
            or len(hybrid.get('tiles', [])) != count):
        raise ValueError('CARD_READER_CONTRACT_CONFLICT')
    gray = _png(reading['crop_png_base64'], reading['crop_sha256'], reading['row_pixel_sha256'])
    if gray.ndim != 2 or gray.shape[0] != 16:
        raise ValueError('CARD_ROW_SHAPE_UNKNOWN')
    if not verify_card_geometry_resolution(gray, hybrid):
        raise ValueError('CARD_GEOMETRY_RESOLUTION_NOT_REPRODUCIBLE')
    suit_config, glyph_config, rank_config, backup, train = _frozen(reading)
    boxes, domains, unknown = [], [], []
    for index, tile in enumerate(hybrid['tiles']):
        x, y, X, Y = tile['box_xyxy_exclusive']
        if (not all(type(n) is int for n in (x, y, X, Y)) or y != 0 or Y != 16
                or not 0 <= x < X <= gray.shape[1] or not 18 <= X-x <= 22
                or boxes and x < boxes[-1][2] or tile.get('card_index') != index
                or tile.get('legacy_rank_template_sha256') != glyph_config
                or tile.get('suit_backup_config_sha256') != suit_config
                or tile.get('suit_backup_mode') != 'BACKUP'):
            raise ValueError('CARD_TILE_PROVENANCE_CONFLICT')
        pixels = _png(tile['tile_png_base64'], tile['tile_sha256'], tile['tile_pixel_sha256'])
        if not np.array_equal(pixels, np.repeat(gray[y:Y, x:X, None], 3, axis=2)):
            raise ValueError('CARD_TILE_PIXELS_CONFLICT')
        rank = tile['final_rank']
        if rank in RANKS and len(rank) == 1:
            if tile['rank_source'] == 'EXACT_VERIFIED_TRAIN_RANK_GLYPH':
                indices = tile.get('rank_exact_train_indices')
                if (tile.get('rank_status') is not None or tile.get('rank_exact_candidate') != rank
                        or not isinstance(indices, list) or not indices
                        or any(type(n) is not int or not 0 <= n < len(train) for n in indices)
                        or {train[n]['ground_truth'][0] for n in indices} != {rank}):
                    raise ValueError('EXACT_RANK_RECEIPT_CONFLICT')
            elif tile['rank_source'] == 'FROZEN_RANK_NCC_PIXEL_FALLBACK':
                ncc = tile['rank_ncc']
                if (tile.get('rank_status') != 'PIXEL_RANK_FALLBACK' or ncc.get('rank_accepted') is not True
                        or ncc.get('rank_result') != rank or not _scores(ncc, 'rank_', rank_config)
                        or ncc.get('rank_template_manifest_sha256') != hybrid['template_manifest_sha256']):
                    raise ValueError('NCC_RANK_RECEIPT_CONFLICT')
            else:
                raise ValueError('ACCEPTED_RANK_SOURCE_UNKNOWN')
            ranks = [rank]
        elif (rank == 'UNRESOLVED' and tile['rank_source'] == 'UNRESOLVED'
              and tile.get('rank_exact_candidate') == 'UNRESOLVED'
              and tile.get('rank_ncc', {}).get('rank_accepted') is False):
            ranks = list(RANKS)
            unknown.append([index, 'rank'])
        else:
            raise ValueError('RANK_CONFLICT')
        primary = tile['primary_suit'] if tile['primary_suit'] in 'shdc' else None
        decision = decide_suit(primary, tile.get('primary_confidence') == 'BOARD_TILE_ACCEPTED', tile)
        # CardHybrid's frozen post-decision veto keeps two uncalibrated channels
        # unresolved even when their labels happen to agree.
        if tile.get('primary_confidence') != 'BOARD_TILE_ACCEPTED' and tile.get('template_confident') is False:
            decision = dict(suit='UNRESOLVED', suit_source='UNRESOLVED',
                            suit_status=tile.get('template_reason') or 'TEMPLATE_LOW_CONFIDENCE')
        if any(tile.get(k) != decision.get(k) for k in ('suit', 'suit_source', 'suit_status')):
            raise ValueError('SUIT_DECISION_CONFLICT')
        suit = tile['suit']
        if suit in 'shdc' and len(suit) == 1:
            if (tile.get('template_confident') is not True or not _scores(tile, 'template_', backup)
                    or tile.get('suit_source') not in {'agreement', 'template_backup'}
                    or tile.get('suit_status') not in {None, 'PRIMARY_UNCALIBRATED_DISAGREEMENT'}
                    or tile.get('raw_suit2_suit') in 'shdc' and tile['raw_suit2_suit'] != tile['template_suit']):
                raise ValueError('ACCEPTED_SUIT_CONFLICT')
            suits = [suit]
        elif (suit == 'UNRESOLVED' and tile.get('suit_source') == 'UNRESOLVED'
              and tile.get('suit_status') == 'TEMPLATE_LOW_CONFIDENCE' and tile.get('template_confident') is False
              and not _scores(tile, 'template_', backup)):
            suits = list('shdc')
            unknown.append([index, 'suit'])
        else:
            raise ValueError('SUIT_CONFLICT')
        if tile['card'] != (rank + suit if len(ranks) == len(suits) == 1 else 'UNRESOLVED'):
            raise ValueError('CARD_COMPONENT_CONFLICT')
        domains.append([r + s for r in ranks for s in suits])
        boxes.append([x, y, X, Y])
    if hybrid['detected_spans_inclusive'] != [[b[0], b[2]-1] for b in boxes]:
        raise ValueError('CARD_SPAN_SET_CONFLICT')
    if complete:
        cards = [d[0] for d in domains]
        if (unknown or reading.get('status') != 'RAW_UNREVIEWED' or reading.get('reason') is not None
                or hybrid.get('reason') is not None or reading.get('cards') != cards or hybrid.get('cards') != cards
                or len(set(cards)) != len(cards)):
            raise ValueError('SUPPORT_CARDS_NOT_INDEPENDENT_COMPLETE_PIXELS')
    elif (len(unknown) != 1 or reading.get('status') != 'UNRESOLVED'
          or reading.get('reason') not in {'SUIT_UNRESOLVED', 'PIXEL_RANK_UNRESOLVED'}
          or reading.get('cards') not in (None, []) or hybrid.get('cards') not in (None, [])):
        raise ValueError('TARGET_SINGLE_UNKNOWN_COMPONENT_REQUIRED')
    return domains, unknown


def group(source, context, count, *, complete=False):
    rows, reading = source['rows'], source['reading']
    if len(rows) not in (2, 3):
        raise ValueError('SOURCE_GROUP_SIZE_UNKNOWN')
    first = rows[0]
    for n, row in enumerate(rows):
        if (any(row.get(k) != first.get(k) or row.get(k) is None for k in
                ('table_id', 'session_id', 'epoch', 'frame_id', 'video_time'))
                or type(row.get('row_index')) is not int or row['row_index'] != first['row_index'] + n
                or row.get('origin') == 'UNRESOLVED_IDENTITY' or row.get('status') in {'ERROR', 'UNRESOLVED'}
                or row.get('canonical_conflict') or row.get('canonical_duplicate_of')
                or _ref(row) not in context or row.get('unique_card_set_proof')):
            raise ValueError('SOURCE_GROUP_CONTEXT_CONFLICT')
        if row.get('source_pts') is not None or row.get('time_base') is not None:
            if Fraction(row['source_pts'] * row['time_base'][0], row['time_base'][1]) != Fraction(
                    reading['source_pts'] * reading['time_base_num'], reading['time_base_den']):
                raise ValueError('SOURCE_GROUP_PTS_CONFLICT')
    domains, unknown = components(reading, rows[-1], count, complete=complete)
    if complete and (first.get('cards') != reading['cards'] or first.get('card_row_id') != _id(rows[-1])
            or first.get('identity_status') != 'CHAT_ROW_SHA_ROUNDTRIP_VERIFIED'
            or first.get('evidence_status') != 'RAW_UNREVIEWED'
            or first.get('card_crop_pixel_sha256') != rows[-1]['row_pixel_sha256']
            or first.get('raw_pixel_response') != reading['raw_pixel_response']):
        raise ValueError('SUPPORT_PROJECTION_CONFLICT')
    return domains, unknown


def prove(case):
    try:
        context = case['context_rows']
        if not context or not HEADER.fullmatch(_text(context[0])):
            raise ValueError('POSITIVE_CONTEXT_HEADER_UNKNOWN')
        hand = HEADER.fullmatch(_text(context[0]))[1]
        key = tuple(context[0].get(k) for k in ('table_id', 'session_id', 'epoch'))
        seen, last = set(), None
        for row in context:
            position = (row['frame_id'], row['row_index'])
            header = HEADER.fullmatch(_text(row))
            if (tuple(row.get(k) for k in ('table_id', 'session_id', 'epoch')) != key
                    or _id(row) in seen or not _id(row) or not SHA.fullmatch(row['row_pixel_sha256'])
                    or any(type(n) is not int for n in position) or last is not None and position <= last
                    or header and header[1] != hand or 'Starting a new hand' in _text(row)):
                raise ValueError('CONTEXT_HAND_BOUNDARY_CONFLICT')
            seen.add(_id(row)); last = position
        role = case['target_role']
        target = case['target']
        board = case['board']
        if len(board['rows']) != 2 or not RIVER.fullmatch(_text(board['rows'][0])):
            raise ValueError('FINAL_BOARD_LITERAL_SOURCE_UNKNOWN')
        if any(board['reading'].get(k) != v for k, v in dict(type='BOARD', street='RIVER', player=None).items()):
            raise ValueError('BOARD_ASSIGNED_ROLE_CONFLICT')
        bdomains, _ = group(board, context, 5, complete=True)
        board_cards = [d[0] for d in bdomains]
        domains, unknown = group(target, context, 5 if role == 'WIN_BEST_FIVE' else 2)
        sources = [target, board]
        if role == 'WIN_BEST_FIVE':
            payout = PAYOUT.fullmatch(' '.join(_text(r) for r in target['rows'][:2]))
            hole = case['hole']
            show = SHOW.fullmatch(' '.join(_text(r) for r in hole['rows'][:-1]))
            if not payout or not show or payout['actor'] != show[1] or len(target['rows']) != 3:
                raise ValueError('WIN_SHOW_ACTOR_CONFLICT')
            if (any(hole['reading'].get(k) != v for k, v in dict(type='SHOWDOWN', street=None, player=show[1]).items())
                    or any(target['reading'].get(k) is not None for k in ('type', 'street', 'player'))):
                raise ValueError('WIN_SHOW_ASSIGNED_ROLE_CONFLICT')
            hdomains, _ = group(hole, context, 2, complete=True)
            hole_cards = [d[0] for d in hdomains]
            actor = payout['actor']; sources.append(hole)
            candidate_filter = lambda cards: (len(set(board_cards + hole_cards)) == 7
                and set(cards) <= set(board_cards + hole_cards)
                and five_rank(cards) == best_rank(board_cards + hole_cards)
                and _description_matches(payout['description'], cards))
        elif role == 'SHOW_HOLE_CARDS':
            show = SHOW.fullmatch(' '.join(_text(r) for r in target['rows'][:-1]))
            winning = case['winning']
            wrows = winning['rows']
            # Native WIN source is verified independently and cannot depend on
            # this target's derived components or a derived support card.
            refs = [_ref(r) for r in wrows]
            end = context.index(refs[-1])
            win = prove_win_tail(wrows, winning['reading'], context[:end+1])
            if not show or not win or show[1] != win['actor']:
                raise ValueError('SHOW_WIN_ACTOR_OR_PROOF_UNKNOWN')
            if (any(target['reading'].get(k) != v for k, v in dict(type='SHOWDOWN', street=None, player=show[1]).items())
                    or any(winning['reading'].get(k) is not None for k in ('type', 'street', 'player'))):
                raise ValueError('SHOW_WIN_ASSIGNED_ROLE_CONFLICT')
            actor = show[1]; sources.append(winning)
            candidate_filter = lambda cards: (len(set(board_cards + cards)) == 7
                and set(win['cards']) <= set(board_cards + cards)
                and five_rank(win['cards']) == best_rank(board_cards + cards)
                and _description_matches(show[2], win['cards']))
        else:
            raise ValueError('TARGET_ROLE_UNSUPPORTED')
        if any(s['reading'].get('hand_id') != hand for s in sources):
            raise ValueError('READING_HAND_CONFLICT')
        for extra in case.get('additional_supports', []):
            kind, src = extra['role'], extra['source']
            if kind == 'BOARD':
                ds, _ = group(src, context, 5, complete=True)
                valid = (RIVER.fullmatch(_text(src['rows'][0])) and [d[0] for d in ds] == board_cards
                         and all(src['reading'].get(k) == v for k,v in dict(type='BOARD',street='RIVER',player=None).items()))
            elif kind == 'SHOW' and role == 'WIN_BEST_FIVE':
                ds, _ = group(src, context, 2, complete=True)
                literal = SHOW.fullmatch(' '.join(_text(r) for r in src['rows'][:-1]))
                valid = (literal and literal[1] == actor and set(d[0] for d in ds) == set(hole_cards)
                         and all(src['reading'].get(k) == v for k,v in dict(type='SHOWDOWN',street=None,player=actor).items()))
            elif kind == 'WIN' and role == 'SHOW_HOLE_CARDS':
                end = context.index(_ref(src['rows'][-1]))
                extra_win = prove_win_tail(src['rows'],src['reading'],context[:end+1])
                valid = (extra_win and extra_win['actor'] == actor and set(extra_win['cards']) == set(win['cards'])
                         and all(src['reading'].get(k) is None for k in ('type','street','player')))
            else:
                valid = False
            if not valid or src['reading'].get('hand_id') != hand:
                raise ValueError('ADDITIONAL_SUPPORT_CONFLICT')
            sources.append(src)
        ids = [_id(s['rows'][-1]) for s in sources]
        if len(set(ids)) != len(ids):
            raise ValueError('CYCLIC_OR_REUSED_TARGET_SOURCE')
        if (board['rows'][0]['frame_id'], board['rows'][0]['row_index']) >= (
                target['rows'][0]['frame_id'], target['rows'][0]['row_index']):
            raise ValueError('FINAL_BOARD_NOT_OBSERVED_BEFORE_TARGET')
        candidates = [list(c) for c in itertools.product(*domains) if len(set(c)) == len(c) and candidate_filter(list(c))]
        if len(candidates) != 1:
            raise ValueError('CARD_SET_NOT_UNIQUE:' + str(len(candidates)))
        payout_fields = (dict(amount=payout['amount'].replace(' ','').replace('\u00a0',''),
                              pot=payout['pot'],hand_class=CLASSES[five_rank(candidates[0])[0]])
                         if role == 'WIN_BEST_FIVE' else {})
        return dict(status='PROVEN', rule=RULE, evidence_kind=role, field_source='DERIVED_UNIQUE_CARD_SET',
                    actor=actor, hand_id=hand, cards=candidates[0], observed_unknown_components=unknown,
                    candidate_domain_size=math.prod(len(d) for d in domains), surviving_candidates=candidates,
                    raw_recognizer_status=target['reading']['status'], source_case=copy.deepcopy(case),
                    target_refs=[_ref(r) for r in target['rows']], context_rows=copy.deepcopy(context),**payout_fields)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, ZeroDivisionError, OSError, OverflowError) as exc:
        return dict(status='UNKNOWN', rule=RULE, reason=str(exc))


def native(row):
    identity = _id(row)
    return isinstance(identity,str) and (':R' in identity or ':F' in identity)


def current_context(rows, end_ref):
    """Complete actual native prefix through the final relied-on source."""
    key=tuple(end_ref.get(k) for k in ('table_id','session_id','epoch'))
    ends=[i for i,row in enumerate(rows) if _id(row)==_id(end_ref)]
    if len(ends)!=1:return []
    available=[r for r in rows[:ends[0]+1] if native(r) and
               tuple(r.get(k) for k in ('table_id','session_id','epoch'))==key]
    headers=[i for i,r in enumerate(available) if HEADER.fullmatch(_text(r))]
    return available[headers[-1]:] if headers else []


def source_groups(rows, readings):
    physical={}
    for row in rows:
        if native(row):
            key=tuple(row.get(k) for k in ('table_id','session_id','epoch','frame_id','row_index'))
            physical.setdefault(key,[]).append(row)
    result=[]
    for reading in readings.values():
        try:
            prefix=(f"T{int(reading['table_id']):02d}",reading['table_session_id'],reading['epoch'],reading['frame_id'])
            index=reading['source_span_index']
            for width in (2,3):
                groups=[physical.get((*prefix,i),[]) for i in range(index-width+1,index+1)]
                if any(len(g)!=1 for g in groups):continue
                group_rows=[g[0] for g in groups]
                literal=' '.join(_text(r) for r in group_rows[:-1])
                show=SHOW.fullmatch(literal)
                win=PAYOUT.fullmatch(literal)
                role='BOARD' if width==2 and RIVER.fullmatch(literal) else 'SHOW' if show else 'WIN' if width==3 and win else None
                if role:
                    result.append(dict(role=role,actor=show[1] if show else win['actor'] if win else None,
                                       source=dict(rows=group_rows,reading=reading)))
        except (KeyError,TypeError,ValueError):
            continue
    return result


def project_unique_card_sets(rows,readings):
    """Saved-source candidate projection. Keep raw values and all rejected readings."""
    for row in rows:row.pop('unique_card_set_proof',None)
    groups=source_groups(rows,readings);count=0
    for target in groups:
        src=target['source'];reading=src['reading'];role=target['role']
        if (role not in {'SHOW','WIN'} or reading.get('status')!='UNRESOLVED'
                or reading.get('reason') not in {'SUIT_UNRESOLVED','PIXEL_RANK_UNRESOLVED'}):continue
        first=src['rows'][0];position=(first['frame_id'],first['row_index'])
        key=tuple(first.get(k) for k in ('table_id','session_id','epoch'))
        related=[g for g in groups if g is not target and tuple(g['source']['rows'][0].get(k) for k in
                 ('table_id','session_id','epoch'))==key and g['source']['reading'].get('hand_id')==reading.get('hand_id')]
        boards=[g for g in related if g['role']=='BOARD' and g['source']['reading'].get('status')=='RAW_UNREVIEWED'
                and (g['source']['rows'][0]['frame_id'],g['source']['rows'][0]['row_index'])<position]
        support_role='SHOW' if role=='WIN' else 'WIN'
        supports=[g for g in related if g['role']==support_role and g['actor']==target['actor']
                  and (g['source']['reading'].get('status')=='RAW_UNREVIEWED' if role=='WIN' else
                       g['source']['reading'].get('reason')=='NO_NEARBY_CARD_EVENT')]
        if role=='WIN':
            supports=[g for g in supports if (g['source']['rows'][0]['frame_id'],g['source']['rows'][0]['row_index'])<position]
        if not boards or not supports:continue
        order=lambda g:(g['source']['rows'][0]['frame_id'],g['source']['rows'][0]['row_index'])
        boards.sort(key=order);supports.sort(key=order)
        all_sources=[src,*[g['source'] for g in boards+supports]]
        end=max((s['rows'][-1] for s in all_sources),key=lambda r:(r['frame_id'],r['row_index']))
        context=current_context(rows,_ref(end))
        case=dict(target_role='WIN_BEST_FIVE' if role=='WIN' else 'SHOW_HOLE_CARDS',target=copy.deepcopy(src),
                  board=copy.deepcopy(boards[0]['source']),context_rows=[_ref(r) for r in context],
                  context_native_time={_id(r):[r.get('source_pts'),r.get('time_base')] for r in context},
                  additional_supports=[dict(role=g['role'],source=copy.deepcopy(g['source'])) for g in boards[1:]+supports[1:]])
        case['hole' if role=='WIN' else 'winning']=copy.deepcopy(supports[0]['source'])
        proof=prove(case)
        if proof['status']=='PROVEN' and verify_current(proof,rows):
            first['unique_card_set_proof']=proof;count+=1
    return count


def verified_group(proof,rows):
    try:
        return (proof.get('rule')==RULE and proof.get('status')=='PROVEN'
                and [_ref(r) for r in rows]==proof['target_refs'] and prove(proof['source_case'])==proof)
    except (KeyError,TypeError,ValueError,IndexError):return False


def verify_current(proof,rows):
    try:
        if proof.get('rule')!=RULE or proof.get('status')!='PROVEN':return False
        current=current_context(rows,proof['context_rows'][-1])
        if [_ref(r) for r in current]!=proof['context_rows']:return False
        for r in current:
            if (r.get('status') in {'ERROR','UNRESOLVED'} or r.get('origin')=='UNRESOLVED_IDENTITY'
                    or r.get('canonical_conflict') or r.get('canonical_duplicate_of')):return False
        case=proof['source_case']
        times=case.get('context_native_time',{})
        if times and set(times)!={_id(r) for r in current}:return False
        for r in current:
            if [r.get('source_pts'),r.get('time_base')]!=times.get(_id(r),[None,None]):return False
        supports=[case['board'],case.get('hole',case.get('winning'))]
        supports.extend(e['source'] for e in case.get('additional_supports',[]))
        # Native support annotations are checked against the current producer's
        # projection as well as the retained pixels. Added eligible supports may
        # not be omitted from the support inventory.
        support_ids={_id(s['rows'][0]) for s in supports}
        annotated_keys=('cards','card_row_id','identity_status','evidence_status',
                        'card_crop_pixel_sha256','raw_pixel_response','recognition_mode','win_best_five_proof')
        current_by_id={_id(r):r for r in current}
        for source in [case['target'],*supports]:
            head=source['rows'][0];actual=current_by_id.get(_id(head))
            if not actual or source is not case['target'] and actual.get('unique_card_set_proof'):return False
            if any(actual.get(k)!=head.get(k) for k in annotated_keys):return False
        for r in current:
            literal=SHOW.fullmatch(_text(r));win=PAYOUT.fullmatch(_text(r))
            eligible=(r.get('cards') and r.get('card_row_id') and r.get('evidence_status')=='RAW_UNREVIEWED'
                      and (RIVER.fullmatch(_text(r)) or literal and literal[1]==proof['actor']))
            native_win=r.get('win_best_five_proof') or {}
            eligible=eligible or (proof['evidence_kind']=='SHOW_HOLE_CARDS' and native_win.get('actor')==proof['actor'])
            if eligible and _id(r) not in support_ids:return False
        for source in [case['target'],*supports]:
            for r in source['rows']:
                actual=current_by_id.get(_id(r))
                if actual is None:return False
                if actual.get('source_pts') is not None or actual.get('time_base') is not None:
                    reading=source['reading']
                    if Fraction(actual['source_pts']*actual['time_base'][0],actual['time_base'][1])!=Fraction(
                            reading['source_pts']*reading['time_base_num'],reading['time_base_den']):return False
        return prove(case)==proof
    except (KeyError,TypeError,ValueError,IndexError,AttributeError,ZeroDivisionError,OverflowError):return False


def validated_rows(rows):
    rows=list(rows);result=[]
    for row in rows:
        proof=row.get('unique_card_set_proof')
        if proof and not verify_current(proof,rows):
            row=dict(row);row.pop('unique_card_set_proof',None)
            row['unique_card_set_conflict']='CURRENT_FULL_SOURCE_CONTEXT_CONFLICT'
        result.append(row)
    return result


def apply_to_hand(hand):
    """Only the complete current hand may grant cards to assembled messages."""
    rows=[r for m in hand['messages'] for r in m['source_rows']]
    for message in hand['messages']:
        # A previously applied proof never survives a failed replay. Preserve
        # the independent native PIXEL_ONLY path and clear only this authority.
        fields=message.get('field_sources',{})
        if fields.get('cards')=='DERIVED_UNIQUE_CARD_SET':
            message['cards']=[];fields['cards']='UNRESOLVED'
            message.pop('card_provenance',None)
        if fields.get('win_best_five')=='DERIVED_UNIQUE_CARD_SET':
            message.pop('win_best_five_cards',None);message.pop('win_best_five_provenance',None)
            fields['win_best_five']='UNRESOLVED'
        group_rows=message['source_rows'];proof=group_rows[0].get('unique_card_set_proof')
        if not proof:continue
        if (proof.get('hand_id')!=hand.get('hand_id') or not verified_group(proof,group_rows)
                or not verify_current(proof,rows) or message.get('player')!=proof['actor']):continue
        provenance={k:copy.deepcopy(proof[k]) for k in ('rule','evidence_kind','actor','hand_id',
                    'target_refs','observed_unknown_components','candidate_domain_size','surviving_candidates')}
        if proof['evidence_kind']=='SHOW_HOLE_CARDS' and message['action']=='SHOW':
            message['cards']=list(proof['cards']);message['field_sources']['cards']='DERIVED_UNIQUE_CARD_SET'
            message['card_provenance']=provenance
        elif proof['evidence_kind']=='WIN_BEST_FIVE' and message['action']=='WIN' and message['amount']==proof['amount']:
            message['win_best_five_cards']=list(proof['cards'])
            message['field_sources']['win_best_five']='DERIVED_UNIQUE_CARD_SET'
            message['win_best_five_provenance']=dict(provenance,amount=proof['amount'],pot=proof['pot'],hand_class=proof['hand_class'])
