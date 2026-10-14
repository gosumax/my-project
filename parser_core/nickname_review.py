"""Source-bound nickname review cases; native images stay inside data URLs."""
import base64
import copy
import hashlib
import json
import math
import re
from pathlib import Path

from .player_registry import PlayerRegistry, digest, actor_observation_has_name, hand_nick_conflicts
from .simple_seat_who import nick_observation_constraint
from .nickname_policy import NICK_CONFIDENCE_MIN, NICK_REVIEW_MIN

REVIEW_CONFIDENCE_MIN = NICK_CONFIDENCE_MIN
MANUAL_NICK_CONFIDENCE_MIN = NICK_REVIEW_MIN
MANUAL_NICK_REVIEW = 'NATIVE_NICK_085_090'


def _manual_nick_score(row):
    score = row.get('confidence')
    return type(score) in (int, float) and math.isfinite(score) and MANUAL_NICK_CONFIDENCE_MIN <= score < REVIEW_CONFIDENCE_MIN


def _rows(path):
    with Path(path).open(encoding='utf-8', buffering=4 * 1024 * 1024) as stream:
        for line, text in enumerate(stream, 1):
            if text.strip():
                yield line, json.loads(text)


def _image(raw, expected, shape=None, expected_png_shape=None):
    """Validate native gray bytes; encode CHAT previews only in memory."""
    import cv2
    import numpy as np
    data = base64.b64decode(raw, validate=True)
    if shape is None:
        pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    else:
        if len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError('Review gray crop shape invalid')
        pixels = np.frombuffer(data, np.uint8).reshape(shape)
    if pixels is None or hashlib.sha256(pixels.tobytes()).hexdigest() != expected:
        raise ValueError('Nickname review crop hash mismatch')
    if expected_png_shape is not None and list(pixels.shape) != list(expected_png_shape):
        raise ValueError('Review native crop geometry mismatch')
    if shape is not None:
        ok, encoded = cv2.imencode('.png', pixels)
        if not ok:
            raise ValueError('Review preview encoding failed')
        data = encoded.tobytes()
    return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')


def _near(a, b):
    """At most two edits: UI ranking only, never association authority."""
    if abs(len(a) - len(b)) > 2:
        return False
    previous = list(range(len(b) + 1))
    for i, c in enumerate(a, 1):
        current = [i]
        for j, d in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j-1] + (c != d)))
        previous = current
    return previous[-1] <= 2


def _actor_row(row, actor):
    """A wrapped amount-only continuation cannot stand in for a nickname."""
    return actor_observation_has_name(row,actor)


def _actor_spelling(actor,row):
    """Display/candidate view only; preserve the complete parsed actor as raw."""
    if row.get('raw_ocr_text','').startswith(actor+' '):
        match=re.match(r'^(?:Dealer|Дилер)\s*;\s*(.+)$',actor,re.I)
        if match:
            return match[1]
    return actor


def prepare_cases(cases):
    """Triage spelling review only; excluded native evidence keeps its authority."""
    prepared=[];audit=[];seen={}
    for original in cases:
        if original.get('error'):
            prepared.append(original)
            continue
        case=copy.deepcopy(original)
        anchors=[o['text'] for o in case['observations'] if o['kind']=='NICK' and o.get('selected',True)]
        kept=[];excluded=[]
        for observation in case['observations']:
            if (observation['kind']=='NICK' and not observation.get('selected',True) and
                not any(_near(observation['text'],anchor) for anchor in anchors)):
                excluded.append({k:v for k,v in observation.items() if k!='image'})
            else:
                kept.append(observation)
        case['observations']=kept
        from .chat_nick_actor import literal_constraint,matches
        chat_names={o.get('display_text',o['text']) for o in kept if o['kind']=='CHAT'}
        resolved=[]
        for observation in kept:
            label=literal_constraint(observation['text']) if observation['kind']=='NICK' else None
            full=[name for name in chat_names if label and label['kind']=='PREFIX' and matches(name,label)]
            if len(full)==1:
                resolved.append(dict(text=observation['text'],chat_nickname=full[0]))
        resolved_names={o['text'] for o in resolved}
        kept=[o for o in kept if not (o['kind']=='NICK' and o['text'] in resolved_names)]
        case['observations']=kept
        nicks=[o for o in kept if o['kind']=='NICK']
        spellings=sorted({o.get('display_text',o['text']) for o in kept})
        conflict=any(_near(a,b) for i,a in enumerate(spellings) for b in spellings[i+1:])
        weak=any(_manual_nick_score(o) for o in nicks)
        entry=dict(original_case_id=original['case_id'],context=original['context'],
                   excluded_spelling_suggestions=excluded,identity_evidence_changed=False)
        if resolved:entry['automatic_truncation_matches']=resolved
        if not nicks or not weak and not conflict:
            entry['queue_status']='NO_UNRESOLVED_SPELLING';audit.append(entry)
            continue
        if 'reason' in case:
            case['reason']='SIMILAR_SPELLING_CONFLICT' if conflict else 'LOW_CONFIDENCE'
        # Exact repeated spelling sets at the same current hand/seat need one
        # spelling decision. This creates no association for hidden intervals.
        key=(*tuple(case['context'][k] for k in ('table_id','session_id','hand_id','seat')),
             tuple(sorted((o['kind'],o['text']) for o in kept)))
        if key in seen:
            first=seen[key]
            first.setdefault('duplicate_contexts',[]).append(dict(case_id=original['case_id'],
                context=original['context'],identity_interval=original['identity_interval']))
            first['case_id']=digest({k:v for k,v in first.items() if k!='case_id'})
            entry['queue_status']='DUPLICATE_SPELLING_CASE';audit.append(entry)
            continue
        if excluded:
            case['excluded_spelling_suggestions']=excluded
        case['case_id']=digest({k:v for k,v in case.items() if k!='case_id'})
        prepared.append(case);seen[key]=case
        entry['queue_status']='REVIEW';audit.append(entry)
    return prepared,audit


def unreviewed_cases(cases,registry):
    result=[]
    for case in cases:
        identities=[registry.resolve(o['text'],o['kind']) for o in case['observations']]
        if (not case.get('error') and not any(o['kind']=='CHAT' for o in case['observations']) and
            not any(identities)):
            continue
        if (not case.get('error') and identities and all(identities) and
            len({identity['player_id'] for identity in identities})==1):
            if case.get('review_scope')!='HAND_SEAT':
                continue
            # Global spelling membership cannot attest a new hand's temporal
            # context. Suppress only a completed decision for this exact scope.
            wanted={(o['kind'],o['text']) for o in case['observations']}
            if case.get('review_scope_key') and any(
                record.get('review_scope')=='HAND_SEAT' and
                record.get('review_scope_key')==case['review_scope_key'] and
                record['context']==case['context'] and record['source_pins']==case['source_pins'] and
                record.get('hand_bounds')==case.get('hand_bounds') and
                wanted<={(o['kind'],o['text']) for o in record['observations']}
                for record in registry.active_reviews.values()):
                continue
        result.append(case)
    return result


def group_spelling_cases(cases):
    """One display for exact repetitions; each source-bound decision stays separate."""
    result=[]; groups={}
    for case in cases:
        chat_names={o.get('display_text',o['text']) for o in case['observations'] if o['kind']=='CHAT'}
        primary={o['text'] for o in case['observations'] if o['kind']=='NICK' and
                 _manual_nick_score(o) and o['text'] in chat_names}
        unrelated=any(o['kind']=='NICK' and o['text'] not in primary and
                      (not _near(o['text'],next(iter(primary),'')) or
                       type(o.get('confidence')) not in (int,float) or
                       not math.isfinite(o['confidence']) or not 0<=o['confidence']<MANUAL_NICK_CONFIDENCE_MIN)
                      for o in case['observations'])
        if (case.get('error') or case.get('binding_blockers') or
                case.get('review_eligibility') != MANUAL_NICK_REVIEW or
                len(primary) != 1 or chat_names != primary or unrelated):
            result.append(case)
            continue
        key=(case['context']['table_id'],case['context']['session_id'],case['context']['seat'],
             case.get('review_scope','INTERVAL'),next(iter(primary)),
             tuple(sorted(o['text'] for o in case['observations'] if o['kind']=='CHAT')))
        if key not in groups:
            display=copy.deepcopy(case)
            groups[key]=display
            result.append(display)
        else:
            display=groups[key]
            group=display.setdefault('review_group',dict(case_ids=[display['case_id']],contexts=[display['context']]))
            group['case_ids'].append(case['case_id'])
            group['contexts'].append(copy.deepcopy(case['context']))
            present={(o['kind'],o['text']) for o in display['observations']}
            display['observations'].extend(copy.deepcopy(o) for o in case['observations']
                                          if (o['kind'],o['text']) not in present)
            if len({o.get('display_text',o['text']) for o in display['observations']})>1:
                display['reason']='SIMILAR_SPELLING_CONFLICT'
    return result


def _hand_cases(provider, source, registry):
    """Suggestions for explicit human hand/seat review, never an automatic bind.

    A CHAT spelling may precede the first readable native NICK. Preserve that
    distinction rather than inventing an earlier physical nickname interval.
    """
    from .simple_saved_policy import hand_key
    result, covered = [], set()
    for hand in provider.hands.values():
        key = hand_key(hand)
        low, high = provider.bounds[key]
        proofs = provider.actor_sources._proofs(hand)
        actors = sorted({m.get('player') or m.get('actor') for m in hand['messages']} - {None})
        sources = {}
        for actor in actors:
            rows = [r for m in hand['messages'] if (m.get('player') or m.get('actor')) == actor
                    for r in m.get('source_rows', ()) if _actor_row(r, actor) and
                    isinstance(r.get('frame_id'), int) and low <= r['frame_id'] <= high and
                    r.get('stable_row_id') and r.get('row_pixel_sha256')]
            if rows:
                sources[actor] = max(rows, key=lambda r: r.get('confidence') or 0)
        for route, segments in provider.lineage.segments.items():
            if route[:2] != key[:2]:
                continue
            route_key = (*key, route[2])
            if any(p.get('status') == 'PROVEN' and p.get('seat') == route[2] and
                   p.get('association', {}).get('registry_evidence', {}).get('current_hand_review')
                   for p in proofs.values()):
                covered.add(route_key)
                continue
            available = [a for a in sources if not (proofs.get(a, {}).get('status') == 'PROVEN' and
                                                   proofs[a].get('seat') != route[2])]
            best = {}
            for row in provider.lineage.observations:
                if (int(str(row['table_id']).removeprefix('T')), row['session_id'], row['seat']) != route:
                    continue
                raw = row.get('raw_text')
                if (not low <= row['frame'] <= high or
                        (nick_observation_constraint(row) or {}).get('kind') != 'EXACT' or
                        not any(_near(raw, _actor_spelling(a,sources[a])) for a in available)):
                    continue
                if raw not in best or (row.get('confidence') or 0) > (best[raw].get('confidence') or 0):
                    best[raw] = row
            candidates = [a for a in available if any(_near(raw, _actor_spelling(a,sources[a])) for raw in best)]
            if not best or not candidates:
                continue
            unselected=[r for r in provider.lineage.observations if
                (int(str(r['table_id']).removeprefix('T')),r['session_id'],r['seat'])==route and
                low<=r['frame']<=high and nick_observation_constraint(r) is not None and
                r['raw_text'] not in best]
            spellings = sorted(set(best) | {_actor_spelling(a,sources[a]) for a in candidates})
            conflict = any(_near(a, b) for i, a in enumerate(spellings) for b in spellings[i+1:])
            manual_identity = any(_manual_nick_score(r) for r in best.values()) and any(
                proofs.get(a,{}).get('status') != 'PROVEN' for a in candidates)
            # A real spelling below the automatic floor may need a human's
            # hand/seat association even when CHAT spells it identically.
            if not conflict and not manual_identity:
                continue
            # The ordinary interval case suffices unless the evidence actually
            # crosses a NICK interval or requires two CHAT spellings.
            native_segments = {id(provider.lineage.for_seat(*route, r['frame'])) for r in best.values()}
            cross = len(native_segments) > 1 or len(candidates) > 1 or any(
                (segment := provider.lineage.for_seat(*route, row['frame'])) is not None and
                any(sources[a]['frame_id'] < segment['start'] or segment['end'] is not None and
                    sources[a]['frame_id'] > segment['end'] for a in candidates)
                for row in best.values())
            if not cross and not manual_identity:
                continue
            observations = []
            for raw, row in sorted(best.items()):
                segment = provider.lineage.for_seat(*route, row['frame'])
                observations.append(dict(id=digest(['NICK', raw]), kind='NICK', text=raw,
                    confidence=row.get('confidence'), selected=True,
                    source=dict(file=str(source/'seat_forced_inputs.jsonl'), line=row['source_line'],
                        frame=row['frame'],pixel_sha256=row['pixel_sha256'],roi=row['roi'],
                        source_pts=row.get('source_pts'),time_base=row.get('time_base'),
                        native_interval=[segment['start'], segment['end']] if segment else None)))
            for actor in candidates:
                row = sources[actor]
                observations.append(dict(id=digest(['CHAT', actor]), kind='CHAT', text=actor,
                    display_text=_actor_spelling(actor,row),
                    confidence=row.get('confidence'), selected=True,
                    source=dict(file=str(source/'physical_rows.jsonl'), row_id=row['stable_row_id'],
                        frame=row['frame_id'],pixel_sha256=row['row_pixel_sha256'],raw_text=row.get('raw_ocr_text'))))
            known = [registry.resolve(raw, 'NICK') for raw in best]
            known = [r for r in known if r is not None]
            case = dict(context=dict(table_id=key[0],session_id=key[1],hand_id=key[2],seat=route[2]),
                review_scope='HAND_SEAT',hand_bounds=[low,high],identity_interval=[low,high],
                observations=observations,source_pins={},
                suggestion=known[0]['canonical'] if known and len({r['player_id'] for r in known})==1 else '',
                reason='SIMILAR_SPELLING_CONFLICT' if conflict else 'LOW_CONFIDENCE',
                native_intervals=[dict(start=s['start'],end=s['end'],label=s['label']) for s in segments
                                  if s['start']<=high and (s['end'] is None or s['end']>=low)])
            if manual_identity:
                case['review_eligibility']=MANUAL_NICK_REVIEW
            if unselected:
                # Human spelling confirmation is still useful to the player
                # database. It cannot erase these conflicting native sources
                # or grant identity for the current hand.
                reason,conflicts,uncertain=hand_nick_conflicts(provider.lineage,route,(low,high),set(best),
                    registry,case['suggestion'] or None,set(actors)-set(candidates))
                case['unmatched_uncertain_nick_candidates']=[dict(raw_text=r['raw_text'],frame=r['frame'],
                    confidence=r.get('confidence'),source_file=r.get('source_file'),source_line=r.get('source_line'),
                    pixel_sha256=r.get('pixel_sha256')) for r in uncertain]
                if conflicts:case['binding_blockers']=[dict(reason=reason,raw_text=r['raw_text'],
                    frame=r['frame'],source_file=r.get('source_file'),source_line=r.get('source_line'),
                    pixel_sha256=r.get('pixel_sha256')) for r in conflicts]
            clears = [b for b in provider.lineage.boundaries[route] if low<=b['frame']<=high and
                      b['kind']!='STABLE_NEW_NICK']
            if clears:
                case['error']='Доказана смена/освобождение места; объединение раздачи запрещено'
            result.append(case)
            covered.add(route_key)
    return result, covered


def build_cases(provider, source):
    """Review each unregistered fixed-seat nickname, plus unresolved variants."""
    from .simple_saved_policy import hand_key
    source = Path(source).resolve()
    actor_provider = provider.actor_sources
    registry = getattr(provider, 'player_registry', None) or PlayerRegistry()
    native_path, chat_path = source / 'seat_forced_inputs.jsonl', source / 'physical_rows.jsonl'
    if not native_path.exists() or not chat_path.exists():
        return []
    cases, wanted_nicks, wanted_chats = [], set(), set()
    for hand in provider.hands.values():
        key = hand_key(hand)
        low, high = provider.bounds[key]
        proofs = actor_provider._proofs(hand)
        actors = sorted({m.get('player') or m.get('actor') for m in hand['messages']} - {None})
        for route, segments in provider.lineage.segments.items():
            if route[:2] != key[:2]:
                continue
            for segment in segments:
                nicks = [r for r in segment['observations'] if low <= r['frame'] <= high and
                         (nick_observation_constraint(r) or {}).get('kind') == 'EXACT']
                if not nicks:
                    continue
                regular_texts = {r['raw_text'] for r in nicks}
                nicks += [r for r in segment['ignored'] if low <= r['frame'] <= high and
                          segment['start'] <= r['frame'] and (segment['end'] is None or r['frame'] <= segment['end']) and
                          (nick_observation_constraint(r) or {}).get('kind') == 'EXACT']
                best = {}
                for row in nicks:
                    raw = row['raw_text']
                    if raw not in best or (row.get('confidence') or 0) > (best[raw].get('confidence') or 0):
                        best[raw] = row
                current = [registry.resolve(raw, 'NICK') for raw in best]
                chat_options = []
                for actor in actors:
                    proof = proofs.get(actor, {})
                    anchor = proof.get('anchor', {})
                    own = (proof.get('status') == 'PROVEN' and proof.get('seat') == route[2] and
                           provider.lineage.for_seat(*route, anchor.get('frame', -1)) is segment)
                    other = proof.get('status') == 'PROVEN' and proof.get('seat') != route[2]
                    if other or not (own or any(_near(raw, actor) for raw in best)):
                        continue
                    rows = [r for m in hand['messages'] if (m.get('player') or m.get('actor')) == actor
                            for r in m.get('source_rows', ())]
                    rows = [r for r in rows if r.get('stable_row_id') and r.get('row_pixel_sha256') and
                            isinstance(r.get('frame_id'), int) and low <= r['frame_id'] <= high and
                            segment['start'] <= r['frame_id'] and
                            (segment['end'] is None or r['frame_id'] <= segment['end']) and
                            _actor_row(r, actor)]
                    if rows:
                        row = max(rows, key=lambda r: r.get('confidence') or 0)
                        chat_options.append((actor, row, own))
                # Keep uncorroborated UI/OCR text out of the queue. Native
                # 85..90 proposals with earlier CHAT use _hand_cases below.
                if not chat_options and not any(current):
                    continue
                spellings = sorted(set(best) | {actor for actor, _, _ in chat_options})
                similar_conflict = any(a != b and _near(a, b) for i, a in enumerate(spellings)
                                       for b in spellings[i+1:])
                scores = [r.get('confidence') for r in best.values()]
                low_confidence = any(not isinstance(s, (int, float)) or not math.isfinite(s) or
                                     s < REVIEW_CONFIDENCE_MIN for s in scores)
                if not low_confidence and not similar_conflict:
                    continue
                overlapping = [s for s in segments if s['start'] <= high and
                               (s['end'] is None or s['end'] >= low)]
                if (all(current) and len({r['player_id'] for r in current}) == 1 and
                        len(overlapping) == 1 and all(own for _, _, own in chat_options)):
                    continue
                observations = []
                for raw, row in sorted(best.items()):
                    observations.append(dict(id=digest(['NICK', raw]), kind='NICK', text=raw,
                        confidence=row.get('confidence'), selected=raw in regular_texts,
                        source=dict(file=str(native_path), line=row['source_line'], frame=row['frame'],
                                    pixel_sha256=row['pixel_sha256'], roi=row['roi'],
                                    source_pts=row.get('source_pts'), time_base=row.get('time_base'))))
                    wanted_nicks.add(row['source_line'])
                for actor, row, own in chat_options:
                    observations.append(dict(id=digest(['CHAT', actor]), kind='CHAT', text=actor,
                        confidence=row.get('confidence'), selected=own,
                        source=dict(file=str(chat_path), row_id=row['stable_row_id'], frame=row['frame_id'],
                                    pixel_sha256=row['row_pixel_sha256'], raw_text=row.get('raw_ocr_text'))))
                    wanted_chats.add((row['stable_row_id'], row['frame_id'], row['row_pixel_sha256']))
                known = [r for r in current if r is not None]
                cases.append(dict(context=dict(table_id=key[0], session_id=key[1], hand_id=key[2], seat=route[2]),
                    identity_interval=[segment['start'], segment['end']], observations=observations,
                    suggestion=(known[0]['canonical'] if known and
                                len({r['player_id'] for r in known}) == 1 else ''),
                    reason=('SIMILAR_SPELLING_CONFLICT' if similar_conflict else 'LOW_CONFIDENCE'),
                    source_pins={}))
    hand_cases, covered = _hand_cases(provider, source, registry)
    cases = [c for c in cases if tuple(c['context'][k] for k in ('table_id','session_id','hand_id','seat')) not in covered]
    cases.extend(hand_cases)
    # Rebuild the retained-source requests after replacing interval duplicates.
    wanted_nicks = {o['source']['line'] for c in cases for o in c['observations'] if o['kind']=='NICK'}
    wanted_chats = {(o['source']['row_id'],o['source']['frame'],o['source']['pixel_sha256'])
                    for c in cases for o in c['observations'] if o['kind']=='CHAT'}
    if not cases:
        return []
    pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (native_path, chat_path)}
    native = {line: row for line, row in _rows(native_path) if line in wanted_nicks}
    chats = {}
    for _, row in _rows(chat_path):
        key = (row.get('row_id'), row.get('frame_id'), row.get('row_pixel_sha256'))
        if key in wanted_chats:
            if key in chats and chats[key] != row:
                raise ValueError('Conflicting physical review crop observations')
            chats[key] = row
    valid = []
    for case in cases:
        try:
            for observation in case['observations']:
                ref = observation['source']
                if observation['kind'] == 'NICK':
                    from .nickname_roster_binding import _seat_rois
                    row = native[ref['line']]
                    if (row['kind'] != 'NICK' or row['seat'] != case['context']['seat'] or
                            row['table_id'] != case['context']['table_id'] or
                            row['session_id'] != case['context']['session_id'] or
                            str(row['hand_id']) != case['context']['hand_id'] or
                            row['source_frame'] != ref['frame'] or row['pixel_sha256'] != ref['pixel_sha256'] or
                            row.get('source_ref') != ref['pixel_sha256'] or row['roi'] != ref['roi'] or
                            row['roi'] != _seat_rois()[f"slot_{case['context']['table_id']:02d}"]['seats'][case['context']['seat']]['nick'] or
                            row['ocr_response']['raw_text'] != observation['text']):
                        raise ValueError('Review native NICK source mismatch')
                    encoded = row.get('png_base64') or row.get('source_png_base64')
                else:
                    row = chats[(ref['row_id'], ref['frame'], ref['pixel_sha256'])]
                    if (row.get('table_id') != case['context']['table_id'] or
                            row.get('table_session_id') != case['context']['session_id'] or
                            row.get('pixel_dtype') != 'uint8'):
                        raise ValueError('Review CHAT source mismatch')
                    encoded = row.get('row_pixels_base64')
                if not encoded:
                    raise ValueError('Review source crop unavailable')
                observation['image'] = _image(encoded, ref['pixel_sha256'],
                    row['pixel_shape'] if observation['kind'] == 'CHAT' else None,
                    [row['roi'][3]-row['roi'][1], row['roi'][2]-row['roi'][0]] if observation['kind'] == 'NICK' else None)
            case['source_pins'] = pins
            if case.get('review_scope') == 'HAND_SEAT':
                case['review_scope_key'] = digest(dict(context=case['context'],source_pins=pins,
                    observations=[dict(kind=o['kind'],text=o['text'],source=o['source']) for o in case['observations']]))
            case['case_id'] = digest(case)
            valid.append(case)
        except (ValueError, KeyError, TypeError) as error:
            # Source failures are visible cases, never replaced by an OCR guess.
            case.update(error=str(error), source_pins=pins)
            case['case_id'] = digest(case)
            valid.append(case)
    valid,audit=prepare_cases(valid)
    provider.nickname_review_queue_audit=audit
    return sorted(unreviewed_cases(valid,registry), key=lambda c: c['reason'] != 'SIMILAR_SPELLING_CONFLICT')


def write_cases(provider, source, output):
    output = Path(output)
    cases = build_cases(provider, source)
    path = output / 'nickname_review_cases.json'
    path.write_text(json.dumps(dict(schema='poker.nickname-review.v1', source=str(Path(source).resolve()),
        review_confidence_min=REVIEW_CONFIDENCE_MIN, manual_nick_confidence_min=MANUAL_NICK_CONFIDENCE_MIN,
        queue_audit=getattr(provider,'nickname_review_queue_audit',[]),
        database=str((getattr(provider, 'player_registry', None) or PlayerRegistry()).path), cases=cases),
        ensure_ascii=False, indent=2), encoding='utf-8')
    return path


PAGE = r'''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Проверка ников</title>
<style>
:root{color-scheme:dark;font:16px system-ui;background:#101720;color:#e8edf5}body{max-width:1100px;margin:28px auto;padding:0 22px}
h1{font-size:28px;margin-bottom:8px}p{color:#afbdd1;line-height:1.5}.toolbar,.actions{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button,input,select{font:inherit;border:1px solid #3b4b61;border-radius:9px;padding:11px 16px;background:#1c2939;color:inherit}
button{cursor:pointer}button:disabled{opacity:.4;cursor:default}.primary{background:#4278da;border-color:#6494ec}
input{min-width:240px}section{background:#182230;border:1px solid #314057;border-radius:15px;padding:22px;margin:22px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.crop{background:#101720;border-radius:10px;padding:15px;overflow:auto}.variants{display:flex;gap:12px;flex-wrap:wrap;margin-top:20px}.variant{padding:12px 16px;border:1px solid #425674;border-radius:9px;background:#101720}
.crop img{height:64px;max-width:none;image-rendering:pixelated;display:block;margin:16px 0}.crop small{display:block;color:#9fb1cc;overflow-wrap:anywhere}
.raw{font-family:ui-monospace,monospace;font-size:20px}
#status{white-space:pre-wrap;line-height:1.5}.error{color:#ffbd8c}.ok{color:#90d6ac}.badge{color:#93b6ef;font-size:14px}details{margin-top:16px;color:#aab9ce}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
</style><body><h1>Проверка ников игроков</h1>
<p>Посмотрите на ник на кропе и введите правильное написание. Подтверждение применится ко всем показанным вариантам и сохранится в общей базе игроков.</p>
<div class="toolbar"><button id="prev">← Назад</button><span id="counter"></span><button id="next">Далее →</button><select id="jump"></select></div>
<section><div class="badge" id="context"></div><h2 id="heading"></h2><p id="why"></p><div class="grid" id="crops"></div>
<p>Варианты распознавания:</p><div class="variants" id="variants"></div>
<details id="sources"><summary>Исходные данные для проверки</summary><div class="grid" id="chatCrops"></div><pre id="provenance"></pre></details></section>
<div class="actions"><input id="canonical" placeholder="Правильный ник" autocomplete="off"><button class="primary" id="save">Подтвердить</button><button id="skip">Пропустить</button></div>
<p id="status" role="status"></p><button id="reprocess">Пересчитать WHO и HH по сохранённым данным</button><p id="job" role="status"></p><p>Подтверждённая запись применяется при следующем расчёте WHO и HH. Новые похожие строки требуют проверки. Конфликтующие соответствия автоматически не объединяются.</p>
<script>
const token=location.hash.slice(1);let cases=[],index=0,confirmed=new Set(),confirmations={};
const el=id=>document.getElementById(id);const text=(id,value)=>{el(id).textContent=value};
function render(){const c=cases[index];text('counter',cases.length?`${index+1} / ${cases.length}`:'Нет случаев для проверки');for(const id of ['crops','variants','chatCrops'])el(id).replaceChildren();el('sources').open=false;el('canonical').disabled=!c;el('skip').disabled=!c;
el('prev').disabled=index===0;el('next').disabled=index>=cases.length-1;el('save').disabled=!c||!!c.error;el('jump').value=index;
if(!c){text('heading','Все варианты проверены');text('context','');text('why','');text('provenance','');el('canonical').value='';return}
const saved=confirmations[c.case_id],repeats=c.review_group?.case_ids.length||1;text('context',`T${String(c.context.table_id).padStart(2,'0')} · раздача ${c.context.hand_id} · ${c.context.seat}${repeats>1?' · повторяется в '+repeats+' раздачах':''}`);text('heading',saved?'Сохранено в базе':'Сверьте написания');el('canonical').value=saved?.canonical||c.suggestion||'';text('save',saved?'Обновить подтверждение':'Подтвердить');
const weak=c.observations.filter(o=>o.kind==='NICK'&&o.confidence>=.85&&o.confidence<.90);text('why',c.reason==='LOW_CONFIDENCE'?`Причина проверки: уверенность 85–89,99% — ${weak.map(o=>o.text+' ('+(o.confidence*100).toFixed(2)+'%)').join(', ')}. Уверенные кропы показаны для сравнения.`:'Причина проверки: разные похожие написания. Высокий confidence сам по себе не устраняет этот конфликт.');
for(const o of c.observations){const box=document.createElement('div');box.className='crop';const label=document.createElement('div');label.textContent=o.kind==='NICK'?o.text:o.kind+' · '+o.text;label.className='raw';box.append(label);
if(o.image){const img=document.createElement('img');img.src=o.image;img.alt=`${o.kind}, исходный кроп, кадр ${o.source.frame}`;box.append(img)}
const info=document.createElement('small');info.textContent=`Кадр ${o.source.frame} · уверенность ${o.confidence==null?'неизвестна':(o.confidence*100).toFixed(1)+'%'}${o.kind==='CHAT'?' · '+(o.source.raw_text||''):''}`;box.append(info);el(o.kind==='NICK'?'crops':'chatCrops').append(box)}
const groups=new Map();for(const o of c.observations){const name=o.display_text||o.text;if(!groups.has(name))groups.set(name,[]);groups.get(name).push(o)}
for(const name of groups.keys()){const label=document.createElement('span');label.className='variant raw';label.textContent=name;el('variants').append(label)}
text('provenance',JSON.stringify({context:c.context,repeated_contexts:c.review_group?.contexts,interval:c.identity_interval,binding_blockers:c.binding_blockers,sources:c.observations.map(o=>o.source)},null,2));text('status',c.error?'Ошибка источника: '+c.error:c.binding_blockers?.length?'Написания можно сохранить в базе. Привязка к текущей раздаче пока не доказана: есть другие неразобранные NICK-кропы.':repeats>1?'Одно подтверждение сохранит это написание для всех '+repeats+' показанных раздач.':'');el('status').className=c.error||c.binding_blockers?.length?'error':''}
function move(n){index=Math.max(0,Math.min(cases.length-1,index+n));render()}
el('prev').onclick=()=>move(-1);el('next').onclick=()=>move(1);el('skip').onclick=()=>move(1);el('jump').onchange=()=>{index=Number(el('jump').value);render()};
el('save').onclick=async()=>{const c=cases[index],selected=c.observations.map(o=>o.id);el('save').disabled=true;
try{const response=await fetch('/confirm',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:JSON.stringify({case_id:c.case_id,canonical:el('canonical').value,selected})});const result=await response.json();if(!response.ok)throw Error(result.error||'Ошибка сохранения');await loadCases();text('status',`Сохранено: ${result.canonical}.`+(result.conflicts.length?' Есть конфликтующие варианты — автоматическая привязка для них отключена.':c.binding_blockers?.length?' Написания добавлены в базу; привязка к текущей раздаче пока не доказана.':' Подтверждённые варианты доступны парсеру.'));el('status').className=c.binding_blockers?.length||result.conflicts.length?'error':'ok'}catch(error){text('status',error.message);el('status').className='error';el('save').disabled=false}};
async function jobStatus(){const r=await fetch('/job',{headers:{'X-Review-Token':token}}),job=await r.json();text('job',job.status==='RUNNING'?'Пересчитываются WHO и HH…':job.status==='IDLE'?'':`${job.status==='COMPLETE'?'Пересчёт завершён':'Ошибка пересчёта'}\n${job.output||''}\n${job.log||''}`);el('reprocess').disabled=job.status==='RUNNING';if(job.status==='RUNNING')setTimeout(jobStatus,3000)}
el('reprocess').onclick=async()=>{el('reprocess').disabled=true;try{const r=await fetch('/reprocess',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':token},body:'{}'}),job=await r.json();if(!r.ok)throw Error(job.error);jobStatus()}catch(e){text('job',e.message);el('reprocess').disabled=false}};
async function loadCases(){const r=await fetch('/cases',{headers:{'X-Review-Token':token}});if(!r.ok)throw Error('Доступ к проверке отклонён');const data=await r.json();cases=data.cases;index=Math.max(0,Math.min(index,cases.length-1));confirmations=data.confirmations||{};el('jump').replaceChildren();for(let i=0;i<cases.length;i++){const o=document.createElement('option'),c=cases[i];o.value=i;o.textContent=`T${String(c.context.table_id).padStart(2,'0')} / ${c.context.hand_id} / ${c.context.seat}`;el('jump').append(o)}render()}
loadCases().then(jobStatus).catch(e=>text('status',e.message));
</script></body></html>'''
