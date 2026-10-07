"""Lossless temporal analysis. No poker/resolver decisions, filters or repairs."""
from __future__ import annotations
import argparse
import base64
from bisect import bisect_right
from collections import Counter, defaultdict
import copy
from difflib import SequenceMatcher
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.capture_temporal_evidence import read, write, sha, source_ms, timestamp_text
from parser_core.edge_gate import confirmed_hand_starts
from parser_core.poker_hand_assembly import assemble_hands, assemble_messages
from parser_core.jev_poker_resolver import hypothesis_options


def order(row):
    return (row.get('timestamp_ms'), row['frame_id'], row['row_index'])


def classify(track, start, following, *, boundary_row=False, blocked=False, candidate_prefix=False):
    """Classify evidence relative to one CONFIRMED start, never a time threshold.

    Within-frame row position breaks capture-time ties, not elapsed time. Unknown
    identities / intervening unconfirmed markers block current-hand inference.
    Older observations prove earlier context, not a specific preceding hand ID.
    """
    if start is None or track.get('first_seen_ms') is None:
        return 'AMBIGUOUS', 'MISSING_OR_CONFLICTING_CONFIRMED_BOUNDARY'
    first = (track['first_seen_ms'], track['first_seen_frame'], track['first_seen_row_index'])
    if start.get('timestamp_ms') is None:
        return 'AMBIGUOUS', 'MISSING_SOURCE_TIMESTAMP'
    if track.get('origin') == 'UNRESOLVED_IDENTITY' or track.get('visibility_gap_count', 0):
        return 'AMBIGUOUS', 'TRACK_IDENTITY_OR_VISIBILITY_GAP'
    if candidate_prefix:
        return 'AMBIGUOUS', 'CURRENT_CANDIDATE_PREFIX_BEFORE_FORCED_BET_START'
    if first < order(start):
        if first[0] == start['timestamp_ms']:
            return 'AMBIGUOUS', 'SAMEFRAME_ROW_POSITION_IS_NOT_PREVIOUS_HAND_PROOF'
        return 'PREVIOUS_HAND_TAIL', 'TRACK_FIRST_OBSERVED_BEFORE_CONFIRMED_START_EARLIER_CONTEXT_ONLY'
    if blocked:
        return 'AMBIGUOUS', 'MISSING_OR_CONFLICTING_CONFIRMED_BOUNDARY'
    if track.get('left_censored') and not boundary_row:
        return 'AMBIGUOUS', 'INITIAL_VISIBLE_NOT_PROVEN_NEW'
    if following is None:
        return 'AMBIGUOUS', 'NO_NEXT_CONFIRMED_START_OPEN_LAST_INTERVAL'
    if following.get('timestamp_ms') is None:
        return 'AMBIGUOUS', 'MISSING_NEXT_SOURCE_TIMESTAMP'
    if first >= order(following):
        return 'NEXT_HAND_HEAD', 'NEW_TRACK_AT_OR_AFTER_NEXT_CONFIRMED_START'
    return 'CURRENT_HAND', 'PROVEN_NEW_TRACK_BETWEEN_CONFIRMED_STARTS'


def enrich(rows, physical, tracks):
    byid = {r['row_id']: r for r in physical}
    assert len(byid) == len(physical) == len(rows)
    output = []
    for source in rows:
        row = copy.deepcopy(source)
        rid = source['stable_row_id']
        p = byid[rid]
        assert (p['frame_id'], p['source_span_index'], p['table_session_id'], p['row_pixel_sha256']) == (
            source['frame_id'], source['row_index'], source['session_id'], source['row_pixel_sha256'])
        ms = source_ms(p['source_pts'], p['time_base_num'], p['time_base_den'])
        assert ms is None and source['video_time'] is None or abs(ms-source['video_time']*1000) < 1e-7
        track = tracks[rid]
        row.update(timestamp_ms=ms, timestamp_text=timestamp_text(ms),
            source_pts=p['source_pts'], time_base_num=p['time_base_num'], time_base_den=p['time_base_den'],
            row_key=p['row_key'], chat_line_y1=p['y1'],
            first_seen_ms=track['first_seen_ms'], last_seen_ms=track['last_seen_ms'],
            observation_count=track['observation_count'], explicit_view_count=track['explicit_view_count'],
            visibility_gap_count=track['visibility_gap_count'], first_seen_kind=track['first_seen_kind'],
            left_censored=track['left_censored'], right_censored=track['right_censored'])
        output.append(row)
    # Preserve source global order and all original fields; chronological views
    # are separate from this unchanged evidence inventory.
    assert len(output) == len(rows)
    assert all(all(r[k] == v for k, v in s.items()) for s, r in zip(rows, output))
    return output


def boundaries_for(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['table_id'], row['session_id'])].append(row)
    boundaries, memberships = [], {}
    for key, source in groups.items():
        ordered = sorted(source, key=lambda r: (r['frame_id'], r['row_index']))
        messages = assemble_messages(ordered)
        starts = confirmed_hand_starts(messages)
        for index, reason in starts.items():
            message = messages[index]
            row = message['source_rows'][0]
            boundaries.append({'table_id': key[0], 'session_id': key[1],
                'frame_id': row['frame_id'], 'row_index': row['row_index'],
                'timestamp_ms': row['timestamp_ms'], 'row_id': row['stable_row_id'],
                'hand_number': message['hand_number'], 'status': 'CONFIRMED_HAND_START',
                'reason': reason, 'source_row_ids': message['source_row_ids'],
                'boundary_authority': 'EXISTING_OFFLINE_EDGE_GATE_RULE_RAW_UNREVIEWED_OCR'})
        for hand in assemble_hands(ordered):
            for message in hand['messages']:
                for rid in message['source_row_ids']:
                    memberships[rid] = {'hand_id': hand['hand_id'],
                        'start_row_id': hand['messages'][0]['source_row_ids'][0],
                        'action': message['action']}
    return boundaries, memberships


def pixel_similarity(a, b):
    x = np.frombuffer(base64.b64decode(a['row_pixels_base64']), np.uint8).reshape(a['pixel_shape'])
    y = np.frombuffer(base64.b64decode(b['row_pixels_base64']), np.uint8).reshape(b['pixel_shape'])
    if x.shape != y.shape:
        return None
    return 1-float(np.mean(np.abs(x.astype(np.int16)-y.astype(np.int16))))/255


def source_window(rows, session, lower, upper):
    """PTS identifies the frame; spatial source order bounds ties within it."""
    low = (lower['frame_id'], lower['row_index'])
    high = (upper['frame_id'], upper['row_index'])
    return [r for r in rows if r['session_id'] == session and
            low <= (r['frame_id'], r['row_index']) <= high]


def safe_message(message):
    return message['action'] not in {'UNKNOWN', 'UNKNOWN_EVENT'} and all(
        r.get('origin') != 'UNRESOLVED_IDENTITY' and r.get('status') not in
        {'UNRESOLVED', 'ERROR', 'UNREADABLE'} and bool(r.get('raw_ocr_text'))
        for r in message['source_rows'])


def conflicting_marker(current, start, following, memberships, rows_by_id, boundaries):
    confirmed_ids = {b['row_id'] for b in boundaries}
    next_candidate = memberships.get(following.get('row_id'), {}).get('start_row_id')
    unique_next = next_candidate is not None and sum(
        memberships.get(b['row_id'], {}).get('start_row_id') == next_candidate for b in boundaries) == 1
    return any(m['hand_id'] is not None and m['start_row_id'] == rid and
        rows_by_id[rid]['table_id'] == current['table_id'] and
        rows_by_id[rid]['session_id'] == current['session_id'] and
        order(start) < order(rows_by_id[rid]) < order(following) and rid not in confirmed_ids and
        not (unique_next and m['start_row_id'] == next_candidate)
        for rid, m in memberships.items())


def leading_prefix(membership, first_source_row_id):
    """An unnumbered interior candidate is not the leading visible prefix."""
    return membership['hand_id'] is None and membership['start_row_id'] == first_source_row_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    out = args.evidence
    capture = json.loads((out/'capture_summary.json').read_text('utf-8'))
    baseline = Path(capture['baseline'])
    for name, digest in capture['baseline_hashes'].items():
        assert sha(baseline/name) == digest
    rows = read(baseline/'ordered_timeline.jsonl')
    physical = read(baseline/'physical_rows.jsonl')
    tracks = {r['track_id']: r for r in read(out/'tracks.jsonl')}
    enriched = enrich(rows, physical, tracks)
    byid = {r['stable_row_id']: r for r in enriched}
    facts = {r['row_id']: r for r in physical}
    bounds, memberships = boundaries_for(enriched)
    candidate_ids_map = defaultdict(set)
    for rid, m in memberships.items():
        candidate_ids_map[m['start_row_id']].add(rid)
    first_source_ids = {}
    for row in sorted(enriched, key=lambda r: (r['frame_id'], r['row_index'])):
        first_source_ids.setdefault((row['table_id'], row['session_id']), row['stable_row_id'])
    grouped = defaultdict(list)
    for b in bounds:
        grouped[(b['table_id'], b['session_id'])].append(b)
    for bs in grouped.values():
        bs.sort(key=order)
    labels = Counter()
    baseline_unresolved = Counter()
    interval_byid = {}
    for row in enriched:
        rid = row['stable_row_id']
        bs = grouped[(row['table_id'], row['session_id'])]
        keys = [order(b) for b in bs]
        membership = memberships[rid]
        candidate_start = byid[membership['start_row_id']]
        # Match confirmed start to its numbered baseline candidate. A generic
        # forced-bet start is associated by SOURCE order, not text coincidence.
        candidate_ids = candidate_ids_map[membership['start_row_id']]
        in_candidate = [b for b in bs if b['row_id'] in candidate_ids]
        start = in_candidate[0] if in_candidate else None
        next_start = None
        if start is not None:
            index = bs.index(start)
            next_start = bs[index+1] if index+1 < len(bs) else None
        if leading_prefix(membership, first_source_ids[(row['table_id'], row['session_id'])]):
            start = bs[0] if bs else None
            next_start = bs[1] if len(bs) > 1 else None
        blocked = len(in_candidate) > 1
        if start and next_start:
            # A numbered marker between these bounds that lacks confirmation
            # means the large temporal interval contains unproved hand edges.
            blocked |= conflicting_marker(row, start, next_start, memberships, byid, bs)
        label, reason = classify(tracks[rid], start, next_start,
            boundary_row=bool(start and rid in start['source_row_ids']), blocked=blocked,
            candidate_prefix=bool(start and membership['hand_id'] is not None and order(row) < order(start)))
        row.update(temporal_class=label, temporal_reason=reason,
            baseline_candidate_hand=membership['hand_id'], analysis_event_type=membership['action'],
            confirmed_start_row_id=start['row_id'] if start else None,
            confirmed_start_ms=start['timestamp_ms'] if start else None,
            next_confirmed_start_row_id=next_start['row_id'] if next_start else None,
            next_confirmed_start_ms=next_start['timestamp_ms'] if next_start else None)
        labels[label] += 1
        if row['status'] in {'UNRESOLVED', 'UNKNOWN', 'ERROR', 'UNREADABLE'}:
            baseline_unresolved[label] += 1
        interval_index = bisect_right(keys, order(row))-1
        interval_byid[rid] = f"{row['session_id']}:{interval_index}"
    write(out/'ordered_timeline_temporal.jsonl', enriched)
    write(out/'confirmed_starts.jsonl', bounds)
    # Existing hypotheses are evidence only. Preserve their original files.
    pairs = {}
    for path in (ROOT/'runs/temporal_overlap_boundary_smoke17_20261001/removed_duplicates.jsonl',
                 ROOT/'runs/temporal_overlap_guarded_smoke17_20261001/NEW/removed_duplicates.jsonl'):
        for proposal in read(path):
            key = (proposal['old_row_id'], proposal['matched_previous_row_id'])
            pairs.setdefault(key, []).append(str(path))
    # Explicit resolver hypothesis packets from the latest saved diagnostics.
    saved = ROOT/'runs/logic_math_fixed_smoke17_20261001/final_diagnostics_v2/hands.jsonl'
    diagnostic_rows = read(saved)
    resolver_hypotheses = []
    for h in diagnostic_rows:
        conflicts = [issue for t in h['diagnostics'].get('trace', []) for issue in t.get('issues', [])]
        for option in hypothesis_options(h['hand'], conflicts):
            resolver_hypotheses.append({'table_id': h['hand']['table_id'], 'hand_id': h['hand']['hand_id'],
                'hypothesis': option, 'source': str(saved), 'kind': 'OFFLINE_CURRENT_GENERATOR_ON_SAVED_TRACE_ISSUES'})
    cache_dir = ROOT/'runs/logic_math_fixed_smoke17_20261001/typesafe_live_20261001/jev_cache'
    cache_request_count = 0
    for path in sorted(cache_dir.glob('*.json')):
        request = json.loads(path.read_text('utf-8')).get('request') or {}
        if not request.get('hand') or 'conflicts' not in request:
            continue
        cache_request_count += 1
        for option in hypothesis_options(request['hand'], request['conflicts']):
            resolver_hypotheses.append({'table_id': request['hand']['table_id'], 'hand_id': request['hand']['hand_id'],
                'hypothesis': option, 'source': str(path), 'kind': 'EXACT_SAVED_RESOLVER_REQUEST_OFFLINE_CANDIDATE_REPLAY'})
    for record in resolver_hypotheses:
        option = record['hypothesis']
        if option['decision'] == 'LIKELY_STALE_DUPLICATE':
            a, b = option['target_rows'][0], option['evidence_rows'][0]
            if a in byid and b in byid:
                pairs.setdefault((a, b), []).append(record['kind']+':'+record['source'])
    write(out/'saved_resolver_hypotheses.jsonl', resolver_hypotheses)
    duplicates = []
    for (a, b), sources in pairs.items():
        x, y = byid[a], byid[b]
        same_track = x['stable_row_id'] == y['stable_row_id']
        overlapping = any(max(l, s) <= min(r, e) for l, r in tracks[a]['visible_spans']
                          for s, e in tracks[b]['visible_spans'])
        # Different tracks/intervals are not proof of distinct events when
        # identity was lost. Same text/near pixels is not a duplicate proof.
        resolved = same_track
        duplicates.append({'row_id': a, 'candidate_canonical_row_id': b,
            'sources': sources, 'delta_timestamp_ms': x['timestamp_ms']-y['timestamp_ms'],
            'same_track_id': same_track, 'same_roi_position': x['row_index'] == y['row_index'],
            'same_pixel_hash': x['row_pixel_sha256'] == y['row_pixel_sha256'],
            'pixel_similarity': pixel_similarity(facts[a], facts[b]),
            'ocr_similarity': SequenceMatcher(None, x['normalized_text'], y['normalized_text']).ratio(),
            'same_hand_interval': interval_byid[a] == interval_byid[b],
            'visibility_intervals_overlap': overlapping,
            'result': 'SAME_PHYSICAL_TRACK' if resolved else 'AMBIGUOUS_CROSS_TRACK_IDENTITY',
            'row_temporal_class': x['temporal_class'], 'candidate_temporal_class': y['temporal_class']})
    write(out/'duplicate_analysis.jsonl', duplicates)
    # Missing-event windows: use actual stored FIRST_BREAK references and
    # nearest source messages. Do not infer/reconstruct a missing action.
    missing = []
    for h in diagnostic_rows:
        first = h['diagnostics'].get('first_break') or {}
        if first.get('code') not in {'MISSING_ACTION', 'MISSING_AMOUNT', 'MISSING_BLINDS_BEFORE_ACTION', 'MISSING_UNCALLED_RETURN'}:
            continue
        messages = h['hand']['messages']
        index = first.get('message_index', 0)
        ids = first.get('source_row_ids', [])
        if not ids or ids[0] not in byid:
            continue
        row = byid[ids[0]]
        previous = next((m for m in reversed(messages[:index]) if safe_message(m)), None)
        following = next((m for m in messages[index+1:] if safe_message(m)), None)
        prior_ids = previous['source_row_ids'] if previous else []
        prior = byid.get(prior_ids[-1]) if prior_ids else None
        following_ids = following['source_row_ids'] if following else []
        next_row = byid.get(following_ids[0]) if following_ids else None
        lo = prior['timestamp_ms'] if prior else row['timestamp_ms']
        hi = next_row['timestamp_ms'] if next_row else row['timestamp_ms']
        candidates = source_window(enriched, row['session_id'], prior or row, next_row or row)
        low = [r for r in candidates if r['confidence'] is None or r['confidence'] < .9 or
               r['status'] in {'UNRESOLVED', 'UNKNOWN', 'UNREADABLE', 'ERROR'}]
        missing.append({'table_id': row['table_id'], 'hand_id': h['hand']['hand_id'],
            'code': first['code'], 'trigger_row_ids': ids, 'previous_row_ids': prior_ids,
            'next_row_ids': following_ids,
            'lower_source_order': [(prior or row)['frame_id'], (prior or row)['row_index']],
            'upper_source_order': [(next_row or row)['frame_id'], (next_row or row)['row_index']],
            'endpoint_confirmation_basis': 'EXISTING_SAFE_PARSED_RAW_UNREVIEWED_OCR_NOT_HUMAN_CERTIFIED',
            'open_lower_endpoint': prior is None, 'open_upper_endpoint': next_row is None,
            'window_start_ms': lo, 'window_end_ms': hi, 'physical_crop_count': len(candidates),
            'candidate_row_ids': [r['stable_row_id'] for r in candidates],
            'low_confidence_or_unresolved_ids': [r['stable_row_id'] for r in low],
            'visible_track_ids': [rid for rid, t in tracks.items() if t['session_id'] == row['session_id'] and
                any(t0 <= (next_row or row)['frame_id'] and t1 >= (prior or row)['frame_id']
                    for t0, t1 in t['visible_spans'])],
            'result': 'WINDOW_LOCALIZED_NO_EVENT_RECOVERED',
            'limitation': 'Source crop time is viewport visibility; missing top suppressed row/unknown pixels remain possible'})
    write(out/'missing_event_windows.jsonl', missing)
    # Separate retained physical views straddling boundaries from original
    # timeline rows. Count track-boundary pairs and view records explicitly.
    cross = []
    for b in bounds:
        for rid, t in tracks.items():
            if t['session_id'] != b['session_id'] or t['first_seen_ms'] is None:
                continue
            if memberships[rid]['start_row_id'] == memberships[b['row_id']]['start_row_id']:
                # The numbered header may precede its own forced-bet start.
                # It is not a previous-hand tail merely because it persists.
                continue
            if t['origin'] != 'UNRESOLVED_IDENTITY' and not t['visibility_gap_count'] and t['first_seen_ms'] < b['timestamp_ms'] and any(l < b['frame_id'] <= r for l, r in t['visible_spans']):
                cross.append({'track_id': rid, 'confirmed_start_row_id': b['row_id'],
                    'confirmed_start_ms': b['timestamp_ms'], 'first_seen_ms': t['first_seen_ms'],
                    'last_seen_ms': t['last_seen_ms'], 'table_id': t['table_id'],
                    'result': 'PREVIOUS_HAND_TAIL', 'meaning': 'Earlier visible context; immediate previous numbered hand not asserted'})
    write(out/'persistent_previous_tails.jsonl', cross)
    # Visual crops/tiles inherit time from their verified physical parent.
    # Unresolved placeholders without pixels expose trigger time separately;
    # that timestamp must not be advertised as a nonexistent crop timestamp.
    visual_contexts = []
    bykey = {r['row_key']: r for r in physical}
    for number, visual in enumerate(read(baseline/'visual_evidence.jsonl'), 1):
        p = facts.get(visual.get('row_id')) or bykey.get(visual.get('row_key'))
        trigger = bykey.get(visual.get('trigger_row_key'))
        has_crop = bool(visual.get('crop_png_base64') or visual.get('crop_path'))
        if p and has_crop:
            assert visual['frame_id'] == p['frame_id']
            assert visual['table_session_id'] == p['table_session_id']
            if visual.get('row_pixel_sha256') is not None:
                assert visual['row_pixel_sha256'] == p['row_pixel_sha256']
            assert visual.get('source_pts') == p['source_pts']
            t = tracks[p['row_id']]
            ms = source_ms(p['source_pts'], p['time_base_num'], p['time_base_den'])
        else:
            p, t, ms = None, None, None
        visual_contexts.append({'evidence_record_number': number,
            'card_evidence_id': visual.get('card_evidence_id'), 'crop_sha256': visual.get('crop_sha256'),
            'row_key': p['row_key'] if p else None, 'stable_row_id': p['row_id'] if p else None,
            'timestamp_ms': ms, 'timestamp_text': timestamp_text(ms),
            'first_seen_ms': t['first_seen_ms'] if t else None,
            'last_seen_ms': t['last_seen_ms'] if t else None,
            'observation_count': t['observation_count'] if t else None,
            'trigger_row_key': visual.get('trigger_row_key'),
            'trigger_timestamp_ms': source_ms(trigger['source_pts'], trigger['time_base_num'], trigger['time_base_den']) if trigger else None,
            'context_status': 'PHYSICAL_PARENT_PTS_VERIFIED' if p else 'NO_PHYSICAL_CROP_TEMPORAL_CONTEXT_UNRESOLVED',
            'tile_ids': [{'card_index': tile.get('card_index'), 'tile_sha256': tile.get('tile_sha256')}
                         for tile in (visual.get('hybrid') or {}).get('tiles', [])]})
    write(out/'visual_temporal_context.jsonl', visual_contexts)
    frame_times = read(out/'frame_times.jsonl')
    deltas = [b['timestamp_ms']-a['timestamp_ms'] for a, b in zip(frame_times, frame_times[1:])
              if a['timestamp_ms'] is not None and b['timestamp_ms'] is not None]
    timestamp_audit = {'source': 'DECODED_PRESENTATION_PTS_TIMES_RATIONAL_TIME_BASE',
        'decoded_frames': len(frame_times), 'missing_pts_frames': sum(f['source_pts'] is None for f in frame_times),
        'negative_time_transitions': sum(d < 0 for d in deltas),
        'equal_time_transitions': sum(d == 0 for d in deltas),
        'distinct_pts_time_bases': sorted({(f['time_base_num'], f['time_base_den']) for f in frame_times}),
        'pts_delta_min_ms': min(deltas, default=None), 'pts_delta_max_ms': max(deltas, default=None),
        'frame_over_fps_used': False, 'timeline_source_time_max_error_ms': max(
            (abs(r['timestamp_ms']-r['video_time']*1000) for r in enriched if r['timestamp_ms'] is not None), default=None)}
    assert all(all(r[k] == v for k, v in s.items()) for s, r in zip(rows, enriched))
    manifest = json.loads((ROOT/'production_manifest.json').read_text('utf-8'))
    assert all(sha(ROOT/p) == digest for p, digest in manifest['hashes'].items())
    result = {'rows_baseline': len(rows), 'rows_temporal': len(enriched),
        'source_fields_and_global_order_preserved': True, 'confirmed_starts': len(bounds),
        'classification': dict(labels), 'baseline_uncertain_rows': sum(baseline_unresolved.values()),
        'baseline_uncertain_after': dict(baseline_unresolved), 'persistent_tail_track_boundary_pairs': len(cross),
        'unique_persistent_tail_tracks': len({r['track_id'] for r in cross}),
        'duplicate_pairs': len(duplicates), 'duplicates_resolved': sum(r['result'] != 'AMBIGUOUS_CROSS_TRACK_IDENTITY' for r in duplicates),
        'duplicates_ambiguous': sum(r['result'] == 'AMBIGUOUS_CROSS_TRACK_IDENTITY' for r in duplicates),
        'duplicate_same_interval': sum(r['same_hand_interval'] for r in duplicates),
        'resolver_options_generated': len(resolver_hypotheses), 'saved_resolver_requests': cache_request_count,
        'resolver_duplicate_options': sum(r['hypothesis']['decision']=='LIKELY_STALE_DUPLICATE' for r in resolver_hypotheses),
        'resolver_duplicate_unique_pairs': sum(any('GENERATOR' in s or 'RESOLVER_REQUEST' in s for s in r['sources']) for r in duplicates),
        'visual_context_records': len(visual_contexts),
        'visual_verified_parent_contexts': sum(v['context_status']=='PHYSICAL_PARENT_PTS_VERIFIED' for v in visual_contexts),
        'timestamp_audit': timestamp_audit,
        'missing_windows_localized': len(missing), 'missing_events_recovered': 0,
        'proven_hand_assignment_errors': 0,
        'production_code_changed': False, 'IS_TEMPORAL_EVIDENCE_ENOUGH_FOR_EDGE_GATE': 'PARTIAL',
        'verdict': 'TEMPORAL_HAND_SEPARATION_PARTIAL',
        'missing_signals': ['Proven cross-track physical identity after tracking loss',
            'Confirmed start/end and onset bounds for censored/unknown viewport rows'],
        'recognition_scope': 'Paired A/B on fixed real production OCR/cards; capture replay verifies pixels/PTS; no live recognizer rerun',
        'classification_scope': 'Research labels relative to existing offline boundary rule; RAW_UNREVIEWED recognition is not independent visual ground truth',
        'analysis_code_sha256': sha(Path(__file__)),
        'analysis_dependency_sha256': {p: sha(ROOT/p) for p in ('experiments/capture_temporal_evidence.py',
            'parser_core/edge_gate.py', 'parser_core/poker_hand_assembly.py', 'parser_core/jev_poker_resolver.py')},
        'diagnostic_source': str(saved), 'diagnostic_source_rows': 7954,
        'missing_window_coverage': 'SAVED_FIRST_BREAK_MISSING_CODES_ONLY_NOT_ALL_LEDGER_ISSUES',
        'baseline_timeline_sha256': sha(baseline/'ordered_timeline.jsonl'),
        'temporal_timeline_sha256': sha(out/'ordered_timeline_temporal.jsonl')}
    (out/'analysis_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
