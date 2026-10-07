"""Read-only source-bound transport/quality audit, independent of segment counts."""
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.poker_reconstruction.timeline import parse_text


def read(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def table(value):
    return int(str(value).removeprefix('T'))


def verified_events():
    """Localize original reviewed events only at their exact first source frame.

    Multiple stitched tracks may corroborate one physical row. They share one
    geometry requirement, not several fabricated event rows. Best-frame track
    geometry is never reused for a different source observation.
    """
    audit=ROOT/'runs/chat_sequence_audit_20261006'
    original=json.loads((audit/'verified_omissions.json').read_text('utf-8'))
    snapshots=defaultdict(set);firsts=defaultdict(set)
    for hand in original['reviewed_hands']:
        for row in hand['lines']:
            snapshots[hand['table'],hand['snapshot_frame'],hand['snapshot_sha256'],row['track']].add(tuple(row['span']))
    for event in read(audit/'stitched_sequence/events.jsonl'):
        for row in event['source_rows']:
            for track in row['source_tracks']:
                firsts[row['table_id'],row['first_frame'],track].add(tuple(row['first_span']))
    result=[]
    for event in original['events']:
        spans=[];proof=[];known=True
        for track in event['source_tracks']:
            found=snapshots[event['table_id'],event['frame'],event.get('source_chat_sha256'),track]
            origin='VERIFIED_HAND_SNAPSHOT_EXACT_FRAME_AND_CHAT_HASH'
            if len(found)!=1:
                found=firsts[event['table_id'],event['frame'],track]
                origin='STITCHED_SOURCE_FIRST_FRAME_GEOMETRY'
            if len(found)!=1:
                known=False;proof.append(dict(track=track,status='UNKNOWN_SOURCE_FRAME_SPAN'));continue
            span=next(iter(found));spans.append(span)
            proof.append(dict(track=track,span=span,source_frame=event['frame'],origin=origin))
        result.append(dict(event,source_spans=sorted(set(spans)) if known else [],
                           source_span_provenance=proof))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--production', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    verified = verified_events()
    source_db=sqlite3.connect((ROOT/'runs/chat_sequence_audit_20261006/source_reconstruction/source_pixels.sqlite').as_uri()+'?mode=ro',uri=True)
    source_db.row_factory=sqlite3.Row
    admissions={(r['table'],r['frame_after']):r for r in read(args.production/'physical_chat_admission.jsonl')}
    final=json.loads((args.production/'final_summary.json').read_text('utf-8'))
    metadata=json.loads((args.production/'run_metadata.json').read_text('utf-8'))
    full_run=final.get('frames')==10100 and metadata.get('frames_requested')==10100
    physical = read(args.production/'physical_rows.jsonl')
    ocr = read(args.production/'raw_ocr_results.jsonl')
    timeline = read(args.production/'ordered_timeline.jsonl')
    hands = read(args.production/'hand_histories.jsonl')
    records = read(args.production/'logical_hand_analysis.jsonl')
    stack_receipts=read(args.production/'stack_start_stack.jsonl') if (args.production/'stack_start_stack.jsonl').exists() else []
    reconstruction=read(args.production/'poker_reconstruction.jsonl') if (args.production/'poker_reconstruction.jsonl').exists() else []
    stack_math=read(args.production/'start_stack_math.jsonl') if (args.production/'start_stack_math.jsonl').exists() else []
    physical_by_id = defaultdict(list)
    ocr_by_id = defaultdict(list)
    timeline_by_id = defaultdict(list)
    for row in physical:
        physical_by_id[row['row_id']].append(row)
    for row in ocr:
        ocr_by_id[row['row_id']].append(row)
    for row in timeline:
        timeline_by_id[row['stable_row_id']].append(row)
    totals = defaultdict(Counter)
    events = []
    kinds = dict(HAND_START='START_HAND', ANTE='ANTE', SMALL_BLIND='SB', BIG_BLIND='BB',
                 DEAL='DEAL', FOLD='FOLD', UNCALLED_RETURN='RETURN', SHOW='SHOW', WIN='WIN',
                 NO_SHOW='INFO', PLAYER_LEAVE='INFO')
    for event in verified:
        # Resolve the verified event directly against this native run. Reset-ID
        # target-pair experiments and interrupted replays cannot supply live IDs.
        receipt=admissions.get((event['table_id'],event['frame']))
        source=source_db.execute('select * from chats where table_id=? and frame=?',
            (event['table_id'],event['frame'])).fetchone()
        gray=zlib.decompress(source['gray_zlib']) if source else None
        source_ok=bool(source and hashlib.sha256(gray).hexdigest()==source['sha256']
            and (not event.get('source_chat_sha256') or source['sha256']==event['source_chat_sha256']) and receipt
            and receipt['source_pixel_hash_after']==source['sha256'])
        matched=[]
        for span in event['source_spans']:
            candidates=[r for r in (receipt or {}).get('admissions',[]) if abs(sum(r['span'])-sum(span))<=4]
            matched.append(candidates[0] if len(candidates)==1 else None)
        ids=list(dict.fromkeys(r['row_id'] for r in matched if r))
        admitted=bool(source_ok and matched and all(matched) and len(ids)==len(matched))
        receipts = []
        for span,admission in zip(event['source_spans'],matched):
            if not admission:continue
            rid=admission['row_id']
            rows = physical_by_id[rid]
            exact=False
            if source_ok and len(rows)==1:
                row=rows[0];y=row['y1'];height,width=row['pixel_shape'];tb=json.loads(source['time_base'])
                cropped=gray[y*source['width']:(y+height)*source['width']]
                a,b=admission['span'];raw_band=gray[a*width:b*width]
                exact=(row['table_id']==event['table_id'] and row['frame_id']==event['frame']
                    and width==source['width'] and y<=span[0]<span[1]<=y+height
                    and row['source_pts']==source['pts']
                    and [row['time_base_num'],row['time_base_den']]==tb
                    and hashlib.sha256(cropped).hexdigest()==row['row_pixel_sha256']
                    and base64.b64decode(row['row_pixels_base64'])==cropped
                    and hashlib.sha256(raw_band).hexdigest()==admission['row_pixel_sha256'])
            admitted = admitted and exact
            receipts.append(dict(row_id=rid, exact_source_row=exact,
                                 physical_count=len(rows), ocr_count=len(ocr_by_id[rid]),
                                 timeline_count=len(timeline_by_id[rid])))
        ocr_ok = admitted and all(len(ocr_by_id[rid]) == 1 and
            ocr_by_id[rid][0].get('raw_ocr_response', {}).get('status') == 'OK' for rid in ids)
        timeline_ok = ocr_ok and all(len(timeline_by_id[rid]) == 1 and
            timeline_by_id[rid][0]['row_pixel_sha256'] == physical_by_id[rid][0]['row_pixel_sha256'] for rid in ids)
        observed_rows = sorted((timeline_by_id[rid][0] for rid in ids if len(timeline_by_id[rid]) == 1),
                               key=lambda r: r['row_index'])
        observed_text = ' '.join(r['raw_ocr_text'] for r in observed_rows)
        parsed = parse_text(observed_text, {'rows': observed_rows})
        recognized = bool(timeline_ok and parsed.kind == kinds[event['kind']])
        if event['kind'] == 'HAND_START':
            recognized = recognized and parsed.source.get('parsed_hand_id') == event['hand_number']
        # INFO alone is not sufficient to establish the requested semantic type.
        if event['kind'] == 'PLAYER_LEAVE':
            recognized = recognized and 'has left the table' in observed_text.lower()
        if event['kind'] == 'NO_SHOW':
            recognized = recognized and any(s in observed_text.lower() for s in ("doesn't show", 'does not show'))
        counter = totals[event['kind']]
        counter.update(physical_verified=1, admitted=int(admitted), OCR=int(ocr_ok),
                       timeline=int(timeline_ok), semantic_type_recognized=int(recognized))
        events.append(dict(event_id=event['event_id'], table=event['table_id'], frame=event['frame'],
            kind=event['kind'], source_chat_sha256=source['sha256'] if source else None,
            source_span_provenance=event['source_span_provenance'], row_receipts=receipts,
            admitted=bool(admitted), OCR=bool(ocr_ok), timeline=bool(timeline_ok),
            semantic_type_recognized=bool(recognized), raw_ocr_text=observed_text,
            source_spans=event['source_spans'],physical_new_status='PROVEN' if admitted else 'UNKNOWN',
            legacy_classification=(receipt or {}).get('old_classification'),
            legacy_reason=(receipt or {}).get('old_reason'),
            admission_proof=(receipt or {}).get('proof'),
            authority='EXACT_NATIVE_SOURCE_RASTER_PTS_AND_LIVE_PIXEL_HASH; TYPE_CHECK_IS_NOT_FULL_EVENT_CORRECTNESS'))
    denominator = json.loads((ROOT/'runs/full_hh_production_20261006/denominator_review/reviewer_verdict.json').read_text('utf-8'))
    hand_audit = []
    for source in denominator['rows']:
        matched = [h for h in hands if table(h['table']) == source['table'] and
                   h['hand_number'] == source['hand']]
        hand_audit.append(dict(source=source, matching_hh_segments=[h['segment_id'] for h in matched],
            exported_statuses=[h['status'] for h in matched],
            certified_candidates=sum(h['certified'] for h in matched),
            independent_full_action_review='NOT_YET_PROVEN'))
    total = sum(c['physical_verified'] for c in totals.values())
    delivered = sum(c['timeline'] for c in totals.values())
    legacy = [r for r in physical if ':R' in r['row_id']]
    baseline=ROOT/'runs/smoke19_batched_storage_full_20261006_013548/production'
    old_legacy=read(baseline/'physical_rows.jsonl')
    keys=('row_id','table_id','frame_id','source_span_index','y1','epoch','origin','source_pts',
          'time_base_num','time_base_den','row_pixel_sha256','pixel_shape')
    projection=lambda rows:sorted([{k:r.get(k) for k in keys} for r in rows],
        key=lambda r:(r['table_id'],r['frame_id'],r['source_span_index']))
    legacy_exact=projection(legacy)==projection(old_legacy)
    summary = dict(known_704={k:dict(v) for k,v in totals.items()},
        known_704_timeline=delivered, known_704_physical=total,
        known_704_transport_recall=delivered/total,
        PHYSICAL_CHAT_RECALL=None,
        PHYSICAL_CHAT_RECALL_status='704_VERIFIED_LOWER_BOUND_ONLY; FULL_SEMANTIC_UNIVERSE_NOT_PROVEN',
        physical_complete_hands_reviewed=denominator['PHYSICAL_COMPLETE_HANDS'],
        physical_denominator_scope=denominator['scope'], END_PARTIAL_reviewed=denominator['END_PARTIAL'],
        exported_statuses=dict(Counter(h['status'] for h in hands)),
        exported_certified_hh=sum(h['certified'] for h in hands),
        independently_correct_hh=0 if not any(h['certified'] for h in hands) else None,
        HH_COMPLETE_RATE=0.0 if not any(h['certified'] for h in hands) else None,
        HH_COMPLETE_RATE_scope='REVIEWED_57_COMPLETE_HAND_INVENTORY; EXHAUSTIVE_DENOMINATOR_NOT_PROVEN',
        logic_statuses=dict(Counter(r['diagnostics']['logic_status'] for r in records)),
        math_statuses=dict(Counter(r['diagnostics']['math_status'] for r in records)),
        unknown_actions=sum(m['action']=='UNKNOWN_EVENT' for r in records for m in r['observed_hand']['messages']),
        unknown_amounts=sum(m.get('amount') is None for r in records for m in r['observed_hand']['messages']
            if m['action'] in {'ANTE','SMALL_BLIND','BIG_BLIND','CALL','BET','RAISE','UNCALLED_RETURN','WIN'}),
        start_stack_statuses=dict(Counter(r['status'] for r in stack_receipts)),
        start_stack_blockers=dict(Counter(r.get('reason') for r in stack_receipts if r['status']!='START_STACK_PROVEN')),
        start_stack_math_pass=sum(r.get('status')=='PASS' for r in stack_math),
        first_visible_source_proofs=sum((r.get('first_visible_proof') or {}).get('status')=='PROVEN' for r in stack_math),
        derived_validation_accepted=sum(r.get('derived_final_validation',{}).get('accepted',False) for r in reconstruction),
        reconstruction_output_eligible=sum(r.get('output_eligible',False) for r in reconstruction),
        reconstruction_statuses=dict(Counter(r.get('reconstruction_status',r.get('status')) for r in reconstruction)),
        HH_PARTIAL=sum(h['status']=='PARTIAL' for h in hands),
        START_PARTIAL_exported=sum(h['status']=='START_PARTIAL' for h in hands),
        END_PARTIAL_exported=sum(h['status']=='END_PARTIAL' for h in hands),
        HH_INVALID_status='NOT_A_DISTINCT_EXPORT_STATUS; SEE_RETAINED_VALIDATION_AND_BLOCKERS',
        hh_blockers=dict(Counter(b for h in hands for b in h.get('blockers',[]))),
        matrix=json.loads((args.production/'full_hh_production_matrix.json').read_text('utf-8')),
        performance={k:final.get(k) for k in ('wall_seconds','elapsed_seconds','capture_seconds','rtf','RTF') if k in final},
        runtime_errors=final.get('errors'),
        physical_rows=len(physical), OCR_rows=len(ocr), timeline_rows=len(timeline),
        legacy_admitted=len(legacy), legacy_OCR=sum(len(ocr_by_id[r['row_id']])==1 for r in legacy),
        legacy_timeline=sum(len(timeline_by_id[r['row_id']])==1 for r in legacy),
        legacy_expected=len(old_legacy), legacy_source_projection_exact=legacy_exact,
        frames=final.get('frames'), full_10100_completed=full_run,
        audit_scope='NATIVE_FULL_RUN' if full_run and legacy_exact else 'NATIVE_RUN_OR_BASELINE_PARITY_UNPROVEN',
        duplicate_physical_ids=sum(len(v)-1 for v in physical_by_id.values()),
        duplicate_OCR_ids=sum(len(v)-1 for v in ocr_by_id.values()),
        png_count=sum(1 for _ in args.production.rglob('*.png')))
    for name, payload in [('known_704_transport.json',events),('physical_hand_matches.json',hand_audit),
                          ('summary.json',summary)]:
        (args.output/name).write_text(json.dumps(payload,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(summary,ensure_ascii=True))


if __name__ == '__main__':
    main()
