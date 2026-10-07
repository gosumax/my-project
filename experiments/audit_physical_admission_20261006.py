"""Bounded saved-pixel audit of supplemental admission; not a production runner."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import zlib
from fractions import Fraction
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.chat_tracker import AdjacentChatTracker
from parser_core.physical_chat_admission import PhysicalChatAdmission
from parser_core.ram_first_stage import FirstStage


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-frame', type=int, default=10099)
    p.add_argument('--target-pairs-only',action='store_true')
    p.add_argument('--stage-on',action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.stage_on and args.target_pairs_only:
        raise ValueError('Full stage parity requires continuous observations')
    audit = ROOT / 'runs/chat_sequence_audit_20261006'
    db = sqlite3.connect((audit/'source_reconstruction/source_pixels.sqlite').as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    verified = json.loads((audit/'verified_omissions.json').read_text('utf-8'))
    source_positions=defaultdict(set)
    snapshot_positions=defaultdict(set)
    for hand in verified['reviewed_hands']:
        for row in hand['lines']:
            snapshot_positions[hand['table'],hand['snapshot_frame'],hand['snapshot_sha256'],row['track']].add(tuple(row['span']))
    with (audit/'stitched_sequence/events.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            for row in json.loads(line)['source_rows']:
                for track in row['source_tracks']:
                    source_positions[row['table_id'],row['first_frame'],track].add(tuple(row['first_span']))
    targets = {(e['table_id'], e['frame']) for e in verified['events']}
    target_records = {}
    counts = Counter()
    started = time.perf_counter()
    source_hash = hashlib.sha256((ROOT/'parser_core/chat_tracker.py').read_bytes()).hexdigest()
    baseline_hash = hashlib.sha256((ROOT/'runs/full_hh_production_20261006/before/files/parser_core/chat_tracker.py').read_bytes()).hexdigest()
    assert source_hash == baseline_hash, 'Legacy tracker changed'
    baseline=ROOT/'runs/smoke19_batched_storage_full_20261006_013548/production'
    stage=None;observer={};legacy_rows=[]
    if args.stage_on:
        metadata=json.loads((baseline/'run_metadata.json').read_text('utf-8'))
        stage=FirstStage(Path('unused'),metadata['table_sessions'],args.max_frame+1,root=ROOT,
            physical_admission=True,tracker_observer=lambda r:observer.update(transition=r),
            admission_observer=lambda r:observer.update(receipt=r))
    rows_stream=(args.output/'stage_rows.jsonl').open('w',encoding='utf-8')
    with (args.output/'transitions.jsonl').open('w',encoding='utf-8') as stream:
        for table in range(1, 10):
            tracker = AdjacentChatTracker(table, f'audit:T{table:02}')
            fallback = PhysicalChatAdmission(table, f'audit:T{table:02}')
            if args.target_pairs_only:
                sources=[]
                for _,frame in sorted(k for k in targets if k[0]==table and k[1]<=args.max_frame):
                    pair=db.execute('select * from chats where table_id=? and frame<=? order by frame desc limit 2',(table,frame)).fetchall()[::-1]
                    sources.extend((row,index==0) for index,row in enumerate(pair))
            else:
                sources=((row,False) for row in db.execute('select * from chats where table_id=? and frame<=? order by frame', (table,args.max_frame)))
            for row,reset in sources:
                if reset:
                    tracker=AdjacentChatTracker(table,f'audit:T{table:02}')
                    fallback=PhysicalChatAdmission(table,f'audit:T{table:02}')
                gray = np.frombuffer(zlib.decompress(row['gray_zlib']),np.uint8).reshape(row['height'],row['width'])
                assert hashlib.sha256(gray.tobytes()).hexdigest() == row['sha256']
                if stage is None:
                    transition = tracker.process(gray, row['frame'])
                    receipt = fallback.process(gray, transition)
                else:
                    observer.clear()
                    time_base=Fraction(*json.loads(row['time_base']))
                    emitted=list(stage.chat_rows(gray,frame_id=row['frame'],table_id=table,
                        session_id=metadata['table_sessions'][str(table)],source_pts=row['pts'],
                        time_base=time_base,video_time=float(row['pts']*time_base)))
                    transition=observer['transition'];receipt=observer.get('receipt')
                    for item in emitted:
                        record=dict(row_id=item.stable_row_id,table_id=item.table_id,frame_id=item.frame_id,
                            source_span_index=item.source_span_index,y1=item.y1,epoch=item.epoch,origin=item.origin,
                            source_pts=item.source_pts,time_base_num=item.time_base_num,time_base_den=item.time_base_den,
                            row_pixel_sha256=item.pixel_sha256,pixel_shape=list(item.pixels.shape))
                        rows_stream.write(json.dumps(record)+'\n')
                        if ':R' in item.stable_row_id:legacy_rows.append(record)
                counts['views'] += 1
                counts[transition['classification']] += 1
                counts['legacy_rows'] += len(transition['OCR_queue_row_ids'])
                if receipt:
                    counts['fallback_rows'] += len(receipt['admissions'])
                    counts['fallback_'+receipt['proof']['reason']] += 1
                    stream.write(json.dumps(receipt)+'\n')
                if (table,row['frame']) in targets:
                    target_records[table,row['frame']] = dict(transition=transition, fallback=receipt)
            print(json.dumps(dict(table=table,counts=counts,wall=time.perf_counter()-started)),flush=True)
    rows_stream.close()
    results=[]
    totals=defaultdict(Counter)
    for event in verified['events']:
        record=target_records.get((event['table_id'],event['frame']))
        original_spans=event.get('source_spans')
        source_spans=[]
        span_provenance=[]
        for track in event['source_tracks']:
            positions=snapshot_positions[event['table_id'],event['frame'],event.get('source_chat_sha256'),track]
            origin='VERIFIED_HAND_SNAPSHOT_EXACT_FRAME_AND_CHAT_HASH'
            if len(positions)!=1:
                positions=source_positions[event['table_id'],event['frame'],track]
                origin='STITCHED_SOURCE_FIRST_FRAME_GEOMETRY'
            if len(positions)!=1:
                source_spans=[]
                span_provenance.append(dict(track=track,status='UNKNOWN_SOURCE_FRAME_SPAN'))
                break
            span=list(next(iter(positions)))
            source_spans.append(span)
            span_provenance.append(dict(track=track,span=span,source_frame=event['frame'],origin=origin))
        covered=[]
        if record and source_spans:
            for span in source_spans:
                matches=[a for a in (record['fallback'] or {}).get('admissions',[])
                         if abs(sum(a['span'])-sum(span))<=4]
                covered.append([a['row_id'] for a in matches])
        recovered=bool(covered) and all(len(x)==1 for x in covered)
        result=dict(event, original_audit_source_spans=original_spans,
            source_spans=source_spans,source_span_provenance=span_provenance,
            source_span_scope='EXACT_SNAPSHOT_OR_FIRST_FRAME_ONLY; BEST_FRAME_TRACK_SPAN_NOT_REUSED',
            supplemental_rows=covered,
            physically_new_status='PROVEN' if recovered else 'UNKNOWN',
            fallback_recoverable=recovered, OCR_status='NOT_RUN',timeline_status='NOT_RUN',
            legacy_classification=record['transition']['classification'] if record else None,
            legacy_reason=record['transition']['reason'] if record else None,
            legacy_OCR_admissions=record['transition']['OCR_queue_row_ids'] if record else [],
            proof=(record['fallback'] or {}).get('proof') if record else None)
        results.append(result)
        totals[event['kind']]['physical_verified']+=1
        totals[event['kind']]['fallback_recoverable']+=recovered
    (args.output/'verified_704_after.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),'utf-8')
    summary=dict(counts=counts,known_omissions={k:dict(v) for k,v in totals.items()},
                 legacy_code_hash_equal=True,wall_seconds=time.perf_counter()-started,
                 scope=('TARGET_PAIRS_ONLY; RESET_IDS_NOT_FULL_PARITY' if args.target_pairs_only else
                        'SAVED_CHANGED_GRAY_VIEWS; NO_NATIVE_RGB_CALLBACK_OR_OCR_PARITY'))
    if stage is not None:
        with (baseline/'physical_rows.jsonl').open(encoding='utf-8') as stream:
            expected=[json.loads(line) for line in stream if line.strip()]
        fields=('row_id','table_id','frame_id','source_span_index','y1','epoch','origin',
                'source_pts','time_base_num','time_base_den','row_pixel_sha256','pixel_shape')
        expected=[{k:r[k] for k in fields} for r in expected if r['frame_id']<=args.max_frame]
        key=lambda r:(r['table_id'],r['frame_id'],r['source_span_index'])
        actual=sorted(legacy_rows,key=key);expected=sorted(expected,key=key)
        summary['legacy_full_rows_expected']=len(expected)
        summary['legacy_full_rows_actual']=len(actual)
        summary['legacy_full_rows_exact']=actual==expected
        summary['stage_on']=True
        summary['legacy_mismatches']=[dict(expected=e,actual=a) for e,a in zip(expected,actual) if e!=a][:5]
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
    print(json.dumps(summary),flush=True)
    if stage is not None:
        assert summary['legacy_full_rows_exact'], 'Legacy source IDs/order/geometry/PTS/pixels changed'


if __name__=='__main__':
    main()
