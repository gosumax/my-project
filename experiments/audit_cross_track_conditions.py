"""Read-only causal endpoint diagnostics; never supplies classifier decisions."""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.capture_temporal_evidence import read, sha, write
from experiments.run_cross_track_linker import Frames
from experiments.run_ram_crop_metadata import image_inventory
from experiments.cross_track_linker import _exact,digest
from parser_core.chat_rows import physical_spans,align_upward,match_rows,match_row_blocks,canonical_row,row_distance
from parser_core.layout import load_fixed_layout


def depth_measure(record,before,after):
    old,new=record['old'],record['new'];m=record['motion']
    assert max(old['last_frame'],new['first_frame'])<=record['decision_frame']
    assert (digest(before),digest(after))==(m['before_hash'],m['after_hash'])
    os,ns=physical_spans(before)[1:],physical_spans(after)[1:]
    oi,ni=old['row_index']-1,new['row_index']-1;shift=m['shift_px']
    result=dict(pair_id=record['pair_id'],endpoint_frames=[old['last_frame'],new['first_frame']],decision_frame=record['decision_frame'])
    if m['target_new_index'] is None:
        candidates=[];target_candidates=[];geometry=[];all_target=[]
        for n,sn in enumerate(ns):
            residual=abs(sum(os[oi])/2-sum(sn)/2-shift)
            distance=row_distance(before,os[oi],after,sn)
            all_target.append(dict(new_index=n,center_residual=residual,distance=distance))
            if residual<=7:
                geometry.append(n)
                if distance<=10:target_candidates.append(n)
            for o,so in enumerate(os):
                delta=abs(sum(so)/2-sum(sn)/2-shift)
                if delta<=7:
                    dist=row_distance(before,so,after,sn)
                    if dist<=10:candidates.append((dist+delta,n,o))
        greedy={};used=set();skips=[]
        for score,n,o in sorted(candidates):
            if n not in greedy and o not in used:greedy[n]=o;used.add(o)
            elif o==oi:skips.append(dict(new_index=n,new_consumed=n in greedy,old_consumed=o in used,score=score))
        crossing=[o for n,o in sorted(greedy.items())]!=sorted(greedy.values())
        reason=('ALIGNMENT_NOT_PROVEN' if not m['aligned'] else 'NO_POSITION_ELIGIBLE_NEW_ROW' if not geometry else
                'PIXEL_DISTANCE_OVER_10_FOR_POSITION_ELIGIBLE_ROWS' if not target_candidates else
                'CROSSED_ASSIGNMENT_RESET' if crossing else 'GREEDY_ONE_TO_ONE_COMPETITION')
        reproduced=match_rows(before,os,after,ns,shift) if m['aligned'] else {}
        assert not any(o==oi for o in reproduced.values())
        result.update(kind='MAPPING_MISSING',mapping_reason=reason,position_eligible_indices=geometry,
            target_eligible_indices=target_candidates,greedy_skips=skips,crossed_assignment_reset=crossing,
            all_target_candidates=all_target,min_center_residual=min(r['center_residual'] for r in all_target),
            min_row_distance=min(r['distance'] for r in all_target),
            min_distance_position_eligible=min((r['distance'] for r in all_target if r['center_residual']<=7),default=None))
        return result
    mapped=m['target_new_index'];assert mapped==ni
    x=before[os[oi][0]:os[oi][1],:-12];y=after[ns[ni][0]:ns[ni][1],:-12]
    shape_equal=x.shape==y.shape
    h,w=min(x.shape[0],y.shape[0]),min(x.shape[1],y.shape[1]);xc,yc=x[:h,:w],y[:h,:w]
    delta=np.abs(xc.astype(np.int16)-yc.astype(np.int16));mask=delta>0
    edge=np.zeros((h,w),bool);edge[:2]=True;edge[-2:]=True;edge[:,:2]=True;edge[:,-2:]=True
    background=(xc<=70)&(yc<=70);active=(xc>70)|(yc>70)
    def stats(region):
        vals=delta[region];return dict(pixels=int(region.sum()),changed=int((mask&region).sum()),mae=float(vals.mean()) if vals.size else None,max=int(vals.max()) if vals.size else None)
    offsets=[]
    for dy in range(-2,3):
        xa=max(0,dy);ya=max(0,-dy);size=min(x.shape[0]-xa,y.shape[0]-ya)
        d=np.abs(x[xa:xa+size].astype(np.int16)-y[ya:ya+size].astype(np.int16))
        offsets.append(dict(vertical_offset=dy,overlap_height=size,mae=float(d.mean()),changed_pixels=int((d>0).sum()),exact_overlap=bool(not np.any(d))))
    best=min(offsets,key=lambda r:(r['mae'],abs(r['vertical_offset'])))
    result.update(kind='MAPPED_NATIVE_MISMATCH',old_tight_shape=list(x.shape),new_tight_shape=list(y.shape),shape_equal=shape_equal,
        tight_center_residual=abs(sum(os[oi])/2-sum(ns[ni])/2-shift),registered_row_distance=row_distance(before,os[oi],after,ns[ni]),
        overlap_mae=float(delta.mean()),overlap_max=int(delta.max()),overlap_changed_pixels=int(mask.sum()),
        overlap_pixels=int(mask.size),crop_outer_2px=stats(edge),crop_inner=stats(~edge),background=stats(background),active=stats(active),
        intensity_class_changed_pixels=int(((xc>70)!=(yc>70)).sum()),vertical_offset_diagnostics=offsets,best_vertical_offset=best,
        pure_vertical_translation_within_2px=bool(best['exact_overlap'] and best['overlap_height']>=min(x.shape[0],y.shape[0])-2),
        changed_row_counts=[int(v) for v in mask.sum(axis=1)],changed_col_counts=[int(v) for v in mask.sum(axis=0)],
        diagnosis='Gray-level or support changes measured; antialiasing/compression cause is not proven by raster differences alone.')
    return result


def measure(record,before,after,before_ids,after_ids,origins):
    old,new=record['old'],record['new']; motion=record['motion']
    assert max(old['last_frame'],new['first_frame'])<=record['decision_frame']
    assert (digest(before),digest(after))==(motion['before_hash'],motion['after_hash'])
    os,ns=physical_spans(before)[1:],physical_spans(after)[1:]
    assert len(os)==len(before_ids) and len(ns)==len(after_ids)
    oi,ni=old['row_index']-1,new['row_index']-1
    assert before_ids[oi]==old['track_id'] and after_ids[ni]==new['track_id']
    a=align_upward(before,after)
    assert (a.proven,a.shift_px)==(motion['aligned'],motion['shift_px'])
    matches=match_rows(before,os,after,ns,a.shift_px) if a.proven else {}
    blocks=match_row_blocks([canonical_row(before,s) for s in os],[canonical_row(after,s) for s in ns])
    block_pairs={(o,n) for b in blocks if b.length>=3 for n,o in b.mapping.items()}
    neighbors=[]
    for n,o in matches.items():
        if o==oi or n==ni:continue
        residual=abs(sum(os[o])/2-sum(ns[n])/2-a.shift_px)
        exact=_exact(before,os[o],after,ns[n])
        neighbors.append(dict(old_index=o,new_index=n,old_id=before_ids[o],new_id=after_ids[n],
            center_residual=residual,native_exact=exact,in_block_ge3=(o,n) in block_pairs,
            strict=bool(exact and residual<=1 and (o,n) in block_pairs),
            changed_id=before_ids[o]!=after_ids[n],new_origin=origins[after_ids[n]]))
    strict=sum(n['strict'] for n in neighbors)
    coherent=sum(n['center_residual']<=1 for n in neighbors)
    changed=sum(n['changed_id'] and n['new_origin']=='UNRESOLVED_IDENTITY' for n in neighbors)
    residual=abs(sum(os[oi])/2-sum(ns[ni])/2-a.shift_px) if a.proven else None
    distance=row_distance(before,os[oi],after,ns[ni])
    return dict(pair_id=record['pair_id'],decision_frame=record['decision_frame'],
        endpoint_frames=[old['last_frame'],new['first_frame']],original_verdict=record['link_status'],
        aligned=a.proven,shift_px=a.shift_px,target_center_residual=residual,
        target_y_existing_tolerance_pass=residual<=7 if residual is not None else None,
        target_row_distance=distance,target_existing_distance_pass=distance<=10,
        target_native_exact=_exact(before,os[oi],after,ns[ni]),
        old_span_count=len(os),new_span_count=len(ns),potential_neighbors=min(len(os)-1,len(ns)-1),
        physically_less_than_two_neighbors=min(len(os)-1,len(ns)-1)<2,
        old_edge=oi in (0,len(os)-1),new_edge=ni in (0,len(ns)-1),
        strict_independent_neighbors=strict,threshold_matched_neighbors=len(neighbors),
        center_coherent_threshold_neighbors=coherent,
        changed_id_unresolved_neighbors=changed,neighbors=neighbors,
        literal_unresolved_identity_neighbor_veto=False,
        note='Independent viewport neighbors are descriptive; no target block membership or classifier replacement is inferred.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--depth',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);began=time.perf_counter()
    temporal=ROOT/'runs/temporal_context_smoke17_20261001'
    baseline=ROOT/'runs/production_smoke17_20261001_013909'
    relationships=ROOT/'runs/cross_track_linker_smoke17_20261001/relationships.jsonl'
    tracks_path=ROOT/'runs/ram_metadata_smoke17_20261001/tracks.jsonl'
    manifest=json.loads((ROOT/'production_manifest.json').read_text('utf-8'))
    paths=[ROOT/name for name in manifest['hashes']]+[ROOT/'production_manifest.json',
        ROOT/'experiments/cross_track_linker.py',ROOT/'experiments/run_cross_track_linker.py',relationships,
        tracks_path,temporal/'chat_states.jsonl',temporal/'frame_times.jsonl',baseline/'run_metadata.json']
    source=json.loads((baseline/'run_metadata.json').read_text('utf-8'));paths.append(Path(source['video']))
    print(json.dumps({'phase':'HASH_AND_IMAGE_PREFLIGHT'}),flush=True)
    hashes={str(path):sha(path) for path in paths}
    assert all(hashes[str(ROOT/name)]==expected for name,expected in manifest['hashes'].items())
    assert hashes[source['video']]==source['source_video_sha256']
    inventory=image_inventory()
    records=read(relationships);assert len(records)==590
    tracks={r['track_id']:r for r in read(tracks_path)}
    origins={rid:r['origin'] for rid,r in tracks.items()}
    states=defaultdict(dict)
    for r in read(temporal/'chat_states.jsonl'):states[r['session_id']][r['source_frame']]=r['track_ids']
    frame_keys={sid:sorted(values) for sid,values in states.items()}
    def state(sid,fid):return states[sid][frame_keys[sid][bisect_right(frame_keys[sid],fid)-1]]
    times=read(temporal/'frame_times.jsonl')
    layout=load_fixed_layout(ROOT/'configs/nine_table_layout.json',ROOT/'configs/layout_847x404.json',ROOT/'configs/chat_log_bottom_rois.json')
    reader=Frames(Path(source['video']),times,layout);result=[];started=time.perf_counter();last=started
    try:
        for i,r in enumerate(records):
            o,n=r['old'],r['new'];assert n['first_frame']==r['decision_frame']
            before=reader.get(o['last_frame'],o['table_id']);after=reader.get(n['first_frame'],n['table_id'])
            result.append(depth_measure(r,before,after) if args.depth else measure(r,before,after,state(o['session_id'],o['last_frame']),state(n['session_id'],n['first_frame']),origins))
            del before,after
            if time.perf_counter()-last>=20:
                print(json.dumps({'phase':'ENDPOINTS','pairs':i+1,'seconds':time.perf_counter()-started}),flush=True);last=time.perf_counter()
    finally:images_end=reader.close()
    replay_seconds=time.perf_counter()-started
    print(json.dumps({'phase':'POSTFLIGHT','pairs':len(result)}),flush=True)
    hashes_after={str(path):sha(path) for path in paths};assert hashes_after==hashes
    after_inventory=image_inventory();assert after_inventory==inventory
    if args.depth:
        missing=[r for r in result if r['kind']=='MAPPING_MISSING'];pixels=[r for r in result if r['kind']=='MAPPED_NATIVE_MISMATCH']
        summary=dict(pairs=len(result),mapping_missing=len(missing),mapping_reasons=dict(Counter(r['mapping_reason'] for r in missing)),
            mapped_native_mismatch=len(pixels),shape_equal=sum(r['shape_equal'] for r in pixels),
            shape_different=sum(not r['shape_equal'] for r in pixels),
            pure_vertical_translation_within_2px=sum(r['pure_vertical_translation_within_2px'] for r in pixels),
            best_nonzero_vertical_offset=sum(r['best_vertical_offset']['vertical_offset']!=0 for r in pixels),
            inner_pixels_changed_pairs=sum(r['crop_inner']['changed']>0 for r in pixels),
            active_pixels_changed_pairs=sum(r['active']['changed']>0 for r in pixels),
            background_pixels_changed_pairs=sum(r['background']['changed']>0 for r in pixels),
            intensity_class_changed_pairs=sum(r['intensity_class_changed_pixels']>0 for r in pixels),
            hashes_unchanged=True,source_hashes=hashes,image_writes=0,image_inventory_count=len(inventory),
            replay_seconds=replay_seconds,total_seconds=time.perf_counter()-began,roi_requests=reader.requests,
            decoded_seek_frames=reader.decoded,roi_peak_images=reader.peak_images,roi_images_at_end=images_end,
            caveat='Existing thresholds only; offset search and gray-level descriptors never enter verdicts. No antialiasing/compression causal attribution.')
        write(args.output/'mapping_rejection_details.jsonl',missing);write(args.output/'pixel_change_details.jsonl',pixels)
        (args.output/'pixel_change_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
        print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'}),flush=True)
        return
    bucket=lambda key:dict(Counter('2+' if r[key]>=2 else str(r[key]) for r in result))
    summary=dict(pairs=len(result),strict_independent_neighbor_buckets=bucket('strict_independent_neighbors'),
        threshold_matched_neighbor_buckets=bucket('threshold_matched_neighbors'),
        center_coherent_threshold_neighbor_buckets=bucket('center_coherent_threshold_neighbors'),
        viewport_edges=sum(r['old_edge'] or r['new_edge'] for r in result),
        physically_less_than_two_neighbors=sum(r['physically_less_than_two_neighbors'] for r in result),
        pairs_with_changed_unresolved_neighbor=sum(r['changed_id_unresolved_neighbors']>0 for r in result),
        pairs_with_two_threshold_neighbors_but_less_than_two_same_id_neighbors=sum(r['threshold_matched_neighbors']>=2 and sum(not n['changed_id'] for n in r['neighbors'])<2 for r in result),
        strict_neighbors_changed_unresolved=sum(n['strict'] and n['changed_id'] and n['new_origin']=='UNRESOLVED_IDENTITY' for r in result for n in r['neighbors']),
        literal_unresolved_identity_veto_pairs=0,target_y_pass=sum(r['target_y_existing_tolerance_pass'] is True for r in result),
        target_y_fail=sum(r['target_y_existing_tolerance_pass'] is False for r in result),
        target_y_unknown=sum(r['target_y_existing_tolerance_pass'] is None for r in result),
        target_distance_pass=sum(r['target_existing_distance_pass'] for r in result),
        target_native_exact=sum(r['target_native_exact'] for r in result),
        native_exact_among_gap1=sum(r['target_native_exact'] and record['features']['delta_frame']==1 for r,record in zip(result,records)),
        replay_seconds=replay_seconds,total_seconds=time.perf_counter()-began,roi_requests=reader.requests,
        decoded_seek_frames=reader.decoded,roi_peak_images=reader.peak_images,roi_peak_bytes=reader.peak_bytes,
        roi_images_at_end=images_end,image_writes=0,image_inventory_count=len(inventory),
        hashes_unchanged=True,source_hashes=hashes,original_verdict_counts=dict(Counter(r['original_verdict'] for r in result)),
        causal_safety='Only endpoint snapshots and ROIs <= new first frame. Origin labels are fixed first-observation metadata. Terminal lifetimes are not inspected.',
        caveat='Independent neighbors need not share a contiguous target block and cannot replace the frozen decision gate. Changed UNRESOLVED IDs are correlation only, not a gate veto.')
    write(args.output/'independent_neighbors.jsonl',result)
    (args.output/'neighbor_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

