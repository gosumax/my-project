"""Offline Smoke17 relationship experiment. Source rows/production stay immutable."""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import threading
import time
import weakref
import av
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.capture_temporal_evidence import read,sha,write
from experiments.run_ram_crop_metadata import image_inventory
from experiments.cross_track_linker import (TrackSnapshot,motion_proof,decide,text_similarity,
                                          SAME,DIFFERENT,AMBIGUOUS,digest,overlap_at)
from parser_core.ram_first_stage import png_compatible_gray
from parser_core.chat_rows import physical_spans,padded_row_span,align_upward,match_rows
from parser_core.layout import load_fixed_layout
from parser_core.edge_gate import classify_edges


class Frames:
    """Three ROI cache slots: before/after/optional one-frame bridge; no image IO."""
    def __init__(self,video,times,layout):
        self.source=av.open(str(video));self.stream=self.source.streams.video[0]
        self.times=times;self.slots={f'T{s.slot_id:02}':s for s in layout.slots}
        self.cache=OrderedDict();self.references=[];self.decoded=0;self.requests=0
        self.peak_images=self.peak_bytes=0
    def get(self,fid,table):
        key=(fid,table);self.requests+=1
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        stamp=self.times[fid];pts=stamp['source_pts']
        self.source.seek(pts,stream=self.stream,backward=True,any_frame=False)
        found=None
        for frame in self.source.decode(self.stream):
            self.decoded+=1
            if frame.pts==pts:
                base=frame.time_base or self.stream.time_base
                assert (base.numerator,base.denominator)==(stamp['time_base_num'],stamp['time_base_den'])
                rgb=frame.to_ndarray(format='rgb24')
                x1,y1,x2,y2=self.slots[table].chat_box
                found=png_compatible_gray(rgb[y1:y2,x1:x2]);del rgb
                break
            if frame.pts is not None and frame.pts>pts:break
        if found is None:raise ValueError(f'PTS frame not recovered: {fid}/{pts}')
        found.flags.writeable=False
        self.cache[key]=found
        while len(self.cache)>3:self.cache.popitem(last=False)
        self.references=[ref for ref in self.references if ref() is not None]
        self.references.append(weakref.ref(found))
        live=[ref() for ref in self.references if ref() is not None]
        self.peak_images=max(self.peak_images,len(live));self.peak_bytes=max(self.peak_bytes,sum(v.nbytes for v in live))
        return found
    def close(self):
        self.cache.clear();self.source.close();gc.collect()
        return sum(ref() is not None for ref in self.references)


def causal_old_frame(track,new_frame,current_ids,old_id):
    if old_id in current_ids:
        return new_frame
    observed=[min(r,new_frame-1) for l,r in track['visible_spans'] if l<new_frame]
    if not observed:raise ValueError('No causal old observation')
    return max(observed)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter()
    print(json.dumps({'phase':'PREFLIGHT'}),flush=True)
    manifest=json.loads((ROOT/'production_manifest.json').read_text('utf-8'))
    pinned={name:sha(ROOT/name) for name in manifest['hashes']};assert pinned==manifest['hashes']
    temporal=ROOT/'runs/temporal_context_smoke17_20261001'
    ram=ROOT/'runs/ram_metadata_smoke17_20261001'
    baseline=ROOT/'runs/production_smoke17_20261001_013909'
    input_paths=[temporal/n for n in ('duplicate_analysis.jsonl','chat_states.jsonl','frame_times.jsonl',
        'confirmed_starts.jsonl','persistent_previous_tails.jsonl')]+[ram/'tracks.jsonl']+[
        baseline/n for n in ('physical_rows.jsonl','ordered_timeline.jsonl','run_metadata.json')]
    input_hashes={str(path.relative_to(ROOT)):sha(path) for path in input_paths}
    code_hashes={name:sha(ROOT/name) for name in ('experiments/cross_track_linker.py','experiments/run_cross_track_linker.py')}
    source=json.loads((baseline/'run_metadata.json').read_text('utf-8'));assert sha(source['video'])==source['source_video_sha256']
    pairs=read(temporal/'duplicate_analysis.jsonl');assert len(pairs)==590
    tracks={t['track_id']:t for t in read(ram/'tracks.jsonl')}
    events=read(baseline/'ordered_timeline.jsonl');events_byid={r['stable_row_id']:r for r in events}
    times=read(temporal/'frame_times.jsonl')
    assert len(times)==source['frames_requested']
    assert all(isinstance(r['source_pts'],int) and r['source_frame']==i for i,r in enumerate(times))
    assert len({r['source_pts'] for r in times})==len(times),'ExactPTSlookuprequiresuniquePTS'
    bounds=read(temporal/'confirmed_starts.jsonl');tails=read(temporal/'persistent_previous_tails.jsonl')
    physical={}
    with (baseline/'physical_rows.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line);r.pop('row_pixels_base64');physical[r['row_id']]=r
    states=defaultdict(dict)
    with (temporal/'chat_states.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line);states[r['session_id']][r['source_frame']]=tuple(tracks[rid]['track_id'] for rid in r['track_ids'])
    state_frames={sid:sorted(data) for sid,data in states.items()}
    def state(sid,fid):
        frames=state_frames[sid];i=bisect_right(frames,fid)-1
        return states[sid][frames[i]] if i>=0 else ()
    def state_frame(sid,fid):
        frames=state_frames[sid];i=bisect_right(frames,fid)-1
        return frames[i] if i>=0 else None
    layout=load_fixed_layout(ROOT/'configs/nine_table_layout.json',ROOT/'configs/layout_847x404.json',ROOT/'configs/chat_log_bottom_rois.json')
    inventory=image_inventory()
    process=psutil.Process();gc.collect();rss_start=process.memory_info().rss;peak=[rss_start];stop=threading.Event()
    def sample():
        while not stop.wait(.05):peak[0]=max(peak[0],process.memory_info().rss)
    sampler=threading.Thread(target=sample,daemon=True);sampler.start()
    reader=Frames(Path(source['video']),times,layout)
    records=[];last_progress=time.perf_counter();capture_started=last_progress
    def endpoint(rid,fid,gray,ids,*,first=False):
        t=tracks[rid];idx=ids.index(rid);all_spans=physical_spans(gray)
        assert len(ids)==len(all_spans)-1
        a,b=padded_row_span(all_spans,idx+1,gray.shape[0]);pixels=gray[a:b]
        spans=tuple((l,min(r,fid)) for l,r in t['visible_spans'] if l<=fid)
        event=events_byid[rid]
        if first:
            saved=physical[rid]
            assert (fid,idx+1,a,digest(pixels))==(saved['frame_id'],saved['source_span_index'],saved['y1'],saved['row_pixel_sha256'])
        return TrackSnapshot(rid,f"T{t['table_id']:02}",t['session_id'],t['first_seen_frame'],fid,
            t['first_seen_ms'],times[fid]['timestamp_ms'],idx+1,a,digest(pixels),event['raw_ocr_text'],
            event['normalized_text'],event['confidence'],spans,event['frame_id'])
    try:
        for index,pair in enumerate(sorted(pairs,key=lambda p:tracks[p['row_id']]['first_seen_frame'])):
            old_id,new_id=pair['candidate_canonical_row_id'],pair['row_id']
            old_t,new_t=tracks[old_id],tracks[new_id];t=new_t['first_seen_frame'];sid=new_t['session_id']
            old_frame=causal_old_frame(old_t,t,state(sid,t),old_id)
            table=f"T{new_t['table_id']:02}"
            before=reader.get(old_frame,table);after=reader.get(t,table)
            before_ids,after_ids=state(sid,old_frame),state(sid,t)
            old=endpoint(old_id,old_frame,before,before_ids)
            new=endpoint(new_id,t,after,after_ids,first=True)
            bridge=None
            if t-old_frame==2:bridge=reader.get(t-1,table)
            proof=motion_proof(before,after,old.row_index-1,new.row_index-1,before_ids,after_ids,
                old_frame,t,bridge=bridge,bridge_frame=t-1 if bridge is not None else None)
            boundaries=tuple(b['frame_id'] for b in bounds if b['session_id']==sid and old_frame<b['frame_id']<=t)
            prev_f=state_frame(sid,t-1);previous=reader.get(prev_f,table) if prev_f is not None else None
            previous_ids=state(sid,t-1)
            onset=None;immediate={}
            if previous is not None:
                a=align_upward(previous,after);old_spans=physical_spans(previous)[1:];new_spans=physical_spans(after)[1:]
                matching=match_rows(previous,old_spans,after,new_spans,a.shift_px) if a.proven else {}
                minimum=max(1,min(len(old_spans),len(new_spans))//2) if a.shift_px==0 else 3
                immediate=dict(before_frame=prev_f,shift_px=a.shift_px,alignment_proven=a.proven,
                    alignment_score=a.score if np.isfinite(a.score) else None,
                    matched_rows=len(matching),required_matches=minimum,
                    production_alignment_rejected=not a.proven or len(matching)<minimum)
                # For a causal immediate onset proof the preceding changed
                # ROI must itself be the immediately preceding decoded frame.
                if prev_f==t-1 and old_spans:
                    anchor_old=next(iter(matching.values()),0)
                    onset=motion_proof(previous,after,anchor_old,new.row_index-1,previous_ids,after_ids,prev_f,t)
            simultaneous=(old.row_index,new.row_index) if old_id in after_ids else None
            result=decide(old,new,proof,t,boundary_frames=boundaries,onset_proof=onset,
                          simultaneous_positions=simultaneous)
            observations=[]
            if proof.aligned and proof.shift_px>0:observations.append('chat scroll / vertical shift')
            if proof.local_loss_bridge:observations.append('temporary disappearance')
            if len(old.visible_spans)>1 or len(new.visible_spans)>1:observations.append('visibility gap')
            if proof.before_hash!=proof.after_hash:observations.append('ROI change')
            if immediate.get('production_alignment_rejected'):observations.append('ambiguous alignment')
            if old.pixel_hash!=new.pixel_hash:observations.append('pixel change')
            if result.link_status==DIFFERENT and old.normalized_text==new.normalized_text:observations.append('same text but new physical row')
            if old.session_id!=new.session_id or old.table_id!=new.table_id:observations.append('session/table boundary')
            primary=('ambiguous alignment' if immediate.get('production_alignment_rejected') else
                     'UNKNOWN / insufficient evidence')
            old_full={k:old_t[k] for k in ('first_seen_ms','last_seen_ms','visible_spans','observation_count','frame_id','pixel_hash','row_key')}
            new_full={k:new_t[k] for k in old_full}
            records.append(dict(pair_id=f'{old_id}->{new_id}',old=asdict(old),new=asdict(new),
                old_terminal_metadata=old_full,new_terminal_metadata=new_full,
                terminal_metadata_is_future_audit_only=True,
                source_ocr_parents={rid:{k:tracks[rid][k] for k in ('source_frame','source_row_key','source_pixel_hash')} for rid in (old_id,new_id)},
                features=dict(delta_ms=new.first_ms-old.last_ms,delta_frame=t-old_frame,delta_y=new.y1-old.y1,
                    text_similarity=text_similarity(old,new),pixel_similarity=proof.pixel_similarity,
                    same_table=old.table_id==new.table_id,same_session=old.session_id==new.session_id,
                    causal_visibility_overlap=overlap_at(old,new,t),terminal_interleaved=new_t['first_seen_frame']<old_t['last_seen_frame'],
                    hand_boundary_frames=boundaries),motion=asdict(proof),onset=asdict(onset) if onset else None,
                immediate_transition=immediate,primary_break_cause=primary,observed_categories=observations,
                **asdict(result),sources=pair['sources']))
            del before,after,previous,bridge
            if time.perf_counter()-last_progress>30:
                print(json.dumps(dict(phase='PAIRS',pairs=index+1,wall_seconds=time.perf_counter()-capture_started)),flush=True)
                last_progress=time.perf_counter()
    finally:
        images_end=reader.close();stop.set();sampler.join()
    extraction_seconds=time.perf_counter()-capture_started
    counts=Counter(r['link_status'] for r in records)
    review=[];shortfalls={}
    for status in (SAME,DIFFERENT,AMBIGUOUS):
        selected=[r for r in records if r['link_status']==status][:20]
        review.extend(selected)
        shortfalls[status]=max(0,20-len(selected))
    # Simulation preserves every source field and row. Attach relationships to
    # copies only; current gate has no relationship consumer. No epoch edits.
    baseline_edges=classify_edges(events).rows
    copied=[dict(r) for r in events];copy_byid={r['stable_row_id']:r for r in copied}
    same_links=[r for r in records if r['link_status']==SAME]
    for r in same_links:
        copy_byid[r['new_track_id']]['cross_track_relationship']=dict(linked_track_id=r['old_track_id'],link_status=SAME,link_reason=r['link_reason'])
    simulated=classify_edges(copied).rows
    assert all(all(n[k]==v for k,v in o.items()) for o,n in zip(events,copied))
    unresolved=[r for r in baseline_edges if r['edge_status']=='UNRESOLVED_EDGE']
    resolved=[(o,n) for o,n in zip(baseline_edges,simulated) if o['edge_status']=='UNRESOLVED_EDGE' and n['edge_status']!='UNRESOLVED_EDGE']
    assert not resolved,'Metadata-only current gate should remain identical'
    candidates={r['new_track_id'] for r in same_links}
    simulation=dict(baseline_unresolved=len(unresolved),potentially_resolved_under_existing_gate=len(resolved),
        still_unresolved=sum(r['edge_status']=='UNRESOLVED_EDGE' for r in simulated),
        unresolved_with_proven_relationship=sum(r['stable_row_id'] in candidates for r in unresolved),
        linking_does_not_help_current_gate=len(unresolved),
        reason='Existing gate does not consume relationship metadata; epochs/boundaries/source rows unchanged',
        decisions_changed=0,source_rows_before=len(events),source_rows_after=len(copied))
    write(args.output/'edge_simulation_rows.jsonl',({k:r[k] for k in ('stable_row_id','frame_id','edge_status')} for r in simulated))
    # Same precomputed pairs/frames/features: paired rule cost, extraction is
    # shared and disclosed separately. No future terminal fields enter decide.
    benchmarks=[]
    for mode in ('metadata','linker'):
        samples=[];mode_rss_start=process.memory_info().rss;mode_rss_peak=mode_rss_start
        for repeat in range(20):
            tick=time.perf_counter()
            for r in records:
                old=TrackSnapshot(**r['old']);new=TrackSnapshot(**r['new'])
                text_similarity(old,new)
                if mode=='linker':
                    from experiments.cross_track_linker import MotionProof
                    pr=MotionProof(**r['motion']);op=MotionProof(**r['onset']) if r['onset'] else None
                    decide(old,new,pr,r['decision_frame'],boundary_frames=tuple(r['features']['hand_boundary_frames']),onset_proof=op,
                        simultaneous_positions=(old.row_index,new.row_index) if r['features']['causal_visibility_overlap'] else None)
            samples.append(time.perf_counter()-tick)
            mode_rss_peak=max(mode_rss_peak,process.memory_info().rss)
        benchmarks.append(dict(mode=mode,pairs=590,repeats=20,median_seconds=float(np.median(samples)),
            min_seconds=min(samples),max_seconds=max(samples),rss_start_bytes=mode_rss_start,
            rss_sampled_peak_bytes=mode_rss_peak,image_disk_writes=0,
            memory_measurement='RSS at each repeat boundary; exact allocation peak not measured'))
    overhead=(benchmarks[1]['median_seconds']/benchmarks[0]['median_seconds']-1)*100
    categories=('chat scroll / vertical shift','temporary disappearance','visibility gap','ROI change','ambiguous alignment',
                'same text but new physical row','pixel change','missing observation','session/table boundary','UNKNOWN / insufficient evidence')
    causes={c:sum(c in r['observed_categories'] or c==r['primary_break_cause'] for r in records) for c in categories}
    persistent_ids={p['track_id'] for p in tails}
    persistent_pairs=[r for r in records if r['old_track_id'] in persistent_ids]
    # Check complete source-lifetimes retrospectively for contradictions. This
    # does not feed an earlier SAME decision.
    false_overlap=[]
    for r in same_links:
        a,b=tracks[r['old_track_id']],tracks[r['new_track_id']]
        if any(max(l,s)<=min(e,t) for l,e in a['visible_spans'] for s,t in b['visible_spans']):false_overlap.append(r['pair_id'])
    assert not false_overlap,'Retrospective false-link overlap found'
    assert input_hashes=={str(path.relative_to(ROOT)):sha(path) for path in input_paths}
    assert sha(source['video'])==source['source_video_sha256']
    assert pinned=={name:sha(ROOT/name) for name in pinned}
    assert code_hashes=={name:sha(ROOT/name) for name in code_hashes}
    after_inventory=image_inventory();created=set(after_inventory)-set(inventory)
    modified={k for k in set(inventory)&set(after_inventory) if inventory[k]!=after_inventory[k]}
    write(args.output/'image_inventory_audit.jsonl',[dict(before=len(inventory),after=len(after_inventory),new_files=sorted(created),modified_files=sorted(modified))])
    assert not created and not modified
    write(args.output/'relationships.jsonl',records)
    write(args.output/'case_audit.jsonl',review)
    result=dict(pair_count=590,before_ambiguous=590,after_counts={s:counts[s] for s in (SAME,DIFFERENT,AMBIGUOUS)},
        category_observations=causes,category_semantics='Overlapping observations, not all causal diagnoses',
        primary_break_causes=dict(Counter(r['primary_break_cause'] for r in records)),
        physical_extraction_seconds=extraction_seconds,decoded_seek_frames=reader.decoded,roi_requests=reader.requests,
        peak_roi_image_objects=reader.peak_images,peak_roi_payload_bytes=reader.peak_bytes,roi_images_at_end=images_end,
        rss_start_bytes=rss_start,rss_peak_bytes=peak[0],performance=benchmarks,analysis_overhead_percent=overhead,
        performance_scope='Same590precomputedpairset,20repeats; extraction excluded and separately timed; notvideoRTF',
        audit_counts=dict(Counter(r['link_status'] for r in review)),audit_shortfalls=shortfalls,
        persistent_previous_pairs=296,candidates_with_persistent_old_track=len(persistent_pairs),
        persistent_old_same_links=sum(r['link_status']==SAME for r in persistent_pairs),retrospective_overlap_false_links=false_overlap,
        edge_simulation=simulation,new_png_files=0,new_jpg_files=0,new_webp_files=0,source_data_changed=False,
        production_code_changed=False,ocr_runs=0,manifest_sha256=sha(ROOT/'production_manifest.json'),
        input_hashes=input_hashes,experiment_hashes=code_hashes,total_seconds=time.perf_counter()-began)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
