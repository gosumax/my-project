"""Causal, metadata-only text-first simulation; never edits source identities."""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
import psutil
from parser_core.edge_gate import classify_edges
from parser_core.production_integrity import load_manifest, validate_integrity

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def content_pass(record):
    """An unavailable or failed comparison must never become truthy support."""
    value=record.get('within_causal_neighbor_control_envelope')
    return value is True or value=='PASS'

def readable_text(record, cutoff):
    if not record or record.get('frame_id',cutoff+1)>cutoff:return False
    return bool(str(record.get('normalized_text') or '').strip() and str(record.get('raw_ocr_text') or '').strip())

def repeat_flags(text):
    s=text.lower();flags=[]
    for label, pattern in [('ANTE',r'\bante\b'),('SB_BB',r'\b(?:small blind|big blind|sb|bb)\b'),('FOLD',r'\bfolds?\b'),('CHECK',r'\bchecks?\b')]:
        if re.search(pattern,s):flags.append(label)
    if re.fullmatch(r'[\d\s.,$€₽]+',s):flags.append('AMOUNT_ONLY')
    if len(s.split())<=2:flags.append('SHORT_TEXT')
    if re.search(r'dealer:\s+\S+\s+(?:folds?|checks?|calls?|bets?|raises?)\b',s):flags.append('NICK_ACTION')
    return flags

def evaluate(pair, old_ids, new_ids, history_ids, texts, neighbors, geometry, support=False):
    t=pair['decision_frame'];old=pair['old'];new=pair['new'];text=old['normalized_text']
    def matching(ids,cutoff=t):
        return [i for i in ids if readable_text(texts.get(i),cutoff) and texts[i]['normalized_text']==text]
    old_match=matching(old_ids,old['last_frame']);new_match=matching(new_ids)
    history_match=sorted(set(matching(history_ids))-{old['track_id'],new['track_id']})
    # Unknown OCR on any viewport row prevents a claim of text uniqueness.
    unknown=sorted(set([i for i in old_ids if not readable_text(texts.get(i),old['last_frame'])]+[i for i in new_ids if not readable_text(texts.get(i),t)]))
    mapping_context=(pair['motion']['aligned'] and old['table_id']==new['table_id'] and old['session_id']==new['session_id'] and old['last_frame']<new['first_frame']==t and not pair['features']['causal_visibility_overlap'])
    possible=[i for i in new_match if geometry.get(i) is not None and geometry[i]<=7] if mapping_context else []
    mapping=('UNIQUE_TEXT_GEOMETRY_MAPPING' if len(possible)==1 else 'MULTIPLE_TEXT_GEOMETRY_CANDIDATES' if len(possible)>1 else 'NO_TEXT_GEOMETRY_MAPPING')
    text_class='TEXT_DUPLICATED_IN_VIEWPORT' if len(old_match)>1 or len(new_match)>1 else 'TEXT_DUPLICATED_IN_TIME_WINDOW' if history_match else 'TEXT_UNIQUE'
    coherent=[n for n in neighbors['neighbors'] if n['center_residual']<=1 and n['old_id']==n['new_id']]
    gap=pair['features']['delta_frame'];immediate=gap==1 or (gap==2 and pair['motion']['local_loss_bridge'])
    conditions=dict(text_identical=bool(text) and text==new['normalized_text'], text_unique=text_class=='TEXT_UNIQUE' and not unknown and len(old_match)==len(new_match)==1,
        same_table_session=old['table_id']==new['table_id'] and old['session_id']==new['session_id'],causal_order=old['last_frame']<new['first_frame']==t,
        unique_geometry=mapping=='UNIQUE_TEXT_GEOMETRY_MAPPING' and possible==[new['track_id']],no_simultaneous=not pair['features']['causal_visibility_overlap'],
        movement=pair['motion']['aligned'],two_consistent_neighbors=len(coherent)>=2,immediate_continuity=immediate,no_boundary=not pair['features']['hand_boundary_frames'],
        supporting_content=bool(support))
    reasons=[k.upper() for k,v in conditions.items() if not v]
    flags=repeat_flags(text)
    # Repeat categories are not a text veto; positive geometry/neighbors/content
    # remain mandatory, and duplicated occurrences are always blocked.
    return dict(pair_id=pair['pair_id'],decision_frame=t,text_class=text_class,old_viewport_text_count=len(old_match),new_viewport_text_count=len(new_match),
        history_other_same_text_ids=history_match,unknown_viewport_text_ids=unknown,candidate_ids=new_match,geometry_candidates=possible,mapping=mapping,
        same_id_coherent_neighbors=len(coherent),mapping_context_proven=mapping_context,conditions=conditions,blocking_reasons=reasons,repeat_danger_flags=flags,
        original_unmapped=pair['motion']['target_new_index'] is None,simulation_verdict='TEXT_FIRST_SAME_CANDIDATE' if not reasons else 'AMBIGUOUS')

def main():
    started=time.perf_counter();process=psutil.Process();rss_begin=process.memory_info().rss
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'runs/text_first_identity_smoke17_20261001');ap.add_argument('--content',type=Path);args=ap.parse_args()
    manifest=load_manifest(ROOT);validate_integrity(ROOT,manifest,strict_hashes=True)
    base=ROOT/'runs/production_smoke17_20261001_013909';temp=ROOT/'runs/temporal_context_smoke17_20261001';audit=ROOT/'runs/cross_track_condition_audit_smoke17_20261001'
    paths=[ROOT/'production_manifest.json',Path(__file__).resolve(),ROOT/'experiments/cross_track_linker.py',ROOT/'experiments/run_cross_track_linker.py',base/'ordered_timeline.jsonl',temp/'chat_states.jsonl',audit/'independent_neighbors.jsonl',audit/'mapping_rejection_details.jsonl',ROOT/'runs/cross_track_linker_smoke17_20261001/relationships.jsonl']
    if args.content:paths.append(args.content.resolve())
    before={str(p):digest(p) for p in paths};pairs=read(ROOT/'runs/cross_track_linker_smoke17_20261001/relationships.jsonl');assert len(pairs)==590
    events=read(base/'ordered_timeline.jsonl');texts={r['stable_row_id']:r for r in events};ns={r['pair_id']:r for r in read(audit/'independent_neighbors.jsonl')};reject={r['pair_id']:r for r in read(audit/'mapping_rejection_details.jsonl')}
    states=defaultdict(list)
    for s in read(temp/'chat_states.jsonl'):states[s['session_id']].append(s)
    for ss in states.values():ss.sort(key=lambda x:x['source_frame'])
    frames={k:[s['source_frame'] for s in v] for k,v in states.items()}
    content={r['pair_id']:r for r in read(args.content)} if args.content else {}
    results=[]
    for p in pairs:
        ss=states[p['old']['session_id']];ff=frames[p['old']['session_id']];oi=bisect_right(ff,p['old']['last_frame'])-1;ni=bisect_right(ff,p['decision_frame'])-1
        old_ids=ss[oi]['track_ids'];new_ids=ss[ni]['track_ids'];history=[x for s in ss[max(0,ni-12):ni] for x in s['track_ids']]
        n=ns[p['pair_id']];g={}
        if p['pair_id'] in reject:
            for c in reject[p['pair_id']]['all_target_candidates']:
                if c['new_index']<len(new_ids):g[new_ids[c['new_index']]]=c['center_residual']
        else:g[p['new_track_id']]=n['target_center_residual']
        if p['pair_id'] in content and 'all_target_candidates' in content[p['pair_id']]:
            g={new_ids[c['new_index']]:c['center_residual'] for c in content[p['pair_id']]['all_target_candidates'] if c['new_index']<len(new_ids)}
        if p['pair_id'] in content and 'all_new_row_geometry' in content[p['pair_id']]:
            g={c.get('new_id',new_ids[c['new_index']]):c['center_residual'] for c in content[p['pair_id']]['all_new_row_geometry'] if c['new_index']<len(new_ids)}
        # Registered distance is positive support, never an exact-raster veto.
        support=False
        if p['pair_id'] in content: support=content_pass(content[p['pair_id']])
        r=evaluate(p,old_ids,new_ids,history,texts,n,g,support);r['supporting_registered_row_distance']=n['target_row_distance']
        if p['pair_id'] in content:
            c=content[p['pair_id']];target=c['target'];best=target['best_alignment']
            r['supporting_content']=dict(status=c['within_causal_neighbor_control_envelope'],control_count=c['causal_neighbor_control_count'],control_max_foreground_mae=c['control_max_foreground_mae'],control_max_support_xor_ratio=c['control_max_support_xor_ratio'],foreground_mae=best['foreground_union']['mae'],dx=best['dx'],dy=best['dy'],before_bbox=target['before']['bbox'],after_bbox=target['after']['bbox'],missing_foreground_before=best['missing_foreground_before'],missing_foreground_after=best['missing_foreground_after'])
        r['false_merge_audit']=dict(nearby_same_text_ids=r['history_other_same_text_ids'],second_viewport_candidate=len(r['candidate_ids'])>1,boundary_conflict=not r['conditions']['no_boundary'],causal_simultaneous=not r['conditions']['no_simultaneous'],physical_conflict_proven=False,physical_conflict_note='No calibrated strong DIFFERENT content rule; envelope failure is insufficient support.',all_causal_conditions_pass=all(r['conditions'].values()))
        r['POST_HOC_CHECK']=dict(old_reappears_after_decision=any(b>p['decision_frame'] for a,b in p['old_terminal_metadata']['visible_spans']),does_not_change_verdict=True)
        results.append(r)
    copied=[dict(r) for r in events];byid={r['stable_row_id']:r for r in copied}
    for p,r in zip(pairs,results):
        if r['simulation_verdict']=='TEXT_FIRST_SAME_CANDIDATE':byid[p['new_track_id']]['cross_track_relationship']={'linked_track_id':p['old_track_id'],'link_status':'TEXT_FIRST_SAME_CANDIDATE'}
    old_edge=classify_edges(events).rows;new_edge=classify_edges(copied).rows
    assert all(all(copy[k]==v for k,v in source.items()) for source,copy in zip(events,copied))
    assert [r['edge_status'] for r in old_edge]==[r['edge_status'] for r in new_edge]
    validate_integrity(ROOT,manifest,strict_hashes=True)
    after={str(p):digest(p) for p in paths};assert before==after
    summary=dict(pair_count=len(results),frozen_counts={'SAME':0,'DIFFERENT':0,'AMBIGUOUS':590},simulation_counts=dict(Counter(r['simulation_verdict'] for r in results)),different_candidates=0,
        all_mapping=dict(Counter(r['mapping'] for r in results)),previously_unmapped_mapping=dict(Counter(r['mapping'] for r in results if r['original_unmapped'])),text_classes=dict(Counter(r['text_class'] for r in results)),
        blocking_reasons=dict(Counter(x for r in results for x in r['blocking_reasons'])),repeat_dangers=dict(Counter(x for r in results for x in r['repeat_danger_flags'])),
        blocked_duplicate_or_multiple=sum(r['simulation_verdict']=='AMBIGUOUS' and (r['text_class']!='TEXT_UNIQUE' or len(r['geometry_candidates'])>1) for r in results),
        pairs_with_unknown_viewport_text=sum(bool(r['unknown_viewport_text_ids']) for r in results),unknown_viewport_track_count=len({i for r in results for i in r['unknown_viewport_text_ids']}),
        edge_before=sum(r['edge_status']=='UNRESOLVED_EDGE' for r in old_edge),edge_after=sum(r['edge_status']=='UNRESOLVED_EDGE' for r in new_edge),source_rows=len(events),source_hashes_unchanged=True,hashes_before=before,hashes_after=after,
        image_writes=0,production_integrity='STRICT_HASHES_PASS',run_stage='CONTENT_CONTROL_SIMULATION' if args.content else 'INITIAL_METADATA_MAPPING_ONLY',content_input=str(args.content) if args.content else None,policy='Causal OCR source frames; previous 12 changed states; <=7 geometry; >=2 same-ID neighbors <=1 residual; local causal neighbor foreground/control envelope required; gap1/gap2 local bridge; no native exact veto. Registered row distance is descriptive. No new DIFFERENT proof.',verdict='TEXT_FIRST_MAPPING_PARTIAL')
    summary['candidate_post_hoc_old_reappearance']=sum(r['simulation_verdict']=='TEXT_FIRST_SAME_CANDIDATE' and r['POST_HOC_CHECK']['old_reappears_after_decision'] for r in results)
    summary['content_control_counts']=dict(Counter(r.get('supporting_content',{}).get('status','NO_EVIDENCE') for r in results))
    summary['performance']=dict(wall_seconds=time.perf_counter()-started,pairs_per_second=len(results)/(time.perf_counter()-started),rss_begin=rss_begin,rss_end=process.memory_info().rss,rss_scope='Process begin/end samples; not peak; metadata loading, strict integrity, hashes and edge simulation included')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'relationships.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in summary.items() if not k.startswith('hashes')},ensure_ascii=False))

if __name__=='__main__':main()
