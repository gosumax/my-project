"""Saved evidence only: blocker causes and qualified PASS pixel provenance."""
import base64,csv,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from parser_core.dealing_start_stack import NUMERIC
from parser_core.seat_forced_binding import POST
RUN=ROOT/'runs/smoke19_full_start_stack_validation_20261003/acceptance_ON';OUT=RUN.parent/'validation'
def rows(n):
    with (RUN/(n+'.jsonl')).open(encoding='utf-8') as f:
        for line,l in enumerate(f,1):
            if l.strip():yield line,json.loads(l)
def main():
    facts=[r for _,r in rows('start_stack_facts')];math=[r for _,r in rows('start_stack_math')];readings=defaultdict(list)
    for line,r in rows('start_stack_readings'):readings[(r['table_id'],r['hand_id'],r['seat'])].append(r)
    seat=json.loads((RUN/'seat_forced_summary.json').read_text('utf-8'));blocked={k.split("'")[-2] for k in seat['forced_snapshot_blockers']};postledger=[];bad=[]
    for line,r in rows('seat_forced_inputs'):
        text=(r['ocr_response'].get('raw_text') or '').strip()
        if r['kind']!='CHAT' or 'posts' not in text.lower():continue
        match=POST.fullmatch(text);bound=any(r['source_ref'] in b['source_refs'] for b in seat['bound_posts'])
        status='BOUND' if bound else 'SNAPSHOT_BLOCKED' if r['hand_id'] in blocked else 'UNBOUND_COMPLETE_SNAPSHOT'
        item=dict(table=r['table_id'],hand=r['hand_id'],frame=r['source_frame'],pts=r['source_pts'],time_base=json.dumps(r['time_base']),line=line,raw_text=text,confidence=r['ocr_response'].get('score'),pixel_sha256=r['pixel_sha256'],row_grammar_valid=match is not None,actor=match.group(1) if match else 'UNKNOWN',role=match.group(2) if match else 'UNKNOWN',amount=match.group(3) or 'UNKNOWN_ADJACENT_WRAP' if match else 'UNKNOWN',binding_status=status)
        postledger.append(item)
        if not match:bad.append(item)
    input_rows=[r for _,r in rows('seat_forced_inputs') if r['kind']=='CHAT']
    for b in bad:
        b['following_rows']=[{k:r[k] for k in ['table_id','hand_id','source_frame','source_pts','time_base','index','source_ref','ocr_response','pixel_sha256']} for r in input_rows if r['table_id']==b['table'] and r['hand_id']==b['hand'] and r['source_frame']==b['frame'] and r['index'] in [next(q['index'] for q in input_rows if q['pixel_sha256']==b['pixel_sha256'])+1,next(q['index'] for q in input_rows if q['pixel_sha256']==b['pixel_sha256'])+2]]
        b['fail_predicate']='POST requires whole Small Blind role in one row; only numeric amount may wrap. Split role text is not repaired.'
    categories=[];unstable=[]
    for f in facts:
        rr=readings[(f['table'],f['hand_id'],f['seat'])]
        if f['blocker']=='FIRST_NUMERIC_TRIPLE_NOT_STABLE':unstable.append(dict(table=f['table'],hand=f['hand_id'],seat=f['seat'],frames=f['frames'],readings=[{k:r[k] for k in ['source_frame','raw_text','confidence','pixel_sha256']} for r in rr if r['source_frame'] in f['frames']]))
        if f['blocker']!='NUMERIC_STACK_NOT_VISIBLE_OR_FIRST_TRIPLE_INCOMPLETE':continue
        qualified=[r for r in rr if r['ocr_status']=='OK' and NUMERIC.fullmatch((r['raw_text'] or '').strip()) and r['confidence']>=.95]
        low=[r for r in rr if NUMERIC.fullmatch((r['raw_text'] or '').strip()) and r['confidence']<.95]
        category='NO_JOBS' if not rr else 'QUALIFIED_NUMERIC_INCOMPLETE' if qualified else 'NUMERIC_LOW_CONFIDENCE' if low else 'NONNUMERIC_OR_EMPTY'
        categories.append(dict(table=f['table'],hand=f['hand_id'],seat=f['seat'],category=category,readings=len(rr),numeric_low_conf_count=len(low),qualified_numeric_count=len(qualified),raw_text_counts=dict(Counter(r['raw_text'] for r in rr)),first_frame=min((r['source_frame'] for r in rr),default=None),last_frame=max((r['source_frame'] for r in rr),default=None)))
    profile=json.loads((ROOT/'configs/layout_847x404.json').read_text('utf-8'));passes=[]
    for m in math:
        if m['status']!='PASS':continue
        checks=[]
        for r in m['evidence']:
            gray=cv2.imdecode(np.frombuffer(base64.b64decode(r['png_base64']),np.uint8),cv2.IMREAD_GRAYSCALE)
            checks.append(dict(frame=r['source_frame'],png_sha=hashlib.sha256(gray.tobytes()).hexdigest()==r['pixel_sha256'],roi=r['roi']==profile['slots'][f"slot_{m['table']:02d}"]['seats'][m['seat']]['stack'],context=(r['table_id'],r['session_id'],r['hand_id'],r['seat'])==(m['table'],m['session'],m['hand_id'],m['seat']),pts=r['source_pts']==r['source_frame']*512 and r['time_base']==[1,15360],raw_qualified=r['ocr_status']=='OK' and bool(NUMERIC.fullmatch(r['raw_text'].strip())) and r['confidence']>=.95))
        source_matches=[]
        for evidence in m['evidence']:
            matches=[r for r in readings[(m['table'],m['hand_id'],m['seat'])] if r['source_frame']==evidence['source_frame']]
            fields=['table_id','session_id','hand_id','seat','source_frame','source_pts','time_base','roi','pixel_sha256','raw_text','confidence','ocr_status']
            source_matches.append(len(matches)==1 and all(matches[0][k]==evidence[k] for k in fields))
        forced_checks=[]
        refs=[ref for group in m.get('standard_ante_source_refs',[])+m.get('special_forced_source_refs',[]) for ref in group]
        for ref in refs:
            matches=[r for r in input_rows if r['table_id']==m['table'] and r['session_id']==m['session'] and r['hand_id']==m['hand_id'] and r['source_ref']==ref]
            valid=len(matches)==1
            if valid:
                r=matches[0];pixels=cv2.imdecode(np.frombuffer(base64.b64decode(r['png_base64']),np.uint8),cv2.IMREAD_GRAYSCALE)
                valid=hashlib.sha256(pixels.tobytes()).hexdigest()==ref and r['source_pts']==r['source_frame']*512 and r['time_base']==[1,15360] and r['source_frame']==m['deal_complete_frame'] and r['ocr_response']['status']=='OK' and r['ocr_response']['score']>=.95
            forced_checks.append(dict(source_ref=ref,source_count=len(matches),valid=valid))
        passes.append(dict(table=m['table'],hand=m['hand_id'],seat=m['seat'],frames=m['frames'],pixel_checks=checks,actual_live_sources_equal=source_matches,forced_source_checks=forced_checks,all_checks=all(all(v for k,v in c.items() if k!='frame') for c in checks) and all(source_matches) and all(r['valid'] for r in forced_checks)))
    pending=[];headers=[];provisional=[]
    for line,r in rows('start_stack_live_receipts'):
        if r.get('table_id')==1 and r.get('hand_id')=='2878365444' and 20<=r.get('source_frame',-1)<=45 and r.get('stage')=='STACK_PENDING_HEADER':pending.append(dict(line=line,frame=r['source_frame'],pts=r['source_pts'],crop_hash=r['crop_hashes'].get('seat7')))
    for line,r in rows('hand_state_receipts'):
        if r.get('table_id')==1 and r.get('hand_id')=='2878365444' and 26<=r.get('source_frame',-1)<=40:headers.append(dict(line=line,**r))
    for line,r in rows('start_stack_capture'):
        if r.get('table_id')==1 and r.get('seat')=='seat7' and r.get('field')=='stack' and 26<=r.get('source_frame',-1)<=39:provisional.append({k:r[k] for k in ['source_frame','source_pts','time_base','roi','pixel_sha256']})
    report=dict(bad_grammar=bad,physical_posts_status_counts=dict(Counter(r['binding_status'] for r in postledger)),physical_posts_total=len(postledger),nonnumeric27_categories=dict(Counter(r['category'] for r in categories)),nonnumeric27=categories,unstable=unstable,PASS59_pixel_checks=passes,PASS59_pixel_checks_all=all(p['all_checks'] for p in passes),gap_evidence=dict(pending=pending,header_callbacks=headers,provisional_raw=provisional,actual_reading_frames=[r['source_frame'] for r in readings[(1,'2878365444','seat7')]],cause='BOUNDED_PENDING_HEADER_BUFFER_EXPIRED; sources26..39 withheld beyond5source retention and evicted before samehand release; originalprovisionalRGBpixels retained separately but notruntimequeue repair',call_path='scripts/run_smoke14_ram.py:on_table(frame callbacks)->StartStackLiveConnection.observe_frame->_observe_frame pending_stacks cap5/pop0->completed source_readings->first_visible_stack_triple expected26 missing returnsUNKNOWN',bug_verdict='No unproven late replacement; explicit bounded expiry retained as UNKNOWN. Per-frame eviction receipts not emitted; exact eviction sequence inferred from source-ordered cap5 and persisted withheld/header sources.'))
    with (OUT/'all180_physical_post_candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in postledger for k in r)));w.writeheader();w.writerows(postledger)
    (OUT/'blocker_audit.json').write_text(json.dumps(report,indent=2),'utf-8');print(json.dumps({k:report[k] for k in ['physical_posts_status_counts','physical_posts_total','nonnumeric27_categories','PASS59_pixel_checks_all']},ensure_ascii=False))
if __name__=='__main__':main()
