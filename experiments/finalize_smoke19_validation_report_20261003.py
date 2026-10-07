"""Complete saved-evidence ledgers and review contacts; no production changes."""
import base64,csv,json
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image,ImageDraw
from io import BytesIO
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/smoke19_full_start_stack_validation_20261003'
RUN=OUT/'acceptance_ON';VAL=OUT/'validation';REVIEW=OUT/'review'
def read(name):return json.loads((RUN/name).read_text('utf-8'))
def rows(name):
    with (RUN/name).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)
def key(r):return r['table_id'],r['session_id'],r['hand_id']
def dump(name,values):
    if not values:return
    with (VAL/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
def main():
    seat=read('seat_forced_summary.json');live=read('start_stack_live_summary.json')
    inputs=list(rows('seat_forced_inputs.jsonl'));headers=defaultdict(list)
    for r in rows('hand_state_receipts.jsonl'):
        if r.get('status')=='HAND_HEADER_CONFIRMED':headers[key(r)].append(r)
    deals={key(r):r for r in rows('deal_complete_receipts.jsonl') if r.get('evidence_status')=='PROVEN'}
    handrows=[]
    for t in rows('dealing_cards_triggers.jsonl'):
        k=key(t);d=deals.get(k);h=headers[k]
        handrows.append(dict(table=k[0],session=k[1],hand=k[2],header_source_frame=t['source_frame'],
            header_pts=t['source_pts'],header_hash=t['header_pixel_sha256'],trigger_source_frame=t['trigger_source_frame'],
            trigger_pts=t['trigger_source_pts'],trigger_hash=t['pixel_hash'],deal_frame=d['source_frame'] if d else 'UNKNOWN',
            ncc_positive_frames=json.dumps(d['card_stable_frames']) if d else 'UNKNOWN',
            internal_header_source_matches=any(r['source_frame']==t['source_frame'] for r in h),
            blocker=None if d else 'TABLE_DEAL_COMPLETE_NOT_PROVEN'))
    dump('all71_hand_contexts.csv',handrows)
    rr=defaultdict(dict)
    for r in rows('start_stack_readings.jsonl'):rr[(*key(r),r['seat'])][r['source_frame']]=r
    seatrows=[];canonical=Counter()
    for f in rows('start_stack_facts.jsonl'):
        k=(f['table'],f['session'],f['hand_id']);s=f['seat'];frames=f['frames'];sources=rr[(*k,s)]
        h=next(h for h in live['hands'] if key(h)==k);a=h['canonical_attempts'].get(s)
        status=a['status'] if a else 'UNKNOWN';canonical[status]+=1
        end=max(frames) if frames else max(sources,default=f['deal_complete_frame'])
        gaps=[n for n in range(f['deal_complete_frame']+1,end+1) if n not in sources]
        seatrows.append(dict(table=k[0],session=k[1],hand=k[2],seat=s,deal=f['deal_complete_frame'],
            first_numeric=f.get('first_numeric_visible_frame'),frames=json.dumps(frames),
            raw_texts=json.dumps([sources.get(n,{}).get('raw_text') for n in frames]),
            confidences=json.dumps([sources.get(n,{}).get('confidence') for n in frames]),
            source_pts=json.dumps([sources.get(n,{}).get('source_pts') for n in frames]),
            hashes=json.dumps([sources.get(n,{}).get('pixel_sha256') for n in frames]),
            displayed=f['displayed_stack'],canonical_status=status,
            canonical_blocker=a.get('blocker') if a else 'NUMERIC_STACK_NOT_VISIBLE_OR_FIRST_TRIPLE_INCOMPLETE',
            source_gaps=json.dumps(gaps),coverage_end=end,epoch=f.get('occupancy_epoch')))
    dump('all144_seat_groups.csv',seatrows)
    blockers={(r['table'],r['session'],r['hand_id'],r['actor']):r['blocker'] for r in seat['historical_binding_blockers']}
    unbound=[]
    with (VAL/'forced_unbound.csv').open(encoding='utf-8-sig') as stream:old=list(csv.DictReader(stream))
    for p in old:
        refs=json.loads(p['source_refs']);src=next(r for r in inputs if r['source_ref']==refs[0])
        k=key(src);nick=[dict(seat=r['seat'],frame=r['source_frame'],pts=r['source_pts'],raw=r['ocr_response'].get('raw_text'),
            score=r['ocr_response'].get('score'),hash=r['pixel_sha256']) for r in inputs
            if r['kind']=='NICK' and key(r)==k and r['ocr_response'].get('raw_text','').strip()==p['actor']]
        unbound.append(dict(table=k[0],session=k[1],hand=k[2],actor=p['actor'],role=p['role'],amount=p['amount'],
            forced_frame=src['source_frame'],forced_pts=src['source_pts'],time_base=json.dumps(src['time_base']),
            source_refs=p['source_refs'],nickname_sources=json.dumps(nick),seat='UNKNOWN',epoch='UNKNOWN',
            blocker=blockers.get((*k,p['actor']),'UNKNOWN')))
    dump('forced_unbound_detailed.csv',unbound)
    mathrows=[]
    for m in rows('start_stack_math.jsonl'):
        bets=m.get('forced_bets') or {}
        mathrows.append(dict(table=m['table'],session=m['session'],hand=m['hand_id'],seat=m['seat'],
            displayed=m['displayed_stack'],ante_added=bets.get('ANTE','UNKNOWN'),SB_added=bets.get('SB',0),
            BB_added=bets.get('BB',0),special_added=json.dumps({k:v for k,v in bets.items() if k not in {'ANTE','SB','BB'}}),
            forced_bets_proven=m.get('status')=='PASS',START_STACK=m['START_STACK'],status=m['status'],blocker=m.get('blocker'),
            vision_fact_file='start_stack_facts.jsonl',frames=json.dumps(m['frames']),epoch=m.get('occupancy_epoch')))
    dump('math_debits_detailed.csv',mathrows)
    # Original saved crops only. These contact sheets do not change raw inputs.
    wanted={(9,'2878365800'),(5,'2878365932'),(1,'2878365962')}
    cards=[r for r in inputs if r['kind']=='CHAT' and (r['table_id'],r['hand_id']) in wanted]
    sheet=Image.new('RGB',(900,max(1,len(cards))*65),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(cards):
        y=i*65;draw.text((5,y),f"T{r['table_id']} hand{r['hand_id']} f{r['source_frame']} row{r['index']} conf{r['ocr_response'].get('score'):.4f}",fill='black')
        im=Image.open(BytesIO(base64.b64decode(r['png_base64']))).convert('RGB');im=im.resize((im.width*2,im.height*2))
        sheet.paste(im,(5,y+17))
    sheet.save(REVIEW/'grammar_three_hands.png')
    summary=dict(canonical_statuses=dict(canonical),hand_contexts=len(handrows),seat_groups=len(seatrows),
        detailed_unbound=len(unbound),math_rows=len(mathrows),
        internal_source_binding_conflicts=sum(not r['internal_header_source_matches'] for r in handrows),
        independent_missed_false_triggers='NOT_MEASURED',
        observed_auto_all_in_groups=sum(bool(f.get('auto_all_in_observed')) for f in rows('start_stack_facts.jsonl')),
        ordinary_math_PASS_without_historical_binding=sum(m['status']=='PASS' and not any(
            b['table']==m['table'] and b['hand_id']==m['hand'] and b['seat']==m['seat'] for b in seat['bound_posts']) for m in mathrows),
        caveat='No assertion of visually verified absence of events over the entire video.')
    (VAL/'final_field_coverage.json').write_text(json.dumps(summary,indent=2),'utf-8');print(json.dumps(summary))
    # Final decision is deliberately partial despite passing automated contracts.
    import hashlib
    saved=json.loads((VAL/'validation_summary.json').read_text('utf-8'))
    frozen=json.loads((OUT/'acceptance_candidate/frozen_hashes.json').read_text('utf-8'))
    drift=[n for n,h in frozen.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
    verdict=dict(verdict='SMOKE19_FULL_VALIDATION_PARTIAL',production_code_changed=False,
        production_promotion=False,default_flags_false=True,registered_hash_drift=drift,
        full_run_exit0=True,full_frames=10100,internal_guards=saved['guards'],counts=saved['counts'],
        remaining_validation_limits=['Independent marker misses and false positives NOT_MEASURED',
            '53 header contexts without positive TABLE confirmation',
            'Only9 distinct hands with any mathPASS; requested10 sample shortfall1',
            'Manual visual review bounded; not exhaustive source annotation'],
        root_visual_review=['numeric_first_triples_contact_sheet','three split-role snapshots',
            'table9 hand2878365800 TABLE source1338'],
        source_review=json.loads((REVIEW/'source_extraction_summary.json').read_text('utf-8')),
        evidence_report='REPORT.md')
    (VAL/'FINAL_VERDICT.json').write_text(json.dumps(verdict,ensure_ascii=False,indent=2),'utf-8')
    index=REVIEW/'index.html';html=index.read_text('utf-8')
    if 'Original-source review' not in html:
        html=html.replace('<h2>Contact sheets</h2>',
            '<h2>Original-source review</h2><p><a href="source_frames.html">71 CHAT and54 TABLE source crops</a> '
            '<a href="source_frame_provenance.json">Frame/PTS/hash provenance</a> '
            '<a href="grammar_three_hands.png">Three split-role snapshots</a></p>'
            '<p>Root performed bounded contact-sheet and grammar-snapshot visual checks; exhaustive review remains incomplete.</p>'
            '<h2>Contact sheets</h2>')
        index.write_text(html,'utf-8')
if __name__=='__main__':main()
