"""Bounded comparison against latest fifteen posts and original 13/8 evidence."""
import importlib.util
import json, hashlib, difflib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/historical_forced_binding_integration_20261003'
RUN=OUT/'acceptance_ON'
LATEST=ROOT/'runs/deal_complete_fix_chat_review_20261003/acceptance_ON'
def main():
    spec=importlib.util.spec_from_file_location('prior_analysis',ROOT/'experiments/analyze_pending_header_roi_acceptance_20261003.py')
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    prior.OUT=OUT;prior.RUN=RUN;prior.main()
    d=prior.read(OUT/'acceptance_analysis.json')
    a=prior.read(LATEST/'seat_forced_summary.json');b=prior.read(RUN/'seat_forced_summary.json')
    extra=[r for r in b['bound_posts'] if r not in a['bound_posts']]
    missing=[r for r in a['bound_posts'] if r not in b['bound_posts']]
    previous_sequence=[r for r in b['bound_posts'] if r in a['bound_posts']]
    streams=['physical_rows','raw_ocr_results','ordered_timeline','context_decisions','removed_duplicates','validated_hands','visual_evidence','hand_analysis']
    protected=[dict(stream=s,equal=prior.clean(prior.rows(LATEST/(s+'.jsonl')))==prior.clean(prior.rows(RUN/(s+'.jsonl')))) for s in streams]
    readings=prior.rows(RUN/'start_stack_readings.jsonl')
    h2=[r for r in readings if r['table_id']==3 and r['hand_id']=='2876719777']
    coverage={s:[f for f in range(2531,2567) if f not in {r['source_frame'] for r in h2 if r['seat']==s}] for s in ['seat'+str(i) for i in range(1,9)]}
    old24=[r for r in prior.rows(LATEST/'start_stack_readings.jsonl') if r['table_id']==3 and r['hand_id']=='2876719777' and 2564<=r['source_frame']<=2566]
    new24=[r for r in h2 if 2564<=r['source_frame']<=2566]
    fields=['table_id','session_id','hand_id','source_frame','source_pts','time_base','seat','roi','pixel_sha256','raw_text','confidence','ocr_status']
    comparison=dict(latest_bound_count=len(a['bound_posts']),new_bound_count=len(b['bound_posts']),previous15_exact=not missing and previous_sequence==a['bound_posts'],previous15_sequence_exact=previous_sequence==a['bound_posts'],added_posts=extra,only_added_bankiir_ante700=len(extra)==1 and extra[0]['actor']=='bankiir' and extra[0]['role']=='ANTE' and extra[0]['amount']==700 and extra[0]['seat']=='seat3' and extra[0]['source_frame']==2530 and extra[0]['table']==3 and extra[0]['hand_id']=='2876719777' and extra[0]['session']=='table_abb26b4f0632d61cda00071f',
      facts_exact=prior.rows(LATEST/'start_stack_facts.jsonl')==prior.rows(RUN/'start_stack_facts.jsonl'),math_exact=prior.rows(LATEST/'start_stack_math.jsonl')==prior.rows(RUN/'start_stack_math.jsonl'),
      triples_exact=prior.read(LATEST/'start_stack_live_summary.json')['hands']==prior.read(RUN/'start_stack_live_summary.json')['hands'],protected_latest=protected,h2_source_gaps=coverage,
      alignment24_exact=len(new24)==len(old24)==24 and all(all(x[k]==y[k] for k in fields) for x,y in zip(old24,new24)),
      default_false=not d['default_flags']['enable_dealing_cards_hand_state_trigger'] and not d['default_flags']['enable_start_stack_capture'])
    allowed={'PRODUCTION.md','parser_core/seat_forced_binding.py','tests/test_seat_forced_binding.py'}
    before=prior.read(OUT/'before/production_manifest.json')
    changed=[n for n,h in before['hashes'].items() if prior.sha(ROOT/n)!=h]
    d['protected_registered_hashes_unchanged']=set(changed)<=allowed
    d['changed_registered_files']=changed
    comparison['duplicate_bound_posts']=len(b['bound_posts'])-len({json.dumps(r,sort_keys=True) for r in b['bound_posts']})
    comparison['no_extra_bindings']=len(extra)==1 and not missing and comparison['duplicate_bound_posts']==0
    comparison['historical_receipts']={k:v for k,v in b.items() if 'historical' in k or 'late' in k}
    target_receipts=[r for r in b.get('historical_binding_receipts',[]) if r.get('table')==3 and r.get('hand_id')=='2876719777' and r.get('actor')=='bankiir']
    comparison['target_historical_anchor_exact']=len(target_receipts)==1 and target_receipts[0]['forced_source_frame']==2530 and target_receipts[0]['binding_anchor_frame']==2566 and target_receipts[0]['label_confirmation_frame']==2566 and target_receipts[0]['seat']=='seat3' and target_receipts[0]['epoch']=='T03_SEAT3_EPOCH_1'
    comparison['target_late_literal_conflict_exact']=len(target_receipts)==1 and any(r['actor']=='bankir' and r['frames']==[2580,2581,2582] for r in target_receipts[0]['later_literal_conflicts'])
    comparison['target_no_alias_merge']=len(target_receipts)==1 and target_receipts[0]['actor']=='bankiir' and all(r['actor']=='bankiir' for r in target_receipts[0]['label_evidence'])
    comparison['protected_changed_subset']=set(changed)<=allowed
    d['latest_comparison']=comparison
    (OUT/'acceptance_analysis.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(comparison,ensure_ascii=False))
if __name__=='__main__':main()
