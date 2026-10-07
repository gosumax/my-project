"""Aggregate saved independent visual labels; conservative shared disagreement."""
from collections import Counter
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parser_core.edge_gate import classify_edges
OUT=ROOT/'runs/text_first_precision_review_smoke17_20261001'

def read(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
def measure(rows):
    c=Counter(r['visual_label'] for r in rows)
    s,d,u=[c.get(k,0) for k in ('VISUALLY_SAME','VISUALLY_DIFFERENT','VISUALLY_UNCERTAIN')]
    return dict(reviewed=len(rows),SAME=s,DIFFERENT=d,UNCERTAIN=u,confirmed_precision=s/(s+d) if s+d else None,
                uncertain_fraction=u/len(rows) if rows else None)

def main():
    sealed=read(OUT/'sealed_evidence.jsonl');evidence={r['case_id']:r for r in sealed}
    assignments=json.loads((OUT/'review_assignments.json').read_text('utf-8'))
    raw={};reviewer_counts={};validation_errors=[]
    for who in ('A','B'):
        path=OUT/f'visual_{who}.jsonl';rows=read(path) if path.exists() else []
        seen=set()
        for r in rows:
            cid=r['case_id']
            if cid in seen:validation_errors.append(f'Duplicate reviewer {who}/{cid}')
            seen.add(cid)
            if cid not in assignments[who]:validation_errors.append(f'Unassigned {who}/{cid}')
            if r['label'] not in ('VISUALLY_SAME','VISUALLY_DIFFERENT','VISUALLY_UNCERTAIN'):validation_errors.append(f'Invalid label {who}/{cid}')
            if r.get('review_phase')!='CAUSAL':validation_errors.append(f'Noncausal label {who}/{cid}')
            raw.setdefault(cid,{})[who]=r
        reviewer_counts[who]=dict(saved=len(rows),required=len(assignments[who]),complete=set(assignments[who])==seen,labels=dict(Counter(r['label'] for r in rows)))
    assert not validation_errors,validation_errors
    receipts=[json.loads(p.read_text('utf-8')) for p in (OUT/'render_receipts').glob('*.json')]
    displayed={cid for r in receipts if not r['posthoc'] for cid in r['cases']}
    assert all(cid in displayed for cid in raw),'Label without source display receipt'
    actual=[];pending=[];disagreements=[];shared_complete=0;shared_agree=0
    for cid,e in evidence.items():
        rr=raw.get(cid,{})
        required=[who for who in ('A','B') if cid in assignments[who]]
        if any(who not in rr for who in required):pending.append(cid);continue
        labels={r['label'] for r in rr.values()}
        shared=len(required)>1
        if shared:
            shared_complete+=1;shared_agree+=len(labels)==1
        label=next(iter(labels)) if len(labels)==1 else 'VISUALLY_UNCERTAIN'
        if len(labels)>1:disagreements.append(dict(case_id=cid,raw_labels={who:r['label'] for who,r in rr.items()},final_label=label,policy='Conservative disagreement -> UNCERTAIN; no root override'))
        actual.append(dict(case_id=cid,pair_id=e['pair_id'],categories=e['categories'],
            original_simulation_verdict=e['original_simulation_verdict'],visual_label=label,
            review_status='REVIEWED',raw_judgments=rr,shared_review=shared,consensus=len(labels)==1,
            admission_label_hidden_during_review=True,causal_verdict_unchanged=True))
    admitted=[r for r in actual if r['original_simulation_verdict']=='TEXT_FIRST_SAME_CANDIDATE']
    rejected=[r for r in actual if r['original_simulation_verdict']=='AMBIGUOUS']
    categories=('NORMAL','ANTE','SB_BB','FOLD','NICK_ACTION','GAP2','WEAK_CONTENT','MAX_Y_RESIDUAL','POST_HOC_RISK','SHORT_TEXT','AMOUNT_ONLY')
    stats={cat:measure([r for r in admitted if cat in r['categories']]) for cat in categories}
    false_pos=[r for r in admitted if r['visual_label']=='VISUALLY_DIFFERENT']
    false_neg=[r for r in rejected if r['visual_label']=='VISUALLY_SAME']
    # Current gate consumes no relationship metadata. Add only visual SAME links
    # to copies, never alter source fields, epochs or admission logic.
    source=read(ROOT/'runs/production_smoke17_20261001_013909/ordered_timeline.jsonl')
    copied=[dict(r) for r in source];byid={r['stable_row_id']:r for r in copied}
    eligible=[r for r in admitted if r['visual_label']=='VISUALLY_SAME']
    for r in eligible:
        e=evidence[r['case_id']]
        byid[e['new']['track_id']]['cross_track_relationship']={'linked_track_id':e['old']['track_id'],'link_status':'VISUALLY_SAME','review_case_id':r['case_id']}
    assert all(all(n[k]==v for k,v in o.items()) for o,n in zip(source,copied))
    baseline=classify_edges(source).rows;sim=classify_edges(copied).rows
    assert [r['edge_status'] for r in baseline]==[r['edge_status'] for r in sim]
    edge=dict(before=sum(r['edge_status']=='UNRESOLVED_EDGE' for r in baseline),after=sum(r['edge_status']=='UNRESOLVED_EDGE' for r in sim),
        visually_validated_candidate_relationships=len(eligible),source_rows=len(source),source_fields_unchanged=True,
        actual_gate_changed=False,theoretical_removed_under_current_gate=0,
        reason='Current gate has no consumer for relationship metadata; no hypothetical new gate policy invented.')
    complete=all(r['complete'] for r in reviewer_counts.values())
    verdict='TEXT_FIRST_PRECISION_UNSAFE' if false_pos else 'TEXT_FIRST_PRECISION_PARTIAL'
    # A high subset precision alone cannot establish safe deployment. Require
    # full coverage and no unresolved material pattern, including reappearance.
    if complete and not false_pos and admitted and not any(r['visual_label']=='VISUALLY_UNCERTAIN' for r in admitted):
        verdict='TEXT_FIRST_PRECISION_PARTIAL'  # posthoc adjudication/report decides whether high is justified
    summary=dict(status='COMPLETE_CAUSAL_REVIEW' if complete else 'PARTIAL_CAUSAL_REVIEW',task_complete=False,
        sample_total=len(sealed),reviewers=reviewer_counts,reviewed_unique=len(actual),pending=len(pending),
        all_sample=measure(actual),prospective_candidates=measure(admitted),ambiguous_false_negative_sample=measure(rejected),
        candidate_categories=stats,ambiguous_categories={cat:measure([r for r in rejected if cat in r['categories']]) for cat in categories},
        false_positives=len(false_pos),observed_false_negative_candidates=len(false_neg),
        shared_cases_complete=shared_complete,shared_exact_agreement=shared_agree,shared_disagreements=disagreements,
        edge_simulation=edge,verdict=verdict,image_writes=0,source_data_changed=False,production_code_changed=False,
        scope='AI-assisted visual validation of frozen dataset; candidate coverage280, AMBIGUOUS stratified sample50, no full recall claim.',
        uncertainty_policy='UNKNOWN images/history/duplicates and reviewer disagreement remain VISUALLY_UNCERTAIN; excluded from precision denominator.')
    (OUT/'independent_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in actual),encoding='utf-8')
    (OUT/'pending_cases.json').write_text(json.dumps(pending),encoding='utf-8')
    (OUT/'visual_false_positives.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in false_pos),encoding='utf-8')
    (OUT/'visual_false_negatives.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in false_neg),encoding='utf-8')
    (OUT/'independent_precision_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('status','reviewed_unique','pending','all_sample','prospective_candidates','ambiguous_false_negative_sample','false_positives','observed_false_negative_candidates','shared_disagreements','verdict')},ensure_ascii=False))

if __name__=='__main__':main()
