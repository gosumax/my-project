"""Read-only acceptance of an official FULL_HH_PRODUCTION output directory."""
import argparse
from collections import Counter
import json
from pathlib import Path


def lines(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit(output):
    def load(name):
        p=output/name
        return json.loads(p.read_text('utf-8')) if p.exists() else {}
    matrix=load('full_hh_production_matrix.json').get('modules',{})
    physical=lines(output/'physical_rows.jsonl');ocr=lines(output/'raw_ocr_results.jsonl')
    timeline=lines(output/'ordered_timeline.jsonl');hands=lines(output/'hand_histories.jsonl')
    observed=lines(output/'hand_analysis.jsonl')
    kinds=Counter(m['action'] for h in observed for m in h['observed_hand']['messages'])
    ids=Counter(r['row_id'] for r in physical);ocr_ids=Counter(r['row_id'] for r in ocr)
    final=load('final_summary.json');stack=load('stack_evidence_summary.json')
    errors=load('errors_retries.json')
    checks=dict(matrix_all_on=bool(matrix) and all(v is True for v in matrix.values()),
        headers=kinds['HAND_HEADER']+kinds['HAND_ID']>0,
        ante=kinds['ANTE']>0,small_blind=kinds['SMALL_BLIND']>0,big_blind=kinds['BIG_BLIND']>0,
        dealing=kinds['DEAL']+kinds['DEALING_CARDS']>0,
        ordinary_actions=sum(kinds[k] for k in ('FOLD','CHECK','CALL','BET','RAISE','WIN'))>0,
        hand_ids=any(h['hand_number'] is not None for h in hands),segmentation=len(hands)>0,
        stack_observations=stack.get('observations',0)>0,
        stack_numeric=stack.get('numeric_observations',0)>0,
        start_stack_consumer=stack.get('start_stack_enabled') is True and (output/'stack_start_stack.jsonl').exists(),
        hh_formed=bool(hands),png_zero=not any(output.rglob('*.png')),
        physical_to_ocr_exact=bool(ids) and ids==ocr_ids and all(v==1 for v in ids.values()),
        ocr_responses=bool(ocr) and all(r.get('raw_ocr_response',{}).get('status')=='OK' for r in ocr),
        final_exists=bool(final),runtime_errors_zero=(output/'errors_retries.json').exists() and not errors.get('errors'))
    result=dict(verdict='PASS' if all(checks.values()) else 'FAIL',checks=checks,
        failures=[k for k,v in checks.items() if not v],physical_rows=len(physical),ocr_rows=len(ocr),
        timeline_rows=len(timeline),action_types=dict(kinds),hh_hands=len(hands),
        hh_certified=sum(h['certified'] for h in hands),
        scope='SHORT_CONNECTION_AND_TRANSPORT_ACCEPTANCE; NOT_HH95_OR_PHYSICAL_RECALL')
    (output/'short_acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    result=audit(a.output);print(json.dumps(result,ensure_ascii=True));raise SystemExit(0 if result['verdict']=='PASS' else 1)
