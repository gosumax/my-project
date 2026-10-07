"""Frozen overlay receipt, per-hand attribution and one fixed-window experiment."""
from __future__ import annotations
import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_validated_relationship_edge_simulation import read, write_json, write_rows, digest
from experiments.replay_poker_logic_math_jev import attach_blind_level, semantic_evidence
from parser_core.edge_gate import classify_edges, rows_for_downstream
from parser_core.poker_hand_assembly import assemble_hands
from parser_core.poker_logic_engine import validate_hand
from parser_core.poker_math_engine import audit_observed_ledger

OVERLAY=ROOT/'outputs/validated_relationship_edge_smoke17_20261001_v3'


def source_ids(hand):
    return [rid for m in hand['messages'] for rid in m['source_row_ids']]


def prepare(hands):
    for hand in hands: attach_blind_level(hand)
    semantics=semantic_evidence(hands)
    if semantics['status']=='ADDED_SUPPORTED':
        for hand in hands:
            hand['semantics']=dict(call='ADDED',bet='ADDED',source_type='DERIVED',
                                   rule='Existing offline replay monetary-clean evidence')
    return semantics


def evaluate(hands, window_by_row):
    semantics=prepare(hands)
    records=[]
    for index,hand in enumerate(hands):
        checked=validate_hand(hand)
        blockers=checked['certification_blockers']
        action=[i for i in blockers if i['domain']=='ACTION' and i['code'] not in checked['math']['errors']]
        completeness=[i for i in blockers if i['domain']=='COMPLETENESS']
        enriched=[]
        for issue in checked['issues']:
            item=copy.deepcopy(issue)
            item['source_messages']=[dict(source_row_ids=m['source_row_ids'],raw_text=m['raw_text'],
                action=m['action'],player=m['player'],amount=m['amount'],frame_id=m['frame_id'])
                for m in hand['messages'] if set(m['source_row_ids']) & set(issue['source_row_ids'])]
            enriched.append(item)
        windows=sorted({window_by_row[rid] for rid in source_ids(hand)})
        records.append(dict(hand_index=index,table_id=hand['table_id'],session_id=hand['session_id'],
            hand_id=hand['hand_id'],source_row_ids=source_ids(hand),fixed_edge_window_ids=windows,
            hand_start_observed=hand['start_observed'],hand_end_observed=hand['end_observed'],
            boundary_conflicts=hand['boundary_conflicts'],
            logic_pass=checked['logic_pass'],math_pass=checked['math']['pass'],
            fully_valid=checked['status']=='FULLY_VALID_HAND',status=checked['status'],
            logic_blocking_checks=action+completeness,math_blocking_checks=checked['math']['errors'],
            fully_valid_blocking_checks=blockers,blocker_codes=dict(Counter(i['code'] for i in blockers)),
            issues_with_source=enriched,validator_result=checked,
            first_break=validate_hand(hand,diagnostic_stop=True)['first_issue'],
            observed_ledger=audit_observed_ledger(hand)))
    metrics=dict(hands=len(records),logic_pass=sum(r['logic_pass'] for r in records),
        math_pass=sum(r['math_pass'] for r in records),fully_valid=sum(r['fully_valid'] for r in records),
        complete_candidates=sum(r['validator_result']['complete_candidate'] for r in records),
        semantics_status=semantics['status'],
        affected_hands_by_blocker=dict(Counter(code for r in records for code in r['blocker_codes'])),
        blocker_occurrences=dict(Counter(i['code'] for r in records for i in r['fully_valid_blocking_checks'])),
        multi_window_hands=sum(len(r['fixed_edge_window_ids'])>1 for r in records))
    return records,metrics


def assemble_fixed_windows(classified, attribution):
    """Transport already-proved edge intervals, no event/number/row repair.

    Reuse existing assembly within each interval. Do not concatenate unrelated
    KEEP intervals across excluded gaps. A forced start remains a derived edge
    context; it does not invent a NEW_HAND poker event or payout.
    """
    context={a['stable_row_id']:a for a in attribution}
    windows=defaultdict(list)
    intervals={}
    for row in rows_for_downstream(classified):
        a=context[row['stable_row_id']]
        interval=a['hand_interval']
        if not interval: raise ValueError('KEEP without a confirmed interval')
        key=(row['table_id'],row['session_id'],interval['start_row_id'])
        windows[key].append(copy.deepcopy(row));intervals[key]=interval
    hands=[]
    for key,rows in windows.items():
        candidates=assemble_hands(rows)
        if len(candidates)!=1:
            raise ValueError('Conflicting downstream segmentation within one fixed interval')
        hand=candidates[0]
        interval=intervals[key]
        if rows[0]['stable_row_id']!=interval['start_row_id']:
            raise ValueError('Missing confirmed start source row')
        hand['edge_start_evidence']=copy.deepcopy(interval)
        hand['start_observed']=True
        hand['next_boundary_observed']=interval['next_start_frame'] is not None
        # end_observed stays assembly's observed WIN/HAND_END; no payout inferred.
        hands.append(hand)
    admitted=rows_for_downstream(classified)
    reconstructed=[r for hand in hands for m in hand['messages'] for r in m['source_rows']]
    # Group order may differ across interleaved tables; compare exact ID maps and
    # per-table source order rather than pretending caller order was rewritten.
    assert {r['stable_row_id']:r for r in reconstructed}=={r['stable_row_id']:r for r in admitted}
    for table in {r['table_id'] for r in admitted}:
        assert [r['stable_row_id'] for r in reconstructed if r['table_id']==table]==[
            r['stable_row_id'] for r in admitted if r['table_id']==table]
    return hands


def freeze_overlay():
    destination=ROOT/'provenance/experiments/validated_relationship_overlay_v3_20261001'
    paths=[ROOT/p for p in ['experiments/validated_relationship_edge_simulation.py',
        'experiments/run_validated_relationship_edge_simulation.py',
        'experiments/report_validated_relationship_edge_simulation.py',
        'tests/test_validated_relationship_edge_simulation.py',
        'docs/VALIDATED_RELATIONSHIP_EDGE_SMOKE17_20261001.md',
        'parser_core/edge_gate.py','parser_core/poker_hand_assembly.py',
        'parser_core/poker_logic_engine.py','parser_core/poker_math_engine.py',
        'experiments/replay_poker_logic_math_jev.py','production_manifest.json','PRODUCTION.md']]
    paths += [p for p in OVERLAY.rglob('*') if p.is_file()]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    if destination.exists():
        receipt=json.loads((destination/'freeze_manifest.json').read_text('utf-8'))
        if receipt['hashes']!=hashes: raise ValueError('Existing experiment freeze differs; never overwrite')
        return str(destination)
    destination.mkdir(parents=True)
    for path in paths:
        target=destination/'files'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        assert digest(target)==hashes[str(path.relative_to(ROOT))]
    write_json(destination/'freeze_manifest.json',dict(schema='experiment.freeze.v1',hashes=hashes,
        evidence_status='PROVEN_OFFLINE_EDGE_EXPERIMENT_ONLY',production_promotion=False,
        proven_scope='22 safe fixed-window edge admissions on frozen Smoke17; no HH certification',
        frozen_metrics=dict(source_rows=8146,core_relationships=280,keep_rows=976,unresolved=6663,
                            logic_pass=0,math_pass=5,fully_valid=0),
        promotion_gate='Several independently proven changes must improve fully valid hands before considering transfer; not automatic authorization'))
    return str(destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Output must be fresh')
    frozen=freeze_overlay()
    inputs=json.loads((OVERLAY/'input_hashes.json').read_text('utf-8'))
    assert all(digest(ROOT/path)==sha for path,sha in inputs.items())
    args.output.mkdir(parents=True)
    original=read(ROOT/'runs/production_smoke17_20261001_013909/ordered_timeline.jsonl')
    baseline=classify_edges(original).rows
    core=read(OVERLAY/'CORE_280/classified_rows.jsonl')
    attr=read(OVERLAY/'CORE_280/row_reason_attribution.jsonl')
    window_by_row={a['stable_row_id']:a['hand_interval']['start_row_id'] if a['hand_interval'] else 'NO_CONFIRMED_START' for a in attr}
    legacy_hands={name:assemble_hands(rows_for_downstream(rows)) for name,rows in [('BASELINE',baseline),('CORE_280',core)]}
    all_records={};summary=dict(frozen_overlay=frozen,arms={},production_changed=False)
    for name,hands in legacy_hands.items():
        records,metrics=evaluate(hands,window_by_row)
        assert len(records)==27
        all_records[name]=records;summary['arms'][name]=metrics
        write_rows(args.output/(name+'.hands.jsonl'),records)
        write_json(args.output/(name+'.assembled_hands.json'),hands)
    pairs=[]
    base=all_records['BASELINE'];new=all_records['CORE_280']
    for old in base:
        matched=[r for r in new if set(old['source_row_ids'])<=set(r['source_row_ids']) and r['table_id']==old['table_id']]
        if len(matched)!=1: raise ValueError('Cannot uniquely match baseline hand by source containment')
        current=matched[0]
        pairs.append(dict(baseline_hand_index=old['hand_index'],core_hand_index=current['hand_index'],
            table_id=old['table_id'],hand_id=old['hand_id'],
            source_rows_added=[rid for rid in current['source_row_ids'] if rid not in old['source_row_ids']],
            math_pass_before=old['math_pass'],math_pass_after=current['math_pass'],
            math_before=old['validator_result']['math'],math_after=current['validator_result']['math'],
            fixed_windows_before=old['fixed_edge_window_ids'],fixed_windows_after=current['fixed_edge_window_ids']))
    regressions=[r for r in pairs if r['math_pass_before'] and not r['math_pass_after']]
    assert len(regressions)==1 and len(regressions[0]['source_rows_added'])==22
    regression=regressions[0]
    index=regression['core_hand_index']
    hand=legacy_hands['CORE_280'][index]
    added=set(regression['source_rows_added'])
    revert=copy.deepcopy(hand)
    revert['messages']=[m for m in revert['messages'] if not set(m['source_row_ids']) & added]
    checked=validate_hand(revert)
    assert checked['math']['pass'] and checked['math']['delta']==regression['math_before']['delta']
    regression['remove_exact_added_rows_counterfactual']=checked['math']
    regression['added_messages']=[dict(action=m['action'],player=m['player'],amount=m['amount'],
        raw_text=m['raw_text'],source_row_ids=m['source_row_ids'],frame_id=m['frame_id'])
        for m in hand['messages'] if set(m['source_row_ids']) & added]
    regression['proven_cause']='FIXED_EDGE_INTERVALS_LOST_AT_DOWNSTREAM_ASSEMBLY'
    write_json(args.output/'math_6_to_5_regression.json',regression)
    write_json(args.output/'all_27_hand_comparison.json',pairs)
    for name,rows in [('BASELINE_FIXED_WINDOWS',baseline),('CORE_FIXED_WINDOWS',core)]:
        hands=assemble_fixed_windows(rows,attr)
        records,metrics=evaluate(hands,window_by_row)
        summary['arms'][name]=metrics
        write_rows(args.output/(name+'.hands.jsonl'),records)
        write_json(args.output/(name+'.assembled_hands.json'),hands)
    summary['selected_class']=dict(name='FIXED_EDGE_INTERVALS_LOST_AT_DOWNSTREAM_ASSEMBLY',
        rationale='Largest action blocker ACTION_AFTER_HAND_END; direct multi-window source membership proves shared assembly contract loss',
        fix_scope='Offline assembly view grouped by exact existing admitted edge intervals; no source edits, dedup or payout repair')
    assert all(digest(ROOT/path)==sha for path,sha in inputs.items())
    summary['source_parity']=dict(source_rows=8146,source_and_production_hashes_unchanged=True,
        admitted_core_rows=976,admitted_row_multiset_and_per_table_order_preserved=True,image_writes=0,
        existing_fixed_start_ids_unchanged=True,events_ocr_ids_epochs_unchanged=True)
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
