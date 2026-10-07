"""Bounded saved-Smoke17 replay, JSON sidecars only, no image/video/OCR work."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.validated_relationship_edge_simulation import simulate, contexts
from parser_core.edge_gate import classify_edges, rows_for_downstream


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deep_size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen: return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(deep_size(k, seen)+deep_size(v, seen) for k,v in value.items())
    elif isinstance(value, (list, tuple, set)):
        size += sum(deep_size(v, seen) for v in value)
    return size


def load_validated(review):
    freeze = json.loads((review/'causal_review_freeze.json').read_text('utf-8'))
    for name, sha in freeze['hashes'].items():
        if digest(review/name) != sha: raise ValueError('Frozen review receipt changed: '+name)
    sample=json.loads((review/'sample_summary.json').read_text('utf-8'))
    pinned={str(k).replace('\\','/'):v for k,v in sample['hashes_before'].items()}
    inputs=['runs/cross_track_linker_smoke17_20261001/relationships.jsonl',
            'runs/text_first_identity_smoke17_20261001/relationships.jsonl']
    for name in inputs:
        matches=[sha for path,sha in pinned.items() if path.endswith('/'+name)]
        if len(matches)!=1 or digest(ROOT/name)!=matches[0]:
            raise ValueError('Frozen candidate evidence hash mismatch: '+name)
    frozen={r['pair_id']:r for r in read(ROOT/inputs[0])}
    prospective={r['pair_id']:r for r in read(ROOT/inputs[1])}
    ocr={r['row_id']:r for r in read(ROOT/'runs/production_smoke17_20261001_013909/raw_ocr_results.jsonl')}
    sealed = {r['case_id']: r for r in read(review/'sealed_evidence.jsonl')}
    labels = read(review/'independent_results.jsonl')
    if len(labels) != 330 or len(sealed) != 330: raise ValueError('Review cohort mismatch')
    projected = []
    for label in labels:
        evidence = sealed[label['case_id']]
        if evidence['pair_id'] != label['pair_id'] or evidence['original_simulation_verdict'] != label['original_simulation_verdict']:
            raise ValueError('Review/evidence join mismatch')
        f=frozen[label['pair_id']]
        if evidence['old']!=f['old'] or evidence['new']!=f['new'] or (
                evidence['boundary_frames']!=f['features']['hand_boundary_frames']):
            raise ValueError('Sealed endpoints/lifetimes differ from pinned candidate evidence')
        if evidence['pair_id']!=f['old']['track_id']+'->'+f['new']['track_id']:
            raise ValueError('Endpoint pair ID mismatch')
        if label['original_simulation_verdict']!=prospective[label['pair_id']]['simulation_verdict']:
            raise ValueError('Cohort does not match frozen TEXT-FIRST admission')
        if label['visual_label'] != 'VISUALLY_SAME': continue
        if label['review_status'] != 'REVIEWED': raise ValueError('Unreviewed SAME')
        old, new = evidence['old'], evidence['new']
        ocr_ok={}
        for side,endpoint in [('old',old),('new',new)]:
            parent=f['source_ocr_parents'][endpoint['track_id']]
            record=ocr[endpoint['track_id']]
            response=record['raw_ocr_response']
            if (record['row_key'],record['frame_id'],record['row_pixel_sha256'])!=(
                    parent['source_row_key'],parent['source_frame'],parent['source_pixel_hash']):
                raise ValueError('Raw OCR source-parent provenance mismatch')
            ocr_ok[side]=response.get('status')=='OK' and not response.get('error') and response.get('raw_text')==endpoint['raw_text']
        projected.append(dict(pair_id=label['pair_id'], case_id=label['case_id'],
            cohort='CORE_280' if label['original_simulation_verdict']=='TEXT_FIRST_SAME_CANDIDATE' else 'AMBIGUOUS_33_SAMPLE',
            visual_label=label['visual_label'], relationship='VALIDATED_SAME_PHYSICAL_OCCURRENCE',
            relationship_source='TEXT_FIRST_VISUAL_VALIDATION', old_track_id=old['track_id'], new_track_id=new['track_id'],
            table_id=old['table_id'], session_id=old['session_id'],
            old_first_frame=old['first_frame'], new_first_frame=new['first_frame'],
            old_last_frame=old['last_frame'], new_last_frame=new['last_frame'],
            old_pixel_hash=old['pixel_hash'], new_pixel_hash=new['pixel_hash'],
            old_source_pixel_hash=f['source_ocr_parents'][old['track_id']]['source_pixel_hash'],
            new_source_pixel_hash=f['source_ocr_parents'][new['track_id']]['source_pixel_hash'],
            old_source_ocr_ok=ocr_ok['old'],new_source_ocr_ok=ocr_ok['new'],
            boundary_frames=evidence.get('boundary_frames', [])))
    core = [r for r in projected if r['cohort']=='CORE_280']
    if len(core)!=280 or len(projected)!=313: raise ValueError('Validated counts mismatch')
    return core, projected, labels


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_rows(path, rows):
    with path.open('x', encoding='utf-8') as stream:
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False)+'\n')


def breakdown(attribution):
    unresolved = [r for r in attribution if r['new_edge_status']=='UNRESOLVED_EDGE']
    return dict(total=len(unresolved),
        missing_end_among_unresolved=sum(bool(r['hand_interval']) and not r['hand_interval']['end_reason'] for r in unresolved),
        exclusive_combinations=dict(Counter('|'.join(r['reason_after']) for r in unresolved)),
        overlapping_predicates=dict(Counter(reason for r in unresolved for reason in r['reason_after'])))


def dry_run(classified):
    """Reuse existing offline deterministic assembly/logic/math, no JEV/repair."""
    from parser_core.poker_hand_assembly import assemble_hands
    from parser_core.poker_logic_engine import validate_hand
    from experiments.replay_poker_logic_math_jev import attach_blind_level, semantic_evidence
    filtered=rows_for_downstream(classified)
    frozen=copy.deepcopy(filtered)
    hands=assemble_hands(copy.deepcopy(filtered))
    for hand in hands: attach_blind_level(hand)
    semantics=semantic_evidence(hands)
    if semantics['status']=='ADDED_SUPPORTED':
        for hand in hands:
            hand['semantics']=dict(call='ADDED',bet='ADDED',source_type='DERIVED',
                rule='Existing replay monetary-clean support; offline simulation only')
    checked=[validate_hand(hand) for hand in hands]
    assert filtered==frozen
    return dict(admitted_rows=len(filtered),hands_assembled=len(hands),
        complete_candidates=sum(r['complete_candidate'] for r in checked),
        logic_pass=sum(r['logic_pass'] for r in checked),math_pass=sum(r['math']['pass'] for r in checked),
        fully_valid=sum(r['status']=='FULLY_VALID_HAND' for r in checked),
        semantics_status=semantics['status'],live_calls=0,source_copy_unchanged=True,
        decisive_issues=dict(Counter(i['code'] for r in checked for i in r['issues'])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--repeats', type=int, default=5)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Output must be fresh')
    if not 3<=args.repeats<=9: parser.error('Use 3..9 bounded timing repeats')
    source_dir=ROOT/'runs/production_smoke17_20261001_013909'
    review=ROOT/'runs/text_first_precision_review_smoke17_20261001'
    files=[source_dir/name for name in ['ordered_timeline.jsonl','physical_rows.jsonl','raw_ocr_results.jsonl',
                                      'visual_evidence.jsonl','ordered_timeline.xlsx','run_metadata.json']]
    files += [review/name for name in ['sample_summary.json','sealed_evidence.jsonl','independent_results.jsonl','causal_review_freeze.json',
                                      'visual_A.jsonl','visual_B.jsonl','verification.json']]
    files += [ROOT/'runs/cross_track_linker_smoke17_20261001/relationships.jsonl',
              ROOT/'runs/text_first_identity_smoke17_20261001/relationships.jsonl']
    files += [ROOT/'parser_core/edge_gate.py',ROOT/'parser_core/poker_hand_assembly.py',ROOT/'production_manifest.json']
    manifest=json.loads((ROOT/'production_manifest.json').read_text('utf-8'))
    files += [ROOT/path for path in manifest['hashes']]
    before={str(p.relative_to(ROOT)):digest(p) for p in files}
    rows=read(source_dir/'ordered_timeline.jsonl')
    core, extended, labels=load_validated(review)
    baseline=classify_edges(rows)
    counts=dict(Counter(r['edge_status'] for r in baseline.rows))
    if len(rows)!=8146 or counts!={'PARTIAL_START':441,'KEEP':954,'PARTIAL_END':66,'UNRESOLVED_EDGE':6685}:
        raise ValueError('Frozen baseline counts mismatch')
    frozen=read(ROOT/'outputs/edge_gate_20261001_smoke17_final/ordered_timeline.edge.jsonl')
    if frozen!=baseline.rows: raise ValueError('Frozen baseline row parity mismatch')
    row_context=contexts(rows,baseline)
    args.output.mkdir(parents=True)
    modes=[('CORE_280',core)]+([('EXTENDED_313_SAMPLE',extended)] if args.extended else [])
    report=dict(source_timeline=str(source_dir/'ordered_timeline.jsonl'), baseline_edge=counts,
                total_rows=8146, modes={}, production_code_changed=False)
    report['baseline_unresolved_reasons']=breakdown(simulate(rows,[],baseline,row_context).attribution)
    for mode, links in modes:
        result=simulate(rows,links,baseline,row_context)
        mode_dir=args.output/mode
        mode_dir.mkdir()
        # Compare every original field, nested evidence, IDs and order.
        assert [{k:v for k,v in r.items() if k!='edge_status'} for r in result.rows]==rows
        changes=[r for r in result.attribution if r['old_edge_status']!=r['new_edge_status']]
        new_keep=[r for old,r in zip(baseline.rows,result.rows) if old['edge_status']!='KEEP' and r['edge_status']=='KEEP']
        for change in changes:
            assert change.get('window_continuity_proofs'), 'Silent reclassification'
        write_rows(mode_dir/'classified_rows.jsonl',result.rows)
        write_rows(mode_dir/'row_reason_attribution.jsonl',result.attribution)
        write_rows(mode_dir/'status_changes.jsonl',changes)
        write_rows(mode_dir/'relationships_used.jsonl',result.relationships)
        write_json(mode_dir/'continuity_chains.json',result.chains)
        base_times,sim_times=[],[]
        # Alternation bounds cache/order bias. Excludes JSON I/O and image work.
        for iteration in range(args.repeats):
            for name in (['baseline','simulation'] if iteration%2==0 else ['simulation','baseline']):
                start=time.perf_counter()
                measured=classify_edges(rows) if name=='baseline' else simulate(rows,links)
                (base_times if name=='baseline' else sim_times).append(time.perf_counter()-start)
        base_med,sim_med=statistics.median(base_times),statistics.median(sim_times)
        lookup_times=[]
        # Overlay plus attribution cost, with frozen gate/context precomputed.
        for _ in range(args.repeats):
            start=time.perf_counter();simulate(rows,links,baseline,row_context)
            lookup_times.append(time.perf_counter()-start)
        stats=dict(relationships_loaded=len(links), relationships_used=len(result.relationships),
            edge_counts=dict(Counter(r['edge_status'] for r in result.rows)),status_changes=len(changes),
            identity_only_resolutions=sum(r['old_edge_status']=='UNRESOLVED_EDGE' and r['new_edge_status']!='UNRESOLVED_EDGE' for r in changes),
            unresolved_reasons=breakdown(result.attribution),
            unresolved_rows_touching_relationship=sum(r['veto_retained'] and bool(r['continuity_component_pair_ids']) for r in result.attribution),
            ownership_evidence=dict(Counter(r['ownership_evidence'] for r in result.attribution)),
            components=len(result.chains), multi_link_components=sum(len(c['pair_ids'])>1 for c in result.chains),
            chain_conflicts=dict(Counter(reason for c in result.chains for reason in c['conflicts'])),
            downstream=dict(new_keep_rows=len(new_keep),dry_run=dry_run(result.rows) if new_keep else 'NOT_REQUIRED_NO_NEW_KEEP'),
            performance=dict(baseline_seconds=base_times,simulation_seconds=sim_times,baseline_median=base_med,
                simulation_median=sim_med,overhead_percent=(sim_med/base_med-1)*100,
                overlay_attribution_median=statistics.median(lookup_times),
                relationship_metadata_deep_bytes=deep_size(result.relationships),
                continuity_components_deep_bytes=deep_size(result.chains),
                scope='End-to-end offline gate plus overlay/attribution; separate overlay cost includes deepcopy, not lookup alone'))
        report['modes'][mode]=stats
    # Real negative cohorts never enter either SAME-only allowlist.
    core_pairs={r['pair_id'] for r in core};ext_pairs={r['pair_id'] for r in extended}
    negatives=[dict(case_id=r['case_id'],pair_id=r['pair_id'],visual_label=r['visual_label'],
                    continuity_admitted_core=r['pair_id'] in core_pairs,continuity_admitted_extended=r['pair_id'] in ext_pairs)
               for r in labels if r['visual_label']!='VISUALLY_SAME']
    assert len(negatives)==17 and not any(r['continuity_admitted_core'] or r['continuity_admitted_extended'] for r in negatives)
    prospective=read(ROOT/'runs/text_first_identity_smoke17_20261001/relationships.jsonl')
    blocked=[r for r in prospective if 'TEXT_UNIQUE' in r.get('blocking_reasons',[])]
    controls=[dict(pair_id=r['pair_id'],blocking_reasons=r['blocking_reasons'],
                   core_continuity_admitted=r['pair_id'] in core_pairs,
                   extended_continuity_admitted=r['pair_id'] in ext_pairs) for r in blocked]
    assert not any(r['core_continuity_admitted'] for r in controls)
    write_rows(args.output/'negative_visual_controls.jsonl',negatives)
    write_rows(args.output/'duplicate_text_controls.jsonl',controls)
    report['negative_controls']=dict(visual=dict(Counter(r['visual_label'] for r in negatives)),
        duplicate_text_blocked_count=len(controls),core_admitted=0,
        extended_validated_exceptions=sum(r['extended_continuity_admitted'] for r in controls),
        boundary_conflicts_retained='All exact baseline window vetoes preserved; targeted regression required')
    after={str(p.relative_to(ROOT)):digest(p) for p in files}
    assert before==after
    report['source_parity']=dict(rows_before=8146,rows_after=8146,all_fields_and_order_equal=True,
        frozen_baseline_classification_equal=True,input_file_hashes_unchanged=True,
        pixels='No decode or pixel operation; pixel evidence hashes and source artifact bytes preserved',
        image_writes=0,production_registered_file_bytes_unchanged=True)
    report['verdict']=('VALIDATED_RELATIONSHIPS_HELP_PARTIALLY' if report['modes']['CORE_280']['identity_only_resolutions']
                       else 'VALIDATED_RELATIONSHIPS_DO_NOT_HELP_EDGE_GATE')
    report['limitations']=['Only proved previous-hand-tail observations can discharge unsafe marker veto in a fixed window.',
        'Conflicting numbers, multiple NEW_HAND, epoch discontinuity and unproven boundaries remain vetoes.',
        'No new hand boundaries, duplicate event suppression, OCR correction, downstream repair or production promotion.']
    write_json(args.output/'input_hashes.json',before)
    write_json(args.output/'simulation_report.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__': main()
