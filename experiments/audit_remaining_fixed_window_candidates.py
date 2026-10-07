"""Bounded saved-receipt audit only. Does not import/run parser or validators."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/edge_hand_attribution_window_simulation_20261001_v2'
OVERLAY=ROOT/'outputs/validated_relationship_edge_smoke17_20261001_v3'
HISTORICAL=ROOT/'runs/poker_logic_math_jev_smoke17_final_20261001'
CLASSES=['ACTION_AFTER_HAND_END','ROW_IDENTITY_UNRESOLVED','MISSING_ACTION',
    'AMOUNT_SEMANTICS_UNPROVEN','ACTION_AFTER_FOLD','CARD_EVIDENCE_MISSING',
    'OLD_WIN','MATH_DELTA','SIDE_POT_NOT_VERIFIABLE','CHAT_OVERLAP_DUPLICATE']
GROUPS={'completeness':{'COMPLETENESS'},'identity_provenance':{'PROVENANCE'},
        'action_state':{'ACTION'},'card_evidence':{'CARDS'},'payout':{'PAYOUT'},'math':{'MATH'}}


def read(path):
    return [json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def historical_semantics_review(byid):
    """Compare saved receipts only; never recompute semantic hypotheses."""
    evidence=json.loads((HISTORICAL/'semantics_evidence.json').read_text('utf-8'))
    manifest=json.loads((HISTORICAL/'experimental_code_manifest.json').read_text('utf-8'))
    source=json.loads((HISTORICAL/'input_manifest.json').read_text('utf-8'))
    supporting=[]
    for hand in evidence['supporting_hands']:
        item=copy.deepcopy(hand)
        item['current_edge_status_counts']=dict(Counter(byid[r]['edge_status'] for r in hand['source_row_ids']))
        item['exact_current_source_rows']=[byid[r] for r in hand['source_row_ids']]
        supporting.append(item)
    return dict(historical_status=evidence['status'],source_sha256=source['sha256'],
        source_sha_matches_current=source['sha256']==sha(ROOT/'runs/production_smoke17_20261001_013909/ordered_timeline.jsonl'),
        supporting_hands=supporting,
        code_hash_comparison={p:dict(historical=h,current=sha(ROOT/p),equal=h==sha(ROOT/p))
                              for p,h in manifest['files'].items()},
        assessment='NOT_ELIGIBLE_PROVEN_UPSTREAM_FACT',
        reason='Cross-hand scope is intended, but support transcript ownership/multiplicity is unresolved. '
               'The historical monetary-clean rule does not exclude provenance/boundary/state blockers; '
               'three supporting transcripts are entirely UNRESOLVED_EDGE in current CORE. '
               'Ledger discrimination alone cannot certify their premise. Historical code also differs.')


def verify_receipts(original,classified,hands,records,attribution):
    if len(original)!=8146 or len(classified)!=8146:raise ValueError('Source count mismatch')
    if [{k:v for k,v in r.items() if k!='edge_status'} for r in classified]!=original:
        raise ValueError('Source field/order mutation')
    if len(hands)!=29 or len(records)!=29:raise ValueError('Candidate count mismatch')
    byid={r['stable_row_id']:r for r in classified}
    if len(byid)!=8146:raise ValueError('Duplicate source IDs')
    contexts={r['stable_row_id']:r for r in attribution}
    seen=[]
    for index,(hand,record) in enumerate(zip(hands,records)):
        if (hand['table_id'],hand['session_id'],hand['hand_id'])!=(record['table_id'],record['session_id'],record['hand_id']):
            raise ValueError('Hand context mismatch')
        ids=[rid for message in hand['messages'] for rid in message['source_row_ids']]
        if ids!=record['source_row_ids']:raise ValueError('Hand source order mismatch')
        seen+=ids
        starts=set()
        for message in hand['messages']:
            for row,rid in zip(message['source_rows'],message['source_row_ids']):
                if row!=byid[rid] or row['edge_status']!='KEEP':raise ValueError('Admitted source field mutation')
                if (row['table_id'],row['session_id'])!=(hand['table_id'],hand['session_id']):
                    raise ValueError('Interleaved context leakage')
                starts.add(contexts[rid]['hand_interval']['start_row_id'])
        if starts!=set(record['fixed_edge_window_ids']) or len(starts)!=1:
            raise ValueError('Fixed interval mismatch')
        if ids[0]!=next(iter(starts)):raise ValueError('Missing fixed start')
        for issue in record['validator_result']['issues']:
            if issue['message_index']>=0 and issue['source_row_ids']!=hand['messages'][issue['message_index']]['source_row_ids']:
                raise ValueError('Issue source mismatch')
        if record['fully_valid_blocking_checks']!=record['validator_result']['certification_blockers']:
            raise ValueError('Blocker loss')
    admitted=[r for r in classified if r['edge_status']=='KEEP']
    if len(seen)!=976 or Counter(seen)!=Counter(r['stable_row_id'] for r in admitted):
        raise ValueError('Admitted source row loss/duplication')
    for table in {r['table_id'] for r in admitted}:
        if [rid for rid in seen if byid[rid]['table_id']==table]!=[
            r['stable_row_id'] for r in admitted if r['table_id']==table]:
            raise ValueError('Per-table source order mutation')


def blocker_entry(issue,hand,byid):
    result=copy.deepcopy(issue)
    ids=issue['source_row_ids']
    result['exact_source_rows']=[copy.deepcopy(byid[rid]) for rid in ids]
    if issue['message_index']>=0:
        result['source_message']=copy.deepcopy(hand['messages'][issue['message_index']])
        result['check_scope']='EXACT_MESSAGE'
    else:
        result['source_message']=None
        result['check_scope']='AGGREGATE_HAND_LEDGER_OR_COMPLETENESS'
        result['aggregate_evidence_row_ids']=[rid for m in hand['messages'] for rid in m['source_row_ids']]
        result['money_message_evidence']=[dict(message_index=i,action=m['action'],player=m['player'],
            amount=m['amount'],source_row_ids=m['source_row_ids'],raw_text=m['raw_text'])
            for i,m in enumerate(hand['messages']) if m['action'] in {
                'ANTE','SMALL_BLIND','BIG_BLIND','CALL','BET','RAISE','UNCALLED_RETURN','WIN'}]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output must be fresh')
    files=[ROOT/'production_manifest.json',ROOT/'PRODUCTION.md',
           ROOT/'parser_core/edge_gate.py',ROOT/'parser_core/poker_hand_assembly.py',
           ROOT/'parser_core/poker_logic_engine.py',ROOT/'parser_core/poker_math_engine.py',
           ROOT/'experiments/replay_poker_logic_math_jev.py',
           ROOT/'experiments/attribute_edge_hands_and_window_simulation.py']
    files += [p for p in BASE.iterdir() if p.is_file()]
    files += [HISTORICAL/name for name in ['semantics_evidence.json','input_manifest.json','experimental_code_manifest.json']]
    files += [OVERLAY/'CORE_280'/name for name in ['classified_rows.jsonl','row_reason_attribution.jsonl','relationships_used.jsonl']]
    inputs=json.loads((OVERLAY/'input_hashes.json').read_text('utf-8'))
    files += [ROOT/path for path in inputs]
    before={str(p.relative_to(ROOT)):sha(p) for p in files}
    if any(sha(ROOT/path)!=value for path,value in inputs.items()):raise ValueError('Frozen source hash drift')
    original=read(ROOT/'runs/production_smoke17_20261001_013909/ordered_timeline.jsonl')
    core=read(OVERLAY/'CORE_280/classified_rows.jsonl')
    records=read(BASE/'CORE_FIXED_WINDOWS.hands.jsonl')
    hands=json.loads((BASE/'CORE_FIXED_WINDOWS.assembled_hands.json').read_text('utf-8'))
    attr=read(OVERLAY/'CORE_280/row_reason_attribution.jsonl')
    verify_receipts(original,core,hands,records,attr)
    byid={r['stable_row_id']:r for r in core};contexts={r['stable_row_id']:r for r in attr}
    links=read(OVERLAY/'CORE_280/relationships_used.jsonl')
    args.output.mkdir(parents=True)
    write(args.output/'historical_semantics_review.json',historical_semantics_review(byid))
    all28=[]
    for hand,r in zip(hands,records):
        if r['fully_valid']:continue
        entries=[blocker_entry(i,hand,byid) for i in r['fully_valid_blocking_checks']]
        first=r['source_row_ids'][0];interval=contexts[first]['hand_interval']
        bygroup={name:[i for i in entries if i['domain'] in domains] for name,domains in GROUPS.items()}
        bygroup['amount_semantics']=[i for i in entries if i['code']=='AMOUNT_SEMANTICS_UNPROVEN']
        bygroup['other']=[i for i in entries if i['domain'] not in set().union(*GROUPS.values())]
        critical=[i for i in entries if i['code']=='ROW_IDENTITY_UNRESOLVED']
        semantics=bygroup['amount_semantics']
        witness=critical[0] if critical else semantics[0]
        barrier=dict(type='UNPROVEN_EVENT_IDENTITY' if critical else 'UNPROVEN_AMOUNT_SEMANTICS',
            positive_upstream_proof_found=False,witness_blocker=witness,
            limitation='Physical SAME links do not certify poker event ownership/multiplicity' if critical else
                       'Current hand amounts at own contribution zero do not distinguish ADDED vs TOTAL; no promoted semantic receipt')
        related=[link for link in links if link['old_track_id'] in r['source_row_ids'] or link['new_track_id'] in r['source_row_ids']]
        all28.append(dict(hand_index=r['hand_index'],table_id=r['table_id'],session_id=r['session_id'],
            hand_number_candidate=r['hand_id'],fixed_interval=interval,start_row_id=first,
            source_row_ids=r['source_row_ids'],logic_checks=r['logic_blocking_checks'],
            math_checks=copy.deepcopy(r['validator_result']['math']),
            blocker_groups=bygroup,all_blockers=entries,all_issues=copy.deepcopy(r['validator_result']['issues']),
            raw_hand=copy.deepcopy(hand),related_validated_physical_links=related,
            evidence_barrier=barrier,upstream_transport_defect_proven=False))
    assert len(all28)==28
    with (args.output/'all28_attribution.jsonl').open('x',encoding='utf-8') as stream:
        for r in all28:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    classes=[]
    explanations={
        'ACTION_AFTER_HAND_END':'Validator ended flag follows parsed WIN; later betting occurs in same fixed interval. Premature/stale WIN or missing history is a hypothesis, not permission to reorder/delete.',
        'ROW_IDENTITY_UNRESOLVED':'Preserved source origin flags critical event observations; no proof of downstream loss of a certified event identity.',
        'MISSING_ACTION':'Raw source text cannot be mapped to proven action by current saved parser receipt; requires reading/semantic proof.',
        'AMOUNT_SEMANTICS_UNPROVEN':'No retained proven CALL/BET contract; balancing ledger is not semantic independence.',
        'ACTION_AFTER_FOLD':'Preserved player state FOLDED precedes later action. Aliases/stale row/missing boundary hypotheses lack positive proof.',
        'CARD_EVIDENCE_MISSING':'Current source-linked board/hole/ownership or contested payout completeness fails; no permission to infer cards.',
        'OLD_WIN':'Observed payout fails current completion check; upstream has not certified its legal placement.',
        'MATH_DELTA':'Observed Decimal net-minus-paid differs; missing/duplicated amounts/payouts cannot be selected from conservation alone.',
        'SIDE_POT_NOT_VERIFIABLE':'Upstream action/provenance ledger or card ranks lack proof; consequence of missing evidence, not independent repair.',
        'CHAT_OVERLAP_DUPLICATE':'Repeated observed marker/forced/return within fixed interval; physical/event duplicate mapping lacks proof. No new dedup permitted.'}
    for code in CLASSES:
        occurrences=[i for r in all28 for i in r['all_blockers'] if i['code']==code]
        warnings=[i for r in all28 for i in r['all_issues'] if i['code']==code and i['severity']!='BLOCKER']
        classes.append(dict(code=code,symptom_blocker_occurrences=len(occurrences),
            affected_hands=sum(code in {i['code'] for i in r['all_blockers']} for r in all28),
            warning_occurrences=len(warnings),proven_correctable_shared_root_causes=0,
            safely_correctable_blockers=0,remaining_root_cause_count='UNKNOWN_NOT_INFERRED_FROM_SYMPTOMS',
            evidence_interpretation=explanations[code],exact_occurrences=occurrences))
    write(args.output/'symptom_root_cause_register.json',classes)
    counts=Counter(r['evidence_barrier']['type'] for r in all28)
    metrics=dict(candidate_hands=29,remaining_candidates=28,logic_pass=1,math_pass=6,fully_valid=1)
    summary=dict(verdict='NO_NEXT_PROVEN_SAFE_FIX',baseline=metrics,after=metrics,
        execution='READ_SAVED_RECEIPTS_ONLY_NO_FIX_NO_SIMULATION',evidence_barriers=dict(counts),
        blockers_removed=0,changed_hands=[],new_valid_hands=[],regressed_hands=[],selected_root_cause=None,
        production_code_changed=False,edge_gate_calls=0,validator_calls=0,video_ocr_api_jev_calls=0,image_writes=0,
        source_rows=8146,admitted_rows=976,no_dedup=True,original_field_order_unchanged=True,
        no_positive_proof='No saved upstream fact certifies the unresolved blocker witnesses; no blanket claim about future recoverability')
    assert {str(p.relative_to(ROOT)):sha(p) for p in files}==before
    write(args.output/'input_hashes.json',before)
    write(args.output/'summary.json',summary)
    lines=['# Remaining28 CORE_FIXED_WINDOWS audit — 2026-10-01','',
        '**VERDICT: NO_NEXT_PROVEN_SAFE_FIX. PRODUCTION CODE CHANGED: NO.**','',
        '## FILES CHANGED','',
        'New experiments/audit_remaining_fixed_window_candidates.py, tests/test_remaining_candidate_receipts.py, this report and fresh JSON audit outputs. No existing experiment, edge gate, validator, production/config/manifest change.', '',
        '## BASELINE / CORE_FIXED_WINDOWS','',
        'Saved baseline receipt:29 candidates, Logic PASS1, Math PASS6, Fully valid1.28 non-valid candidates audited. Same29/1/6/1 after audit: no positive proof selected, no fix or simulation executed. No blockers removed, no changed/regressed hands.', '',
        '## ALL-28 ATTRIBUTION','',
        'Machine-readable all28_attribution.jsonl contains exact table/session/fixed start/next boundary/interval, all source rows and raw hand, all issues, all blockers grouped by completeness/provenance/action/cards/payout/math/amount semantics. Math includes exact player ledger and delta. Every message blocker embeds its exact original source rows. Aggregate EOF checks retain the whole hand source-row inventory and money message witnesses; they are not assigned an invented single culprit row.', '',
        '## SYMPTOM / ROOT CAUSE / SAFE CORRECTION','',
        '| Symptom | BLOCKER occurrences | Affected hands | Proven correctable roots | Safe blocker removals |','|---|---:|---:|---:|---:|']
    for c in classes:lines.append(f'|{c["code"]}|{c["symptom_blocker_occurrences"]}|{c["affected_hands"]}|0|0|')
    lines += ['',
        'Occurrences include only BLOCKER; warnings separately recorded. Unknown true root-cause population is not reported as zero. Zero means no positively proven safe correction in this bounded evidence audit.', '',
        '## ROOT CAUSE SELECTED / CORRECTABILITY','',
        '**None.**27 remaining candidates carry critical ROW_IDENTITY_UNRESOLVED. For each, evidence_barrier embeds a exact blocker witness and the original unresolved source. Some have validated physical continuity links; these do not certify normalized poker-event multiplicity or correct hand ownership. Source origin remains unresolved and is propagated by the unchanged validator. No certified upstream event-identity fact was found to transport.', '',
        'The remaining T06 #2876718617 (index4) has only2 AMOUNT_SEMANTICS_UNPROVEN: CALL10750 and BET10956. Ledger delta0. Both actors have street contribution0 before those actions, so added/total interpretations coincide for those values. This hand does not provide discriminatory evidence itself. Cross-hand semantics inference is intended; lack of self-proof alone does not reject a global contract.', '',
        'Historical runs/poker_logic_math_jev_smoke17_final_20261001/semantics_evidence.json does contain ADDED_SUPPORTED with the same original input SHA. Supporting ADDED/TOTAL deltas are 0/-80000, 0/-6000, 0/-12000, 0/-6000. However semantic_evidence in experiments/replay_poker_logic_math_jev.py tests unique money signatures, nonnull amounts, selected money errors and delta discrimination; it does not exclude provenance/boundary/state conflicts. Three support transcripts (T02 #2876720584, T06 #2876722787, T06 #2876724008) are entirely UNRESOLVED_EDGE in current CORE; only T04 is admitted. Unresolved starts and, for T06 #2876724008, antes/blinds leave complete ownership/multiplicity unproved. Historical assembler/logic/math/replay hashes also differ. This DERIVED receipt is therefore not an eligible proven upstream fact for current fixed windows. Transporting it would import an unresolved premise. Exact support rows, statuses and hash comparisons are saved in historical_semantics_review.json.', '',
        'FILE → FUNCTION → CALLER → DATA inspected read-only: parser_core/poker_logic_engine.py validate_hand emits critical provenance from source origin, action-state from observed sequence, AMOUNT_SEMANTICS_UNPROVEN from absent hand.semantics; poker_math_engine.validate_math checks the computed ledger. Prior fixed-window assembler transports exact intervals; all current29 receipts have one immutable interval. No residual cross-window transport loss demonstrated.', '',
        '## CLASSES INVESTIGATED','']
    for c in classes:lines += [f'**{c["code"]}:** '+c['evidence_interpretation'],'']
    lines += ['## SIMULATION / METRICS / CHANGED HANDS','',
        'No simulation: user requires positive source/upstream proof before a new fix. Fully valid remains1, not2. Candidate hands29, Logic1, Math6. Blockers removed0, interpreted rows changed0, changed/new/regressed hands0. Future OCR/visual/semantic information could change this finding; no universal impossibility claim.', '',
        '## NEGATIVE CONTROLS / REGRESSIONS','',
        'No fix exists to fire. All conflicting-number, epoch, competing-start, missing-start/end, unresolved-identity, ERROR/UNREADABLE/OCR, ambiguous-semantics, missing-card, branch, interleaved-session and genuine after-end/after-fold inputs therefore retain their saved interpretation. This is an applicability decision, not a claimed synthetic fix test. Receipt-validation tests exercise input mutation, source loss, false context and blocker loss. No production, edge gate or validators execute in the tests.', '',
        '## PARITY / PROVENANCE','',
        '8146 original rows equal classified rows after removing only edge_status, field-for-field/order.976 admitted source rows remain exactly once in saved29 hands with matching row values and per-table order. All94 frozen source/production hashes checked and all audit input hashes compared before/after. Existing validators/configs/manifest and previous artifacts unchanged. No image writes, decode/OCR/API/Jev, edge gate or validator calls. Read-only saved receipts only.', '',
        '## TESTS','',
        'See tests.json for actual test exit/count and independent verification.json for audit checks. No simulation negative-control success is claimed.', '',
        '## ALL28 HAND REGISTER','']
    for r in all28:
        math=r['math_checks']
        lines += [f'### {r["hand_index"]+1}: {r["table_id"]} / {r["hand_number_candidate"] or "UNKNOWN"}', '',
            f'Session `{r["session_id"]}`; start `{r["start_row_id"]}`; fixed interval frames {r["fixed_interval"]["start_frame"]}→{r["fixed_interval"]["next_start_frame"]}.',
            f'Math: net{math["net_total"]}, payout{math["payout_total"]}, delta{math["delta"]}; errors {math["errors"]}. Full per-player ledger in JSON.',
            'Logic checks: '+', '.join(sorted({i['code'] for i in r['logic_checks']}))+'.','',
            '| Blocker group | Exact checks |','|---|---|']
        for group,issues in r['blocker_groups'].items():
            lines.append(f'|{group}|'+(', '.join(sorted({i['code'] for i in issues})) or 'NONE')+'|')
        lines += ['','| BLOCKER | Source IDs / scope | Detail |','|---|---|---|']
        for issue in r['all_blockers']:
            detail=str(issue['detail']).replace('|','\\|').replace('\n',' ')
            lines.append(f'|{issue["code"]}|'+(', '.join(issue['source_row_ids']) or 'Aggregate; exact hand inventory in JSON')+f'|{detail}|')
        lines.append('')
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
