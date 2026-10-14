"""Actual saved-source direct provider and synthetic negatives; no HH acceptance."""
import copy, hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/hh95_evidence_20261008';SRC=Path('C:/Users/Admin/.codex/tmp/hh95_source_cache_20261008/full_v1')
from parser_core import direct_start_stack as draft
from parser_core.adjacent_stack_carry import SavedCarrySources
@pytest.fixture(scope='module')
def native():
    old=json.loads((OUT/'direct_first_stack_native_cases.json').read_text('utf-8'))['cases']
    records=[json.loads(line) for line in (OUT/'carry_service_full_replay_v7/logical_hand_analysis.jsonl').open(encoding='utf-8') if line.strip()]
    records=[r for r in records if r['observed_hand']['table_id']=='T01']
    wanted={(draft.native_key(c),c['actor']) for c in old}
    joint=json.loads((OUT/'carry_service_full_replay_v7/joint_start_stack_audit.json').read_text('utf-8'))['rows']
    selected=[r for r in joint if ((r['table_id'],r['session_id'],r['hand_id']),r['actor']) in wanted]
    assert len(selected)==5
    registry=SavedCarrySources(SRC,records,[])
    provider=draft.DirectStartStackProvider(SRC,records,SimpleNamespace(rows=selected),registry)
    inv=provider.inventory;inv._tested_provider=provider
    inv._tested_records=records;inv._tested_joint_rows=selected
    cases=[inv.case_for(draft.native_key(c),c['actor'],c['seat']) for c in old]
    return inv,cases
@pytest.mark.parametrize('index,value',[(0,16383),(1,None),(2,20963),(3,15774),(4,None)])
def test_fresh_full_native_proof(native,index,value):
    inv,cases=native;provider=inv._tested_provider;case=cases[index];k=(draft.native_key(case),case['actor']);p=provider.proofs.get(k) or next(row for row in provider.audit if row['hand_key']==list(k[0]) and row['actor']==case['actor'])
    if value is None:
        assert p['status']=='UNKNOWN';assert p['reason']=='FIRST_VISIBLE_UNEXPLAINED_PRE_NUMERIC_PIXELS'
    else:
        assert p['status']=='PROVEN';assert p['calculated_start_stack']==value
        assert p['forced_amount_added_back']==70 and [(r['role'],r['amount']) for r in p['own_forced_posts']]==[('ANTE',70)]
        assert p['first_visible_proof']['start_frame']==cases[index]['source_context']['header_frame']
        assert p['scope']=='SOURCE_REPLAYED_DIRECT_FIRST_NUMBER_NO_OCCUPANCY_EPOCH'
def number(c,frame):return next(r for r in c['raw_direct_stack_readings'] if r['source_frame']==frame)
def test_foreign_seat_high_confidence_transplant(native):
    _,cases=native;c=copy.deepcopy(cases[0]);r=number(c,2132);donor=number(cases[1],2132)
    r['ocr_response']=copy.deepcopy(donor['ocr_response'])
    for k in ['raw_text','confidence','ocr_status']:r[k]=donor[k]
    assert r['confidence']>.95
    with pytest.raises(ValueError,match='DIRECT_OCR_CALLBACK_ITEM_CONFLICT'):draft.full_readings(c)
@pytest.mark.parametrize('mutation',['foreign_frame','missing','rejected','scaled_outer'])
def test_every_callback_identity(native,mutation):
    _,cases=native;c=copy.deepcopy(cases[3] if mutation=='scaled_outer' else cases[0])
    if mutation=='rejected':r=next(r for r in c['raw_direct_stack_readings'] if draft.effective_numeric(r,.95,allow_scaled=True) is None)
    elif mutation=='scaled_outer':r=next(r for r in c['raw_direct_stack_readings'] if 'same_source_scale_evidence' in r['ocr_response'])
    else:r=number(c,2132)
    if mutation=='missing':r['ocr_response'].pop('item_id')
    else:r['ocr_response']['item_id']='1:999999:0'
    with pytest.raises(ValueError,match='DIRECT_OCR_CALLBACK_ITEM_CONFLICT'):draft.full_readings(c)
@pytest.mark.parametrize('mutation',['shortened','stale'])
def test_case_inventory_required(native,mutation):
    inv,cases=native;c=copy.deepcopy(cases[0])
    if mutation=='shortened':c['raw_stack_captures']=c['raw_stack_captures'][1:]
    else:c['source_context']['header_frame']+=1
    p=draft.prove(c,inv);assert p['status']=='UNKNOWN' and p['reason']=='FRESH_COMPLETE_WINDOW_INVENTORY_REQUIRED'
@pytest.fixture(scope='module')
def sql_native(native):
    _,cases=native;c=cases[2];sql,*_=draft.numeric_sources(c,c['database_metadata'])
    row=next(copy.deepcopy(r) for f,r in sorted(sql.items()) if f==5102 and r['ocr_status']=='OK')
    return row
def test_sql_different_accepted_number(sql_native):
    r=copy.deepcopy(sql_native);r['raw_text']='20894'
    with pytest.raises(ValueError,match='INDEPENDENT_SQL_NUMERIC_CONFLICT'):draft.verify_sql_conflicts({5102:r},5056,[5102,5103,5104],20893)
def test_sql_earlier_accepted_number(sql_native):
    r=copy.deepcopy(sql_native)
    with pytest.raises(ValueError,match='INDEPENDENT_SQL_EARLIER_ACCEPTED_NUMBER'):draft.verify_sql_conflicts({5101:r},5056,[5102,5103,5104],20893)
def test_legitimate_sql_change_after_triple(sql_native):
    r=copy.deepcopy(sql_native);r['raw_text']='20800'
    assert draft.verify_sql_conflicts({5105:r},5056,[5102,5103,5104],20893)==[]
def test_direct_raw_response_consistency(native):
    _,cases=native;c=copy.deepcopy(cases[0]);number(c,2132)['ocr_response']['raw_text']='16314'
    with pytest.raises(ValueError,match='DIRECT_STACK_RESPONSE_CONFLICT'):draft.full_readings(c)
def test_direct_native_rgb_join(native):
    _,cases=native;c=copy.deepcopy(cases[0]);cap=next(r['raw_record'] for r in c['raw_stack_captures'] if r['raw_record']['source_frame']==2132);cap['pixel_sha256']='0'*64
    with pytest.raises(ValueError,match='DIRECT_STACK_NATIVE_PIXELS_CONFLICT'):draft.full_readings(c)

def test_runtime_five_candidate_verdicts(native):
    inv,_=native;provider=inv._tested_provider
    assert len(provider.audit)==5
    assert [r['status'] for r in provider.audit].count('PROVEN')==3
    assert [r['status'] for r in provider.audit].count('UNKNOWN')==2

@pytest.mark.parametrize('case_index',[0,2])
def test_actual_for_hand_reproduces_all_native_proof_json(native,case_index):
    inv,cases=native;provider=inv._tested_provider
    context=draft.native_key(cases[case_index])
    record=next(r for r in inv._tested_records if draft.native_key(r['observed_hand'])==context)
    actual=provider.for_hand(record['observed_hand'])
    expected=[p for (saved,_),p in provider.proofs.items() if saved==context]
    assert actual==expected
    assert json.dumps(actual,sort_keys=True)==json.dumps(expected,sort_keys=True)

@pytest.mark.parametrize('case_index',[0,2,3])
def test_native_join_receipt_has_pixels_and_no_fabricated_sql_identity(native,case_index):
    inv,cases=native;case=cases[case_index]
    proof=inv._tested_provider.proofs[(draft.native_key(case),case['actor'])]
    receipt=draft.join_receipt(proof)
    assert receipt['proof_id']==draft.proof_id(proof)
    assert receipt['calculated_start_stack']==proof['calculated_start_stack']
    assert [r['source_frame'] for r in receipt['native_numeric_sources']]==proof['numeric_frames']
    assert all(r['item_id'] and r['pixel_sha256'] and r['source_rgb_sha256'] for r in receipt['native_numeric_sources'])
    assert not receipt.get('source_observation_ids') and not receipt.get('source_state_ids')
    assert receipt.get('occupancy_epoch') is None
    assert proof['occupancy_epoch'] is None and proof['manufactured_sql_observation_ids'] is False
    from parser_core.first_stack_ocr import direct_numeric_policy
    assert receipt['numeric_ocr_policy']==proof['first_visible_proof']['numeric_ocr_policy']==direct_numeric_policy()


def test_saved_direct_body_rejects_bound_scaled_conflict_under_calibrated_policy(native):
    from test_direct_numeric_policy import attach_bound_scale
    inv,cases=native;case=copy.deepcopy(cases[0])
    point=draft.prove_actor_point(case['source_context'],case['actor'])
    frames={r['source_frame'] for r in point['nickname_sources']}
    for r in case['raw_direct_stack_readings']:
        if r['source_frame'] in frames:
            r['confidence']=.90;r['ocr_response']['score']=.90
            attach_bound_scale(r,'999999')
    # Isolate the body guard after inventory validation. This synthetic negative
    # is not a native-source proof or an alternative inventory implementation.
    inventory=SimpleNamespace(matches=lambda _:True,hashes=inv.hashes)
    result=draft._prove_body(case,inventory,lambda:None)
    assert result['status']=='UNKNOWN' and result['reason']=='FIRST_VISIBLE_FIRST_NUMERIC_TRIPLE_NOT_PROVEN'

@pytest.mark.parametrize('mutation',['foreign_hand','stale_hand'])
def test_provider_foreign_or_stale_current_hand_rejected(native,mutation):
    inv,cases=native;provider=inv._tested_provider;context=draft.native_key(cases[0])
    hand=copy.deepcopy(next(r['observed_hand'] for r in inv._tested_records if draft.native_key(r['observed_hand'])==context))
    if mutation=='foreign_hand':hand['hand_id']='foreign-hand'
    else:hand['messages'][-1]['source_rows'][-1]['raw_ocr_text']='stale-current-hand'
    assert provider.for_hand(hand)==[]

@pytest.mark.parametrize('mutation',['case','record','cached_proof','cached_equal_value_wrong_seat'])
def test_provider_altered_case_record_or_cached_proof_has_no_authority(native,mutation):
    inv,cases=native;case=cases[0];provider=copy.copy(inv._tested_provider)
    saved_key=(draft.native_key(case),case['actor']);stored_case,record=provider.cases[saved_key]
    hand=record['observed_hand'];provider.cases=dict(provider.cases);provider.proofs=dict(provider.proofs)
    if mutation=='case':
        provider.cases[saved_key]=(dict(stored_case,raw_stack_captures=stored_case['raw_stack_captures'][:-1]),record)
    elif mutation=='record':
        changed=copy.deepcopy(record);changed['observed_hand']['messages']=changed['observed_hand']['messages'][:-1]
        provider.cases[saved_key]=(stored_case,changed)
    elif mutation=='cached_equal_value_wrong_seat':provider.proofs[saved_key]=dict(provider.proofs[saved_key],seat='seat1')
    else:provider.proofs[saved_key]=dict(provider.proofs[saved_key],calculated_start_stack=999999)
    assert provider.for_hand(hand)==[]

@pytest.mark.parametrize('mutation',['changed','deleted'])
def test_provider_real_runtime_preseal_failure(native,tmp_path,mutation):
    inv,cases=native;provider=copy.copy(inv._tested_provider)
    path=tmp_path/'runtime_dependency.bin';path.write_bytes(b'retained')
    provider.runtime_files+=((str(path),hashlib.sha256(path.read_bytes()).hexdigest()),)
    if mutation=='changed':path.write_bytes(b'changed')
    else:path.unlink()
    context=draft.native_key(cases[0]);record=next(r for r in inv._tested_records if draft.native_key(r['observed_hand'])==context)
    assert provider.for_hand(record['observed_hand'])==[]

def test_provider_postseal_failure_returns_no_native_proof(native,monkeypatch):
    inv,cases=native;provider=inv._tested_provider;seen=[];original=draft.SourceSealBatch.finish
    def failed_finish(batch):
        seen.append(original(batch));return False
    monkeypatch.setattr(draft.SourceSealBatch,'finish',failed_finish)
    context=draft.native_key(cases[0]);record=next(r for r in inv._tested_records if draft.native_key(r['observed_hand'])==context)
    assert provider.for_hand(record['observed_hand'])==[]
    assert seen==[True]

def test_constructor_postseal_failure_keeps_all_native_proofs_unpublished(native,monkeypatch):
    inv,_=native;provider=draft.DirectStartStackProvider.__new__(draft.DirectStartStackProvider)
    original=draft.SourceSealBatch.finish;original_prove=draft.DirectStartStackProvider._prove;seen=[];statuses=[]
    def observed_prove(instance,*args,**kwargs):
        result=original_prove(instance,*args,**kwargs);statuses.append(result['status']);return result
    def failed_finish(batch):
        assert provider.cases=={} and provider.proofs=={} and provider.audit==[]
        seen.append(original(batch));return False
    monkeypatch.setattr(draft.DirectStartStackProvider,'_prove',observed_prove)
    monkeypatch.setattr(draft.SourceSealBatch,'finish',failed_finish)
    with pytest.raises(ValueError,match='DIRECT_BATCH_POST_SOURCE_SEAL_CHANGED'):
        provider.__init__(SRC,inv._tested_records,SimpleNamespace(rows=inv._tested_joint_rows),inv._tested_provider.registry)
    assert statuses.count('PROVEN')==3 and statuses.count('UNKNOWN')==2
    assert seen==[True]
    assert provider.cases=={} and provider.proofs=={} and provider.audit==[]

def test_no_fresh_direct_inventory(native):
    _,cases=native
    assert draft.prove(cases[0],None)['reason']=='FRESH_COMPLETE_WINDOW_INVENTORY_REQUIRED'

@pytest.fixture(scope='module')
def consumer_native(native,tmp_path_factory):
    from parser_core.stack_evidence import StackEvidenceProvider
    from parser_core.joint_stack_sources import JointStackSourceProvider
    from parser_core.poker_reconstruction_layer import reconstruct_production_hands,reconstruction_config
    from parser_core.hh_export import export_hand_histories
    inv,cases=native;case=cases[0];provider=inv._tested_provider;context=draft.native_key(case)
    record=next(r for r in inv._tested_records if draft.native_key(r['observed_hand'])==context)
    sql=StackEvidenceProvider(SRC/'stack_pipeline/timeline.sqlite',video_sha256=inv.video_sha256)
    # Restore the actual retained v7 joint proof store, using its real replay
    # method. This supplies no fabricated success and adds no direct authority.
    joint=JointStackSourceProvider.__new__(JointStackSourceProvider);joint.proofs={}
    for line in (OUT/'carry_service_full_replay_v7/joint_start_stack_proofs.jsonl').open(encoding='utf-8'):
        proof=json.loads(line);key=(proof['table_id'],proof['session_id'],proof['hand_id'])
        if key==context:joint.proofs.setdefault(key,[]).append(proof)
    sql.joint_sources=joint;sql.direct_sources=provider
    config=dict(reconstruction_config(),enabled=True,first_visible_stack_contract=True,strict_source_uniqueness=True)
    results,_=reconstruct_production_hands([record],config=config,stack_provider=sql)
    result=results[0];path=tmp_path_factory.mktemp('direct_consumers_native')
    sql.write_audit(path,[record],start_stack=True,reconstruction_results=[result])
    stack_rows=[json.loads(line) for line in (path/'stack_start_stack.jsonl').read_text('utf-8').splitlines()]
    (path/'poker_reconstruction.jsonl').write_text(json.dumps(result)+'\n','utf-8')
    export_hand_histories(path,[record],direct_provider=provider)
    hh=json.loads((path/'hand_histories.jsonl').read_text('utf-8').splitlines()[0])
    return inv,case,record,sql,config,result,stack_rows,hh

def test_actual_direct_solver_audit_and_hh_consume_native_receipt(consumer_native):
    _,case,_,_,_,result,stacks,hh=consumer_native
    assert result['joined_start_stacks'][case['actor']]==16383
    receipt=next(r for r in result['direct_start_stack_join_receipts'] if r['actor']==case['actor'])
    stack=next(r for r in stacks if r['player']==case['actor'])
    assert stack['status']=='START_STACK_PROVEN' and stack['calculated_start_stack']==16383
    assert stack['direct_binding_receipt']==receipt
    assert stack['source_observation_id'] is None and stack['source_state_id'] is None and stack['occupancy_epoch'] is None
    player=next(r for r in hh['players'] if r['player']==case['actor'])
    assert player['START_STACK_status']=='PROVEN' and player['START_STACK']==16383

@pytest.mark.parametrize('mutation',['missing_provider','forged_value','foreign_seat','foreign_frame'])
def test_alleged_direct_receipt_cannot_grant_solver_audit_hh_or_carry_authority(consumer_native,tmp_path,mutation):
    from parser_core.poker_reconstruction_layer import reconstruct_production_hands
    from parser_core.hh_export import export_hand_histories
    from parser_core.adjacent_stack_carry import independent_stacks
    inv,case,record,sql,config,_,_,_=consumer_native
    provider=copy.copy(inv._tested_provider);provider.cases={}
    proof=provider.proofs[(draft.native_key(case),case['actor'])];receipt=draft.join_receipt(proof)
    if mutation=='forged_value':receipt['calculated_start_stack']=999999
    elif mutation=='foreign_seat':receipt['seat']='seat1'
    elif mutation=='foreign_frame':receipt['native_numeric_sources'][0]['source_frame']+=1
    source=copy.copy(sql)
    if mutation=='missing_provider':source.__dict__.pop('direct_sources',None)
    else:source.direct_sources=provider
    results,_=reconstruct_production_hands([record],config=config,stack_provider=source)
    result=results[0]
    assert case['actor'] not in result['joined_start_stacks']
    assert not result['direct_start_stack_join_receipts']
    result['direct_start_stack_join_receipts']=[receipt]
    source.write_audit(tmp_path,[record],start_stack=True,reconstruction_results=[result])
    stacks=[json.loads(line) for line in (tmp_path/'stack_start_stack.jsonl').read_text('utf-8').splitlines()]
    target=next(r for r in stacks if r['player']==case['actor'])
    assert target['status']=='START_STACK_UNKNOWN' and target['calculated_start_stack'] is None
    alleged=dict(target,status='START_STACK_PROVEN',seat_id=case['seat'],calculated_start_stack=receipt['calculated_start_stack'],
        source_observation_id=None,source_state_id=None,occupancy_epoch=None,direct_binding_receipt=receipt)
    stacks=[alleged if r['player']==case['actor'] else r for r in stacks]
    (tmp_path/'stack_start_stack.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in stacks),'utf-8')
    (tmp_path/'poker_reconstruction.jsonl').write_text(json.dumps(result)+'\n','utf-8')
    export_hand_histories(tmp_path,[record],direct_provider=None if mutation=='missing_provider' else provider)
    hh=json.loads((tmp_path/'hand_histories.jsonl').read_text('utf-8').splitlines()[0])
    assert next(r for r in hh['players'] if r['player']==case['actor'])['START_STACK'] is None
    assert case['actor'] not in independent_stacks(record['observed_hand'],dict(seed_math_facts=[]),source)

@pytest.mark.parametrize('field,value',[('source_observation_id',999999),('source_state_id','FOREIGN_SQL_STATE'),('occupancy_epoch',999)])
def test_hh_direct_receipt_cannot_claim_sql_identity_or_epoch(consumer_native,tmp_path,field,value):
    from parser_core.hh_export import export_hand_histories
    inv,case,record,_,_,result,stacks,_=consumer_native
    altered=copy.deepcopy(stacks)
    target=next(r for r in altered if r['player']==case['actor']);target[field]=value
    (tmp_path/'stack_start_stack.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in altered),'utf-8')
    (tmp_path/'poker_reconstruction.jsonl').write_text(json.dumps(result)+'\n','utf-8')
    export_hand_histories(tmp_path,[record],direct_provider=inv._tested_provider)
    hh=json.loads((tmp_path/'hand_histories.jsonl').read_text('utf-8').splitlines()[0])
    player=next(r for r in hh['players'] if r['player']==case['actor'])
    assert player['START_STACK_status']=='UNKNOWN' and player['START_STACK'] is None

@pytest.mark.parametrize('mutation',['equal_value_wrong_seat','changed_value','changed_native_frame'])
def test_actual_positive_provider_requires_exact_receipt_in_audit_and_hh(consumer_native,tmp_path,mutation):
    from parser_core.hh_export import export_hand_histories
    inv,case,record,sql,_,original,_,_=consumer_native
    result=copy.deepcopy(original)
    receipt=next(r for r in result['direct_start_stack_join_receipts'] if r['actor']==case['actor'])
    if mutation=='equal_value_wrong_seat':receipt['seat']='seat1'
    elif mutation=='changed_value':receipt['calculated_start_stack']+=1
    else:receipt['native_numeric_sources'][0]['source_frame']+=1
    # The provider remains the real positive native provider. Equality against
    # its freshly reproduced receipt must reject the serialized mutation.
    sql.write_audit(tmp_path,[record],start_stack=True,reconstruction_results=[result])
    stacks=[json.loads(line) for line in (tmp_path/'stack_start_stack.jsonl').read_text('utf-8').splitlines()]
    target=next(r for r in stacks if r['player']==case['actor'])
    assert target['status']=='START_STACK_UNKNOWN' and target['calculated_start_stack'] is None
    alleged=dict(target,status='START_STACK_PROVEN',seat_id=receipt['seat'],calculated_start_stack=receipt['calculated_start_stack'],
        source_observation_id=None,source_state_id=None,occupancy_epoch=None,direct_binding_receipt=receipt)
    stacks=[alleged if r['player']==case['actor'] else r for r in stacks]
    (tmp_path/'stack_start_stack.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in stacks),'utf-8')
    (tmp_path/'poker_reconstruction.jsonl').write_text(json.dumps(result)+'\n','utf-8')
    export_hand_histories(tmp_path,[record],direct_provider=inv._tested_provider)
    hh=json.loads((tmp_path/'hand_histories.jsonl').read_text('utf-8').splitlines()[0])
    assert next(r for r in hh['players'] if r['player']==case['actor'])['START_STACK'] is None

def test_carry_direct_stack_still_requires_native_roster_seat_check(consumer_native):
    from parser_core.adjacent_stack_carry import independent_stacks,roster
    inv,case,record,sql,*_=consumer_native;hand=record['observed_hand']
    proof=inv._tested_provider.registry.roster_proof(hand)
    try:mapping=roster(hand,proof)
    except ValueError:
        with pytest.raises(ValueError):independent_stacks(hand,dict(seed_math_facts=[]),sql)
    else:
        assert mapping[case['actor']]==case['seat']
        assert independent_stacks(hand,dict(seed_math_facts=[]),sql)[case['actor']]==16383
