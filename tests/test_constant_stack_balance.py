"""Native control plus synthetic negative mutations; no HH authority."""
import base64, copy, hashlib, importlib.util, json, sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from parser_core.adjacent_stack_carry import SavedCarrySources
from parser_core.first_stack_ocr import RULE, proof_config, scale_pixels
OUT=ROOT/'runs/hh95_evidence_20261008';SRC=Path('C:/Users/Admin/.codex/tmp/hh95_source_cache_20261008/full_v1')
from parser_core import constant_stack_balance as draft
@pytest.fixture(scope='module')
def native():
    path=OUT/'constant_balance_four_t05_2878365932_bvg72_native_case_20261008_161502.json'
    blob=json.loads(path.read_text('utf-8'));case=blob['case'];metadata=blob['database_metadata_raw']
    records=[]
    with (OUT/'carry_service_full_replay_v7/logical_hand_analysis.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['observed_hand']['table_id'] in {'T05','T06'}:records.append(r)
    record=next(r for r in records if r['observed_hand']['hand_id']==case['hand_id'])
    chosen=[record]+[r for r in records if r['observed_hand']['table_id']=='T06' and r['observed_hand']['hand_id'] in {'2878365610','2878365876'}]
    from types import SimpleNamespace
    audit=json.loads((OUT/'carry_service_full_replay_v7/joint_start_stack_audit.json').read_text('utf-8'))
    wanted={('2878365932','bvg72'),('2878365610','mmm156'),('2878365876','bbobyy')}
    bound=[r for r in audit['rows'] if (r.get('hand_id'),r.get('actor')) in wanted]
    assert len(bound)==3
    provider=draft.ConstantStackBalanceProvider(SRC,records,SimpleNamespace(rows=bound))
    registry=provider.registry;inventory=provider.inventory
    registry._tested_constant_provider=provider
    registry._tested_joint_rows=bound
    return case,metadata,record,registry,inventory
def test_real_native_positive_control(native):
    case,metadata,record,registry,inventory=native
    # Constructor has already run the actual full native proof for this case.
    result=registry._tested_constant_provider.proofs[(draft.key(record['observed_hand']),case['actor'])]
    assert result['status']=='PROVEN',result
    assert result['calculated_start_stack']==30364
    assert result['ledger_seed_start_stacks']=={} and result['first_visible_authority'] is False
    assert result['scope']=='SOURCE_REPLAYED_CONSTANT_BALANCE_NO_OCCUPANCY_EPOCH'
def test_no_fresh_inventory(native):
    case,metadata,record,registry,_=native
    assert draft.prove(case,metadata,record,registry)['reason']=='FRESH_INDEPENDENT_INVENTORY_REQUIRED'
@pytest.mark.parametrize('field',['source_context','full_hand_source_context','raw_stack_captures','raw_sql_observations','raw_sql_visual_states','raw_direct_stack_readings'])
@pytest.mark.parametrize('mutation',['omit','alter'])
def test_fresh_inventory_rejects_case_subset_or_changes(native,field,mutation):
    case,metadata,record,registry,inventory=native;bad=dict(case)
    if mutation=='omit':bad[field]={} if isinstance(case[field],dict) else []
    elif isinstance(case[field],dict):bad[field]=dict(case[field],capture_end_inclusive=case[field]['capture_end_inclusive']-1)
    else:
        bad[field]=[dict(x) for x in case[field]];bad[field][0]=dict(bad[field][0],mutation_test_only=True)
    assert not inventory.matches(bad,metadata,record,registry)
    assert draft.prove(bad,metadata,record,registry,inventory)['reason']=='FRESH_NATIVE_INVENTORY_CONFLICT'
@pytest.mark.parametrize('mutation',['omit','alter'])
def test_fresh_metadata_inventory(native,mutation):
    case,metadata,record,registry,inventory=native
    bad=[] if mutation=='omit' else [dict(r) for r in metadata]
    if bad:bad[0]['value']='null'
    assert not inventory.matches(case,bad,record,registry)
    assert draft.prove(case,bad,record,registry,inventory)['reason']=='FRESH_NATIVE_INVENTORY_CONFLICT'
def test_native_interval_truncated_complete_record_and_both_registries(native):
    case,metadata,record,registry,inventory=native;bad=copy.deepcopy(record)
    assert bad['lifecycle']['status']=='COMPLETE'
    bad['observed_hand']['messages']=bad['observed_hand']['messages'][:-1]
    reg=copy.copy(registry);reg.records=[bad if r['observed_hand']['hand_id']==case['hand_id'] else r for r in registry.records]
    inv=copy.copy(inventory);inv.records=[bad if r['observed_hand']['hand_id']==case['hand_id'] else r for r in inventory.records]
    assert inv.matches(case,metadata,bad,reg)
    with pytest.raises(ValueError,match='COMPLETE_NATIVE_HAND_COVERAGE_CONFLICT'):
        draft.complete_hand_interval(bad,reg,case['actor'],case['seat'])
def test_removed_native_tail_from_registry(native):
    case,_,record,registry,_=native;reg=copy.copy(registry)
    target=[r for r in registry.native if r['table_id']=='T05' and r['session_id']==case['session_id'] and case['constant_interval'][0]<=r['frame_id']<=case['constant_interval'][1]][-1]
    reg.native=[r for r in registry.native if r is not target]
    with pytest.raises(ValueError,match='COMPLETE_NATIVE_HAND_COVERAGE_CONFLICT'):
        draft.complete_hand_interval(record,reg,case['actor'],case['seat'])
def test_removed_next_native_header_admission(native):
    case,_,record,registry,_=native;reg=copy.copy(registry);boundary=case['constant_interval'][1]+1
    reg.headers=[h for h in registry.headers if not(h['table_id']==5 and h['session_id']==case['session_id'] and h['source_frame']==boundary and h['status']=='ADMITTED')]
    with pytest.raises(ValueError,match='INDEPENDENT_NEXT_HEADER_ADMISSION_UNKNOWN'):
        draft.native_hand_bounds(record['observed_hand'],reg)
def test_unknown_service_source_remains_unproven(native):
    case,_,record,_,_=native;hand=copy.deepcopy(record['observed_hand'])
    message=hand['messages'][-1];message['raw_text']='Dealer: unknown transfer ???';message['action']='TABLE_INFO'
    for r in message['source_rows']:r['raw_ocr_text']='Dealer: unknown transfer ???';r['event_type']='TABLE_INFO'
    with pytest.raises(ValueError,match='INDEPENDENT_COMPLETE_LEDGER_UNPROVEN|INDEPENDENT_LEDGER_SOURCE_REVALIDATION_UNPROVEN|UNPARSED_HAND_SOURCE|UNCLASSIFIED_HAND_SERVICE'):
        draft.independent_ledger(hand)
def test_real_source_seal_change_after_load_helper(tmp_path):
    path=tmp_path/'native_source.txt';path.write_bytes(b'original retained source')
    inv=draft.FreshConstantSources.__new__(draft.FreshConstantSources);inv.source=tmp_path
    inv.native_who=None
    inv.hashes={'native_source.txt':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert inv.sources_unchanged();path.write_bytes(b'changed retained source');assert not inv.sources_unchanged()
def test_direct_receipt_pixel_mutation_rejected(native):
    case,*_=native;bad=dict(case);bad['raw_direct_stack_readings']=[dict(next(r for r in case['raw_direct_stack_readings'] if r['source_frame']==1965))]
    bad['raw_direct_stack_readings'][0]['pixel_sha256']='0'*64
    with pytest.raises(ValueError,match='DIRECT_STACK_NATIVE_PIXELS_CONFLICT'):draft.direct_readings(bad)
def test_direct_receipt_raw_response_mismatch_rejected(native):
    case,*_=native;bad=dict(case);bad['raw_direct_stack_readings']=[dict(next(r for r in case['raw_direct_stack_readings'] if r['source_frame']==1965))]
    bad['raw_direct_stack_readings'][0]['raw_text']='999'
    with pytest.raises(ValueError,match='DIRECT_STACK_RESPONSE_CONFLICT'):draft.direct_readings(bad)
@pytest.fixture(scope='module')
def verified_numbers(native):
    case,metadata,*_=native
    readings,bindings,observations,states=draft.numeric_sources(case,metadata)
    direct=draft.direct_readings(case);frames=case['numeric_frames']
    selected=[b for b in bindings if b['frame'] in frames]
    return case,readings,selected,observations,states,direct,case['constant_interval'][0]+2,frames,case['constant_interval'][1]
def agreement(data,readings=None,observations=None,states=None,direct=None):
    case,rr,bindings,oo,ss,dd,overlay_end,frames,high=data
    return draft.verify_numeric_agreement(case,30294,rr if readings is None else readings,bindings,oo if observations is None else observations,ss if states is None else states,dd if direct is None else direct,overlay_end,frames,high)
def test_source_verified_final_agreement_control(verified_numbers):
    ids,direct=agreement(verified_numbers)
    assert ids==[1144,1561] and len(direct)==3
@pytest.mark.parametrize('frames,low,high',[([1965,1966,1967],1918,1966),([1965,1967,1968],1918,3232),([1919,1920,1921],1918,3232)])
def test_numeric_triple_bound_or_nonconsecutive_rejection(frames,low,high):
    with pytest.raises(ValueError,match='NUMERIC_TRIPLE_CONTEXT_CONFLICT'):draft.verify_numeric_triple_frames(frames,low,high)
def test_old_state_current_recurrence_numeric_conflict(verified_numbers):
    case,readings,_,obs,states,_,_,frames,_=verified_numbers
    rr=dict(readings);f=frames[-1];rr[f]=dict(rr[f],raw_text='999999')
    oo={k:dict(v) for k,v in obs.items()};ss={k:dict(v) for k,v in states.items()}
    sid=rr[f]['numeric_source_evidence']['source_state_id']
    ss[sid]['frame_index']=case['constant_interval'][0]-100
    for o in oo.values():
        if o['state_id']==sid:o['frame_index']=ss[sid]['frame_index']
    # Synthetic late-predicate input. It tests the exact predicate used by prove;
    # it does not claim that altered OCR has a native acceptance receipt.
    with pytest.raises(ValueError,match='INDEPENDENT_NATIVE_NUMERIC_CONFLICT'):agreement(verified_numbers,readings=rr,observations=oo,states=ss)
def test_other_accepted_sql_numeric_conflict(verified_numbers):
    states={k:dict(v) for k,v in verified_numbers[4].items()}
    states['S01561'].update(ocr_raw='999999',ocr_numeric='999999')
    with pytest.raises(ValueError,match='INDEPENDENT_ACCEPTED_NUMERIC_CONFLICT'):agreement(verified_numbers,states=states)
def test_independent_direct_accepted_conflict(verified_numbers):
    direct=[dict(r) for r in verified_numbers[5]];direct[0]['value']=999999
    with pytest.raises(ValueError,match='INDEPENDENT_DIRECT_NUMERIC_CONFLICT'):agreement(verified_numbers,direct=direct)
def synthetic_source_bound_scaled_row(case):
    row=copy.deepcopy(next(r for r in case['raw_direct_stack_readings'] if r['source_frame']==1965))
    row.update(raw_text='999999',confidence=.5,ocr_status='OK')
    row['ocr_response'].update(raw_text='999999',score=.5,status='OK')
    gray,enlarged=scale_pixels(row['png_base64'],row['pixel_sha256']);model,slots=proof_config()
    item=f"{row['table_id']}:{row['source_frame']}:{list(slots['slot_05']['seats']).index(row['seat'])}"
    row['ocr_response']['item_id']=item
    row['ocr_response']['same_source_scale_evidence']=dict(rule=RULE,status='OBSERVED',model_name=model['model_name'],weights_sha256=model['weights_sha256'],primary_response=dict(status='OK',raw_text='999999',score=.5),source_item_id=item,source_pixel_sha256=row['pixel_sha256'],source_shape=list(gray.shape),scaled_shape=list(enlarged.shape),scaled_pixel_sha256=hashlib.sha256(enlarged.tobytes()).hexdigest(),response=dict(item_id=item,status='OK',raw_text='999999',score=.999))
    return row
def test_exact_source_scaled_conflict_reaches_agreement(native,verified_numbers):
    case,*_=native;row=synthetic_source_bound_scaled_row(case)
    bad=dict(case,raw_direct_stack_readings=[row]);accepted=draft.direct_readings(bad)
    assert accepted[0]['value']==999999
    with pytest.raises(ValueError,match='INDEPENDENT_DIRECT_NUMERIC_CONFLICT'):agreement(verified_numbers,direct=accepted)
def test_rejected_direct_without_rgb_retained_but_no_numeric_authority():
    blob=json.loads((OUT/'constant_balance_four_t06_2878365876_bbobyy_native_case_20261008_161502.json').read_text('utf-8'));case=blob['case']
    row=next(r for r in case['raw_direct_stack_readings'] if r['source_frame']==1770)
    bad=dict(case,raw_direct_stack_readings=[row]);before=copy.deepcopy(bad['raw_direct_stack_readings'])
    assert not any(c['raw_record']['source_frame']==1770 for c in case['raw_stack_captures'])
    assert draft.direct_readings(bad)==[]
    assert bad['raw_direct_stack_readings']==before
def test_primary_accepted_direct_without_rgb_blocked(native):
    case,*_=native;row=next(r for r in case['raw_direct_stack_readings'] if r['source_frame']==1965)
    bad=dict(case,raw_direct_stack_readings=[row],raw_stack_captures=[c for c in case['raw_stack_captures'] if c['raw_record']['source_frame']!=1965])
    with pytest.raises(ValueError,match='DIRECT_STACK_NATIVE_CAPTURE_MISSING'):draft.direct_readings(bad)
def test_scaled_accepted_direct_without_rgb_blocked(native):
    case,*_=native;row=synthetic_source_bound_scaled_row(case)
    bad=dict(case,raw_direct_stack_readings=[row],raw_stack_captures=[c for c in case['raw_stack_captures'] if c['raw_record']['source_frame']!=1965])
    with pytest.raises(ValueError,match='DIRECT_STACK_NATIVE_CAPTURE_MISSING'):draft.direct_readings(bad)
def test_selected_old_state_birth_not_postforced(verified_numbers):
    case,rr,bb,oo,ss,dd,overlay_end,frames,high=verified_numbers
    observations={k:dict(v) for k,v in oo.items()};states={k:dict(v) for k,v in ss.items()}
    sid=bb[0]['state_id'];states[sid]['frame_index']=case['constant_interval'][0]-100
    for o in observations.values():
        if o['state_id']==sid:o['frame_index']=states[sid]['frame_index']
    with pytest.raises(ValueError,match='SELECTED_NUMERIC_NATIVE_BIRTH_NOT_POST_FORCED'):agreement(verified_numbers,observations=observations,states=states)
@pytest.fixture(scope='module')
def empty_native(native):
    _,_,_,registry,_=native
    record=next(r for r in registry.records if r['observed_hand']['table_id']=='T06' and r['observed_hand']['hand_id']=='2878365610')
    h=record['observed_hand'];context=(h['table_id'],h['session_id'],h['hand_id'])
    event=next(e for e in registry.summary['state_receipts'] if e.get('table')==6 and e.get('session')==h['session_id'] and e.get('hand_id')==h['hand_id'] and e.get('kind')=='EMPTY')
    return registry,record,context,event,'seat4'
def test_actual_positive_other_empty(empty_native):
    registry,_,context,event,target=empty_native
    assert draft.positive_other_empty(event,registry,context,target)
@pytest.mark.parametrize('field,value',[('source_ref','0'*64),('hand_id','foreign'),('seat','seat4'),('frame',99999),('evidence_status','UNKNOWN'),('kind','REPLACE')])
def test_other_empty_source_context_target_rejections(empty_native,field,value):
    registry,_,context,event,target=empty_native;bad=dict(event);bad[field]=value
    assert not draft.positive_other_empty(bad,registry,context,target)
def test_other_empty_ambiguous_deal_rejected(empty_native):
    registry,_,context,event,target=empty_native;reg=copy.copy(registry)
    deal=next(d for d in registry.deals if d['hand_id']==context[2] and d['table_id']==6)
    reg.deals=registry.deals+[deal]
    assert not draft.positive_other_empty(event,reg,context,target)
def test_empty_then_occupied_barrier(empty_native):
    registry,record,_,event,target=empty_native;reg=copy.copy(registry);reg.summary=dict(registry.summary)
    changed=dict(event,kind='OCCUPIED',frame=event['frame']+1,epoch=999,source_ref='synthetic-test-only')
    reg.summary['state_receipts']=registry.summary['state_receipts']+[changed]
    with pytest.raises(ValueError,match='SEAT_CONTINUITY_BARRIER'):draft.complete_hand_interval(record,reg,'mmm156',target)
@pytest.mark.parametrize('number,actor,filename',[('2878365610','mmm156','constant_balance_four_t06_2878365610_mmm156_native_case_20261008_161502.json'),('2878365876','bbobyy','constant_balance_four_t06_2878365876_bbobyy_native_case_20261008_161502.json')])
def test_native_t06_full_proof_after_empty_partition(native,number,actor,filename):
    _,_,_,registry,inventory=native;blob=json.loads((OUT/filename).read_text('utf-8'));case=blob['case']
    record=next(r for r in registry.records if r['observed_hand']['table_id']=='T06' and r['observed_hand']['hand_id']==number)
    assert inventory.matches(case,blob['database_metadata_raw'],record,registry)
    result=registry._tested_constant_provider.proofs[(draft.key(record['observed_hand']),actor)]
    path=OUT/f'constant_balance_runtime_test_t06_{number}_{actor}_result_{__import__("time").strftime("%Y%m%d_%H%M%S",__import__("time").gmtime())}.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    assert result['status']=='PROVEN',dict(status=result['status'],reason=result['reason'])
    assert result['ledger_seed_start_stacks']=={} and result['first_visible_authority'] is False
    assert result['scope']=='SOURCE_REPLAYED_CONSTANT_BALANCE_NO_OCCUPANCY_EPOCH'
def test_runtime_provider_constructor_bounded_native_candidates(native):
    _,_,_,registry,_=native;provider=registry._tested_constant_provider
    assert len(provider.audit)==3 and all(r['status']=='PROVEN' for r in provider.audit),provider.audit
def test_runtime_provider_current_hand_reproduces_proof(native):
    case,_,record,registry,_=native;provider=registry._tested_constant_provider
    proofs=provider.for_hand(record['observed_hand'])
    assert len(proofs)==1 and proofs[0]['actor']==case['actor'] and proofs[0]['calculated_start_stack']==30364
@pytest.mark.parametrize('mutation',['foreign_hand','stale_source_hand'])
def test_provider_foreign_or_stale_hand_has_no_authority(native,mutation):
    _,_,record,registry,_=native;hand=copy.deepcopy(record['observed_hand'])
    if mutation=='foreign_hand':hand['hand_id']='foreign-test-context'
    else:hand['messages'][-1]['source_rows'][-1]['raw_ocr_text']='stale-current-source-mutation'
    assert registry._tested_constant_provider.for_hand(hand)==[]
def test_provider_forged_cached_proof_is_rejected_by_actual_source_replay(native):
    case,_,record,registry,_=native;provider=copy.copy(registry._tested_constant_provider)
    provider.proofs=dict(provider.proofs);context=draft.key(record['observed_hand']);k=(context,case['actor'])
    provider.proofs[k]=dict(provider.proofs[k],calculated_start_stack=999999)
    assert provider.for_hand(record['observed_hand'])==[]
@pytest.fixture(scope='module')
def stack_sql(native):
    from parser_core.stack_evidence import StackEvidenceProvider
    _,metadata,*_=native;meta={r['key']:json.loads(r['value']) for r in metadata}
    return StackEvidenceProvider(SRC/'stack_pipeline/timeline.sqlite',video_sha256=meta['video_sha256'])
@pytest.mark.parametrize('mutation',['forged_value','foreign_context','stale_source_id'])
def test_constant_receipt_cannot_grant_solver_audit_hh_authority(native,stack_sql,tmp_path,mutation):
    from parser_core.poker_reconstruction_layer import reconstruct_production_hands,reconstruction_config
    from parser_core.hh_export import export_hand_histories
    case,_,record,registry,_=native;context=draft.key(record['observed_hand']);key=(context,case['actor'])
    provider=copy.copy(registry._tested_constant_provider)
    # Actual runtime for_hand observes no matching candidate, while arbitrary
    # serialized proofs/receipts remain present. No validator is mocked.
    provider.cases={}
    receipt=draft.join_receipt(provider.proofs[key])
    if mutation=='forged_value':receipt['calculated_start_stack']=999999
    elif mutation=='foreign_context':receipt['hand_id']='foreign-test-context'
    else:receipt['source_observation_ids']=[99999999];receipt['source_state_ids']=['STALE_TEST_STATE']
    sql=copy.copy(stack_sql);sql.constant_sources=provider
    config=dict(reconstruction_config(),enabled=True,first_visible_stack_contract=True,strict_source_uniqueness=True)
    results,_=reconstruct_production_hands([record],config=config,stack_provider=sql)
    result=results[0]
    assert case['actor'] not in result['joined_start_stacks']
    assert not result['constant_stack_balance_join_receipts']
    # A forged serialized reconstruction receipt also cannot grant audit use.
    result['constant_stack_balance_join_receipts']=[receipt]
    sql.write_audit(tmp_path,[record],start_stack=True,reconstruction_results=[result])
    stack_rows=[json.loads(line) for line in (tmp_path/'stack_start_stack.jsonl').read_text('utf-8').splitlines()]
    target=next(r for r in stack_rows if r['player']==case['actor'])
    assert target['status']=='START_STACK_UNKNOWN' and target['calculated_start_stack'] is None
    # Supply the alleged stack explicitly to test HH's independent replay gate.
    alleged=dict(target,status='START_STACK_PROVEN',seat_id=case['seat'],calculated_start_stack=receipt['calculated_start_stack'],source_observation_id=receipt['source_observation_ids'][0],source_state_id=receipt['source_state_ids'][0],occupancy_epoch=None,constant_binding_receipt=receipt)
    stack_rows=[alleged if r['player']==case['actor'] else r for r in stack_rows]
    (tmp_path/'stack_start_stack.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in stack_rows),'utf-8')
    (tmp_path/'poker_reconstruction.jsonl').write_text(json.dumps(result)+'\n','utf-8')
    export_hand_histories(tmp_path,[record],constant_provider=provider)
    hh=json.loads((tmp_path/'hand_histories.jsonl').read_text('utf-8').splitlines()[0])
    actor=next(r for r in hh['players'] if r['player']==case['actor'])
    assert actor['START_STACK'] is None and not hh['certified']

@pytest.fixture
def seal_sources(tmp_path):
    """Small real files exercise seals, without claiming native proof authority."""
    from types import SimpleNamespace
    source=tmp_path/'source';source.mkdir()
    data=source/'shared.jsonl';data.write_bytes(b'{"source":"retained"}\n')
    config=tmp_path/'config.json';config.write_bytes(b'{"enabled":true}')
    expected=hashlib.sha256(data.read_bytes()).hexdigest()
    registry=SimpleNamespace(source=source,hashes={'shared.jsonl':expected},
        config_path=config,config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    inventory=SimpleNamespace(source=source,hashes={'shared.jsonl':expected})
    return registry,inventory,data,config

def test_batch_real_file_changed_before_begin(seal_sources):
    registry,inventory,data,_=seal_sources
    batch=draft.SourceSealBatch(registry,inventory,())
    data.write_bytes(b'changed retained bytes')
    assert not batch.begin()
    assert batch.check_phase('before')=='BATCH_SOURCE_SEAL_NOT_OPEN'

def test_batch_real_file_deleted_before_begin(seal_sources):
    registry,inventory,data,_=seal_sources
    batch=draft.SourceSealBatch(registry,inventory,())
    data.unlink()
    assert not batch.begin()

@pytest.mark.parametrize('which',['source','config'])
def test_batch_real_source_or_config_changed_after_begin(seal_sources,which):
    registry,inventory,data,config=seal_sources
    batch=draft.SourceSealBatch(registry,inventory,())
    assert batch.begin()
    (data if which=='source' else config).write_bytes(b'mid-batch changed bytes')
    assert not batch.finish()
    assert batch.check_phase('after')=='BATCH_SOURCE_SEAL_NOT_OPEN'

def test_batch_real_file_deleted_after_begin(seal_sources):
    registry,inventory,data,_=seal_sources
    batch=draft.SourceSealBatch(registry,inventory,())
    assert batch.begin();data.unlink()
    assert not batch.finish()

@pytest.mark.parametrize('which',['inventory','runtime'])
def test_batch_expected_overlap_conflict(seal_sources,which):
    registry,inventory,data,_=seal_sources
    runtime=()
    if which=='inventory':inventory.hashes={'shared.jsonl':'0'*64}
    else:runtime=((str(data),'0'*64),)
    with pytest.raises(ValueError,match='BATCH_EXPECTED_SOURCE_HASH_CONFLICT'):
        draft.SourceSealBatch(registry,inventory,runtime)

def test_batch_closed_and_reentrant_contract(seal_sources):
    registry,inventory,_,_=seal_sources
    batch=draft.SourceSealBatch(registry,inventory,())
    assert batch.check_phase('before')=='BATCH_SOURCE_SEAL_NOT_OPEN'
    assert not batch.finish()
    assert batch.begin()
    with pytest.raises(ValueError,match='BATCH_ALREADY_OPEN'):batch.begin()
    assert batch.finish()
    assert not batch.finish()
    assert batch.check_phase('after')=='BATCH_SOURCE_SEAL_NOT_OPEN'

@pytest.mark.parametrize('which',['seat','frame'])
@pytest.mark.parametrize('accepted',[True,False])
def test_direct_callback_item_foreign_seat_or_frame(native,which,accepted):
    case,*_=native
    row=copy.deepcopy(next(r for r in case['raw_direct_stack_readings'] if r['source_frame']==1965))
    if not accepted:
        row.update(confidence=.5);row['ocr_response']['score']=.5
    table,frame,seat=row['ocr_response']['item_id'].split(':')
    row['ocr_response']['item_id']=f'{table}:{int(frame)+1 if which=="frame" else frame}:{int(seat)+1 if which=="seat" else seat}'
    with pytest.raises(ValueError,match='DIRECT_OCR_CALLBACK_ITEM_CONFLICT'):
        draft.direct_readings(dict(case,raw_direct_stack_readings=[row]))

@pytest.mark.parametrize('actor',['bvg72','mmm156','bbobyy'])
def test_three_native_batch_proof_json_equals_standalone(native,actor):
    _,_,_,registry,_=native;provider=registry._tested_constant_provider
    saved_key=next(k for k in provider.cases if k[1]==actor)
    case,record=provider.cases[saved_key]
    standalone=provider._prove(case,record)
    # Full JSON equality includes reasons, native row/binding order and ledger.
    assert standalone==provider.proofs[saved_key]
    assert json.dumps(standalone,sort_keys=True)==json.dumps(provider.proofs[saved_key],sort_keys=True)

def test_provider_preseal_real_runtime_file_failure_has_no_authority(native,tmp_path):
    _,_,record,registry,_=native;provider=copy.copy(registry._tested_constant_provider)
    dependency=tmp_path/'runtime_dependency.dat';dependency.write_bytes(b'original')
    provider.runtime_files=provider.runtime_files+((str(dependency),hashlib.sha256(dependency.read_bytes()).hexdigest()),)
    dependency.write_bytes(b'changed')
    assert provider.for_hand(record['observed_hand'])==[]

def test_provider_missing_runtime_file_has_no_authority(native,tmp_path):
    _,_,record,registry,_=native;provider=copy.copy(registry._tested_constant_provider)
    dependency=tmp_path/'runtime_dependency.dat';dependency.write_bytes(b'original')
    provider.runtime_files=provider.runtime_files+((str(dependency),hashlib.sha256(dependency.read_bytes()).hexdigest()),)
    dependency.unlink()
    assert provider.for_hand(record['observed_hand'])==[]

def test_provider_postseal_failure_returns_no_proof(native,monkeypatch):
    _,_,record,registry,_=native;provider=registry._tested_constant_provider
    original=draft.SourceSealBatch.finish;seen=[]
    def failed_finish(batch):
        seen.append(original(batch));return False
    monkeypatch.setattr(draft.SourceSealBatch,'finish',failed_finish)
    assert provider.for_hand(record['observed_hand'])==[]
    assert seen==[True]  # Inject only failure after the actual content check.

@pytest.mark.parametrize('which',['case','record'])
def test_provider_altered_case_or_record_cannot_pass_batch(native,which):
    case,_,record,registry,_=native;provider=copy.copy(registry._tested_constant_provider)
    provider.cases=dict(provider.cases);saved_key=(draft.key(record['observed_hand']),case['actor'])
    stored_case,stored_record=provider.cases[saved_key]
    if which=='case':
        changed=dict(stored_case,raw_stack_captures=stored_case['raw_stack_captures'][:-1])
        provider.cases[saved_key]=(changed,stored_record)
    else:
        changed=copy.deepcopy(stored_record);changed['observed_hand']['messages']=changed['observed_hand']['messages'][:-1]
        provider.cases[saved_key]=(stored_case,changed)
    assert provider.for_hand(record['observed_hand'])==[]

def test_constructor_postseal_failure_never_publishes_proven(native,monkeypatch):
    from types import SimpleNamespace
    _,_,_,registry,_=native
    provider=draft.ConstantStackBalanceProvider.__new__(draft.ConstantStackBalanceProvider)
    original=draft.SourceSealBatch.finish;seen=[];candidate_statuses=[]
    original_prove=draft.ConstantStackBalanceProvider._prove
    def observed_prove(instance,*args,**kwargs):
        result=original_prove(instance,*args,**kwargs)
        candidate_statuses.append(result['status']);return result
    def failed_finish(batch):
        assert provider.cases=={} and provider.proofs=={}
        assert not any(row['status']=='PROVEN' for row in provider.audit)
        seen.append(original(batch));return False
    monkeypatch.setattr(draft.ConstantStackBalanceProvider,'_prove',observed_prove)
    monkeypatch.setattr(draft.SourceSealBatch,'finish',failed_finish)
    with pytest.raises(ValueError,match='CONSTANT_BATCH_POST_SOURCE_SEAL_CHANGED'):
        provider.__init__(SRC,registry.records,SimpleNamespace(rows=registry._tested_joint_rows))
    assert seen==[True]
    assert candidate_statuses==['PROVEN','PROVEN','PROVEN']
    assert provider.cases=={} and provider.proofs=={}
    assert not any(row['status']=='PROVEN' for row in provider.audit)
