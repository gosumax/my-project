"""Generic raster parity and native UNKNOWN guards; no annotated success."""
import base64,copy,hashlib,json,sqlite3,sys,time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/hh95_evidence_20261008'
SOURCE=Path('C:/Users/Admin/.codex/tmp/hh95_source_cache_20261008/full_v1')
PROBE=OUT/'sitout_general_matcher_probe_bank3_20261008_191720_result.json'
from parser_core.sitout_text_pattern import SitOutTextPattern,MODEL,SIT_OUT_MATCH_THRESHOLD
from parser_core import sitout_hand_evidence as sitout

@pytest.fixture(scope='module')
def classified():
    expected=json.loads(PROBE.read_text('utf8'));model=SitOutTextPattern();actual={}
    with sqlite3.connect((SOURCE/'stack_pipeline/timeline.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        for row in db.execute('select * from visual_states order by state_id'):
            assert hashlib.sha256(row['raw_gray']).hexdigest()==row['exact_hash']
            gray=np.frombuffer(row['raw_gray'],np.uint8).reshape(row['height'],row['width'])
            actual[row['state_id']]=model.evaluate(gray)
    return expected,actual

def test_all_5543_runtime_classifier_scores_decisions_and_source_hashes_equal_fixed_probe(classified):
    expected,actual=classified;assert len(actual)==expected['state_count']==5543
    for row in expected['all_states']:
        result=actual[row['state_id']]
        assert result['score']==row['score']
        assert result['matched']==(row['score']>=SIT_OUT_MATCH_THRESHOLD)
        assert result['threshold']==.86
        assert result['box']==row['template_boxes'][row['winning_template']]
        assert [r['score'] for r in result['template_scores']]==row['template_scores']
        assert result['numeric_authority'] is False

def test_calibrated_matcher_recovers_all_11_prelabelled_sources(classified):
    expected,actual=classified;rows=expected['positive_results'];assert len(rows)==11
    assert sum(actual[r['state_id']]['matched'] for r in rows)==11
    assert actual['S02998']['matched']
    assert {r['state_id'] for r in rows}.isdisjoint({'S03249','S03837'})

def test_all_954_accepted_numeric_and_native_fold_are_rejected(classified):
    expected,actual=classified
    rows=[r for r in expected['all_states'] if r['category']=='accepted_numeric_negative'];assert len(rows)==954
    assert all(not actual[r['state_id']]['matched'] for r in rows)
    assert not actual['S02935']['matched']
    assert actual['S02935']['score']==pytest.approx(.21854105591773987)

def test_new_matches_among_original_unknowns_require_independent_pixel_labels(classified):
    expected,actual=classified;rows=[r for r in expected['all_states'] if r['category']=='unknown_unreviewed']
    assert len(rows)==4575
    labels=json.loads((ROOT/'tests/fixtures/threshold_calibration/sitout_unknown_labels.json').read_text('utf8'))
    reviewed={r['raw_gray_sha256'] for r in labels['rows'] if r['independent_visual_label']=='SIT_OUT'}
    matched={r['raw_gray_sha256'] for r in rows if actual[r['state_id']]['matched']}
    assert len(reviewed)==12 and matched==reviewed

@pytest.mark.parametrize('mutation',['threshold','method','generic','source_pixel','template_pixel','bbox','shape','empty','float','range','constant'])
def test_damaged_model_contract_or_native_crop_cannot_load(tmp_path,mutation):
    data=json.loads(MODEL.read_text('utf8'));row=data['templates'][0]
    if mutation=='threshold':data['threshold']=.9
    elif mutation=='method':data['method']='TM_CCORR_NORMED'
    elif mutation=='generic':data['generic_pixel_only']=False
    elif mutation=='source_pixel':row['raw_gray'][0][0]^=1
    elif mutation=='template_pixel':row['template_gray'][0][0]^=1
    elif mutation=='bbox':row['template_xyxy'][0]+=1
    elif mutation=='shape':row['template_shape'][0]+=1
    elif mutation=='empty':data['templates']=[]
    elif mutation=='float':row['template_gray'][0][0]=.5
    elif mutation=='range':row['template_gray'][0][0]=256
    else:
        row['template_gray']=[[0]*row['template_shape'][1] for _ in range(row['template_shape'][0])]
        row['template_gray_sha256']=hashlib.sha256(np.asarray(row['template_gray'],np.uint8).tobytes()).hexdigest()
    path=tmp_path/'model.json';path.write_text(json.dumps(data),'utf8')
    with pytest.raises(ValueError):SitOutTextPattern(path)

def test_model_provenance_fields_do_not_route_classifier(tmp_path):
    data=json.loads(MODEL.read_text('utf8'))
    for row in data['templates']:row.update(table_id=999,seat_id='foreign',frame=-1,state_id='foreign')
    path=tmp_path/'model.json';path.write_text(json.dumps(data),'utf8');altered=SitOutTextPattern(path);original=SitOutTextPattern()
    for _,gray in original.templates:
        a=altered.evaluate(gray);b=original.evaluate(gray)
        assert (a['matched'],a['score'],a['box'],a['template_sha256'])==(b['matched'],b['score'],b['box'],b['template_sha256'])

@pytest.mark.parametrize('gray',[np.zeros((1,1),np.uint8),np.zeros((13,81),np.uint8)])
def test_small_or_blank_native_gray_cannot_match(gray):
    result=SitOutTextPattern().evaluate(gray);assert not result['matched'] and not result['numeric_authority']

def test_non_native_float_pixels_rejected():
    with pytest.raises(ValueError,match='SIT_OUT_NATIVE_GRAY_REQUIRED'):SitOutTextPattern().evaluate(np.zeros((13,81),np.float64))

@pytest.fixture(scope='module')
def native():
    records=[json.loads(line) for line in (OUT/'carry_service_full_replay_v7/logical_hand_analysis.jsonl').open(encoding='utf8') if line.strip()]
    stamp=time.strftime('%Y%m%d_%H%M%S',time.gmtime());prefix=OUT/f'sitout_general_native_provider_{stamp}'
    start=time.perf_counter();Path(str(prefix)+'_started.json').write_text(json.dumps(dict(record_count=len(records),scope='BOUNDED_NATIVE_STATUS_DIAGNOSTIC_NO_HH_ACCEPTANCE')),'utf8')
    provider=sitout.SitOutSourceProvider(SOURCE,records)
    counts={};reasons={}
    for row in provider.audit:
        counts[row['status']]=counts.get(row['status'],0)+1
        if row['reason'] is not None:reasons[row['reason']]=reasons.get(row['reason'],0)+1
    receipt=dict(record_count=len(records),candidate_actor_count=len(provider.audit),selected_hand_count=len(provider.inventory.records) if provider.inventory else 0,counts=counts,reasons=reasons,wall_seconds=time.perf_counter()-start,proofs=list(provider.proofs.values()),native_positive_for_hand='PENDING_TEST' if provider.proofs else 'NOT_MEASURED_NO_NATIVE_POSITIVE',HH_acceptance='NOT_MEASURED')
    Path(str(prefix)+'_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),'utf8')
    record=next(r for r in records if r['observed_hand']['table_id']=='T01' and r['observed_hand']['hand_id']=='2878366598')
    case=sitout.case_for(provider.inventory,sitout.native_key(record['observed_hand']),'gosteeer','seat7')
    point=sitout.prove_actor_point(case['source_context'],case['actor']);assert point['status']=='PROVEN',point
    return provider,record,case,point

def test_all_current_v7_native_provider_reports_actual_proofs_or_unknown_without_exemption(native):
    provider,record,case,_=native
    row=next(r for r in provider.audit if r['actor']==case['actor'] and tuple(r['hand_key'])==sitout.native_key(record['observed_hand']))
    assert row['status']=='UNKNOWN',row
    assert not provider.excludes_stack_work(sitout.native_key(record['observed_hand']),case['actor'])
    for proof in provider.proofs.values():
        assert proof['numeric_stack_authority'] is False and proof['synthetic_fold_authority'] is False
        assert proof['ledger_seed_start_stacks']=={}

def test_actual_native_positive_for_hand_replays_without_success_mock(native):
    provider,_,_,_=native
    if not provider.proofs:pytest.skip('NOT_MEASURED_NO_NATIVE_POSITIVE')
    (context,actor),proof=next(iter(provider.proofs.items()));case,record=provider.cases[(context,actor)]
    assert proof in provider.for_hand(record['observed_hand'])

def test_forged_cached_native_target_proof_cannot_bypass_replay(native):
    provider,record,case,_=native;key=(sitout.native_key(record['observed_hand']),case['actor'])
    prior_case=provider.cases.get(key);prior_proof=provider.proofs.get(key)
    try:
        provider.cases[key]=(case,record)
        provider.proofs[key]=dict(status='PROVEN',actor=case['actor'],player_status='SIT_OUT',numeric_stack_authority=False)
        assert provider.for_hand(record['observed_hand'])==[]
    finally:
        if prior_case is None:provider.cases.pop(key,None)
        else:provider.cases[key]=prior_case
        if prior_proof is None:provider.proofs.pop(key,None)
        else:provider.proofs[key]=prior_proof

@pytest.mark.parametrize('field,value',[('hand_id','foreign'),('seat','seat1'),('table_id',9),('session_id','foreign')])
def test_native_case_context_cannot_transfer_inventory(native,field,value):
    provider,record,case,_=native;bad=copy.deepcopy(case);bad[field]=value
    assert not provider.inventory.matches(bad,provider.inventory.metadata,record,provider.registry)

@pytest.mark.parametrize('mutation',['hand','actor_action','lifecycle'])
def test_changed_current_native_record_cannot_reuse_inventory(native,mutation):
    provider,record,case,_=native;bad=copy.deepcopy(record)
    if mutation=='hand':bad['observed_hand']['hand_id']='foreign'
    elif mutation=='lifecycle':bad['lifecycle']['status']='PARTIAL'
    else:
        row=next(m for m in bad['observed_hand']['messages'] if m.get('player')==case['actor'] and m['action']=='FOLD')
        row['action']='CALL'
    assert not provider.inventory.matches(case,provider.inventory.metadata,bad,provider.registry)

@pytest.mark.parametrize('field',['raw_stack_captures','raw_direct_stack_readings','raw_sql_observations','source_context','full_hand_source_context'])
def test_native_inventory_subset_or_alteration_rejected(native,field):
    provider,record,case,_=native;bad=copy.deepcopy(case)
    if isinstance(bad[field],list):
        if bad[field]:bad[field]=bad[field][1:]
        else:bad[field]=[{'foreign':True}]
    else:bad[field]['header_frame']+=1
    assert not provider.inventory.matches(bad,provider.inventory.metadata,record,provider.registry)

@pytest.mark.parametrize('mutation',['gap','hash','frame','seat','bytes','clock'])
def test_actual_native_capture_gap_pixels_context_and_clock_fail(native,mutation):
    _,_,case,point=native;bad=copy.deepcopy(case);frame=point['nickname_sources'][0]['source_frame']
    wrapped=next(w for w in bad['raw_stack_captures'] if w['raw_record']['source_frame']==frame);cap=wrapped['raw_record']
    if mutation=='gap':bad['raw_stack_captures'].remove(wrapped)
    elif mutation=='hash':cap['pixel_sha256']='0'*64
    elif mutation=='frame':cap['source_frame']+=100000
    elif mutation=='seat':cap['seat']='seat1'
    elif mutation=='clock':cap['source_pts']+=1
    else:
        pixels=cv2.imdecode(np.frombuffer(base64.b64decode(cap['png_base64']),np.uint8),1);pixels[0,0,0]^=1
        ok,png=cv2.imencode('.png',pixels);assert ok;cap['png_base64']=base64.b64encode(png).decode()
    with pytest.raises((ValueError,KeyError)):sitout.source_image(bad,frame)

def test_native_missing_header_capture_blocks_initial_phase(native):
    provider,_,case,point=native;bad=copy.deepcopy(case);low=case['source_context']['header_frame']
    bad['raw_stack_captures']=[w for w in bad['raw_stack_captures'] if w['raw_record']['source_frame']!=low]
    with pytest.raises((ValueError,KeyError)):sitout.initial_status(bad,point,provider.inventory)

def test_real_native_late_sitout_cannot_bypass_original_source_window(native):
    _,_,case,_=native
    frame=5395
    assert frame>case['source_context']['capture_end_inclusive']
    saved_frames={w['raw_record']['source_frame'] for w in case['raw_stack_captures']}
    if frame not in saved_frames:
        # source_image indexes the actual retained captures before checking
        # context. Exact absence must remain absence; no SQL gray is a RGB cap.
        with pytest.raises(KeyError) as error:sitout.source_image(case,frame)
        assert error.value.args==(frame,)
    else:
        with pytest.raises(ValueError,match='^SIT_OUT_INITIAL_CAPTURE_CONTEXT_CONFLICT$'):
            sitout.source_image(case,frame)

def test_actual_native_numeric_sources_are_contradictions_only(native):
    provider,_,_,_=native
    case=json.loads((OUT/'constant_balance_four_t05_2878365932_bvg72_native_case_20261008_161502.json').read_text('utf8'))['case']
    conflicts=sitout.status_numeric_conflicts(case,provider.inventory,case['source_context']['capture_end_inclusive'])
    assert conflicts and any(r['kind']=='SQL' for r in conflicts)
    assert all(not any(k in r for k in ('amount','value','displayed_stack','calculated_start_stack')) for r in conflicts)
    point={'nickname_sources':[{'source_frame':case['source_context']['capture_end_inclusive']-2+i} for i in range(3)]}
    with pytest.raises(ValueError,match='SIT_OUT_NUMERIC_SOURCE_CONFLICT'):sitout.initial_status(case,point,provider.inventory)

def test_missing_model_is_not_replaced_by_actor_annotation(tmp_path):
    with pytest.raises(OSError):SitOutTextPattern(tmp_path/'absent.json')

@pytest.mark.parametrize('phase',['before','after'])
def test_real_temporary_source_change_or_deletion_closes_source_batch(tmp_path,phase):
    path=tmp_path/'source.bin';path.write_bytes(b'original-native-source')
    config=tmp_path/'config.json';config.write_bytes(b'{}')
    registry=SimpleNamespace(source=tmp_path,hashes={'source.bin':hashlib.sha256(path.read_bytes()).hexdigest()},config_path=config,config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
    inventory=SimpleNamespace(source=tmp_path,hashes={})
    batch=sitout.SourceSealBatch(registry,inventory,{})
    if phase=='before':path.unlink();assert not batch.begin()
    else:assert batch.begin();path.write_bytes(b'changed-native-source');assert not batch.finish()
