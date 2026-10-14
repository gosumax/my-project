"""Fresh saved-pixel authority must revoke stale projections and survive RAM/JSON."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from parser_core.normalize_chat import normalize
from parser_core.visual_cards import associate_visual_evidence
from parser_core.visual_tile_queue import VisualTileQueue
from parser_core.win_tail_evidence import prove_win_tail
from scripts.replay_saved_quality import (ordinary_replay_rows,
    refresh_saved_card_readings, _VISUAL_FIELDS)

ROOT=Path(__file__).resolve().parents[1]
NATIVE=json.loads((ROOT/'tests/fixtures/saved_pixel_native_sources.json').read_text('utf-8'))


@pytest.fixture
def sealed_validation_bundle(tmp_path):
    from scripts.replay_saved_quality import _validation_sha
    roles=('logical','traces','stacks','seats','who','associations','who_updates',
        'sitout_targets','sitout_histories','no_sb_seats','visual','review','reference_hh',
        'reference_text','original_seals')
    files={role:tmp_path/(role+'.json') for role in roles}
    for role,path in files.items():
        if role!='original_seals':path.write_text('{}\n','utf-8')
    original={str(p.resolve()):_validation_sha(p) for role,p in files.items() if role!='original_seals'}
    files['original_seals'].write_text(json.dumps(original),'utf-8')
    bundle=dict(schema='poker.saved-hh-validation.v1',files={role:dict(path=str(path),
        sha256=_validation_sha(path)) for role,path in files.items()})
    path=tmp_path/'bundle.json';path.write_text(json.dumps(bundle),'utf-8')
    return path,files


def test_saved_validation_rejects_review_locator_and_changed_authority(sealed_validation_bundle):
    from scripts.replay_saved_quality import _validation_bundle
    packet,files=sealed_validation_bundle
    with pytest.raises(ValueError,match='REVIEW_LOCATOR'):
        _validation_bundle(packet,files['review'].with_name('other.json'))
    files['review'].write_text('{"changed":true}','utf-8')
    with pytest.raises(ValueError,match='INPUT_CHANGED:review'):
        _validation_bundle(packet,files['review'])


def test_saved_validation_rejects_tampered_trace_before_execution(sealed_validation_bundle):
    from scripts.replay_saved_quality import _validation_bundle
    packet,files=sealed_validation_bundle
    files['traces'].write_text('{"accepted":true,"edited_action":true}','utf-8')
    with pytest.raises(ValueError,match='INPUT_CHANGED:traces'):
        _validation_bundle(packet,files['review'])


def test_saved_validation_requires_original_source_seal_even_if_packet_rehashed(sealed_validation_bundle):
    from scripts.replay_saved_quality import _validation_bundle,_validation_sha
    packet,files=sealed_validation_bundle
    files['seats'].write_text('{"START_STACK":999999}','utf-8')
    data=json.loads(packet.read_text('utf-8'))
    data['files']['seats']['sha256']=_validation_sha(files['seats'])
    packet.write_text(json.dumps(data),'utf-8')
    with pytest.raises(ValueError,match='SOURCE_CHANGED'):
        _validation_bundle(packet,files['review'])


@pytest.mark.parametrize('same_actor',[False,True])
def test_saved_validation_rejects_ambiguous_who(same_actor):
    from scripts.replay_saved_quality import _SavedValidationActors
    rows=[dict(table_id=1,session_id='s',hand_id='1',player=name,new_status='PROVEN',
        new_proof=dict(actor=name,canonical_actor='a' if same_actor else name,
            seat=f'seat{index}' if same_actor else 'seat1',status='PROVEN',
            table_id='T01',session_id='s',hand_id='1',anchor=dict(frame=10),
            identity_interval=dict(start=1,end=20))) for index,name in enumerate(('a','b'),1)]
    actors=_SavedValidationActors(rows,[])
    with pytest.raises(ValueError,match='WHO_CONFLICT'):
        actors.for_hand(dict(table_id='T01',session_id='s',hand_id='1'))


def test_saved_validation_projects_source_order_aliases_with_same_native_authority():
    from scripts.replay_saved_quality import _SavedValidationActors
    rows=[dict(table_id=1,session_id='s',hand_id='1',player=name,new_status='PROVEN',
        new_proof=dict(actor=name,canonical_actor='111',seat='seat3',anchor=dict(frame=10),
            status='PROVEN',table_id='T01',session_id='s',hand_id='1',
            identity_interval=dict(start=1,end=20),rule_used=name))
        for name in ('11','111')]
    actors=_SavedValidationActors(rows,[]);hand=dict(table_id='T01',session_id='s',hand_id='1')
    proofs=actors.for_hand(hand)
    assert len(proofs)==1 and proofs[0]['actor']=='111' and len(actors.rows[actors.key(hand)])==2
    assert proofs[0]['rule_used']=='111'
    assert actors.actor_map(hand)=={'11':'111','111':'111'}
    audit=actors.projections[actors.key(hand)][0]
    assert [row['player'] for row in audit['raw_rows']]==['11','111']
    assert audit['selected_proof']==proofs[0]
    actors.rows[actors.key(hand)]['11']['new_proof']['anchor']['frame']=11
    with pytest.raises(ValueError,match='WHO_CONFLICT'):
        actors.for_hand(hand)


@pytest.mark.parametrize('field,value',[('identity_interval',dict(start=2,end=20)),
    ('hand_id','2'),('status','UNKNOWN')])
def test_saved_validation_rejects_alias_authority_conflict(field,value):
    from scripts.replay_saved_quality import _SavedValidationActors
    rows=[dict(table_id=1,session_id='s',hand_id='1',player=name,new_status='PROVEN',
        new_proof=dict(actor=name,canonical_actor='111',seat='seat3',anchor=dict(frame=10),
            status='PROVEN',table_id='T01',session_id='s',hand_id='1',
            identity_interval=dict(start=1,end=20))) for name in ('11','111')]
    rows[-1]['new_proof'][field]=value
    with pytest.raises(ValueError,match='WHO_CONFLICT'):
        _SavedValidationActors(rows,[]).for_hand(dict(table_id='T01',session_id='s',hand_id='1'))


def test_saved_validation_rejects_bundle_replacement_with_same_locators(sealed_validation_bundle):
    from scripts.replay_saved_quality import _validation_bundle,_validation_finish_seals
    packet,files=sealed_validation_bundle
    initial=_validation_bundle(packet,files['review'])
    packet.write_text(packet.read_text('utf-8')+'\n','utf-8')
    with pytest.raises(ValueError,match='BUNDLE_CHANGED_DURING_RUN'):
        _validation_finish_seals(initial,packet,files['review'],{})


def test_saved_validation_rejects_execution_source_change(sealed_validation_bundle,tmp_path):
    from scripts.replay_saved_quality import _validation_bundle,_validation_finish_seals,_validation_sha
    packet,files=sealed_validation_bundle
    initial=_validation_bundle(packet,files['review'])
    code=tmp_path/'runner.py';code.write_text('old','utf-8')
    execution={str(code):_validation_sha(code)};code.write_text('new','utf-8')
    with pytest.raises(ValueError,match='EXECUTION_SOURCE_CHANGED'):
        _validation_finish_seals(initial,packet,files['review'],execution)


@pytest.mark.parametrize('missing',['CHAT','MANUAL','SITOUT','FORCED_CLAIM'])
def test_saved_validation_requires_all_consumed_native_parents(tmp_path,missing):
    from parser_core.current_chat_sources import FILES
    from scripts.replay_saved_quality import _validation_native_seals
    source=tmp_path/'source';other=tmp_path/'other'
    manual=tmp_path/'manual.json';status=tmp_path/'status.sqlite'
    review=tmp_path/'review.jsonl'
    review.write_text(json.dumps(dict(manual_source=dict(path=str(manual),sha256='sealed'))) +'\n','utf-8')
    paths={source/name for name in (*FILES,'seat_forced_inputs.jsonl','seat_forced_summary.json','run_metadata.json')}
    paths.update(other/name for name in FILES);paths.update((manual,status))
    pinned={str(path):'sealed' for path in paths}
    removed={'CHAT':source/FILES[0],'MANUAL':manual,'SITOUT':status,'FORCED_CLAIM':other/FILES[0]}[missing]
    del pinned[str(removed)]
    rows=[dict(sitout_sources=[dict(source=dict(file=str(status)))],forced_allin_stack_conflict=True,
               forced_allin_stack_claim=dict(source=str(other)))]
    with pytest.raises(ValueError,match='SOURCE_UNSEALED'):
        _validation_native_seals(source,dict(review=review),pinned,rows)


@pytest.mark.parametrize('changed',['AFTER','VERDICT','FOREIGN_PATH'])
def test_saved_validation_rejects_invalid_historical_native_seal(sealed_validation_bundle,tmp_path,changed):
    from scripts.replay_saved_quality import _validation_bundle,_validation_sha
    packet,files=sealed_validation_bundle
    source=tmp_path/'native';source.mkdir()
    parent=source/('other.jsonl' if changed=='FOREIGN_PATH' else 'start_stack_capture.jsonl')
    parent.write_text('native','utf-8');digest=_validation_sha(parent)
    prior={str(parent):dict(sha256=digest,bytes=parent.stat().st_size)}
    history=dict(before=prior,after=copy.deepcopy(prior),parity='PASS',changed=[])
    if changed=='AFTER':history['after'][str(parent)]['sha256']='different'
    if changed=='VERDICT':history['parity']='FAIL'
    history_path=tmp_path/'history.json';history_path.write_text(json.dumps(history),'utf-8')
    bundle=json.loads(packet.read_text('utf-8'))
    bundle.update(native_source=str(source),native_revalidation_inputs={str(parent):digest},
        native_parent_history=dict(path=str(history_path),sha256=_validation_sha(history_path)))
    packet.write_text(json.dumps(bundle),'utf-8')
    with pytest.raises(ValueError,match='HISTORICAL_PARITY|HISTORY_NOT_PASS|ADDITIONAL_PARENT_INVALID'):
        _validation_bundle(packet,files['review'])


def validation_parity_example():
    return dict(derived_final_validation={section:dict(logic_ms=1.0,
        math=dict(elapsed_ms=0.5,**{'pass':True})) for section in ('validation','diagnostics')},
        authority=dict(start_stack=123),source=dict(logic_ms=10.0))


def test_saved_validation_parity_only_excludes_exact_regenerated_stopwatches():
    from scripts.replay_saved_quality import _validation_difference,_TRACE_VALIDATION_TIMINGS
    old=validation_parity_example();new=copy.deepcopy(old)
    for section in ('validation','diagnostics'):
        new['derived_final_validation'][section].update(logic_ms=22.0)
        new['derived_final_validation'][section]['math']['elapsed_ms']=3.0
    assert _validation_difference(old,new,_TRACE_VALIDATION_TIMINGS) is None
    new['authority']['start_stack']=124
    assert 'authority.start_stack' in _validation_difference(old,new,_TRACE_VALIDATION_TIMINGS)
    new['authority']['start_stack']=123;new['source']['logic_ms']=11.0
    assert 'source.logic_ms' in _validation_difference(old,new,_TRACE_VALIDATION_TIMINGS)


@pytest.mark.parametrize('value',[True,-1.0,float('nan'),float('inf'),None,'1.0'])
def test_saved_validation_parity_rejects_invalid_timing(value):
    from scripts.replay_saved_quality import _validation_difference,_TRACE_VALIDATION_TIMINGS
    old=validation_parity_example();new=copy.deepcopy(old)
    new['derived_final_validation']['validation']['logic_ms']=value
    assert 'INVALID_TIMING' in _validation_difference(old,new,_TRACE_VALIDATION_TIMINGS)


def test_saved_validation_parity_rejects_missing_timing():
    from scripts.replay_saved_quality import _validation_difference,_TRACE_VALIDATION_TIMINGS
    old=validation_parity_example();new=copy.deepcopy(old)
    del new['derived_final_validation']['diagnostics']['math']['elapsed_ms']
    assert 'MISSING_TIMING' in _validation_difference(old,new,_TRACE_VALIDATION_TIMINGS)


def test_saved_validation_hh_parity_preserves_logical_record_timings():
    from scripts.replay_saved_quality import _validation_difference,_HH_VALIDATION_TIMINGS
    old=validation_parity_example();old['reconstruction']=validation_parity_example()
    old['validation']=dict(logic_ms=5.0);new=copy.deepcopy(old)
    new['reconstruction']['derived_final_validation']['validation']['logic_ms']=100.0
    assert _validation_difference(old,new,_HH_VALIDATION_TIMINGS) is None
    new['validation']['logic_ms']=6.0
    assert 'validation.logic_ms' in _validation_difference(old,new,_HH_VALIDATION_TIMINGS)


@pytest.mark.parametrize('field,value',[('accepted',1),('start_stack',123.0)])
def test_saved_validation_parity_rejects_equal_valued_nested_type_change(field,value):
    from scripts.replay_saved_quality import _validation_difference,_TRACE_VALIDATION_TIMINGS
    old=validation_parity_example();old['authority']['accepted']=True
    new=copy.deepcopy(old);new['authority'][field]=value
    assert f'authority.{field}:TYPE' in _validation_difference(old,new,_TRACE_VALIDATION_TIMINGS)


@pytest.mark.parametrize('module,name',[
    ('parser_core.poker_reconstruction.solver','solve'),
    ('parser_core.poker_reconstruction.reconstruct','reconstruct_hand'),
    ('parser_core.poker_reconstruction.state_machine','apply'),
    ('parser_core.simple_saved_policy','saved_stack_observations'),
    ('parser_core.tiny_chat_client','__init__'),
    ('parser_core.ram_first_stage','rows'),
])
def test_saved_validation_forbids_search_and_acquisition(module,name):
    from collections import Counter
    from types import SimpleNamespace
    from scripts.replay_saved_quality import _saved_validation_profile
    guard=_saved_validation_profile(Counter())
    frame=SimpleNamespace(f_globals={'__name__':module},f_code=SimpleNamespace(co_name=name))
    with pytest.raises(RuntimeError,match='FORBIDDEN_CALL'):
        guard(frame,'call',None)


def test_official_saved_validation_cli_rejects_live_profile(tmp_path):
    import subprocess,sys
    result=subprocess.run([sys.executable,str(ROOT/'run_production.py'),
        '--validate-saved-hh','unused.json','--no-sb-review','unused.json',
        '--output',str(tmp_path/'output'),'--profile','FULL_HH_PRODUCTION'],
        cwd=ROOT,capture_output=True)
    assert result.returncode==2 and b'unrecognized arguments' in result.stderr
    assert not (tmp_path/'output').exists()


def inputs(path, records=None, physical=None):
    path.mkdir()
    for name,rows in [('visual_evidence.jsonl',records if records is not None else NATIVE['parents']),
                      ('physical_rows.jsonl',physical if physical is not None else NATIVE['physical'])]:
        (path/name).write_text(''.join(json.dumps(r)+'\n' for r in rows),'utf-8')
    return path


@pytest.fixture(scope='module')
def refreshed(tmp_path_factory):
    base=tmp_path_factory.mktemp('native-pixel-replay')
    source=inputs(base/'source');output=base/'output';output.mkdir()
    before={p.name:p.read_bytes() for p in source.iterdir()}
    live={}
    original=VisualTileQueue.write_evidence
    def capture(queue,path):
        live.update(copy.deepcopy(queue.results))
        original(queue,path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(VisualTileQueue,'write_evidence',capture)
        readings=refresh_saved_card_readings(source,output)
    assert before=={p.name:p.read_bytes() for p in source.iterdir()}
    return source,output,readings,live


def test_native_refresh_keeps_sources_and_proves_complete_partition(refreshed):
    source,output,readings,_=refreshed
    assert len(readings)==3
    assert (output/'visual_evidence_source.jsonl').read_bytes()==(source/'visual_evidence.jsonl').read_bytes()
    for old in NATIVE['parents']:
        new=readings[old['row_key']]
        for field in ('row_key','row_id','row_pixel_sha256','crop_png_base64','crop_sha256','source_pts'):
            assert old[field]==new[field]
        assert new['raw_pixel_response']['rank_config_sha256']==hashlib.sha256(
            (ROOT/'configs/rank_ncc_production.json').read_bytes()).hexdigest()
    resolved=readings['6:2843:17']
    assert resolved['cards']==['7s','Kh','4d'] and not resolved.get('hybrid_authoritative_rejection')
    assert resolved.get('reason') is None
    assert resolved['hybrid']['secondary_partition_proof']['rule']=='SECONDARY_PARTITION_RANK_COMPONENT_BRIDGE_V1'
    receipt=json.loads((output/'saved_card_replay_receipt.json').read_text('utf-8'))
    assert str((ROOT/'models/board_glyphs_v1.json').resolve()) in receipt['runtime_sha256']
    assert len(receipt['runtime_sha256'])>500  # Includes individually pinned TRAIN tiles.


def test_actual_ram_win_proof_matches_persisted_receipt_without_serialization_repair(refreshed):
    _,_,persisted,live=refreshed
    case=copy.deepcopy(NATIVE['win_source'])
    timeline=copy.deepcopy(case['context'])
    for row in timeline:
        row['event_type']=normalize([row['raw_ocr_text']]).event_type
    ram=copy.deepcopy(live);saved=copy.deepcopy(persisted)
    associate_visual_evidence(timeline,ram)
    associate_visual_evidence(timeline,saved)
    key=case['reading']['row_key']
    a=prove_win_tail(case['rows'],ram[key],case['context'])
    b=prove_win_tail(case['rows'],saved[key],case['context'])
    assert a and b
    assert a['cards']==b['cards']==case['reading']['cards']
    assert a['source_refs']==b['source_refs']
    assert json.loads(json.dumps(a))==b


def test_visual_reset_revokes_stale_cards_but_preserves_raw_and_identity():
    rows=copy.deepcopy(NATIVE['projected_triggers'])
    assert rows and all(r.get('cards') for r in rows)
    old=copy.deepcopy(rows)
    clean=ordinary_replay_rows(rows,reset_visual=True)
    assert rows==old
    assert len(clean)==len(rows)
    for native,current in zip(rows,clean):
        assert all(field not in current for field in _VISUAL_FIELDS)
        assert current['event_type']==normalize([native['raw_ocr_text']]).event_type
        assert not any(k in current for k in ('hand_id','street','player'))
        for field in ('stable_row_id','raw_ocr_text','row_pixel_sha256','table_id','session_id','frame_id','row_index','epoch'):
            assert native.get(field)==current.get(field)
    assert ordinary_replay_rows(rows)==rows  # Non-FULL replay unchanged.
    for status in ('ERROR','UNRESOLVED'):
        bad=copy.deepcopy(rows);bad[0]['status']=status
        assert ordinary_replay_rows(bad,reset_visual=True)[0]['status']==status


@pytest.mark.parametrize('mutation',['png','pixel','context','clock','duplicate','unknown_diagnostic'])
def test_corrupt_saved_sources_fail_before_starting_producer(tmp_path,monkeypatch,mutation):
    parents=copy.deepcopy(NATIVE['parents']);physical=copy.deepcopy(NATIVE['physical'])
    if mutation=='png':parents[0]['crop_sha256']='0'*64
    elif mutation=='pixel':physical[0]['row_pixels_base64']='AAAA'
    elif mutation=='context':parents[0]['epoch']+=1
    elif mutation=='clock':parents[0]['video_time']+=1
    elif mutation=='duplicate':parents.append(copy.deepcopy(parents[0]))
    elif mutation=='unknown_diagnostic':parents.append({'status':'UNRESOLVED','cards':['As','Kd']})
    source=inputs(tmp_path/'source',parents,physical);output=tmp_path/'out';output.mkdir()
    def forbidden(*args,**kwargs):pytest.fail('Corrupt input reached producer')
    monkeypatch.setattr(VisualTileQueue,'__init__',forbidden)
    with pytest.raises(ValueError,match='SAVED_CARD_'):
        refresh_saved_card_readings(source,output)
    assert not (output/'visual_evidence.jsonl').exists()


def test_missing_diagnostic_is_preserved_separately_and_not_treated_as_pixels(tmp_path):
    diagnostic=dict(type='SHOWDOWN',street=None,player='Unresolved',table_id='T01',
        table_session_id='source-context',frame_id=1,hand_id=None,trigger_row_key='1:1:1',
        crop_type='CHAT_ROW_CARDS',crop_path=None,raw_vl_response=None,status='UNRESOLVED',
        reason='NO_VALID_ADJACENT_CARD_TILE')
    source=inputs(tmp_path/'source',[*NATIVE['parents'],diagnostic]);output=tmp_path/'out';output.mkdir()
    assert len(refresh_saved_card_readings(source,output))==3
    receipt=json.loads((output/'saved_card_replay_receipt.json').read_text('utf-8'))
    assert receipt['source_missing_diagnostics']==1
    assert (output/'visual_evidence_source.jsonl').read_bytes()==(source/'visual_evidence.jsonl').read_bytes()


def test_changed_model_dependency_is_detected_without_mutating_file(tmp_path,monkeypatch):
    source=inputs(tmp_path/'source');output=tmp_path/'out';output.mkdir()
    target=(ROOT/'models/board_glyphs_v1.json').resolve()
    original=Path.read_bytes;calls=0
    def changed_after_load(path):
        nonlocal calls
        raw=original(path)
        if path.resolve()==target:
            calls+=1
            if calls>=3:return raw+b' '
        return raw
    monkeypatch.setattr(Path,'read_bytes',changed_after_load)
    with pytest.raises(ValueError,match='SOURCE_OR_RUNTIME_CHANGED'):
        refresh_saved_card_readings(source,output)
    assert not (output/'visual_evidence.jsonl').exists()
