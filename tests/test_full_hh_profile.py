from types import SimpleNamespace
import copy
import json
from pathlib import Path
import pytest
from parser_core.full_hh_profile import configure, connection_matrix, require_complete, REQUIRED
from parser_core.hh_export import export_hand_histories
from parser_core.poker_pipeline import analyze_timeline
from parser_core.hand_lifecycle import classify_lifecycle


def test_profile_enables_all_feature_routes():
    args=SimpleNamespace(profile='FULL_HH_PRODUCTION',skip_vl=False,stack_evidence=None)
    assert configure(args)
    assert all(getattr(args,k) for k in ('enable_dealing_cards_hand_state_trigger','enable_start_stack_capture',
        'stack_timeline','stack_start_stack','performance_diagnostics','dealing_start_stack_acceptance'))


def test_missing_required_consumer_rejects_acceptance():
    matrix=dict.fromkeys(REQUIRED,True)
    matrix['START_STACK']=False
    with pytest.raises(ValueError,match='FULL_HH_PRODUCTION_INCOMPLETE: START_STACK'):
        require_complete(matrix)
    with pytest.raises(ValueError,match='FULL_HH_PRODUCTION_INCOMPLETE'):
        configure(SimpleNamespace(profile='FULL_HH_PRODUCTION',skip_vl=True,stack_evidence=None))


def test_matrix_checks_real_connections():
    obj=object()
    kwargs=dict(stage=SimpleNamespace(physical_admission=True),ocr=obj,header_lane=obj,sidecar=obj,
        stack_live=obj,seat_binding=obj,stack_timeline=obj,cards=obj,reconstruction_enabled=True,
        diagnostic=obj,hh_export=export_hand_histories)
    assert all(connection_matrix(**kwargs).values())
    kwargs['seat_binding']=None
    with pytest.raises(ValueError,match='FULL_HH_PRODUCTION_INCOMPLETE'):
        connection_matrix(**kwargs)


@pytest.mark.parametrize('texts,expected',[
    (['Dealer: A folds','Dealer: B wins 10'],'START_PARTIAL'),
    (['Hand #123456','Dealer: A posts Small Blind 5'],'END_PARTIAL'),
    (['Hand #123456','Dealer: A posts Small Blind 5','Dealer: B posts Big Blind 10',
      'Dealer: A folds','Dealer: B gets uncalled bet 5 back','Dealer: B wins 10'],'PARTIAL'),
])
def test_hh_export_preserves_partial_and_does_not_invent_seats_or_stacks(tmp_path,texts,expected):
    rows=[dict(stable_row_id=f's:R{i}',table_id='T01',session_id='s',frame_id=1,row_index=i,
               epoch=1,origin='NEW',raw_ocr_text=t,video_time=0) for i,t in enumerate(texts)]
    records,_=analyze_timeline(rows,ocr_status={r['stable_row_id']:'OK' for r in rows})
    classify_lifecycle(records)
    summary=export_hand_histories(tmp_path,records)
    exported=[json.loads(s) for s in (tmp_path/'hand_histories.jsonl').read_text('utf-8').splitlines()]
    assert len(exported)==1 and exported[0]['status']==expected
    assert [m['raw_text'] for m in exported[0]['actions']]==texts
    assert all(p['seat'] is None and p['START_STACK'] is None for p in exported[0]['players'])
    assert summary['certified_hh']==0 and summary['HH_COMPLETE_RATE'] is None


def test_conflicting_stack_receipts_never_certify(tmp_path):
    record=dict(segment_id='s1',lifecycle=dict(status='COMPLETE'),fully_validated=True,
        validation={},diagnostics={},observed_hand=dict(table_id='T01',session_id='s',hand_id='123',
        messages=[dict(player='Alice',raw_text='Dealer: Alice wins 10')]))
    (tmp_path/'poker_reconstruction.jsonl').write_text(json.dumps(dict(segment_id='s1',output_eligible=True)),encoding='utf-8')
    binding=dict(player='Alice',seat_id=1,table_id='T01',session_id='s',hand_id='123',occupancy_epoch='e1')
    receipts=[dict(segment_id='s1',status='START_STACK_PROVEN',source_observation_id=i,source_state_id=i,
                   calculated_start_stack=v,binding_receipt=binding,**binding) for i,v in [(1,100),(2,200)]]
    (tmp_path/'stack_start_stack.jsonl').write_text('\n'.join(json.dumps(r) for r in receipts),encoding='utf-8')
    summary=export_hand_histories(tmp_path,[record])
    assert summary['certified_hh']==0 and summary['blockers']['START_STACK_CONFLICT']==1


@pytest.mark.parametrize('index', [0, 1, 2])
@pytest.mark.parametrize('conflicting_source', [False, True])
@pytest.mark.parametrize('sidecar_header', [False, True])
def test_shared_live_card_stage_matches_saved_native_proofs(tmp_path, index, conflicting_source, sidecar_header):
    from parser_core.full_hh_profile import project_current_card_evidence
    from parser_core.visual_cards import associate_visual_evidence, project_chat_card_events
    from parser_core.win_tail_evidence import project_win_tails
    from parser_core.card_set_evidence import project_unique_card_sets
    from scripts.replay_saved_quality import ordinary_replay_rows
    from parser_core.canonical_events import canonical_join

    sources=json.loads((Path(__file__).parent/'fixtures/card_set_projection_native_sources.json').read_text('utf-8'))
    source=copy.deepcopy(sources[index])
    rows=ordinary_replay_rows(source['rows'], reset_visual=True)
    (tmp_path/'seat_forced_summary.json').write_text(json.dumps({'forced_snapshots':[]}),encoding='utf-8')
    if sidecar_header:
        # Exercise the real saved-receipt reader and insertion at the same
        # native header location, without an ordinary HAND_ID observation.
        header=rows.pop(0)
        receipt=dict(table_id=header['table_id'],session_id=header['session_id'],
            source_frame=header['frame_id'],source_span_index=header['row_index'],
            source_pts=header.get('source_pts'),time_base=header.get('time_base'),
            pixel_sha256=header['row_pixel_sha256'],hand_id=source['hand_id'])
        admitted=dict(receipt,status='ADMITTED')
        completed=dict(receipt,status='HAND_HEADER_CONFIRMED',
            ocr_response=dict(status='OK',raw_text=header['raw_ocr_text'],score=header['confidence']))
        (tmp_path/'hand_header_ocr.jsonl').write_text(
            json.dumps(admitted)+'\n'+json.dumps(completed)+'\n',encoding='utf-8')
        (tmp_path/'hand_state_receipts.jsonl').write_text(json.dumps(completed)+'\n',encoding='utf-8')
    rows,_=canonical_join(tmp_path,rows)
    if sidecar_header:
        assert rows[0]['source_kind']=='SIDECAR_HEADER_RECEIPT'
        assert rows[0]['event_type']=='HAND_HEADER'
    if conflicting_source:
        for reading in source['readings'].values():
            reading['row_pixel_sha256']='0'*64
    expected_rows,actual_rows=copy.deepcopy(rows),copy.deepcopy(rows)
    expected_readings,actual_readings=copy.deepcopy(source['readings']),copy.deepcopy(source['readings'])
    raw_fields=('stable_row_id','table_id','session_id','frame_id','row_index',
                'raw_ocr_text','row_pixel_sha256','source_pts','time_base','video_time','epoch')
    raw_before=[[r.get(k) for k in raw_fields] for r in actual_rows]

    # The previously registered saved sequence is the comparison contract.
    associate_visual_evidence(expected_rows,expected_readings)
    native=project_chat_card_events(expected_rows,expected_readings)
    winning=project_win_tails(expected_rows,expected_readings)
    derived=project_unique_card_sets(expected_rows,expected_readings)
    counts=project_current_card_evidence(actual_rows,actual_readings)
    assert actual_rows==expected_rows and actual_readings==expected_readings
    assert counts==dict(native_card_events=native,win_best_five_proofs=winning,unique_card_set_proofs=derived)
    assert [[r.get(k) for k in raw_fields] for r in actual_rows]==raw_before
    if conflicting_source:
        assert winning==derived==0
    elif not sidecar_header:
        proofs=[r.get('unique_card_set_proof') or r.get('win_best_five_proof') for r in actual_rows]
        assert any(p and p['cards']==source['expected_cards'] for p in proofs)


def test_shared_live_card_stage_keeps_unbound_sidecar_hand_unknown():
    from parser_core.full_hh_profile import project_current_card_evidence
    source=json.loads((Path(__file__).parent/'fixtures/win_tail_t09_cross_epoch_source.json').read_text('utf-8'))
    rows=copy.deepcopy(source['context'])
    for row in rows:
        row['event_type']='WIN' if row['stable_row_id']==source['rows'][0]['stable_row_id'] else 'UNKNOWN_EVENT'
    # A conflicting numbered context cannot be repaired by routing through FULL.
    rows[0]['raw_ocr_text']='Hand #123'
    rows[0]['event_type']='HAND_HEADER'
    reading=copy.deepcopy(source['reading'])
    reading['hand_id']=source['reading'].get('hand_id') or source['context'][0].get('hand_id') or '2878366656'
    result=project_current_card_evidence(rows,{reading['row_key']:reading})
    assert result['unique_card_set_proofs']==0
    assert result['win_best_five_proofs']==0
