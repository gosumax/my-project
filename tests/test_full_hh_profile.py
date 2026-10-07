from types import SimpleNamespace
import json
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
