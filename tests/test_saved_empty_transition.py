"""A reviewed capture hook cannot waive unrelated ALL57 source pins."""
import copy
import json
from pathlib import Path
import pytest
from scripts.replay_saved_quality import _validation_empty_transition,_validation_sha

NAMES=('parser_core/start_stack_live.py','parser_core/empty_fast_capture.py',
       'off_empty_dispatch.py','off_empty_bound.py','off_empty_pixels.py')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),'utf-8')
    return dict(path=str(path.resolve()),sha256=_validation_sha(path))

@pytest.fixture
def transition(tmp_path):
    closure={}
    for name in NAMES:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(name,'utf-8');closure[name]=_validation_sha(path)
    old=write(tmp_path/'rollback/producer.py','ORIGINAL')
    inputs={str((tmp_path/name).resolve()):digest for name,digest in closure.items()}
    inputs[old['path']]=old['sha256']
    proof=dict(verdict='PASS',decision_mismatches=0,state_mismatches=0,source_receipt_mismatches=0,
        native_initial_hands=57,native_joint_hands=16,fast_capture_batches=1,callback_schedules=3,
        producer_body_equivalence='PASS',new_ocr_calls=0,new_solver_calls=0,new_video_decode_calls=0,
        source_input_sha256=inputs)
    proof_ref=write(tmp_path/'proof.json',proof)
    review=dict(verdict='PASS',runtime_sha256=closure,proof_sha256=proof_ref['sha256'],before_sha256=old['sha256'])
    review_ref=write(tmp_path/'review.json',review)
    data=dict(schema='poker.saved-empty-runtime-transition.v1',runtime_sha256=closure,
        before=old,proof=proof_ref,review=review_ref)
    pins={str((tmp_path/NAMES[0]).resolve()):old['sha256'],str(tmp_path/'unrelated'):'UNCHANGED'}
    return tmp_path,pins,data,proof,review

def test_only_reviewed_producer_execution_pin_advances(transition):
    root,pins,data,*_=transition;_validation_empty_transition(root,pins,data)
    assert pins[str((root/NAMES[0]).resolve())]==data['runtime_sha256'][NAMES[0]]
    assert pins[str(root/'unrelated')]=='UNCHANGED'
    assert pins[data['before']['path']]==data['before']['sha256']

@pytest.mark.parametrize('change',['scope','missing_old_pin','old_bytes','runtime_bytes','proof_bytes',
    'proof_fail','decision','state','source','coverage','schedules','old_binding',
    'review_proof','review_old','review_runtime','proof_runtime','existing_helper_pin','producer_equivalence','acquisition'])
def test_transition_rejects_missing_or_changed_proof(transition,change):
    root,pins,data,proof,review=transition
    if change=='scope':data['runtime_sha256']['parser_core/hh_export.py']='UNAUTHORIZED'
    if change=='missing_old_pin':pins.pop(str((root/NAMES[0]).resolve()))
    if change=='old_bytes':Path(data['before']['path']).write_text('CHANGED')
    if change=='runtime_bytes':(root/NAMES[1]).write_text('CHANGED')
    if change=='proof_bytes':Path(data['proof']['path']).write_text('CHANGED')
    if change=='existing_helper_pin':pins[str((root/NAMES[1]).resolve())]='HISTORICAL'
    if change in ('proof_fail','decision','state','source','coverage','schedules','old_binding','proof_runtime','producer_equivalence','acquisition'):
        if change=='proof_fail':proof['verdict']='FAIL'
        if change=='decision':proof['decision_mismatches']=1
        if change=='state':proof['state_mismatches']=1
        if change=='source':proof['source_receipt_mismatches']=1
        if change=='coverage':proof['native_initial_hands']=56
        if change=='schedules':proof['callback_schedules']=2
        if change=='old_binding':proof['source_input_sha256'].pop(data['before']['path'])
        if change=='proof_runtime':proof['source_input_sha256'][str((root/NAMES[1]).resolve())]='OTHER'
        if change=='producer_equivalence':proof['producer_body_equivalence']='FAIL'
        if change=='acquisition':proof['new_video_decode_calls']=1
        data['proof']=write(root/'proof.json',proof)
        review['proof_sha256']=data['proof']['sha256'];data['review']=write(root/'review.json',review)
    if change.startswith('review_'):
        field={'review_proof':'proof_sha256','review_old':'before_sha256','review_runtime':'runtime_sha256'}[change]
        review[field]='OTHER';data['review']=write(root/'review.json',review)
    with pytest.raises(ValueError,match='SAVED_VALIDATION_TRANSITION'):
        _validation_empty_transition(root,pins,data)
