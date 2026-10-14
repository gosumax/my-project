"""Native pixels test the saved-number contract without decoding video or OCR."""
import copy
import sys
from pathlib import Path

import pytest

from parser_core.first_visible_single_numeric import first_saved_numeric_proof, SAVED_SINGLE_CONTRACT
from parser_core.first_stack_ocr import single_first_numeric_policy
from parser_core.start_stack_visibility import first_visible_proof, verify_first_visible_fact

sys.path.insert(0, str(Path(__file__).parent))
from test_direct_numeric_policy import native_context


def sources():
    c=native_context()
    return c,{r['source_frame']:copy.deepcopy(r) for r in c['readings']}


@pytest.mark.parametrize('prefix',['original','blank','unclassified','missing','none'])
def test_pre_numeric_pixels_and_coverage_do_not_block(prefix):
    _,readings=sources()
    if prefix=='blank':
        for f in range(258,284):
            readings[f]['raw_text']=''
            readings[f]['ocr_status']='NOT_READ'
    elif prefix=='unclassified':
        for f in range(258,284):
            readings[f]['raw_text']='transition overlay'
    elif prefix=='missing':
        del readings[260]
    elif prefix=='none':
        readings={284:readings[284]}
    proof=first_saved_numeric_proof(readings,[284],start_frame=258)
    assert proof['status']=='PROVEN' and proof['displayed_stack']=='17416'
    assert proof['contract']==SAVED_SINGLE_CONTRACT


def test_numeric_at_hand_start_and_no_repetition_required():
    _,r=sources()
    proof=first_saved_numeric_proof({284:r[284]},[284],start_frame=284)
    assert proof['status']=='PROVEN' and len(proof['source_evidence'])==1


def test_saved_number_is_not_limited_to_initial_64_frames():
    _,r=sources()
    proof=first_saved_numeric_proof({284:r[284]},[284],start_frame=200)
    assert proof['status']=='PROVEN' and proof['first_numeric_frame']==284


def test_first_available_accepted_number_cannot_be_replaced_by_later_number():
    _,r=sources()
    proof=first_saved_numeric_proof(r,[285],start_frame=258)
    assert proof['status']=='UNKNOWN' and proof['reason']=='EARLIER_ACCEPTED_SAVED_NUMBER'


@pytest.mark.parametrize('score',[.93,.9,None,True,float('nan')])
def test_existing_primary_numeric_gate_remains(score):
    _,r=sources();r[284]['confidence']=score
    assert first_saved_numeric_proof(r,[284],start_frame=258)['status']=='UNKNOWN'


@pytest.mark.parametrize('field,value',[
    ('pixel_sha256','0'*64),('seat','seat99'),('source_frame',283),
    ('hand_id',None),('session_id',None),('raw_text','Sit out')])
def test_selected_source_and_numeric_guards_remain(field,value):
    _,r=sources();r[284][field]=value
    assert first_saved_numeric_proof(r,[284],start_frame=258)['status']=='UNKNOWN'


def test_before_hand_number_is_not_allowed():
    _,r=sources()
    assert first_saved_numeric_proof(r,[284],start_frame=285)['status']=='UNKNOWN'


def test_new_contract_does_not_enter_historical_generic_fact_authority():
    c,r=sources();fact=copy.deepcopy(c['fact'])
    fact['first_visible_proof']=first_saved_numeric_proof(r,[284],start_frame=258)
    assert not verify_first_visible_fact(fact)
    old=first_visible_proof(r,[284],start_frame=258,confidence=.93,
        single_first_policy=single_first_numeric_policy())
    assert old['status']=='PROVEN' and old['contract']!=SAVED_SINGLE_CONTRACT
