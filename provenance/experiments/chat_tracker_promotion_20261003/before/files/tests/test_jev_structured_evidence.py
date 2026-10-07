"""Source-scoped receipt/admission controls; no network or pipeline launch."""
import copy
import json
import pytest
from experiments.jev_verified_evidence_smoke17 import (
    OUT,BASE,action_context,checked_answer,read,review_proposal,stable_result,verify_dispatch)

@pytest.fixture(scope='module')
def t06():
    hand=json.loads((BASE/'CORE_FIXED_WINDOWS.assembled_hands.json').read_text('utf-8'))[4]
    record=read(BASE/'CORE_FIXED_WINDOWS.hands.jsonl')[4]
    binding=json.loads((OUT/'requests/01_hand_04.binding.json').read_text('utf-8'))
    return hand,record,binding

@pytest.mark.parametrize('index,amount',[(16,'10750'),(27,'10956')])
def test_t06_equal_effects_are_not_discriminatory(t06,index,amount):
    ctx=action_context(t06[0],t06[1],index)
    assert ctx['constraints_trusted'] and ctx['prior_player_street_contribution']=='0'
    assert ctx['compatible_semantics']==['ADDED','TOTAL']
    assert ctx['candidate_effects']['ADDED']['added_amount']==ctx['candidate_effects']['TOTAL']['added_amount']==amount
    assert ctx['pot_observed'] is None and ctx['stack_observed'] is None

@pytest.mark.parametrize('choice',['ADDED','TOTAL','UNRESOLVED'])
def test_confidence_cannot_discriminate_t06(t06,choice):
    target=t06[2]['question_targets']['semantics_m16']
    result=review_proposal(target,dict(choice=choice,confidence=1))
    assert result['status']=='UNRESOLVED' and result['admitted_fact'] is None

@pytest.mark.parametrize('code,index',[('ACTION_AFTER_FOLD',16),('ACTION_AFTER_HAND_END',16),
    ('ACTION_AFTER_ALLIN',16),('ROW_IDENTITY_UNRESOLVED',16),('PLAYER_IDENTITY_CONFLICT',16),
    ('INCOMPLETE_START',-1),('HAND_BOUNDARY_CONFLICT',-1)])
def test_conflicted_state_never_drives_semantics(t06,code,index):
    record=copy.deepcopy(t06[1]);record['fully_valid_blocking_checks'].append(dict(
        code=code,message_index=index,source_row_ids=t06[0]['messages'][16]['source_row_ids']))
    ctx=action_context(t06[0],record,16)
    assert not ctx['constraints_trusted'] and not ctx['compatible_semantics']

@pytest.mark.parametrize('kind',['identity_provenance','card_evidence','payout','completeness','action_state'])
def test_supported_label_does_not_invent_missing_fact(t06,kind):
    target=copy.deepcopy(t06[2]['question_targets']['semantics_m16']);target['kind']=kind
    assert review_proposal(target,dict(choice='SUPPORTED',confidence=1))['admitted_fact'] is None

def test_conditional_prior_call_not_treated_as_observed(t06):
    hand,record=copy.deepcopy(t06[:2]);hand['messages'][15]['action']='CALL';hand['messages'][15]['player']='billmurray'
    record['observed_ledger']['steps'][15]['before']['street_contributions_by_player']['billmurray']='500'
    ctx=action_context(hand,record,16)
    assert ctx['ambiguous_prior_call_indices']==[15] and not ctx['constraints_trusted']

def test_response_must_match_question_contract():
    q=dict(criteria={'ADDED':'','TOTAL':'','UNRESOLVED':''})
    with pytest.raises(ValueError):checked_answer(q,dict(type='choice',choice='VALID_HAND',confidence=1,probabilities={'VALID_HAND':1}))

def test_non_normalized_distribution_cannot_be_evidence():
    q=dict(criteria={'CONTRADICTED':'','SUPPORTED':'','UNRESOLVED':''})
    with pytest.raises(ValueError,match='Probability sum'):
        checked_answer(q,dict(type='choice',choice='UNRESOLVED',confidence=.74,
            probabilities=dict(CONTRADICTED=.16,SUPPORTED=.02,UNRESOLVED=.81)))

@pytest.mark.parametrize('field',['request_file','request_sha256','binding_sha256','source_sha256'])
def test_foreign_or_modified_dispatch_rejected(field):
    raw=json.loads((OUT/'receipts/02_hand_03.raw.json').read_text('utf-8'))
    request=OUT/'requests/02_hand_03.json';binding=OUT/'requests/02_hand_03.binding.json'
    source=OUT/json.loads(binding.read_text('utf-8'))['source_file']
    verify_dispatch(raw,request,binding,source)
    raw[field]='foreign'
    with pytest.raises(ValueError):verify_dispatch(raw,request,binding,source)

def test_runtime_timing_exclusion_keeps_source_and_verdict():
    assert stable_result(dict(logic_ms=1,math=dict(elapsed_ms=2),timestamp_ms=7,status='PARTIAL'))==dict(math={},timestamp_ms=7,status='PARTIAL')

def test_packet_format_does_not_send_image_as_fact():
    for p in (OUT/'requests').glob('*.binding.json'):
        request=json.loads(p.with_name(p.name.replace('.binding.json','.json')).read_text('utf-8'))
        assert {'OBSERVED','DERIVED','CANDIDATES','UNKNOWN'}<=set(request['state'])
        assert request['state']['UNKNOWN']['observed_pot'] is None
        assert request['state']['UNKNOWN']['observed_stacks'] is None
        assert 'row_pixels_base64' not in json.dumps(request)

def test_current_source_has_no_unique_trusted_semantic_candidate():
    targets=[t for p in (OUT/'requests').glob('*.binding.json')
        for t in json.loads(p.read_text('utf-8'))['question_targets'].values() if t['kind']=='AMOUNT_SEMANTICS']
    assert targets
    assert not any(t['derived_action_context']['constraints_trusted'] and len(t['derived_action_context']['compatible_semantics'])==1 for t in targets)
