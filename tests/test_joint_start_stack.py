import copy
import json
import sys
from pathlib import Path

import pytest

from parser_core.joint_start_stack import prove
from parser_core.joint_start_stack import verify_retained,join_receipt

@pytest.fixture
def source():
    return json.loads((Path(__file__).parent/'fixtures/joint_stack_native_sources.json').read_text('utf-8'))

@pytest.mark.parametrize('actor,value',[('kalmar_204',75441),('vasyajin',13160)])
def test_raw_saved_positive(source,actor,value):
    case=next(c for c in source['cases'] if c['actor']==actor)
    before=copy.deepcopy(source)
    proof=prove(case,source['database_metadata_raw'])
    assert proof['status']=='PROVEN',proof.get('reason')
    assert proof['calculated_start_stack']==value
    assert proof['global_occupancy_authority'] is False
    assert source==before

@pytest.mark.parametrize('mutation',[
    'capture_gap','foreign_pts','rgb_hash','sql_raw_pixels','ocr_numeric_raw_conflict',
    'low_confidence','state_context','foreign_nick','foreign_header','own_forced_gap',
    'missing_blind','duplicate_own_ante','late_first_number','current_context_rebind'])
def test_missing_or_conflicting_input_is_unknown(source,mutation):
    case=copy.deepcopy(source['cases'][0]);context=case['source_context']
    states=case['raw_sql_visual_states'];numeric=next(s for s in states if s['status']=='CONFIRMED_NUMERIC')
    if mutation=='capture_gap':case['raw_stack_captures'].pop(15)
    elif mutation=='foreign_pts':case['raw_stack_captures'][20]['raw_record']['source_pts']+=512
    elif mutation=='rgb_hash':case['raw_stack_captures'][20]['raw_record']['pixel_sha256']='0'*64
    elif mutation=='sql_raw_pixels':numeric['raw_gray_base64']='AAAA'
    elif mutation=='ocr_numeric_raw_conflict':numeric['ocr_numeric']='123'
    elif mutation=='low_confidence':
        numeric['ocr_confidence']=.949
        payload=json.loads(numeric['provenance']);payload['state_payload']['ocr_confidence']=.949
        numeric['provenance']=json.dumps(payload)
    elif mutation=='state_context':numeric['seat_id']='seat1'
    elif mutation=='foreign_nick':
        for r in context['nickname_observations_all_within_window']:
            if r['seat']==case['seat']:r['ocr_response']['raw_text']='FOREIGN_ACTOR'
    elif mutation=='foreign_header':context['hand_id']='987654321'
    elif mutation=='own_forced_gap':
        context['ordinary_prefix_timeline']=[r for r in context['ordinary_prefix_timeline'] if not ('kalmar_204 posts Ante' in r['raw_ocr_text'])]
    elif mutation=='missing_blind':
        context['ordinary_prefix_timeline']=[r for r in context['ordinary_prefix_timeline'] if not ('kalmar_204 posts Big Blind' in r['raw_ocr_text'])]
    elif mutation=='duplicate_own_ante':
        context['ordinary_prefix_timeline'].append(copy.deepcopy(next(r for r in context['ordinary_prefix_timeline'] if 'kalmar_204 posts Ante' in r['raw_ocr_text'])))
    elif mutation=='late_first_number':
        observation=next(o for o in case['raw_sql_observations'] if o['observation_id']==case['retained_numeric_observation_id'])
        observation['frame_index']+=1
    elif mutation=='current_context_rebind':case['source_context']['session_id']='FOREIGN_SESSION'
    before=copy.deepcopy(case)
    result=prove(case,source['database_metadata_raw'])
    assert result['status']=='UNKNOWN',(mutation,result)
    assert result['reason']
    assert case==before

@pytest.mark.parametrize('mutation',['missing_config_receipts','duplicate_metadata'])
def test_complete_frozen_sql_metadata_required(source,mutation):
    metadata=copy.deepcopy(source['database_metadata_raw'])
    if mutation=='missing_config_receipts':
        row=next(r for r in metadata if r['key']=='config_hashes')
        row['value']='{}'
    else:
        metadata.append(copy.deepcopy(metadata[0]))
    result=prove(source['cases'][0],metadata)
    assert result['status']=='UNKNOWN'


@pytest.fixture
def record():
    return json.loads((Path(__file__).parent/'fixtures/joint_stack_hand_source.json').read_text('utf-8'))


def test_current_hand_join_replays_every_native_prefix_row(source,record):
    p=prove(source['cases'][0],source['database_metadata_raw'])
    assert verify_retained(p,record['observed_hand'])
    for mutation in ('foreign_hand','source_hash','source_gap','reorder','foreign_pts','error','conflict'):
        hand=copy.deepcopy(record['observed_hand'])
        if mutation=='foreign_hand':hand['hand_id']='OTHER'
        elif mutation=='source_hash':hand['messages'][0]['source_rows'][0]['row_pixel_sha256']='0'*64
        elif mutation=='source_gap':hand['messages'].pop(3)
        elif mutation=='reorder':hand['messages'][3:5]=reversed(hand['messages'][3:5])
        elif mutation=='foreign_pts':hand['messages'][0]['source_rows'][0].update(source_pts=1,time_base=[1,15360])
        elif mutation=='error':hand['messages'][0]['source_rows'][0]['status']='ERROR'
        else:hand['messages'][0]['source_rows'][0]['canonical_conflict']='TEST'
        assert not verify_retained(p,hand),mutation


def test_hh_consumer_accepts_only_replayed_joint_receipt_and_keeps_full_gate(source,record,tmp_path):
    from parser_core.hh_export import export_hand_histories
    p=prove(source['cases'][0],source['database_metadata_raw']);j=join_receipt(p)
    stack=dict(segment_id=record['segment_id'],status='START_STACK_PROVEN',player=p['actor'],
        seat_id=p['seat'],calculated_start_stack=p['calculated_start_stack'],
        source_observation_id=j['source_observation_ids'][0],source_state_id=j['source_state_ids'][0],
        joint_binding_receipt=j)
    (tmp_path/'stack_start_stack.jsonl').write_text(json.dumps(stack)+'\n','utf-8')
    (tmp_path/'joint_start_stack_proofs.jsonl').write_text(json.dumps(p)+'\n','utf-8')
    summary=export_hand_histories(tmp_path,[record])
    out=json.loads((tmp_path/'hand_histories.jsonl').read_text('utf-8'))
    actor=next(a for a in out['players'] if a['player']==p['actor'])
    assert actor['START_STACK']==75441 and actor['seat']=='seat5'
    assert summary['certified_hh']==0 and 'ALL_PLAYER_SEATS_AND_START_STACKS_NOT_PROVEN' in out['blockers']
    stack['calculated_start_stack']+=1
    (tmp_path/'stack_start_stack.jsonl').write_text(json.dumps(stack)+'\n','utf-8')
    export_hand_histories(tmp_path,[record])
    out=json.loads((tmp_path/'hand_histories.jsonl').read_text('utf-8'))
    assert next(a for a in out['players'] if a['player']==p['actor'])['START_STACK'] is None


def test_solver_join_consumes_only_current_source_proof(source,record):
    from types import SimpleNamespace
    from parser_core.poker_reconstruction_layer import reconstruct_production_hands,reconstruction_config
    from parser_core.full_hh_profile import reconstruction_settings
    proof=prove(source['cases'][0],source['database_metadata_raw'])
    class Sources:
        def for_hand(self,hand):
            return [proof] if verify_retained(proof,hand) else []
    provider=SimpleNamespace(joint_sources=Sources())
    results,_=reconstruct_production_hands([record],config=reconstruction_settings(reconstruction_config()),stack_provider=provider)
    assert results[0]['joined_start_stacks']=={'kalmar_204':75441}
    assert results[0]['joint_start_stack_join_receipts']==[join_receipt(proof)]
    record['observed_hand']['messages'][0]['source_rows'][0].update(source_pts=1,time_base=[1,15360])
    results,_=reconstruct_production_hands([record],config=reconstruction_settings(reconstruction_config()),stack_provider=provider)
    assert results[0]['joined_start_stacks']=={}
