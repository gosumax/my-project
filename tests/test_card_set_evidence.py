import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.card_set_evidence import prove, _ref, project_unique_card_sets, verify_current, apply_to_hand
from parser_core.poker_hand_assembly import make_message

SOURCES = json.loads((ROOT/'tests/fixtures/card_set_unique_native_sources.json').read_text('utf-8'))
PROJECTIONS=json.loads((ROOT/'tests/fixtures/card_set_projection_native_sources.json').read_text('utf-8'))
DERIVED_PROJECTIONS=[source for source in PROJECTIONS if source.get('expected_authority')!='NATIVE_PIXEL_WIN_BEST_FIVE']
NATIVE_PROJECTIONS=[source for source in PROJECTIONS if source.get('expected_authority')=='NATIVE_PIXEL_WIN_BEST_FIVE']


@pytest.mark.parametrize('index,cards,size', [(0, ['9h','9s','8d','8s','Ks'],4), (1, ['Td','9c'],13)])
def test_native_source_has_one_complete_assignment_without_rewriting_raw(index,cards,size):
    case=copy.deepcopy(SOURCES[index]);before=copy.deepcopy(case)
    result=prove(case)
    assert result['status']=='PROVEN',result
    assert result['cards']==cards and result['candidate_domain_size']==size
    assert result['surviving_candidates']==[cards]
    assert result['field_source']=='DERIVED_UNIQUE_CARD_SET'
    assert result['raw_recognizer_status']=='UNRESOLVED' and case==before
    assert case['target']['reading']['hybrid_authoritative_rejection'] is True


@pytest.mark.parametrize('index',[0,1])
@pytest.mark.parametrize('mutation', ['foreign_hand','foreign_actor','board_role','board_street','board_actor',
    'target_hash','target_png','tile_hash','response_hash','frame','epoch','rank_confidence',
    'duplicate_component','cyclic_source','missing_support','unknown_support','accepted_component_conflict'])
def test_missing_conflicting_or_unbound_support_cannot_authorize_cards(index,mutation):
    case=copy.deepcopy(SOURCES[index]);target=case['target'];reading=target['reading'];tile=reading['hybrid']['tiles'][0]
    if mutation=='foreign_hand':reading['hand_id']='OTHER'
    elif mutation=='foreign_actor':
        source=case['hole'] if index==0 else target
        source['reading']['player']='OTHER'
    elif mutation=='board_role':case['board']['reading']['type']='SHOWDOWN'
    elif mutation=='board_street':case['board']['reading']['street']='FLOP'
    elif mutation=='board_actor':case['board']['reading']['player']='OTHER'
    elif mutation=='target_hash':reading['row_pixel_sha256']='0'*64
    elif mutation=='target_png':reading['crop_png_base64']='AAAA'
    elif mutation=='tile_hash':tile['tile_pixel_sha256']='0'*64
    elif mutation=='response_hash':reading['raw_pixel_response']['source_sha256']='0'*64
    elif mutation=='frame':reading['frame_id']+=1
    elif mutation=='epoch':reading['epoch']+=1
    elif mutation=='rank_confidence':
        t=case['board']['reading']['hybrid']['tiles'][2 if index==0 else 1]
        t['rank_source']='FROZEN_RANK_NCC_PIXEL_FALLBACK';t['rank_status']='PIXEL_RANK_FALLBACK'
        t['rank_ncc']={'rank_accepted':False}
    elif mutation=='duplicate_component':tile['card']='9c'
    elif mutation=='cyclic_source':
        case['hole' if index==0 else 'winning']=copy.deepcopy(target)
    elif mutation=='missing_support':case.pop('hole' if index==0 else 'winning')
    elif mutation=='unknown_support':case['board']['reading']['status']='UNRESOLVED'
    elif mutation=='accepted_component_conflict':
        reading['hybrid']['tiles'][-1]['suit_status']='SUIT_CONFLICT'
    assert prove(case)['status']=='UNKNOWN',mutation


def test_literal_pair_alone_does_not_repair_the_hole_rank():
    case=copy.deepcopy(SOURCES[1]);case.pop('winning')
    assert prove(case)['status']=='UNKNOWN'


def projected_hand(source):
    rows=copy.deepcopy(source['rows'])
    assert project_unique_card_sets(rows,source['readings'])==1
    head=next(r for r in rows if r.get('unique_card_set_proof'));proof=head['unique_card_set_proof']
    messages=[];i=0
    while i<len(rows):
        width=len(proof['target_refs']) if rows[i] is head else 1
        messages.append(make_message(rows[i:i+width],require_pixel_cards=True));i+=width
    return dict(table_id=head['table_id'],session_id=head['session_id'],hand_id=source['hand_id'],messages=messages),proof


@pytest.mark.parametrize('source',DERIVED_PROJECTIONS,ids=lambda s:s['hand_id'])
def test_actual_factory_and_full_hand_consumer_preserve_raw_and_singleton_has_no_authority(source):
    before=copy.deepcopy(source);hand,proof=projected_hand(source)
    message=next(m for m in hand['messages'] if m['source_rows'][0].get('unique_card_set_proof'))
    assert message['cards']==[] and not message.get('win_best_five_cards')
    apply_to_hand(hand)
    field='cards' if source['expected_role']=='SHOW_HOLE_CARDS' else 'win_best_five_cards'
    assert message[field]==source['expected_cards']
    assert source==before


@pytest.mark.parametrize('index',range(len(DERIVED_PROJECTIONS)))
@pytest.mark.parametrize('mutation',['header','missing_source','source_order','foreign_pts','error',
    'new_accepted_target','support_projection','removed_proof','foreign_hand'])
def test_current_conflict_revokes_previously_assigned_derived_authority(index,mutation):
    source=DERIVED_PROJECTIONS[index];hand,proof=projected_hand(source);apply_to_hand(hand)
    target=next(m for m in hand['messages'] if m['source_rows'][0].get('unique_card_set_proof'))
    rows=[r for m in hand['messages'] for r in m['source_rows']]
    assert verify_current(proof,rows)
    if mutation=='header':rows[0]['raw_ocr_text']='#123'
    elif mutation=='missing_source':hand['messages'].pop(1)
    elif mutation=='source_order':hand['messages'][1:3]=reversed(hand['messages'][1:3])
    elif mutation=='foreign_pts':rows[0].update(source_pts=1,time_base=[1,15360])
    elif mutation=='error':rows[1]['status']='ERROR'
    elif mutation=='new_accepted_target':target['source_rows'][0].update(cards=['Ah','Ad'],evidence_status='RAW_UNREVIEWED')
    elif mutation=='support_projection':
        board_id=proof['source_case']['board']['rows'][0]['stable_row_id']
        next(r for r in rows if r['stable_row_id']==board_id)['cards']=['Ah']*5
    elif mutation=='removed_proof':target['source_rows'][0].pop('unique_card_set_proof')
    elif mutation=='foreign_hand':hand['hand_id']='OTHER'
    apply_to_hand(hand)
    assert target['cards']==[] and not target.get('win_best_five_cards')
    assert target.get('field_sources',{}).get('cards')!='DERIVED_UNIQUE_CARD_SET'
    assert target.get('field_sources',{}).get('win_best_five')!='DERIVED_UNIQUE_CARD_SET'


@pytest.mark.parametrize('source',NATIVE_PROJECTIONS,ids=lambda s:s['hand_id'])
@pytest.mark.parametrize('mutation',[None,'source_hash','foreign_hand','missing_source'])
def test_new_native_win_acceptance_uses_current_pixel_proof_and_retains_guards(source,mutation):
    from parser_core.win_tail_evidence import project_win_tails,verified_win_tail
    rows=copy.deepcopy(source['rows']);readings=copy.deepcopy(source['readings'])
    assert project_win_tails(rows,readings)>=1
    assert project_unique_card_sets(rows,readings)==0
    head=next(r for r in rows if (r.get('win_best_five_proof') or {}).get('cards')==source['expected_cards'])
    proof=head['win_best_five_proof'];by_id={r['stable_row_id']:r for r in rows}
    group=[by_id[ref['stable_row_id']] for ref in proof['source_refs']]
    if mutation=='source_hash':group[-1]['row_pixel_sha256']='0'*64
    elif mutation=='foreign_hand':proof['reading']['hand_id']='OTHER'
    elif mutation=='missing_source':group.pop()
    if mutation:
        assert verified_win_tail(group) is None
    else:
        assert verified_win_tail(group)==proof
        message=make_message(group,require_pixel_cards=True)
        assert message['cards']==[]
        assert message['win_best_five_cards']==source['expected_cards']
        assert message.get('field_sources',{}).get('win_best_five')!='DERIVED_UNIQUE_CARD_SET'
