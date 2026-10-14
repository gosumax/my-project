import copy
import json
from pathlib import Path

import pytest

from parser_core.poker_logic_engine import validate_hand


HANDS=json.loads((Path(__file__).parent/'fixtures/show_class_street_sources.json').read_text('utf-8'))


@pytest.mark.parametrize('hand',HANDS,ids=lambda h:h['hand_id'])
def test_native_show_description_uses_its_observed_street(hand):
    before=copy.deepcopy(hand)
    result=validate_hand(hand)
    assert result['card_status']=='PASS'
    assert not any(i['code']=='CARD_LOGIC_CONFLICT' for i in result['issues'])
    assert hand==before


@pytest.mark.parametrize('hand',HANDS,ids=lambda h:h['hand_id'])
def test_wrong_early_show_class_remains_a_conflict(hand):
    hand=copy.deepcopy(hand)
    target=next(m for m in hand['messages'] if m['action']=='SHOW' and m['player'] in {'tolbin3','rinatromanov19'})
    target['showdown_class']='FOUR_OF_A_KIND'
    result=validate_hand(hand)
    assert result['card_status']=='CONFLICT'
    assert any(i['code']=='CARD_LOGIC_CONFLICT' and 'description' in i['detail'] for i in result['issues'])


@pytest.mark.parametrize('hand',HANDS,ids=lambda h:h['hand_id'])
def test_win_still_uses_final_board(hand):
    hand=copy.deepcopy(hand)
    next(m for m in hand['messages'] if m['action']=='WIN')['showdown_class']='HIGH_CARD'
    result=validate_hand(hand)
    assert result['card_status']=='CONFLICT'


@pytest.mark.parametrize('hand',HANDS,ids=lambda h:h['hand_id'])
def test_missing_early_board_is_not_filled_from_future_runout(hand):
    hand=copy.deepcopy(hand)
    first_show=next(n for n,m in enumerate(hand['messages']) if m['action']=='SHOW')
    for m in hand['messages'][:first_show]:
        if m['action'] in {'FLOP','TURN','RIVER'}:m['cards']=[]
    result=validate_hand(hand)
    assert any(i['code']=='CARD_EVIDENCE_MISSING' and 'described class' in i['detail'] for i in result['issues'])


def test_final_sidepot_winner_comparison_still_rejects_wrong_winner():
    hand=copy.deepcopy(next(h for h in HANDS if h['hand_id']=='2878365523'))
    for m in hand['messages']:
        if m['action']=='WIN':m['player']='rinatromanov19';m['showdown_class']=None
    result=validate_hand(hand)
    assert any(i['code']=='CARD_LOGIC_CONFLICT' and 'side-pot winners' in i['detail'] for i in result['issues'])
