import copy,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from parser_core.hh_text import render_complete
from parser_core.poker_hand_assembly import assemble_messages

@pytest.fixture
def native():return json.loads((Path(__file__).parent/'fixtures/clean_hh_native_source.json').read_text('utf-8'))

def test_native_complete_input_immutable_no_unknown_metadata(native):
    before=copy.deepcopy(native);text=render_complete(native)
    assert native==before
    assert text.count('PokerDom Hand #')==1
    assert not any(t in text for t in ['TABLE_INFO','SourceOCR','blockers','UNKNOWN',' UTC','is the button'])
    assert 'Total pot 5360 | Rake 0' in text

@pytest.mark.parametrize('kind',['TABLE_INFO','HAND_HEADER','HAND_ID','NEW_HAND','BLIND_LEVEL','HAND_END'])
def test_service_messages_do_not_enter_clean_hh(native,kind):
    clean=render_complete(native);native['actions'].insert(2,dict(action=kind,raw_text='SourceOCR audit service',player=None,amount=None,cards=[]))
    before=copy.deepcopy(native);assert render_complete(native)==clean;assert native==before

def test_native_assembled_rebuy_table_info_is_excluded(native):
    rows=json.loads((Path(__file__).parent/'fixtures/service_rebuy_native_source.json').read_text('utf-8'))
    messages=assemble_messages(rows,require_pixel_cards=True,require_adjacency=True)
    assert len(messages)==1 and messages[0]['action']=='TABLE_INFO'
    assert messages[0]['player'] is None and messages[0]['amount'] is None
    clean=render_complete(native);native['actions'].insert(2,messages[0]);before=copy.deepcopy(native)
    assert render_complete(native)==clean and native==before

@pytest.mark.parametrize('kind',['INFO','REBUY'])
def test_arbitrary_non_hh_action_is_explicitly_unsupported(native,kind):
    native['actions'].insert(2,dict(action=kind,raw_text='audit text',player=None,amount=None))
    before=copy.deepcopy(native)
    with pytest.raises(ValueError,match='UNSUPPORTED_POKER_ACTION'):render_complete(native)
    assert native==before

def synthetic():
    return dict(status='COMPLETE',certified=True,blockers=[],table='T01',hand_number='12345',final_pot=600,
        players=[dict(player=p,seat=f'seat{i}',START_STACK=1000,START_STACK_status='PROVEN') for i,p in enumerate(['Alice','Bob'],1)],
        actions=[dict(action='SMALL_BLIND',player='Alice',amount=50),dict(action='BIG_BLIND',player='Bob',amount=100),dict(action='DEAL'),
            dict(action='RAISE',player='Alice',amount=300,amount_proof={'unique':True}),
            dict(action='CALL',player='Bob',amount=200,amount_proof={'unique':True}),
            dict(action='FLOP',cards=['Ac','Kd','2h']),dict(action='BET',player='Alice',amount=100,amount_proof={'unique':True}),
            dict(action='FOLD',player='Bob'),dict(action='UNCALLED_RETURN',player='Alice',amount=100),
            dict(action='WIN',player='Alice',amount=600)])

def test_raise_increment_uses_current_bet_not_own_added_chips():
    h=synthetic();before=copy.deepcopy(h);text=render_complete(h)
    assert 'Alice: raises 200 to 300' in text
    assert 'raises 250 to 300' not in text
    assert 'Bob: calls 200' in text
    assert 'Uncalled bet (100) returned to Alice' in text
    assert h==before

@pytest.mark.parametrize('mutation',['partial','uncertified','blockers','missing_amount_proof','unknown_board','unknown_show','nonzero_rake'])
def test_unproven_or_conflicting_fields_reject_without_mutation(native,mutation):
    if mutation=='partial':native['status']='PARTIAL'
    elif mutation=='uncertified':native['certified']=False
    elif mutation=='blockers':native['blockers']=['UNKNOWN']
    elif mutation=='missing_amount_proof':next(a for a in native['actions'] if a['action']=='CALL').pop('amount_proof',None)
    elif mutation=='unknown_board':next(a for a in native['actions'] if a['action']=='FLOP')['cards']=[]
    elif mutation=='unknown_show':native['actions'].insert(-1,dict(action='SHOW',player=native['players'][0]['player'],cards=[]))
    elif mutation=='nonzero_rake':native['final_pot']+=1
    before=copy.deepcopy(native)
    with pytest.raises(ValueError):render_complete(native)
    assert native==before
