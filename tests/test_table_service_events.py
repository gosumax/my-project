import copy
import json
from pathlib import Path

import pytest

from parser_core.table_service_events import describe_service, service_view
from parser_core.poker_hand_assembly import assemble_messages, observed_wrapped_continuation
from parser_core.poker_reconstruction.assembly import candidate_assemblies

ROWS = json.loads((Path(__file__).parent/'fixtures/service_rebuy_native_source.json').read_text('utf-8'))


def hand(messages):
    return dict(table_id=ROWS[0]['table_id'], session_id=ROWS[0]['session_id'],
                hand_id='2878366772', messages=messages)


def test_native_rebuy_is_one_service_group_without_pot_or_player_authority():
    rows=copy.deepcopy(ROWS); before=copy.deepcopy(rows)
    messages=assemble_messages(rows,require_pixel_cards=True,require_adjacency=True)
    assert len(messages)==1 and messages[0]['action']=='TABLE_INFO'
    assert messages[0]['player'] is None and messages[0]['amount'] is None
    assert messages[0]['source_rows']==rows
    events,exhausted,_=candidate_assemblies(rows)
    assert not exhausted and len(events)==1 and len(events[0])==1
    event=events[0][0]
    assert event.kind=='INFO' and event.player is None and event.amount is None
    assert event.source['service_event']==dict(kind='REBUY',actor='gekcon555',amount=150000,stack_transfer=True)
    assert event.source['rows']==rows and rows==before


@pytest.mark.parametrize('mutation',['frame','row_gap','table','session','epoch'])
def test_rebuy_wrap_cannot_cross_source_context(mutation):
    rows=copy.deepcopy(ROWS)
    key={'frame':'frame_id','row_gap':'row_index','table':'table_id','session':'session_id','epoch':'epoch'}[mutation]
    rows[1][key]=rows[1][key]+1 if isinstance(rows[1][key],int) else 'OTHER'
    assert not observed_wrapped_continuation(rows[:1],rows[1])
    assert len(assemble_messages(rows,require_pixel_cards=True,require_adjacency=True))==2


@pytest.mark.parametrize('text',[
    'gekcon555 makes rebuy(s) and', 'gets 150 000 chips',
    'gekcon555 makes rebuy(s) and gets 150 000 chips calls 200',
    'gekcon555 makes rebuy(s) and gets 150 000 chips0',
    'gekcon555 makes rebuy(s) and gets 15 00 chips',
    'gekcon555 makes rebuy(s) and gets 0 chips',
    'Hand #2878367000 makes rebuy(s) and gets 150 000 chips',
    'Hand#2878367000 makes rebuy(s) and gets 150 000 chips',
    'Dealer: Alice calls 200 makes rebuy(s) and gets 150 000 chips',
    'rebuy...0', 'is All-in0', 'Alice wins 150 000 from the main pot'])
def test_incomplete_or_financial_text_is_not_silently_service(text):
    assert describe_service(text) is None


def test_service_grammar_preserves_reserved_boundary_and_financial_barriers():
    from parser_core.normalize_chat import normalize
    from parser_core.poker_reconstruction.timeline import parse_text
    header='Hand #2878367000 makes rebuy(s) and gets 150 000 chips'
    assert normalize([header]).event_type=='HAND_HEADER'
    assert parse_text(header,{}).kind=='START_HAND'
    financial='Dealer: Alice calls 200 makes rebuy(s) and gets 150 000 chips'
    assert normalize([financial]).event_type!='TABLE_INFO'
    assert parse_text(financial,{}).kind!='INFO'


def test_exact_source_reobservation_is_coalesced_but_all_raw_receipts_survive():
    messages=assemble_messages(copy.deepcopy(ROWS),require_pixel_cards=True,require_adjacency=True)
    original=hand(messages+copy.deepcopy(messages));before=copy.deepcopy(original)
    view=service_view([original])
    assert len(view['events'])==1 and view['exact_source_reobservations']==1
    assert len(view['events'][0]['observations'])==2
    assert not view['stack_authority'] and not view['seat_authority'] and not view['complete_roster_proven']
    assert original==before


def test_same_rebuy_text_at_a_second_physical_source_is_a_second_transfer():
    first=assemble_messages(copy.deepcopy(ROWS),require_pixel_cards=True,require_adjacency=True)[0]
    second=copy.deepcopy(first)
    for row in second['source_rows']:
        row['frame_id']+=50;row['stable_row_id']+=':SECOND'
    view=service_view([hand([first,second])])
    assert len(view['events'])==2 and view['exact_source_reobservations']==0
    assert [e['observation']['amount'] for e in view['events']]==[150000,150000]


def test_conflicting_fields_at_same_source_are_not_deduplicated():
    first=assemble_messages(copy.deepcopy(ROWS),require_pixel_cards=True,require_adjacency=True)[0]
    second=copy.deepcopy(first);second['source_rows'][0]['confidence']=.5
    with pytest.raises(ValueError,match='SERVICE_SAME_SOURCE_FIELDS_CONFLICT'):
        service_view([hand([first,second])])


def test_join_leave_join_preserves_real_roster_transitions_and_idempotent_confirmation():
    messages=[]
    for i,text in enumerate(['Alice has joined the table','Alice has joined the table',
                             'Alice has left the table','Alice has joined the table']):
        row=copy.deepcopy(ROWS[0]);row['frame_id']+=i;row['stable_row_id']+=':'+str(i)
        row['raw_ocr_text']='Dealer: '+text
        messages.append(dict(raw_text=row['raw_ocr_text'],source_rows=[row]))
    view=service_view([hand(messages)])
    assert [e['roster_effect'] for e in view['events']]==[
        'OBSERVED_STATE_TRANSITION','CONFIRMS_SAME_OBSERVED_STATE',
        'OBSERVED_STATE_TRANSITION','OBSERVED_STATE_TRANSITION']
    assert len(view['events'])==4 and view['exact_source_reobservations']==0
    assert view['observed_presence'][0]['state']=='PRESENT'
