import copy,json
from pathlib import Path
import pytest
from parser_core.poker_reconstruction.assembly import candidate_assemblies
from parser_core.poker_reconstruction.timeline import parse_rows
from parser_core.poker_reconstruction.reconstruct import reconstruct_hand
from parser_core.poker_hand_assembly import assemble_messages, observed_wrapped_continuation

@pytest.mark.parametrize('index',[0,1])
def test_native_literal_finished_and_joined_wraps_retain_both_sources(index):
    rows=json.loads((Path(__file__).parent/'fixtures/info_wrap_t08_source.json').read_text('utf-8'))[index]
    before=copy.deepcopy(rows)
    assert observed_wrapped_continuation(rows[:1],rows[1])
    messages=assemble_messages(rows,require_pixel_cards=True,require_adjacency=True)
    assert len(messages)==1 and messages[0]['action']=='TABLE_INFO'
    assert messages[0]['source_rows']==rows and messages[0]['amount'] is None
    assemblies,exhausted,_=candidate_assemblies(rows)
    assert not exhausted and any(len(es)==1 and es[0].kind=='INFO' and es[0].source['rows']==rows for es in assemblies)
    assert rows==before
    rows[1]['frame_id']+=1
    assert not observed_wrapped_continuation(rows[:1],rows[1])
    assert len(assemble_messages(rows,require_pixel_cards=True,require_adjacency=True))==2


def test_literal_wrap_does_not_swallow_financial_suffix():
    rows=json.loads((Path(__file__).parent/'fixtures/info_wrap_t08_source.json').read_text('utf-8'))[0]
    rows[1]['raw_ocr_text']='the tournament calls 100'
    assert not observed_wrapped_continuation(rows[:1],rows[1])

@pytest.fixture(scope='module')
def real():return json.loads((Path(__file__).parent/'fixtures/info_wrap_t05_source.json').read_text('utf-8'))

def reconstruct(rows,hand_id='2878366210'):
    assemblies,exhausted,_=candidate_assemblies(rows)
    result=reconstruct_hand(parse_rows(rows),table='T05',hand_id=hand_id,assemblies=assemblies,
        assembly_exhausted=exhausted,strict_source_uniqueness=True,max_nodes=20000)
    return assemblies,result

def test_real_waiting_notice_has_one_exact_literal_source_group(real):
    rows=copy.deepcopy(real['rows']);before=copy.deepcopy(rows);assemblies,result=reconstruct(rows)
    head=next(r for r in rows if r['raw_ocr_text']=='Dealer: Waiting for iplayupay to')
    tail=next(r for r in rows if r['raw_ocr_text']=='reconnect...')
    assert (head['frame_id'],head['row_index'],tail['frame_id'],tail['row_index'])==(3753,17,3753,18)
    for events in assemblies:
        groups=[e for e in events if head in e.source.get('rows',[])]
        assert len(groups)==1 and groups[0].kind=='INFO' and groups[0].source['rows']==[head,tail]
    assert result['solution_count']==1 and not result['exhausted'] and result['status']=='AUTO_RECOVERED'
    assert rows==before

@pytest.mark.parametrize('mutation',['frame','row_gap','session'])
def test_invalid_adjacency_does_not_gain_literal_wrap_authority(real,mutation):
    rows=copy.deepcopy(real['rows']);head=next(r for r in rows if r['raw_ocr_text']=='Dealer: Waiting for iplayupay to');tail=next(r for r in rows if r['raw_ocr_text']=='reconnect...')
    if mutation=='frame':tail['frame_id']+=1
    elif mutation=='row_gap':tail['row_index']+=1
    else:tail['session_id']='DIFFERENT_SESSION'
    assemblies,_=reconstruct(rows)
    assert not any(head in e.source.get('rows',[]) and tail in e.source.get('rows',[]) for events in assemblies for e in events)


def test_real_opaque_t07_tail_still_withholds_reconstruction():
    data=json.loads((Path(__file__).parent/'fixtures/poker_math_smoke19_source.json').read_text('utf-8'))['hands']['2878367230']
    rows=[copy.deepcopy(r) for m in data['messages'] for r in m['source_rows']]
    for n,r in enumerate(rows):r['source_index']=n
    before=copy.deepcopy(rows);assemblies,result=reconstruct(rows,'2878367230')
    assert any(e.kind=='UNKNOWN' and e.source['raw_text']=='0+03030K' for events in assemblies for e in events)
    assert result['solution_count']==0 and not result['kept_events'] and not result['exhausted']
    assert rows==before
