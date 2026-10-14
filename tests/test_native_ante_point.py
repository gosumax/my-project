import copy,json
from pathlib import Path
import pytest
from parser_core.actor_seat_point import ante_candidates,ANTE

@pytest.fixture
def native():return json.loads((Path(__file__).parent/'fixtures/native_ante_point_sources.json').read_text('utf-8'))

@pytest.mark.parametrize('index',range(5))
def test_five_native_wraps_keep_both_sources(native,index):
    rows=[p['row'] for p in native['cases'][index]['source_pair']];before=copy.deepcopy(rows)
    result=ante_candidates(rows)
    assert len(result)==1 and result[0]['reason'] is None
    assert result[0]['actor']=='ubivashka88888' and result[0]['amount']==625
    assert result[0]['source_rows']==rows and rows==before

@pytest.mark.parametrize('index',range(5))
@pytest.mark.parametrize('mutation',['table','session','frame','index','epoch','hand','pts','time_base','head_confidence','tail_confidence','bool_confidence','nan_confidence','tail_type','missing_tail','video_time'])
def test_broken_wrap_remains_explicit_rejected_candidate(native,index,mutation):
    rows=copy.deepcopy([p['row'] for p in native['cases'][index]['source_pair']]);head,tail=rows
    if mutation=='table':tail['table_id']='T99'
    elif mutation=='session':tail['session_id']='foreign'
    elif mutation=='frame':tail['frame_id']+=1
    elif mutation=='index':tail['row_index']+=1
    elif mutation=='epoch':tail['epoch']+=1
    elif mutation=='hand':tail['hand_id']='foreign-hand'
    elif mutation=='pts':tail['source_pts']=999999
    elif mutation=='time_base':tail['time_base']=[1,999]
    elif mutation=='head_confidence':head['confidence']=.94
    elif mutation=='tail_confidence':tail['confidence']=.94
    elif mutation=='bool_confidence':head['confidence']=True
    elif mutation=='nan_confidence':tail['confidence']=float('nan')
    elif mutation=='tail_type':tail['raw_ocr_text']='625 chips'
    elif mutation=='missing_tail':rows.pop()
    elif mutation=='video_time':tail['video_time']+=1
    result=ante_candidates(rows)
    assert len(result)==1 and result[0]['actor']=='ubivashka88888'
    assert result[0]['reason'] in {'OWN_ANTE_AMOUNT_WRAP_UNPROVEN','OWN_ANTE_SOURCE_CONFIDENCE_UNKNOWN'}

def test_duplicate_own_ante_inventory_is_not_silently_deduplicated(native):
    rows=[p['row'] for p in native['cases'][0]['source_pair']];result=ante_candidates(rows+copy.deepcopy(rows))
    assert len(result)==2 and all(p['reason'] is None for p in result)
    # Literal inventory is not a seat/stack proof; duplicate rejection belongs
    # to native source/own-post validation.
    assert all(set(p)=={'actor','source_rows','reason','amount'} for p in result)

@pytest.mark.parametrize('marker',['Hand #2878365529','Starting a new hand.'])
def test_head_marker_is_never_numeric_tail(native,marker):
    rows=copy.deepcopy([p['row'] for p in native['cases'][0]['source_pair']]);rows[1]['raw_ocr_text']=marker
    result=ante_candidates(rows);assert len(result)==1 and result[0]['reason']=='OWN_ANTE_AMOUNT_WRAP_UNPROVEN'

def test_legacy_single_line_keeps_original_source_shape(native):
    row=copy.deepcopy(native['cases'][0]['source_pair'][0]['row']);row['raw_ocr_text']+=' 625';before=copy.deepcopy(row)
    old=ANTE.fullmatch(row['raw_ocr_text']);result=ante_candidates([row])
    assert result==[dict(actor=old[1],source_rows=[row],reason=None,amount=int(old[2]))]
    assert row==before and len(result[0]['source_rows'])==1
