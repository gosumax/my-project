import copy
import hashlib

import pytest

from experiments.attribute_edge_hands_and_window_simulation import assemble_fixed_windows, evaluate
from experiments.validated_relationship_edge_simulation import simulate
from parser_core.edge_gate import classify_edges
from parser_core.poker_hand_assembly import assemble_hands
from parser_core.poker_logic_engine import validate_hand


def rows(texts):
    return [dict(table_id='T01',session_id='s',stable_row_id=f's:R{i}',sequence_id=i+1,
        frame_id=i,row_index=0,epoch=0,raw_ocr_text=text,status='RAW_UNREVIEWED',
        notes_error='',origin='TRACKED',row_pixel_sha256=hashlib.sha256(text.encode()).hexdigest())
        for i,text in enumerate(texts)]


def complete():
    return ['A posts Small Blind 5','B posts Big Blind 10','Dealing cards',
            'A folds','B gets uncalled bet 5 back','B wins 10']


def test_fixed_windows_stop_cross_hand_concat_without_new_events():
    source=rows(complete()+complete());before=copy.deepcopy(source)
    frozen=classify_edges(source);sidecar=simulate(source,[],frozen)
    assert len(assemble_hands(frozen.rows))==1
    hands=assemble_fixed_windows(frozen.rows,sidecar.attribution)
    assert len(hands)==2 and all(validate_hand(h)['status']=='FULLY_VALID_HAND' for h in hands)
    assert source==before
    assert [rid for h in hands for m in h['messages'] for rid in m['source_row_ids']]==[r['stable_row_id'] for r in source]
    assert all(m['action']!='NEW_HAND' for h in hands for m in h['messages'])


def test_fixed_window_cannot_invent_missing_payout():
    source=rows(complete()+['A posts Small Blind 5','B posts Big Blind 10','Dealing cards',
                         'A folds']+complete())
    sidecar=simulate(source,[])
    hands=assemble_fixed_windows(sidecar.rows,sidecar.attribution)
    assert len(hands)==3
    result=validate_hand(hands[1])
    assert hands[1]['end_observed'] is False
    assert result['status']=='INCOMPLETE_END'
    assert result['status']!='FULLY_VALID_HAND'


def test_confirmed_start_row_must_remain_admitted():
    source=rows(complete());sidecar=simulate(source,[])
    sidecar.rows[0]['edge_status']='UNRESOLVED_EDGE'
    with pytest.raises(ValueError,match='Missing confirmed start'):
        assemble_fixed_windows(sidecar.rows,sidecar.attribution)


def test_keep_without_confirmed_interval_fails_closed():
    source=rows(complete());sidecar=simulate(source,[])
    sidecar.attribution[0]['hand_interval']=None
    with pytest.raises(ValueError,match='KEEP without a confirmed interval'):
        assemble_fixed_windows(sidecar.rows,sidecar.attribution)


def test_no_unknown_epoch_window_is_admitted():
    source=rows(complete());source[-1]['epoch']=1
    sidecar=simulate(source,[])
    assert assemble_fixed_windows(sidecar.rows,sidecar.attribution)==[]


def test_attribution_keeps_completeness_with_action_blockers():
    source=rows(['A calls 5','B wins 10'])
    hands=assemble_hands(source)
    records,_=evaluate(hands,{r['stable_row_id']:'NO_CONFIRMED_START' for r in source})
    codes={i['code'] for i in records[0]['logic_blocking_checks']}
    assert 'INCOMPLETE_START' in codes
    assert 'CALL_WITH_NOTHING_TO_CALL' in codes


def test_interleaved_tables_and_sessions_never_share_windows():
    first=rows(complete());second=rows(complete());third=rows(complete())
    for row in second:
        row.update(table_id='T02',session_id='s2',stable_row_id='s2:'+row['stable_row_id'])
    for row in third:
        row.update(session_id='s3',stable_row_id='s3:'+row['stable_row_id'])
    source=[r for pair in zip(first,second) for r in pair]+third
    sidecar=simulate(source,[])
    hands=assemble_fixed_windows(sidecar.rows,sidecar.attribution)
    assert {(h['table_id'],h['session_id']) for h in hands}=={('T01','s'),('T02','s2'),('T01','s3')}
    assert all(validate_hand(h)['status']=='FULLY_VALID_HAND' for h in hands)
