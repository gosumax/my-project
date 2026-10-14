"""User-approved forced-prefix gate; synthetic mutations are negative controls."""
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from parser_core.seat_forced_binding import parse_forced_snapshot

HAND='12345678'
ROOT=Path(__file__).resolve().parents[1]


def snapshot(kind='plain'):
    lines=[f'Hand #{HAND}','Dealer: Starting a new hand',
           'Dealer: p0 posts Ante 100','Dealer: p1 posts Ante 100',
           'Dealer: p0 posts Small Blind 500','Dealer: p1 posts Big Blind 1 000',
           'Dealer: Dealing cards']
    if kind=='amount_wrap':lines[4:5]=['Dealer: p0 posts Small Blind','500']
    if kind=='small_wrap':lines[4:5]=['Dealer: p0 posts Small','Blind 500']
    if kind=='big_wrap':lines[5:6]=['Dealer: p1 posts Big','Blind 1 000']
    if kind=='state':lines.insert(4,'Dealer: p0 sits out')
    if kind=='allin_wrap':lines.insert(3,'and is All-in')
    return [dict(index=i,text=t,score=.99,source_ref=f'pixel{i}',source_frame=20,
                 table_id=1,session_id='session',hand_id=HAND,epoch=0) for i,t in enumerate(lines)]


GATES=[('plain',0),('plain',1),('plain',2),('plain',3),('plain',4),('plain',5),('plain',6),
       ('amount_wrap',5),('small_wrap',4),('small_wrap',5),('big_wrap',5),('big_wrap',6),
       ('state',4),('allin_wrap',3)]


@pytest.mark.parametrize('kind,index',GATES)
def test_exact_user_boundary_and_old_default(kind,index):
    rows=snapshot(kind);rows[index]['score']=.93
    before=copy.deepcopy(rows)
    assert parse_forced_snapshot(rows,HAND,2,confidence=.93)[0] is not None
    assert parse_forced_snapshot(rows,HAND,2)[0] is None
    assert rows==before
    rows[index]['score']=math.nextafter(.93,0)
    assert parse_forced_snapshot(rows,HAND,2,confidence=.93)[0] is None


INVALID=[float('nan'),float('inf'),-float('inf'),True,False,None,'0.93',-.1,1.1]


@pytest.mark.parametrize('score',INVALID)
@pytest.mark.parametrize('index',[0,2,6])
def test_invalid_row_score_cannot_gain_authority(score,index):
    rows=snapshot();rows[index]['score']=score
    assert parse_forced_snapshot(rows,HAND,2,confidence=.93)[0] is None


@pytest.mark.parametrize('threshold',INVALID)
def test_invalid_requested_threshold_fails_closed(threshold):
    assert parse_forced_snapshot(snapshot(),HAND,2,confidence=threshold)[0] is None


@pytest.mark.parametrize('mutation',[
    'empty','garbage','malformed_post','duplicate_post','foreign_header','foreign_frame',
    'conflicting_role','second_deal','missing_deal','voluntary','missing_ante',
    'foreign_wrap_session','foreign_wrap_hand','foreign_wrap_epoch','foreign_wrap_table','nonadjacent_wrap','zero_post',
])
def test_low_confidence_policy_keeps_source_and_grammar_guards(mutation):
    rows=snapshot('amount_wrap' if 'wrap' in mutation else 'plain')
    for r in rows:r['score']=.93
    if mutation=='empty':rows=[]
    elif mutation=='garbage':rows.insert(4,dict(rows[3],index=3.5,text='unrecognized garbage'))
    elif mutation=='malformed_post':rows[2]['text']='Dealer: p0 posts Ante bananas'
    elif mutation=='duplicate_post':rows.insert(3,dict(rows[2],index=2.5))
    elif mutation=='foreign_header':rows.insert(4,dict(rows[0],index=3.5,text='Hand #12345679'))
    elif mutation=='foreign_frame':rows[2]['source_frame']=21
    elif mutation=='conflicting_role':rows[5]['text']='Dealer: p0 posts Big Blind 1 000'
    elif mutation=='second_deal':rows.append(dict(rows[-1],index=50))
    elif mutation=='missing_deal':rows.pop()
    elif mutation=='voluntary':rows.insert(4,dict(rows[3],index=3.5,text='Dealer: p0 calls 100'))
    elif mutation=='missing_ante':rows.pop(2)
    elif mutation=='foreign_wrap_session':rows[5]['session_id']='FOREIGN'
    elif mutation=='foreign_wrap_hand':rows[5]['hand_id']='OTHER'
    elif mutation=='foreign_wrap_epoch':rows[5]['epoch']=1
    elif mutation=='foreign_wrap_table':rows[5]['table_id']=2
    elif mutation=='nonadjacent_wrap':rows[5]['index']=5.5
    elif mutation=='zero_post':rows[2]['text']='Dealer: p0 posts Ante 0'
    assert parse_forced_snapshot(rows,HAND,2,confidence=.93)[0] is None,mutation


def test_legacy_small_blind_tail_gate_remains_094_at_default():
    rows=snapshot('small_wrap');rows[5]['score']=.94
    assert parse_forced_snapshot(rows,HAND,2)[0] is not None
    rows[5]['score']=math.nextafter(.94,0)
    assert parse_forced_snapshot(rows,HAND,2)[0] is None


def test_declared_shared_policy_has_user_boundary():
    from parser_core.joint_start_stack import FORCED_PREFIX_OCR_POLICY
    assert FORCED_PREFIX_OCR_POLICY['min_confidence']==.93
    assert FORCED_PREFIX_OCR_POLICY['small_blind_tail_min_confidence']==.93
    assert FORCED_PREFIX_OCR_POLICY['policy_id']=='USER_APPROVED_FORCED_PREFIX_OCR_093_20261008_V1'


@pytest.mark.parametrize('case_index',[0,6])
def test_unchanged_native_representative_prefix_old_new(case_index):
    from parser_core.actor_seat_point import source_rows
    from parser_core.joint_start_stack import own_posts
    folder=ROOT/'runs/hh95_evidence_20261008/joint_failure_probe_20261008'
    path=folder/f'case_{case_index:02d}.json'
    receipt=json.loads((folder/'cases.json').read_text('utf-8'))[case_index]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']
    case=json.loads(path.read_text('utf-8'))['case'];context=case['source_context']
    sources=source_rows(context)
    rows=[dict(table_id=context['table_id'],session_id=context['session_id'],hand_id=context['hand_id'],
               epoch=r['epoch'],source_frame=p['frame_id'],index=r['row_index'],source_ref=r['row_pixel_sha256'],
               text=r['raw_ocr_text'],score=r['confidence']) for r,p,_,_ in sources]
    occupied=len({r['text'].split(' posts Ante')[0] for r in rows if ' posts Ante' in r['text']})
    before=copy.deepcopy(rows)
    assert parse_forced_snapshot(rows,context['hand_id'],occupied)[0] is None
    new,reason=parse_forced_snapshot(rows,context['hand_id'],occupied,confidence=.93)
    assert new is not None,reason
    posts=own_posts(context,case['actor'])
    assert posts and all(p['actor']==case['actor'] for p in posts)
    assert rows==before
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']
