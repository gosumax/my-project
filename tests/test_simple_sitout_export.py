"""Current saved SIT OUT classification through the unchanged HH gates."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import tempfile
from sitout_status_fixture import status_source

from parser_core.hh_export import export_hand_histories
from parser_core.hh_text import render_complete, SIMPLE_SITOUT_CONTRACT
from parser_core.simple_stack_policy import POLICY, STACK_NUMERIC_CONFIDENCE_MIN
from parser_core.simple_saved_policy import SIT_OUT_GATE_AUTHORITY
from parser_core.simple_seat_who import SeatIdentityLineage
from test_hh_sitout_contract import example, record_for


def state():
    directory=tempfile.mkdtemp(prefix='sitout-native-test-')
    return dict(table_id=1, session_id='s', hand_id='123456', player='A', seat='seat1',
        policy=POLICY, WHO_status='PROVEN', player_status='SIT_OUT',
        sitout_authority=SIT_OUT_GATE_AUTHORITY,
        reason='NORMAL_SIT_OUT_WITHOUT_NUMERIC', DISPLAYED_STACK=None, START_STACK=None,
        first_numeric_source=None, hand_start=100, hand_end=200,
        sitout_sources=[status_source(directory)])


def exported(tmp_path, current, *, provider_present=True, who_proven=True, other_blockers=()):
    record=record_for(example())
    numeric=dict(status='PROVEN', rule=POLICY, actor='B', seat='seat2',
                 calculated_start_stack=1000)
    identity=[dict(actor='A',seat='seat1')] if who_proven else []
    actor_provider=SimpleNamespace(for_hand=lambda hand:[*identity,dict(actor='B',seat='seat2')])
    simple=SimpleNamespace(for_hand=lambda hand:[numeric], seat_rows=lambda hand:[current],
        actor_sources=SimpleNamespace(canonical_hand=lambda hand:copy.deepcopy(hand)))
    stacks=[dict(segment_id=record['segment_id'],player='A',seat_id='seat1',table_id='T01',
        session_id='s',hand_id='123456',status='START_STACK_UNKNOWN',calculated_start_stack=None,
        DISPLAYED_STACK=None,observed_stack=None,player_status='SIT_OUT'),
        dict(segment_id=record['segment_id'],player='B',seat_id='seat2',table_id='T01',
        session_id='s',hand_id='123456',status='START_STACK_PROVEN',calculated_start_stack=1000,
        simple_binding_receipt=numeric)]
    recon=dict(segment_id=record['segment_id'],output_eligible=not other_blockers,
               output_blockers=list(other_blockers),payout_eligible=True,calculated_pot=10)
    for name,rows in [('stack_start_stack.jsonl',stacks),('poker_reconstruction.jsonl',[recon])]:
        (tmp_path/name).write_text(''.join(json.dumps(r)+'\n' for r in rows),'utf8')
    export_hand_histories(tmp_path,[record],actor_provider=actor_provider,
                         simple_provider=simple if provider_present else None)
    return json.loads((tmp_path/'hand_histories.jsonl').read_text('utf8'))


def test_valid_current_sitout_passes_export_without_legacy_receipt_and_preserves_occupant(tmp_path):
    current=state();before=copy.deepcopy(current)
    hand=exported(tmp_path,current)
    assert hand['certified'] and hand['blockers']==[]
    assert hand['hh_contract']==SIMPLE_SITOUT_CONTRACT
    player=hand['players'][0]
    assert (player['player'],player['seat'],player['ACTOR_status'])==('A','seat1','PROVEN')
    assert player['START_STACK'] is None and player['START_STACK_status']=='UNKNOWN'
    assert 'sitout_binding_receipt' not in player
    assert hand['evidence_axes']['start_stacks']['status']=='UNKNOWN'
    text=render_complete(hand)
    assert 'Seat 1: A (sitting out)' in text and '(0 in chips)' not in text
    assert hand['observed_actions']==hand['actions'] and current==before
    assert STACK_NUMERIC_CONFIDENCE_MIN==.85


def test_current_sitout_does_not_bypass_other_export_blockers(tmp_path):
    hand=exported(tmp_path,state(),other_blockers=['EXISTING_PRODUCTION_VALIDATION_UNRESOLVED'])
    assert not hand['certified']
    assert 'ALL_PLAYER_SEATS_AND_START_STACKS_NOT_PROVEN' not in hand['blockers']
    assert 'EXISTING_PRODUCTION_VALIDATION_UNRESOLVED' in hand['blockers']


def test_mixed_current_and_legacy_player_contracts_render_without_weakening_legacy_checks():
    hand=example();legacy=copy.deepcopy(hand['players'][0])
    legacy['player']='C';legacy['seat']='seat3'
    legacy['sitout_binding_receipt'].update(actor='C',seat='seat3')
    hand['players'].append(legacy)
    hand['actions'].insert(3,dict(action='FOLD',player='C'))
    current=hand['players'][0]
    current.pop('sitout_binding_receipt')
    current.update(ACTOR_status='PROVEN',DISPLAYED_STACK=None,simple_sitout_state=state())
    hand['hh_contract']=SIMPLE_SITOUT_CONTRACT
    assert 'Seat 3: C (sitting out)' in render_complete(hand)
    hand['players'][2]['sitout_binding_receipt']['synthetic_fold_authority']=True
    with pytest.raises(ValueError,match='PROVEN_SEAT_AND_STACK_REQUIRED'):render_complete(hand)


@pytest.mark.parametrize('mutation',['no_provider','unknown_who','no_identity','empty','replacement',
    'no_source','unproven_status','foreign_table','foreign_session','foreign_hand','foreign_seat',
    'foreign_player','outside_hand','numeric','no_bounds','no_policy','no_authority','classifier_unknown'])
def test_invalid_or_unproven_status_is_not_fabricated(tmp_path,mutation):
    current=state()
    if mutation=='unknown_who':current['WHO_status']='UNKNOWN'
    elif mutation=='empty':current['player_status']='EMPTY'
    elif mutation=='replacement':current['player']='replacement'
    elif mutation=='no_source':current['sitout_sources']=[]
    elif mutation=='unproven_status':current['sitout_sources'][0]['status']='UNKNOWN'
    elif mutation=='foreign_table':current['table_id']=2
    elif mutation=='foreign_session':current['session_id']='other'
    elif mutation=='foreign_hand':current['hand_id']='654321'
    elif mutation=='foreign_seat':current['seat']='seat2'
    elif mutation=='foreign_player':current['player']='B'
    elif mutation=='outside_hand':current['sitout_sources'][0]['frame']=201
    elif mutation=='numeric':current['DISPLAYED_STACK']=0
    elif mutation=='no_bounds':current.pop('hand_start')
    elif mutation=='no_policy':current.pop('policy')
    elif mutation=='no_authority':current.pop('sitout_authority')
    elif mutation=='classifier_unknown':current['sitout_sources'][0]['status_classifier']['matched']=False
    hand=exported(tmp_path,current,provider_present=mutation!='no_provider',who_proven=mutation!='no_identity')
    assert not hand['certified'] and 'ALL_PLAYER_SEATS_AND_START_STACKS_NOT_PROVEN' in hand['blockers']
    assert 'simple_sitout_state' not in hand['players'][0]
    assert hand['players'][0]['START_STACK'] is None


def test_sitout_sitin_same_occupant_but_empty_and_replacement_close_identity():
    def nick(frame,name):
        return dict(table_id=1,session_id='s',seat='seat1',frame=frame,raw_text=name,confidence=.97)
    events=[dict(table=1,session='s',seat='seat1',frame=f,kind=k,evidence_status='PROVEN')
            for f,k in [(120,'SIT_OUT'),(180,'SIT_IN')]]
    lineage=SeatIdentityLineage([nick(10,'A')],events)
    assert lineage.for_actor(1,'s','A',100,200,actors=['A'])['status']=='PROVEN'
    empty=SeatIdentityLineage([nick(10,'A'),nick(90,'OPEN SEAT')],events)
    assert empty.for_actor(1,'s','A',100,200,actors=['A'])['status']=='UNKNOWN'
    replaced=SeatIdentityLineage([nick(10,'A'),nick(90,'B'),nick(95,'B')],events)
    assert replaced.for_actor(1,'s','A',100,200,actors=['A'])['status']=='UNKNOWN'
    assert replaced.for_actor(1,'s','B',100,200,actors=['B'])['status']=='PROVEN'


def test_saved_sql_sitout_is_bound_to_its_current_hand_seat_and_native_hash():
    from parser_core.simple_saved_policy import saved_stack_observations, saved_sitout_observations
    source=Path(r'C:\Users\Admin\.codex\tmp\hh95_source_cache_20261008\full_v1')
    session='table_de6df4eff185ac16ec217d53'
    bounds={(6,session,'2878365610'):(465,1164)}
    statuses=saved_sitout_observations(source,bounds,saved_stack_observations(source))
    assert len(statuses)==1
    row=statuses[0]
    assert (row['hand_id'],row['seat'],row['frame'])==('2878365610','seat2',580)
    assert row['authority']==SIT_OUT_GATE_AUTHORITY
    assert row['source']['state_id']=='S00393'
    assert row['source']['pixel_sha256']=='84ba692e1137a071113e0c087a98383ba347156c38579116644ec554a72ce57b'
    assert row['status_classifier']['matched'] and row['status_classifier']['numeric_authority'] is False


@pytest.mark.parametrize('mutation',['hash','classifier_score','model','locator','source_clock','source_seat','receipt_hand'])
def test_cached_status_cannot_replace_native_source_proof(mutation):
    from parser_core.simple_saved_policy import valid_sitout_status
    row=state()['sitout_sources'][0]
    assert valid_sitout_status(row,table_id=1,session_id='s',seat='seat1',hand_id='123456',hand_start=100,hand_end=200)
    if mutation=='hash':row['source']['pixel_sha256']='z'*64
    elif mutation=='classifier_score':row['status_classifier']['score']=.99
    elif mutation=='model':row['status_classifier']['model_sha256']='a'*64
    elif mutation=='locator':row['source']['line']=2
    elif mutation=='source_clock':row['source']['source_pts']+=1
    elif mutation=='source_seat':row['seat']='seat2'
    elif mutation=='receipt_hand':row['hand_id']='other'
    assert not valid_sitout_status(row,table_id=1,session_id='s',seat='seat1',hand_id='123456',hand_start=100,hand_end=200)


def test_parent_capture_png_tampering_is_rejected(tmp_path):
    from parser_core.simple_saved_policy import valid_sitout_status
    row=status_source(tmp_path);path=Path(row['source']['file'])
    original=json.loads(path.read_text('utf8'))
    # Replace pixels with another valid PNG but retain the original parent seal.
    other=tmp_path/'other';other.mkdir();status_source(other,sitout=False)
    original['png_base64']=json.loads((other/'start_stack_capture.jsonl').read_text('utf8'))['png_base64']
    path.write_text(json.dumps(original)+'\n',encoding='utf8')
    assert not valid_sitout_status(row,table_id=1,session_id='s',seat='seat1')


@pytest.mark.parametrize('payload',['','invalid base64','YWJj'])
def test_malformed_native_png_is_unproven_not_an_export_exception(tmp_path,payload):
    from parser_core.simple_saved_policy import valid_sitout_status
    row=status_source(tmp_path);path=Path(row['source']['file'])
    raw=json.loads(path.read_text('utf8'));raw['png_base64']=payload
    path.write_text(json.dumps(raw)+'\n',encoding='utf8')
    assert not valid_sitout_status(row,table_id=1,session_id='s',seat='seat1')


@pytest.mark.parametrize('literal',['1','OPEN SEAT','FOLD','ANTE','Sit out'])
def test_raw_text_without_native_sitout_pixels_has_no_status_authority(tmp_path,literal):
    from parser_core.simple_saved_policy import valid_sitout_status, apply_sitout_gate_status
    row=status_source(tmp_path,sitout=False)
    row['raw_text']=literal
    assert not row['status_classifier']['matched']
    row['status_classifier']['matched']=True
    assert not valid_sitout_status(row,table_id=1,session_id='s',seat='seat1')
    current=state();current.update(player_status=None,sitout_authority=None,sitout_sources=[],reason='NO_PARSEABLE_SAVED_NUMERIC')
    apply_sitout_gate_status(current,[row])
    assert current['player_status'] is None and current['sitout_authority'] is None


@pytest.mark.parametrize('state_id',['S00319','S02935'])
def test_native_saved_ante_and_fold_are_not_sitout(state_id):
    import sqlite3
    from parser_core.simple_saved_policy import _sitout_native_source
    from parser_core.sitout_text_pattern import SitOutTextPattern
    path=Path(r'C:\Users\Admin\.codex\tmp\hh95_source_cache_20261008\full_v1\stack_pipeline\timeline.sqlite')
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as conn:
        observation=conn.execute('SELECT observation_id FROM observations WHERE state_id=? ORDER BY frame_index LIMIT 1',(state_id,)).fetchone()[0]
    _,_,gray,_=_sitout_native_source('SAVED_SQL_VISUAL_STATE',path,observation_id=observation)
    assert not SitOutTextPattern().evaluate(gray)['matched']


def test_numeric_and_native_sitout_reach_existing_export_conflict_policy(tmp_path):
    current=state()
    current.update(DISPLAYED_STACK=1000,START_STACK=1000,first_numeric_source={'raw_text':'1000'},
                   player_status=None,sitout_authority=None,sitout_numeric_conflict=True,
                   reason='SIT_OUT_NUMERIC_CONFLICT')
    hand=exported(tmp_path,current)
    assert not hand['certified'] and 'START_STACK_CONFLICT' in hand['blockers']


def test_sequential_numeric_and_sitout_displays_preserve_numeric_fact():
    from parser_core.simple_saved_policy import apply_sitout_gate_status
    current=state()
    current.update(DISPLAYED_STACK=1000,START_STACK=1000,first_numeric_source={'frame':140,'raw_text':'1000'},
                   player_status=None,sitout_authority=None,reason=None)
    before=copy.deepcopy(current)
    apply_sitout_gate_status(current,current['sitout_sources'])
    assert not current['sitout_numeric_conflict'] and current['player_status'] is None
    assert current['reason']=='NUMERIC_STACK_WITH_SIT_OUT_OBSERVATIONS'
    for field in ('DISPLAYED_STACK','START_STACK','first_numeric_source'):
        assert current[field]==before[field]


@pytest.mark.parametrize('kind',['EMPTY','SIT_IN'])
@pytest.mark.parametrize('event_frame,conflict',[(160,False),(150,True)])
def test_numeric_priority_distinguishes_sequential_and_simultaneous_state_claims(kind,event_frame,conflict):
    from parser_core.simple_saved_policy import apply_sitout_gate_status
    current=state()
    current.update(DISPLAYED_STACK=1000,START_STACK=1000,first_numeric_source={'frame':140,'raw_text':'1000'},
                   player_status=None,sitout_authority=None)
    before=copy.deepcopy(current)
    event=dict(table=1,session='s',seat='seat1',frame=event_frame,kind=kind,evidence_status='PROVEN')
    apply_sitout_gate_status(current,current['sitout_sources'],[event])
    assert bool(current['sitout_state_conflicts']) is conflict
    assert current['sitout_state_observations']==[event]
    assert current['sitout_authority'] is None and current['player_status'] is None
    for field in ('WHO_status','player','seat','DISPLAYED_STACK','START_STACK','first_numeric_source'):
        assert current[field]==before[field]


@pytest.mark.parametrize('seat,confidence,conflict',[('seat1',.85,True),('seat2',.99,False),('seat1',.849,False)])
def test_later_numeric_contradiction_uses_current_route_and_unchanged_threshold(seat,confidence,conflict):
    from parser_core.simple_saved_policy import apply_sitout_gate_status
    current=state()
    current.update(DISPLAYED_STACK=1000,START_STACK=1000,first_numeric_source={'frame':140,'raw_text':'1000'})
    later=dict(table_id=1,session_id='s',seat=seat,frame=150,raw_text='1',confidence=confidence)
    apply_sitout_gate_status(current,current['sitout_sources'],numeric_observations=[later])
    assert current['sitout_numeric_conflict'] is conflict
    assert current['DISPLAYED_STACK']==1000 and current['START_STACK']==1000


@pytest.mark.parametrize('kind',['EMPTY','SIT_IN'])
def test_known_current_state_conflict_cannot_grant_status_exemption(tmp_path,kind):
    from parser_core.simple_saved_policy import apply_sitout_gate_status
    current=state()
    apply_sitout_gate_status(current,current['sitout_sources'],[dict(table=1,session='s',seat='seat1',
        frame=160,kind=kind,evidence_status='PROVEN')])
    assert current['sitout_authority'] is None and current['player_status'] is None
    hand=exported(tmp_path,current)
    assert 'START_STACK_CONFLICT' in hand['blockers'] and not hand['certified']


def test_saved_reader_classifies_numeric_routes_and_reused_sql_observation_clock(tmp_path):
    import sqlite3
    from parser_core.simple_saved_policy import saved_sitout_observations, _sitout_native_source
    row=status_source(tmp_path,frame=150,hand='old')
    _,_,gray,_=_sitout_native_source('SAVED_STACK_CAPTURE',tmp_path/'start_stack_capture.jsonl',line=1)
    path=tmp_path/'stack_pipeline';path.mkdir()
    with sqlite3.connect(path/'timeline.sqlite') as conn:
        conn.execute('CREATE TABLE visual_states(table_id,seat_id,frame_index,raw_gray,height,width,state_id,exact_hash,ocr_raw,ocr_confidence,ocr_numeric,status)')
        conn.execute('CREATE TABLE observations(table_id,seat_id,frame_index,observation_id,session_id,source_pts,source_time_base,state_id)')
        conn.execute('INSERT INTO visual_states VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (1,'seat1',150,gray.tobytes(),*gray.shape,'same',row['source']['pixel_sha256'],'Sit out',.94,None,'UNKNOWN'))
        conn.execute('INSERT INTO observations VALUES(?,?,?,?,?,?,?,?)',(1,'seat1',250,1,'s',250*512,'[1,15360]','same'))
    numeric={(1,'s','seat1'):[dict(table_id=1,session_id='s',seat='seat1',frame=260,raw_text='1000',confidence=.99)]}
    rows=saved_sitout_observations(tmp_path,{(1,'s','new'):(200,300)},numeric)
    assert len(rows)==1 and rows[0]['hand_id']=='new' and rows[0]['frame']==250
    assert rows[0]['source']['state_origin_frame']==150 and rows[0]['source']['source_pts']==250*512
    assert not saved_sitout_observations(tmp_path,{(1,'s','old'):(100,199)},numeric)[0]['source'].get('observation_id')
