import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dealing_persistent_hand_20261003 import PersistentHandStateTrigger

def rows(header=True,old_cards=False):
    rs=[{'pixel_hash':'cards','row_index':3,'row_bounds':[60,76],'status':'cards'}]
    if header:rs.insert(0,{'pixel_hash':'header100','row_index':1,'row_bounds':[20,36],'status':'UNKNOWN','hand_header_id':'100'})
    if old_cards:rs[-1].update(row_index=0,row_bounds=[0,16])
    return rs
def ready():
    s=PersistentHandStateTrigger();s.set_context('T09','session1','instance1',0)
    assert s.confirm_hand('T09','100',0,'header100');return s

def test_initial_unknown_without_header_and_context():
    s=PersistentHandStateTrigger()
    assert s.observe('T09',0,rows(False),True)[1]=='CURRENT_HAND_UNKNOWN'
    s.set_context('T09','s','t',1)
    assert s.observe('T09',1,rows(False),True)[0] is None

def test_header_disappearance_persists_and_triggers_first_good_cards():
    s=ready();assert s.observe('T09',0,rows()[:1],True)[0] is None
    assert s.current_hand_id('T09')=='100'
    event,reason,_=s.observe('T09',1,rows(False),True)
    assert reason=='TRIGGER' and event['header_confirmed_frame']==0 and not event['header_co_visible']
    assert event['trigger_frame']==1 and event['hand_id']=='100'

def test_actions_scroll_and_row_ids_do_not_reopen_latch():
    s=ready();assert s.observe('T09',0,rows(),True)[0]
    assert s.observe('T09',1,[],True)[1]=='ALREADY_TRIGGERED'
    assert s.observe('T09',2,[dict(rows(False)[0],row_index=0)],True)[1]=='ALREADY_TRIGGERED'

def test_new_header_replaces_id_and_visible_old_cards_cannot_fire():
    s=ready();s.observe('T09',0,rows(),True)
    assert s.confirm_hand('T09','101',1,'header101')
    rs=[dict(rows()[0],pixel_hash='header101',hand_header_id='101'),rows(False,True)[0]]
    assert s.observe('T09',1,rs,True)[0] is None
    assert s.observe('T09',2,rows(False),True)[0]['hand_id']=='101'

def test_session_change_table_instance_and_replacement_reset_unknown():
    for session,instance,replacement in [('session2','instance1',False),('session1','instance2',False),('session1','instance1',True)]:
        s=ready();s.observe('T09',0,rows(),True)
        s.set_context('T09',session,instance,1,replacement=replacement)
        assert s.current_hand_id('T09') is None
        assert s.observe('T09',1,rows(False),True)[0] is None
        s.confirm_hand('T09','100',2,'header100')
        assert s.observe('T09',2,rows(),True)[0]

def test_contradictory_new_header_clears_persistent_id():
    s=ready();assert not s.confirm_hand('T09','99',1,'different')
    assert s.current_hand_id('T09') is None
    assert s.observe('T09',1,rows(False),True)[0] is None

def test_history_or_unconfirmed_live_context_clears_id():
    s=ready();assert s.observe('T09',0,rows()[:1],True)[0] is None
    assert s.observe('T09',1,rows(False),live_context_confirmed=False)[0] is None
    assert s.current_hand_id('T09') is None
    assert s.observe('T09',2,rows(False),True)[0] is None

def test_frame_gap_without_new_context_resets_unknown():
    s=ready();s.observe('T09',0,rows()[:1],True)
    assert s.observe('T09',10,rows(False),True)[0] is None
    assert s.current_hand_id('T09') is None

def test_unconfirmed_newer_visible_header_cannot_bind_to_old_hand():
    s=ready();rs=rows()+[{'pixel_hash':'new','row_index':2,'row_bounds':[40,56],'status':'UNKNOWN','hand_header_id':'101'}]
    assert s.observe('T09',0,rs,True)[0] is None
    assert s.current_hand_id('T09') is None

def test_known_older_header_segment_is_not_current_when_header_gone():
    s=ready();s.confirm_hand('T09','101',1,'header101')
    assert s.observe('T09',1,rows(),True)[0] is None

def test_two_tables_keep_independent_current_hand_state():
    s=ready();s.set_context('T03','session1','instance3',0);s.confirm_hand('T03','555',0,'h555')
    assert s.observe('T03',0,rows(False),True)[0]['hand_id']=='555'
    assert s.observe('T09',0,rows(False),True)[0]['hand_id']=='100'

def test_same_hand_two_header_rows_use_bottommost_boundary():
    s=ready();rs=rows()+[{'pixel_hash':'bare100','row_index':2,'row_bounds':[40,56],'status':'UNKNOWN','hand_header_id':'100'}]
    assert s.observe('T09',0,rs,True)[0]['hand_id']=='100'

def test_gap_can_resume_with_independently_confirmed_same_frame_header():
    s=ready();s.observe('T09',0,rows()[:1],True)
    assert s.confirm_hand('T09','101',10,'header101')
    event,_,_=s.observe('T09',10,rows(False),True)
    assert event and event['hand_id']=='101'

def test_reconfirmed_same_hand_header_does_not_reopen_fired_latch():
    s=ready();assert s.observe('T09',0,rows(),True)[0]
    assert s.confirm_hand('T09','100',10,'header100-alias')
    assert s.observe('T09',10,rows(False),True)[1]=='ALREADY_TRIGGERED'
