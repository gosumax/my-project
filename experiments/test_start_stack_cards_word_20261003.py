"""Bounded trigger contract checks, isolated from production."""
from start_stack_cards_word_20261003 import OncePerHandTrigger, THRESHOLD

def test_persistent_word_emits_once():
    t=OncePerHandTrigger()
    events=[t.observe(('T04','H1'),f,1.,[91,355],True,True) for f in range(40)]
    assert sum(e is not None for e in events)==1
    assert events[2]['first_appearance_frame']==0
    assert events[2]['trigger_frame']==2

def test_reappearance_after_gap_never_reemits():
    t=OncePerHandTrigger()
    events=[t.observe('H1',f,1.,[91,355],True,True) for f in [0,1,2,20,21,22]]
    assert sum(e is not None for e in events)==1

def test_new_confirmed_hand_gets_own_trigger():
    t=OncePerHandTrigger()
    events=[t.observe(hand,f,1.,[91,355],True,True) for hand,fs in [('H1',[0,1,2]),('H2',[20,21,22])] for f in fs]
    assert sum(e is not None for e in events)==2

def test_unknown_hand_context_never_emits():
    t=OncePerHandTrigger()
    assert all(t.observe(None,f,1.,[91,355],True,True) is None for f in range(10))
    assert all(t.observe('H1',f,1.,[91,355],False,True) is None for f in range(10))
    assert not t.emitted

def test_three_stable_hits_required_and_jump_resets():
    t=OncePerHandTrigger()
    assert t.observe('H1',0,1.,[91,355],True,True) is None
    assert t.observe('H1',1,1.,[91,355],True,True) is None
    assert t.observe('H1',2,1.,[91,300],True,True) is None
    assert t.observe('H1',3,1.,[91,300],True,True) is None
    ev=t.observe('H1',4,1.,[91,300],True,True)
    assert ev['first_appearance_frame']==2
    assert ev['trigger_frame']==4

def test_below_threshold_and_missing_frames_reset_pending():
    t=OncePerHandTrigger()
    for f,score in [(0,1.),(1,THRESHOLD-.001),(2,1.),(4,1.),(5,1.)]:
        assert t.observe('H1',f,score,[91,355],True,True) is None
    assert t.observe('H1',6,1.,[91,355],True,True)['first_appearance_frame']==4

def test_old_row_in_new_hand_does_not_consume_new_latch():
    t=OncePerHandTrigger()
    old=[t.observe('H1',f,1.,[91,355],True,True) for f in range(3)]
    assert old[-1] is not None
    for f in range(20,30):
        assert t.observe('H2',f,1.,[91,73],True,False) is None
    assert 'H2' not in t.emitted
    fresh=[t.observe('H2',f,1.,[91,355],True,True) for f in range(30,33)]
    assert fresh[-1]['first_appearance_frame']==30
    assert fresh[-1]['trigger_frame']==32

def test_new_hand_context_alone_is_insufficient():
    t=OncePerHandTrigger()
    assert all(t.observe('H2',f,1.,[91,355],confirmed=True) is None for f in range(10))
    assert not t.pending and not t.emitted

def test_unconfirmed_row_resets_preceding_partial_confirmation():
    t=OncePerHandTrigger()
    assert t.observe('H2',0,1.,[91,355],True,True) is None
    assert t.observe('H2',1,1.,[91,355],True,True) is None
    assert t.observe('H2',2,1.,[91,355],True,False) is None
    assert t.observe('H2',3,1.,[91,355],True,True) is None
    assert t.observe('H2',4,1.,[91,355],True,True) is None
    assert t.observe('H2',5,1.,[91,355],True,True)['first_appearance_frame']==3
