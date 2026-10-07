"""Epoch/activity/membership are separate source-ordered facts."""
from parser_core.seat_epoch_state import SeatEpochMachine


def machine():
    m=SeatEpochMachine()
    m.hand(3,'S','10000000',10)
    m.hand(3,'S','10000001',30)
    return m


def test_positive_occupancy_does_not_assume_active():
    m=machine();m.observe(3,'S','seat1','OCCUPIED',12,hand_id='10000000');m.build()
    assert m.rows[3,'S','seat1']['state'] is None
    assert m.membership(3,'S','seat1','10000000')['participant']


def test_late_arrival_events_rebuild_by_source_order():
    m=machine()
    m.observe(3,'S','seat1','OCCUPIED',32,actor='new',hand_id='10000001')
    m.observe(3,'S','seat1','EMPTY',25)
    m.observe(3,'S','seat1','OCCUPIED',12,actor='old',hand_id='10000000')
    m.build()
    assert m.epoch_at(3,'S','seat1',12).endswith('_1')
    assert m.epoch_at(3,'S','seat1',26) is None
    assert m.epoch_at(3,'S','seat1',32).endswith('_2')


def test_sitout_remains_occupied_and_forced_participant():
    m=machine();m.observe(3,'S','seat1','SIT_OUT',9,actor='nick')
    m.observe(3,'S','seat1','OCCUPIED',12,actor='nick',hand_id='10000000');m.build()
    assert m.rows[3,'S','seat1']['state']=='SITOUT'
    assert m.epoch_at(3,'S','seat1',12) is not None
    assert m.membership(3,'S','seat1','10000000')['participant']


def test_explicit_sitin_confirms_active_existing_epoch():
    m=machine();m.observe(3,'S','seat1','SIT_OUT',9,actor='nick')
    m.observe(3,'S','seat1','SIT_IN',10,actor='nick',hand_id='10000000');m.build()
    assert m.rows[3,'S','seat1']['state']=='ACTIVE'
    assert m.epoch_at(3,'S','seat1',9)==m.epoch_at(3,'S','seat1',10)


def test_join_empty_seat_midhand_is_not_retroactive_participation():
    m=machine();m.observe(3,'S','seat1','EMPTY',11)
    m.observe(3,'S','seat1','SIT_IN',15,actor='new',hand_id='10000000')
    m.observe(3,'S','seat1','OCCUPIED',16,actor='new',hand_id='10000000');m.build()
    assert m.rows[3,'S','seat1']['state']=='SEATED_MID_HAND'
    assert not m.membership(3,'S','seat1','10000000')['participant']
    assert m.membership(3,'S','seat1','10000000')['blocker']=='SEATED_MID_HAND'


def test_midhand_occupancy_after_confirmed_empty_is_not_retroactive():
    m=machine();m.observe(3,'S','seat1','EMPTY',11)
    m.observe(3,'S','seat1','OCCUPIED',16,actor='new',hand_id='10000000');m.build()
    assert not m.membership(3,'S','seat1','10000000')['participant']


def test_explicit_replacement_has_new_epoch_and_no_currenthand_membership():
    m=machine();m.observe(3,'S','seat1','OCCUPIED',12,actor='old',hand_id='10000000')
    m.observe(3,'S','seat1','REPLACE',15,actor='new',hand_id='10000000');m.build()
    old,new=m.epoch_at(3,'S','seat1',14),m.epoch_at(3,'S','seat1',15)
    assert old!=new
    assert m.membership(3,'S','seat1','10000000',old)['participant']
    assert not m.membership(3,'S','seat1','10000000',new)['participant']


def test_midhand_join_can_participate_in_next_hand_same_epoch():
    m=machine();m.observe(3,'S','seat1','SIT_IN',15,actor='new',hand_id='10000000')
    m.observe(3,'S','seat1','OCCUPIED',16,hand_id='10000000')
    m.observe(3,'S','seat1','OCCUPIED',32,hand_id='10000001');m.build()
    assert not m.membership(3,'S','seat1','10000000')['participant']
    assert m.membership(3,'S','seat1','10000001')['participant']
    assert m.epoch_at(3,'S','seat1',16)==m.epoch_at(3,'S','seat1',32)


def test_ocr_alias_variants_never_create_epoch_or_assume_replacement():
    m=machine();m.observe(3,'S','seat1','OCCUPIED',12,actor='bankir',hand_id='10000000')
    m.observe(3,'S','seat1','OCCUPIED',32,actor='bankiir',hand_id='10000001');m.build()
    assert m.epoch_at(3,'S','seat1',12)==m.epoch_at(3,'S','seat1',32)
    assert m.rows[3,'S','seat1']['aliases']=={'bankir','bankiir'}


def test_old_session_and_old_epoch_cannot_supply_new_membership():
    m=machine();m.observe(3,'OLD','seat1','OCCUPIED',12,hand_id='10000000')
    m.observe(3,'S','seat1','OCCUPIED',32,hand_id='10000001');m.build()
    assert not m.membership(3,'S','seat1','10000000')['participant']
    assert not m.membership(3,'S','seat1','10000001','wrongepoch')['participant']


def test_unproven_empty_cannot_erase_confirmed_epoch():
    m=machine();m.observe(3,'S','seat1','OCCUPIED',12,hand_id='10000000')
    m.observe(3,'S','seat1','EMPTY',15,evidence_status='UNKNOWN');m.build()
    assert m.epoch_at(3,'S','seat1',16) is not None
    assert any(e['status']=='SEAT_EVIDENCE_UNKNOWN' for e in m.audit)


def test_empty_state_has_no_epoch_or_membership_in_new_hand():
    m=machine();m.observe(3,'S','seat1','EMPTY',9);m.build()
    assert m.rows[3,'S','seat1']['state']=='EMPTY'
    assert m.epoch_at(3,'S','seat1',32) is None
    assert not m.membership(3,'S','seat1','10000001')['participant']
