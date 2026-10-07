"""Physical birth proofs and separate identity on real and synthetic pixels."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from parser_core.chat_tracker import AdjacentChatTracker
from parser_core.physical_chat_admission import PhysicalChatAdmission, physical_tail_proof

FIXTURE = Path(__file__).parent/'fixtures/physical_chat_admission_smoke19.npz'


def run_pair(old, new):
    legacy = AdjacentChatTracker(1, 'S')
    lane = PhysicalChatAdmission(1, 'S')
    first = legacy.process(old, 0)
    assert lane.process(old, first) is None
    transition = legacy.process(new, 1)
    frozen = copy.deepcopy(transition)
    counter = legacy.counter
    safe = copy.deepcopy(legacy.safe_id_state)
    receipt = lane.process(new, transition)
    assert transition == frozen
    assert legacy.counter == counter
    assert legacy.safe_id_state == safe
    return legacy, lane, transition, receipt


def synthetic(count=16):
    a = np.zeros((360, 228), np.uint8)
    rng = np.random.default_rng(531)
    for j in range(count):
        a[8+j*20:16+j*20, 10:190] = rng.integers(90,240,(8,180),dtype=np.uint8)
    return a


def test_fixture_source_hashes():
    with np.load(FIXTURE) as data:
        for entry in json.loads(FIXTURE.with_suffix('.json').read_text('utf-8')):
            assert hashlib.sha256(data[entry['key']].tobytes()).hexdigest() == entry['sha256']


def test_t01_unresolved_new_hand_block_captures_header_posts_and_deal():
    with np.load(FIXTURE) as data:
        legacy,lane,transition,receipt = run_pair(data['T01_2085_before'],data['T01_2085_after'])
        assert transition['classification'] == 'UNRESOLVED'
        assert transition['reason'] == 'INSUFFICIENT_CONFIDENT_ZONES'
        assert transition['OCR_queue_row_ids'] == []
        assert receipt['proof']['status'] == 'PROVEN_PHYSICAL_NEW'
        assert len(receipt['admissions']) == 15
        assert all(a['identity_status']=='UNKNOWN' for a in receipt['admissions'])
        assert receipt['proof']['shift_px'] in (302,303)
        # Header, continuation, eight antes, wrapped SB, BB and Dealing cards.
        assert receipt['admissions'][0]['span'][0] >= 74
        assert receipt['admissions'][-1]['span'][1] == 368
        assert len({a['row_id'] for a in receipt['admissions']})==15
        repeat = lane.process(data['T01_2085_after'],legacy.process(data['T01_2085_after'],2))
        assert not repeat['admissions']


@pytest.mark.parametrize('extra',[1,2])
def test_local_append_without_scroll_uses_blank_region(extra):
    old = synthetic(15)
    new = synthetic(15+extra)
    _,_,transition,receipt = run_pair(old,new)
    assert transition['classification']=='LOCAL_CHANGE'
    assert len(receipt['admissions'])==extra
    assert receipt['proof']['shift_px']==0
    assert all(a['row_id'].startswith('S:F') for a in receipt['admissions'])


def test_local_in_place_change_is_not_new_row():
    old = synthetic()
    new = old.copy()
    new[88:96,10:190] //= 2
    _,_,_,receipt = run_pair(old,new)
    assert not receipt['admissions']


def test_arbitrary_replacement_with_no_retained_pixels_is_unknown():
    old = synthetic()
    new = np.flip(old,axis=1).copy()
    _,_,_,receipt = run_pair(old,new)
    assert not receipt['admissions']
    assert receipt['proof']['status']=='UNKNOWN'


def test_global_scroll_lane_does_not_change_legacy_admission_or_ids():
    old = synthetic(18)
    new = np.roll(old,-20,axis=0)
    new[348:356,10:190] = np.random.default_rng(937).integers(90,240,(8,180),dtype=np.uint8)
    _,_,transition,receipt = run_pair(old,new)
    assert transition['classification']=='GLOBAL_SCROLL'
    assert len(transition['OCR_queue_row_ids'])==1
    assert receipt is None


def test_top_edge_fragment_never_submitted():
    with np.load(FIXTURE) as data:
        _,_,transition,receipt=run_pair(data['T01_2085_before'],data['T01_2085_after'])
    assert transition['new_edge_spans']
    assert all(a['span'][0]>1 for a in receipt['admissions'])


def test_same_pixels_never_read_again():
    old = synthetic()
    _,_,_,receipt=run_pair(old,old.copy())
    assert not receipt['admissions']


def test_repeated_rows_and_bottom_edit_are_not_scroll():
    rng=np.random.default_rng(41)
    old=np.zeros((60,80),np.uint8)
    glyph=rng.integers(90,240,(8,60),dtype=np.uint8)
    for y in (5,25,45):
        old[y:y+8,5:65]=glyph
    new=old.copy()
    new[45:53,5:65]=rng.integers(90,240,(8,60),dtype=np.uint8)
    _,_,_,receipt=run_pair(old,new)
    assert not receipt['admissions']
    assert receipt['proof']['reason']=='RETAINED_TRANSLATION_AMBIGUOUS'


def test_new_row_may_start_in_translated_old_bottom_blank_margin():
    from parser_core.chat_rows import physical_spans
    old=synthetic(16)
    new=np.zeros_like(old)
    new[:300]=old[60:]
    rng=np.random.default_rng(956)
    for y in (268,288,308):
        new[y:y+8,10:190]=rng.integers(90,240,(8,180),dtype=np.uint8)
    proof=physical_tail_proof(old,new,physical_spans(old),physical_spans(new))
    assert proof['status']=='PROVEN_PHYSICAL_NEW'
    assert proof['shift_px']==60 and len(proof['new_indices'])==3
    assert physical_spans(new)[proof['new_indices'][0]][0]<len(old)-proof['shift_px']


@pytest.mark.parametrize('key,minimum',[
    ('T03_6455',3), ('T03_9001',3), ('T03_9028',2), ('T09_8025',9), ('T01_9701',7),
])
def test_real_fold_return_no_show_show_win_and_leave_physical_rows(key,minimum):
    with np.load(FIXTURE) as data:
        _,_,transition,receipt=run_pair(data[key+'_before'],data[key+'_after'])
    assert transition['classification'] in {'LOCAL_CHANGE','UNRESOLVED'}
    assert receipt['proof']['status']=='PROVEN_PHYSICAL_NEW'
    assert len(receipt['admissions'])>=minimum
    assert len({a['row_id'] for a in receipt['admissions']})==len(receipt['admissions'])


def test_first_stage_emits_once_with_pixel_and_unknown_identity_receipt():
    from parser_core.ram_first_stage import FirstStage
    receipts=[]
    stage=FirstStage(Path('unused'),{},0,root=Path('.'),physical_admission=True,
                     admission_observer=receipts.append)
    with np.load(FIXTURE) as data:
        before=data['T01_2085_before']; after=data['T01_2085_after']
        assert list(stage.chat_rows(before,frame_id=2084,table_id=1,session_id='S'))==[]
        rows=list(stage.chat_rows(after,frame_id=2085,table_id=1,session_id='S'))
        assert len(rows)==15
        assert all(r.identity_status=='UNKNOWN' and r.admission_ref for r in rows)
        assert len({r.stable_row_id for r in rows})==15
        assert all(r.pixel_sha256==hashlib.sha256(r.pixels.tobytes()).hexdigest() for r in rows)
        assert list(stage.chat_rows(after,frame_id=2086,table_id=1,session_id='S'))==[]
        assert receipts[0]['source_pixel_hash_after']==hashlib.sha256(after.tobytes()).hexdigest()


def test_weak_retained_lines_do_not_erase_new_tail_with_independent_anchors():
    with np.load(FIXTURE) as data:
        _,_,_,receipt=run_pair(data['T04_1009_before'],data['T04_1009_after'])
    assert len(receipt['admissions'])==13
    assert receipt['proof']['confident_retained_rows']>=2
    assert any(not r['pixel_anchor'] for r in receipt['proof']['retained_rows'])
    assert all(r['identity_status']=='UNKNOWN' for r in receipt['proof']['retained_rows'])


def test_single_short_retained_line_does_not_prove_large_replacement():
    with np.load(FIXTURE) as data:
        _,_,_,receipt=run_pair(data['T08_1672_before'],data['T08_1672_after'])
    assert not receipt['admissions']
    assert receipt['proof']['status']=='UNKNOWN'
