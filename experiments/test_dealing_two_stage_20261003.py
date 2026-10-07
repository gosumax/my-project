"""Behavioral contracts for isolated two-stage NCC and causal admission."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from dealing_two_stage_ncc_20261003 import TwoStageModel,CausalRowFrontier,OncePerHandLatch,HandStateTrigger,bind_current,sha

CONFIG=ROOT/'runs/start_stack_two_stage_ncc_20261003/candidate_v2/frozen_config.json'

def row(seed,index):
    gray=np.random.default_rng(seed).integers(0,255,(16,228),dtype=np.uint8)
    return {'gray':gray,'pixel_hash':sha(gray.tobytes()),'row_index':index,'row_bounds':[index*20,index*20+16]}

def test_gate_reject_skips_classifier():
    model=TwoStageModel(CONFIG);r=model.classify(np.zeros((16,228),np.uint8))
    assert r['status']=='UNKNOWN' and not r['gate_pass']
    assert r['scores'] is None and model.classifier_calls==0

def test_confirmed_literal_has_four_scores():
    model=TwoStageModel(CONFIG);gray=np.full((16,228),30,np.uint8)
    gray[:,4:91]=model.templates['Dealer_Dealing'];gray[:,91:131]=model.templates['cards']
    r=model.classify(gray)
    assert r['gate_pass'] and set(r['scores'])=={'cards','Turn','Flop','River'}
    assert r['winner']=='cards' and r['margin']==r['scores'][r['winner']]-r['scores'][r['runner_up']]

def test_initial_and_reset_identity_are_unknown():
    frontier=CausalRowFrontier()
    assert all(not r['append_proven'] for r in frontier.observe(0,[row(i,i) for i in range(4)]))
    assert all(not r['append_proven'] for r in frontier.observe(1,[row(i+10,i) for i in range(4)]))

def test_unique_ordered_append_and_current_hand_order():
    frontier=CausalRowFrontier();old=[row(i,i) for i in range(4)]
    frontier.observe(0,old);current=[dict(r,row_index=i) for i,r in enumerate(old[1:])]+[row(9,3)]
    result=frontier.observe(1,current);new=result[-1]
    assert not new['append_proven'] and new['birth_frame']==1
    assert new['overlap_append_candidate'] and new['freshness']=='UNKNOWN'
    assert not bind_current(new,'123',1,1)[0]
    # Only an independent receipt, not the visual overlap producer, may
    # exercise the positive binding contract.
    new=dict(new,append_proven=True)
    assert bind_current(new,'123',1,1)[0]
    assert not bind_current(new,'123',1,3)[0]
    assert not bind_current(new,'123',2,0)[0]

def test_once_per_hand_requires_same_track_three_frames():
    latch=OncePerHandLatch(3);candidate={'track_id':1,'row_index':3,'pixel_hash':'abc'}
    assert latch.observe(1,'H1',10,[candidate])[0] is None
    assert latch.observe(1,'H1',11,[candidate])[0] is None
    event,_=latch.observe(1,'H1',12,[candidate]);assert event['first_appearance_frame']==10
    assert all(latch.observe(1,'H1',f,[candidate])[0] is None for f in range(13,100))
    assert latch.observe(1,'H2',100,[candidate])[0] is None

def test_ambiguous_or_gapped_confirmation_does_not_trigger():
    latch=OncePerHandLatch(3);c={'track_id':1,'row_index':3,'pixel_hash':'abc'}
    latch.observe(1,'H',0,[c]);latch.observe(1,'H',1,[c]);assert latch.observe(1,'H',3,[c])[0] is None
    assert latch.observe(1,'H',4,[c,c])[0] is None
    assert latch.observe(1,'H',5,[c])[0] is None

def test_frontier_gap_does_not_carry_identity():
    f=CausalRowFrontier();rows=[row(i,i) for i in range(4)]
    first=f.observe(0,rows);after=f.observe(100,rows)
    assert {r['track_id'] for r in first}.isdisjoint({r['track_id'] for r in after})
    assert all(not r['append_proven'] for r in after)

def test_prior_false_bet_gate_rejected_before_classifier():
    import base64,json
    source=ROOT/'runs/production_poker_smoke17_full_candidate_v2_20261002/physical_rows.jsonl'
    p=next(json.loads(x) for x in source.open(encoding='utf-8') if json.loads(x)['row_pixel_sha256']=='75b20b02b7416f0119d910eaff79ab1a4fad2110e2d5523d0243790cb6bed258')
    gray=np.frombuffer(base64.b64decode(p['row_pixels_base64']),np.uint8).reshape(p['pixel_shape'])
    model=TwoStageModel(CONFIG);result=model.classify(gray)
    assert not result['gate_pass'] and result['status']=='UNKNOWN'
    assert result['scores'] is None and model.classifier_calls==0

def phase_rows(header_index=1,cards_index=3):
    return [dict(row(123,header_index),status='UNKNOWN'),dict(row(456,cards_index),status='cards')]

def test_hand_state_marker_ignores_row_birth_and_track_id():
    s=HandStateTrigger(3);rs=phase_rows();s.confirm_hand('T01','100',0,rs[0]['pixel_hash'])
    for f in range(3):
        shifted=phase_rows(header_index=f,cards_index=f+2)
        for r in shifted:r.update(track_id=100*f+r['row_index'],append_proven=False,freshness='UNKNOWN')
        event,reason,_=s.observe('T01',f,shifted)
    assert reason=='TRIGGER' and event['first_appearance_frame']==0
    assert event['row_birth_required'] is False

def test_closed_hand_does_not_retrigger_after_bets_or_scroll():
    s=HandStateTrigger(3);rs=phase_rows();s.confirm_hand('T01','100',0,rs[0]['pixel_hash'])
    for f in range(3):event,_,_=s.observe('T01',f,rs)
    assert event
    assert s.observe('T01',3,[])[1]=='ALREADY_TRIGGERED'
    assert s.observe('T01',40,phase_rows(0,1))[1]=='ALREADY_TRIGGERED'

def test_new_hand_resets_latch_but_rejects_old_cards_above_header():
    s=HandStateTrigger(3);rs=phase_rows();s.confirm_hand('T01','100',0,rs[0]['pixel_hash'])
    for f in range(3):s.observe('T01',f,rs)
    s.confirm_hand('T01','101',3,rs[0]['pixel_hash'])
    for f in range(3,6):assert s.observe('T01',f,phase_rows(3,1))[0] is None
    for f in range(6,9):event,_,_=s.observe('T01',f,phase_rows(1,3))
    assert event and event['hand_id']=='101'
    assert not s.confirm_hand('T01','100',9,rs[0]['pixel_hash'])

def test_hand_state_unknown_header_and_gap_reset_confirmation():
    s=HandStateTrigger(3);rs=phase_rows()
    assert s.observe('T01',0,rs)[1]=='CURRENT_HAND_UNKNOWN'
    s.confirm_hand('T01','100',0,rs[0]['pixel_hash'])
    s.observe('T01',0,rs);s.observe('T01',1,rs)
    assert s.observe('T01',3,rs)[0] is None
    assert s.observe('T01',4,[rs[1]])[1]=='CURRENT_HAND_HEADER_NOT_UNIQUELY_VISIBLE'
    assert s.observe('T01',5,rs)[0] is None
    assert s.observe('T01',6,rs+[rs[0]])[0] is None

def test_next_hand_header_is_upper_fence_even_before_state_update():
    s=HandStateTrigger(3);rs=phase_rows(0,3);s.confirm_hand('T01','100',0,rs[0]['pixel_hash'])
    later=dict(row(789,2),status='UNKNOWN',hand_header_id='101')
    for f in range(3):
        event,_,eligible=s.observe('T01',f,rs+[later])
        assert event is None and not eligible

def test_default_marker_fires_on_first_confirmed_appearance():
    s=HandStateTrigger();rs=phase_rows();s.confirm_hand('T01','100',12,rs[0]['pixel_hash'])
    event,reason,_=s.observe('T01',12,rs)
    assert reason=='TRIGGER' and event['trigger_frame']==12
    assert s.observe('T01',13,phase_rows(0,2))[1]=='ALREADY_TRIGGERED'

