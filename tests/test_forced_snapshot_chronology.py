import copy
import gzip
import json
from pathlib import Path

import pytest

from parser_core.canonical_events import reconcile_stack_facts
from parser_core.poker_math_engine import calculate_start_stack
from parser_core.start_stack_visibility import verify_first_visible_fact


@pytest.fixture
def native():
    path=Path(__file__).parent/'fixtures/forced_chronology_short_v6.json.gz'
    return json.loads(gzip.decompress(path.read_bytes()))


@pytest.mark.parametrize('seat,total',[('seat1','17786'),('seat2','36667'),('seat3','10180'),
    ('seat4','11125'),('seat5','28330'),('seat6','5744'),('seat7','30434'),('seat8','27491')])
def test_native_callback_offset_retains_exact_forced_set_and_first_number(native,seat,total):
    fact=next(r for r in native['facts'] if r['seat']==seat);original=copy.deepcopy(fact)
    args=dict(forced_snapshots=native['snapshots'])
    old=reconcile_stack_facts(fact,native['events'],**args)
    assert old['status']=='UNKNOWN'
    fixed=reconcile_stack_facts(fact,native['events'],first_visible_contract=True,**args)
    assert fact==original and fixed['vision_fact']==original
    assert fixed['status']=='PROVEN_FOR_MATH'
    chronology=fixed['forced_chronology_proof']
    assert chronology['forced_source_frame']==258>fact['deal_complete_frame']==257
    assert chronology['first_numeric_source_frame']==284
    result=calculate_start_stack(fixed,first_visible_contract=True)
    assert result['status']=='PASS' and result['START_STACK']==total


@pytest.mark.parametrize('mutation',['no_visibility','bare_numeric_frame','missing_post','post_pts','row_pts',
    'post_after_first','post_at_equal_frame_wrong_pts','no_roster','epoch','member','anchor','wrong_seat','changed_prefix'])
def test_offset_never_bypasses_source_roster_chronology_or_first_visibility(native,mutation):
    fact=next(r for r in native['facts'] if r['seat']=='seat5')
    events=native['events'];snap=native['snapshots'][0]
    own=next(r for r in events if r['seat']=='seat5')
    if mutation=='no_visibility':fact.pop('first_visible_proof')
    elif mutation=='bare_numeric_frame':fact['first_numeric_visible_frame']=285
    elif mutation=='missing_post':events.remove(own)
    elif mutation=='post_pts':snap['posts'][0]['source_pts']+=1
    elif mutation=='row_pts':snap['posts'][0]['source_rows'][0]['source_pts']+=1
    elif mutation=='post_after_first':snap['posts'][0]['source_frame']=285
    elif mutation=='post_at_equal_frame_wrong_pts':snap['posts'][0]['source_frame']=284
    elif mutation=='no_roster':
        for event in events:event['canonical_provenance']['binding_receipt'].pop('roster_proof',None)
    elif mutation=='epoch':fact['occupancy_epoch']='wrong'
    elif mutation=='member':fact['participant_proven']=False
    elif mutation in ('anchor','wrong_seat'):
        for event in events:
            proof=event['canonical_provenance']['binding_receipt'].get('roster_proof')
            if proof:
                if mutation=='anchor':proof['binding_anchor_frame']=283
                else:proof['matching']['seat5']='wrong'
    else:snap['posts'][0]['amount']+=1
    result=reconcile_stack_facts(fact,events,forced_snapshots=[snap],first_visible_contract=True)
    assert result['status']=='UNKNOWN'


def test_full_cannot_fall_back_to_frame_only_order_when_source_time_is_invalid(native):
    fact=next(r for r in native['facts'] if r['seat']=='seat5')
    # A frame-only old-order assertion must not bypass the full proof contract.
    assert verify_first_visible_fact(fact)
    for post in native['snapshots'][0]['posts']:
        post['source_frame']=fact['deal_complete_frame']
        post['source_pts']=999999999
    for event in native['events']:
        event['canonical_provenance']['post']['source_frame']=fact['deal_complete_frame']
        event['canonical_provenance']['post']['source_pts']=999999999
    result=reconcile_stack_facts(fact,native['events'],forced_snapshots=native['snapshots'],first_visible_contract=True)
    assert result['status']=='UNKNOWN'
