"""Real source-bound per-actor prefix proof, without a roster complement."""
import copy,json
from pathlib import Path
import pytest
from parser_core.nickname_actor_binding import RULE,prove_actor_binding,binding_receipt,verify_binding_receipt
from parser_core.canonical_events import join_forced_posts,reconcile_stack_facts,_binding_rule_verified
from parser_core.exact_forced_chronology import PREFIX_RULE
from parser_core.poker_math_engine import calculate_start_stack

@pytest.fixture(scope='module')
def native():return json.loads((Path(__file__).parent/'fixtures/nickname_actor_binding_source.json').read_text('utf-8'))

def proof(case):return prove_actor_binding(case['inputs'],case['actor'])

def bound(case):
    p=proof(case);assert p['status']=='PROVEN'
    snapshot=case['inputs']['snapshot'];posts=[x for x in snapshot['posts'] if x['actor']==case['actor']]
    receipt=binding_receipt(p,case['actor'],posts)
    enriched=[dict(x,table=p['context'][0],session=p['context'][1],hand_id=p['context'][2],seat=p['seat'],epoch=p['epoch'],binding_rule=RULE) for x in posts]
    return receipt,enriched

def test_real_t01_own_prefix_has_unique_source_bound_actor(native):
    before=copy.deepcopy(native);p=proof(native)
    assert p['status']=='PROVEN' and p['seat']=='seat8' and p['epoch']=='T01_SEAT8_EPOCH_1'
    assert p['binding_anchor_frame']==2132
    assert p['label_evidence'][0]['frames']==[2132]
    assert p['label_evidence'][0]['constraint']==dict(kind='PREFIX',text='krismoltisa')
    r,posts=bound(native);assert verify_binding_receipt(r,native['inputs']['snapshot'])
    assert native==before

@pytest.mark.parametrize('mutation',['earlier_conflict_later_good','other_compatible_seat','confidence_low','confidence_high','confidence_inf','confidence_bool','duplicate_roster','unknown_membership','missing_trigger','nick_hash'])
def test_prefix_missing_conflicting_or_ambiguous_sources_withheld(native,mutation):
    c=copy.deepcopy(native);i=c['inputs']
    target=lambda frames:[r for r in i['nick_rows'] if r['seat']=='seat8' and r['source_frame'] in frames and not r.get('pre_deal')]
    if mutation=='earlier_conflict_later_good':
        rows=target([2129,2130,2131]);assert len(rows)==3
        for r in rows:r['ocr_response'].update(raw_text='olpro',score=.999,status='OK')
    elif mutation=='other_compatible_seat':
        rows=[r for r in i['nick_rows'] if r['seat']=='seat5' and r['source_frame'] in [2132,2133,2134]];assert len(rows)==3
        for r in rows:r['ocr_response'].update(raw_text='kr...',score=.999,status='OK')
    elif mutation.startswith('confidence_'):
        scores={'confidence_low':.94,'confidence_high':1.1,'confidence_inf':float('inf'),'confidence_bool':True}
        rows=target(range(2108,2172)) if mutation=='confidence_low' else target([2132])
        for r in rows:r['ocr_response']['score']=scores[mutation]
    elif mutation=='duplicate_roster':i['snapshot']['actors'].append(c['actor'])
    elif mutation=='unknown_membership':
        for e in i['state_receipts']:
            if e['table']==1 and e['seat']=='seat8' and e['hand_id']==i['context'][2]:e['evidence_status']='UNKNOWN'
    elif mutation=='missing_trigger':i['hand_triggers']=[]
    else:target([2132])[0]['pixel_sha256']='0'*64
    assert proof(c)['status']=='UNKNOWN'


def test_real_t01_prefix_canonical_and_start_stack_consumer(native):
    original=copy.deepcopy(native);receipt,posts=bound(native);snapshot=native['inputs']['snapshot']
    summary=dict(forced_snapshots=[snapshot],bound_posts=posts,historical_binding_receipts=[receipt])
    joined,report=join_forced_posts(native['timeline_rows'],summary)
    events=[e for e in joined if e.get('canonical_event_id') and e.get('player')==native['actor'] and not e.get('canonical_duplicate_of')]
    events=list({e['canonical_event_id']:e for e in events}.values())
    assert len(events)==2
    assert {e['action_type'] for e in events}=={'ANTE','SMALL_BLIND'}
    fact=reconcile_stack_facts(native['fact'],events,forced_snapshots=[snapshot],first_visible_contract=True,exact_binding_sources=native['inputs'])
    assert fact['status']=='PROVEN_FOR_MATH'
    assert fact['forced_chronology_proof']['rule']==PREFIX_RULE
    value=calculate_start_stack(fact,first_visible_contract=True)
    assert value['status']=='PASS' and value['START_STACK']=='12423'
    assert fact['vision_fact']==native['fact'] and native==original


@pytest.mark.parametrize('boundary',['EMPTY','REPLACE'])
def test_pre_deal_first_nick_cannot_cross_physical_occupant_boundary(native,boundary):
    from parser_core.seat_epoch_state import SeatEpochMachine
    c=copy.deepcopy(native);i=c['inputs'];key=tuple(i['context']);deal=i['deal']['source_frame']
    early=copy.deepcopy(next(r for r in i['nick_rows'] if r['seat']=='seat8' and r['source_frame']==2132))
    early.update(source_frame=deal-3,source_pts=i['deal']['source_pts']-3*512,pre_deal=True)
    early['ocr_response']['score']=.95
    i['nick_rows'].append(early)
    for t in i['hand_triggers']:
        if (t['table_id'],t['session_id'],t['hand_id'])==key:t['source_frame']=deal-4
    assert proof(c)['status']=='PROVEN'
    machine=SeatEpochMachine()
    for t in i['hand_triggers']:machine.hand(t['table_id'],t['session_id'],t['hand_id'],t['source_frame'])
    change=dict(table=key[0],session=key[1],seat='seat8',kind=boundary,frame=deal-2,
                hand_id=None,actor=None,evidence_status='PROVEN',source_ref='physical_boundary',source_index=0)
    for e in [*i['state_receipts'],change]:
        machine.observe(e['table'],e['session'],e['seat'],e['kind'],e['frame'],
                        **{k:e[k] for k in ('actor','hand_id','evidence_status','source_ref','source_index')})
    machine.build();i['state_receipts']=machine.audit
    assert proof(c)['status']=='UNKNOWN'


def test_first_nick_does_not_grant_numeric_chronology_in_next_hand(native):
    from parser_core.exact_forced_chronology import prove
    receipt,posts=bound(native);snapshot=native['inputs']['snapshot']
    summary=dict(forced_snapshots=[snapshot],bound_posts=posts,historical_binding_receipts=[receipt])
    joined,_=join_forced_posts(native['timeline_rows'],summary)
    events=list({e['canonical_event_id']:e for e in joined if e.get('canonical_event_id')
                 and e.get('player')==native['actor'] and not e.get('canonical_duplicate_of')}.values())
    i=copy.deepcopy(native['inputs']);key=i['context']
    i['hand_triggers'].append(dict(table_id=key[0],session_id=key[1],hand_id='next-hand',source_frame=2133))
    assert prove(native['fact'],snapshot,events,i) is None


@pytest.mark.parametrize('mutation',['receipt_rule','post_rule','strip_proof_and_rules'])
def test_prefix_binding_cannot_be_relabelled_as_legacy_exact(native,mutation):
    receipt,posts=bound(native);snapshot=native['inputs']['snapshot']
    if mutation=='receipt_rule':receipt['binding_rule']='FIRST_EXACT_LABEL_HISTORICAL_EPOCH'
    elif mutation=='post_rule':posts[0]['binding_rule']='FIRST_EXACT_LABEL_HISTORICAL_EPOCH'
    else:
        receipt.pop('actor_prefix_proof');receipt['binding_rule']='FIRST_EXACT_LABEL_HISTORICAL_EPOCH';posts[0].pop('binding_rule')
    assert not _binding_rule_verified(receipt,snapshot,posts[0])

@pytest.fixture(scope='module')
def native_empty():return json.loads((Path(__file__).parent/'fixtures/nickname_actor_binding_empty_source.json').read_text('utf-8'))

def test_positive_empty_partition_on_other_seat_preserves_actor_binding(native_empty):
    p=proof(native_empty)
    assert p['status']=='PROVEN' and p['seat']!='seat5'

@pytest.mark.parametrize('mutation',['empty_hash','empty_model','empty_partition','empty_frame_gap','empty_state_ref'])
def test_unproved_empty_partition_never_supplies_binding(native_empty,mutation):
    c=copy.deepcopy(native_empty);deal=c['inputs']['deal'];joint=deal['joint_seat_frames']
    if mutation=='empty_hash':joint[0]['empty_evidence']['seat5']['pixel_sha256']='0'*64
    elif mutation=='empty_model':joint[0]['empty_evidence']['seat5']['model_sha256']='0'*64
    elif mutation=='empty_partition':joint[0]['empty_seats']=[]
    elif mutation=='empty_frame_gap':joint[0]['source_frame']-=1
    else:
        for e in c['inputs']['state_receipts']:
            if e['table']==6 and e['hand_id']==deal['hand_id'] and e['kind']=='EMPTY':e['source_ref']='0'*64
    assert proof(c)['status']=='UNKNOWN'

@pytest.mark.parametrize('swapped_key',[False,True])
def test_real_two_empty_seats_require_matching_source_key(swapped_key):
    from parser_core.nickname_actor_binding import _deal_empty_refs
    deal=json.loads((Path(__file__).parent/'fixtures/nickname_actor_binding_empty_two_source.json').read_text('utf-8'))
    if swapped_key:
        for row in deal['joint_seat_frames']:
            row['empty_evidence']['seat6']=copy.deepcopy(row['empty_evidence']['seat5'])
    refs=_deal_empty_refs(deal)
    if swapped_key:assert refs=={}
    else:assert set(refs)=={'seat5','seat6'}
