"""Real saved native NO_SHOW policy controls; no OCR, capture or solver.

The compact fixture retains full source references, not authority flags. Native
parent files are required deliberately: cached strings alone cannot certify.
"""
import copy
import gzip
import json
from decimal import Decimal
from pathlib import Path

import pytest

from parser_core.hidden_loser_evidence import COMPARISON, RULE, HiddenLoserSource
from parser_core.poker_logic_engine import validate_hand
from parser_core.simple_saved_policy import LineageActorProvider, hand_key, saved_nick_observations
from parser_core.simple_seat_who import SeatIdentityLineage

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=Path(__file__).parent/'fixtures/hidden_loser_saved_three.json.gz'

def rows(path):
    with path.open(encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

@pytest.fixture(scope='module')
def corpus():
    return json.loads(gzip.decompress(FIXTURE.read_bytes()).decode('utf8'))

@pytest.fixture(scope='module')
def source(corpus):
    native=Path(corpus['source'])
    assert native.is_dir(), 'Real native parents required; do not replace with cached authority'
    observed=[c['observed_hand'] for c in corpus['cases'].values()]
    bounds={hand_key(c['observed_hand']):tuple(c['bounds']) for c in corpus['cases'].values()}
    states=json.loads((native/'seat_forced_summary.json').read_text('utf8'))['state_receipts']
    assoc=ROOT/'runs/hh_all_hands_20261009/sitout_export_v13_saved/retained_inputs/nick_occupant_associations.jsonl'
    actor=LineageActorProvider(SeatIdentityLineage(saved_nick_observations(native),states),
        bounds,observed,rows(assoc) if assoc.exists() else [])
    return HiddenLoserSource(native,actor,visual_source=ROOT/corpus['visual_source'])

def case(corpus,number='2878366257'):
    return copy.deepcopy(corpus['cases'][number]['candidate_hand'])

def arguments(hand):
    v=validate_hand(hand)
    players={n:{'state':'FOLDED' if p['folded'] else 'ALL_IN' if p['all_in'] else 'ACTIVE'}
             for n,p in v['players'].items()}
    return dict(players=players,nets={n:Decimal(p['net_contribution']) for n,p in v['players'].items()},
        payouts={m['player']:Decimal(m['amount']) for m in hand['messages'] if m['action']=='WIN'},
        board=copy.deepcopy(v['board']),hole=copy.deepcopy(v['showdown']),ledger_known=True,math=v['math'])

def first(hand,action):
    return next(m for m in hand['messages'] if m['action']==action)

@pytest.mark.parametrize('number',['2878366257','2878366136','2878367230'])
def test_native_hidden_loser_exemption_preserves_unknown_holes(corpus,source,number):
    hand=case(corpus,number); before=copy.deepcopy(hand)
    old=validate_hand(hand)
    new=validate_hand(hand,hidden_card_source=source)
    assert old['status']!='FULLY_VALID_HAND'
    assert new['status']=='FULLY_VALID_HAND', (new['certification_blockers'],source.last_rejection)
    proof=new['hidden_loser_authority']
    assert proof['rule']==RULE
    assert proof['hidden_losers']
    for actor,state in proof['hidden_losers'].items():
        assert state['comparison']==COMPARISON and state['hole_cards'] is None
        assert actor not in new['showdown'] and state['native_no_show']
        assert state['who']['status']=='PROVEN'
    assert new['showdown']==old['showdown'] and hand==before
    assert new['math']['pass'] and new['math']['delta']=='0'
    for flag in ('hole_cards_authority','WHO_authority','START_STACK_authority',
                 'board_authority','pot_authority','side_pot_authority'):
        assert proof[flag] is False

@pytest.mark.parametrize('mutation',[
    'no_no_show','unread_show','show_and_no_show','cross_hand','cross_table','cross_session',
    'cross_actor','bad_hash','bad_clock','failed_frame','source_order','cached_string',
    'unknown_winner','unknown_payout','payout_conflict','two_winners','math_conflict','side_pot',
    'board_incomplete','all_in','known_loser_beats_winner','missing_start','missing_end',
])
def test_mutated_real_evidence_never_grants_authority(corpus,source,mutation):
    hand=case(corpus); args=arguments(hand)
    hidden=first(hand,'NO_SHOW'); win=first(hand,'WIN'); row=hidden['source_rows'][0]
    actor=hidden['player']; winner=win['player']
    if mutation=='no_no_show': hand['messages']=[m for m in hand['messages'] if m['action']!='NO_SHOW']
    elif mutation=='unread_show': hidden['action']='SHOW';hidden['cards']=[]
    elif mutation=='show_and_no_show': hand['messages'].append(dict(copy.deepcopy(hidden),action='SHOW',cards=[]))
    elif mutation=='cross_hand': hand['hand_id']='foreign-hand'
    elif mutation=='cross_table': hand['table_id']='T09'
    elif mutation=='cross_session': hand['session_id']='foreign-session'
    elif mutation=='cross_actor': hidden['player']='foreign-actor'
    elif mutation=='bad_hash': row['row_pixel_sha256']='0'*64
    elif mutation=='bad_clock': row['source_pts']=-1
    elif mutation=='failed_frame': row['ocr_status']='ERROR'
    elif mutation=='source_order': hand['messages'].reverse()
    elif mutation=='cached_string': hidden['source_rows']=[];hidden['source_row_ids']=[]
    elif mutation=='unknown_winner': args['hole'].pop(winner)
    elif mutation=='unknown_payout': win['amount']=None
    elif mutation=='payout_conflict': win['amount']=str(Decimal(win['amount'])+1)
    elif mutation=='two_winners': args['payouts'][actor]=Decimal(1)
    elif mutation=='math_conflict': args['math']=dict(args['math'],**{'pass':False,'delta':'1'})
    elif mutation=='side_pot': args['nets'][actor]+=Decimal(1)
    elif mutation=='board_incomplete': args['board']=args['board'][:4]
    elif mutation=='all_in': args['players'][actor]['state']='ALL_IN'
    elif mutation=='known_loser_beats_winner':
        # Another hidden opponent remains missing, so capability must still
        # independently reject a known opponent beating the native winner.
        args['hole'][actor]=['Ac','As']
    elif mutation=='missing_start': hand['start_observed']=False
    elif mutation=='missing_end': hand['end_observed']=False
    assert source.authorize(hand,**args) is None, mutation
    assert source.last_rejection, mutation

@pytest.mark.parametrize('mutation',['no_no_show','unread_show','bad_hash','failed_frame','cached_string'])
def test_validator_keeps_missing_card_gate_when_native_authority_fails(corpus,source,mutation):
    hand=case(corpus); hidden=first(hand,'NO_SHOW')
    if mutation=='no_no_show':hand['messages']=[m for m in hand['messages'] if m['action']!='NO_SHOW']
    elif mutation=='unread_show':hidden['action']='SHOW';hidden['cards']=[]
    elif mutation=='bad_hash':hidden['source_rows'][0]['row_pixel_sha256']='0'*64
    elif mutation=='failed_frame':hidden['source_rows'][0]['ocr_status']='ERROR'
    else:hidden['source_rows']=[];hidden['source_row_ids']=[]
    v=validate_hand(hand,hidden_card_source=source)
    assert v['status']!='FULLY_VALID_HAND'
    assert not v.get('hidden_loser_authority')
    assert any(i['code']=='CARD_EVIDENCE_MISSING' for i in v['certification_blockers'])

def test_cached_hand_flags_and_fake_provider_do_not_authorize(corpus):
    hand=case(corpus)
    hand['hidden_loser_authority']={'rule':RULE,'hidden_losers':{'mooglle1':{}}}
    class CachedFlags:
        def authorize(self,*a,**kw):return hand['hidden_loser_authority']
    v=validate_hand(hand,hidden_card_source=CachedFlags())
    assert v['status']!='FULLY_VALID_HAND' and not v.get('hidden_loser_authority')

def test_stale_who_is_rejected(corpus,source):
    hand=case(corpus); args=arguments(hand)
    # Use a separate actor provider cache, without changing the shared source.
    clone=copy.copy(source);clone.actor_provider=copy.copy(source.actor_provider)
    clone.actor_provider.cache=copy.deepcopy(source.actor_provider.cache)
    proofs=copy.deepcopy(clone.actor_provider._proofs(hand))
    for proof in proofs.values():proof['status']='UNKNOWN'
    clone.actor_provider.cache[hand_key(hand)]=proofs
    assert clone.authorize(hand,**args) is None

def test_folded_no_show_does_not_receive_live_loser_capability(corpus,source):
    hand=case(corpus); args=arguments(hand); actor=first(hand,'NO_SHOW')['player']
    args['players'][actor]['state']='FOLDED'
    proof=source.authorize(hand,**args)
    assert proof is None or actor not in proof['hidden_losers']

@pytest.mark.parametrize('provider_name',['actor_provider','fresh_actors'])
def test_expired_showdown_who_interval_still_rejects(corpus,source,provider_name):
    hand=case(corpus);args=arguments(hand)
    assert source.authorize(hand,**args), source.last_rejection
    clone=copy.copy(source)
    provider=copy.copy(getattr(source,provider_name))
    provider.cache=copy.deepcopy(provider.cache)
    for proof in provider._proofs(hand).values():
        proof['identity_interval']=[0,1]
    setattr(clone,provider_name,provider)
    assert clone.authorize(hand,**args) is None
    assert 'STALE_WHO_INTERVAL' in clone.last_rejection

def test_no_native_source_cannot_be_replaced_by_valid_cached_strings(corpus,tmp_path,source):
    hand=case(corpus)
    missing=HiddenLoserSource(tmp_path,source.actor_provider)
    assert missing.authorize(hand,**arguments(hand)) is None

def test_cached_winning_five_must_equal_native_receipt(corpus,source):
    hand=case(corpus);args=arguments(hand)
    first(hand,'WIN')['win_best_five_cards']=['Qh','Qc','Ah','Kh','9c']
    assert source.authorize(hand,**args) is None

@pytest.mark.parametrize('failed_view',['hybrid','rejection','response'])
def test_unresolved_native_reading_cannot_reuse_stale_projected_fields(corpus,source,failed_view):
    hand=case(corpus);args=arguments(hand)
    assert source.authorize(hand,**args), source.last_rejection
    clone=copy.copy(source);clone.visual=copy.deepcopy(source.visual)
    river=first(hand,'RIVER')
    key=river['source_rows'][0]['card_row_key']
    reading=clone.visual[key]
    if failed_view=='hybrid':
        reading['hybrid']['cards']=[]
    elif failed_view=='rejection':
        reading['hybrid_authoritative_rejection']=True
    else:
        reading['raw_pixel_response']['status']='ERROR'
    assert clone.authorize(hand,**args) is None

def test_whole_native_frame_failure_blocks_hidden_loser_capability(corpus,source):
    hand=case(corpus);args=arguments(hand)
    assert source.authorize(hand,**args), source.last_rejection
    clone=copy.copy(source)
    table,session,_=hand_key(hand)
    frame=first(hand,'NO_SHOW')['source_rows'][0]['frame_id']
    clone.frame_failures=[*source.frame_failures,dict(table_id=table,session_id=session,
        frame=frame,reason='ValueError:INJECTED_NATIVE_FRAME_FAILURE',
        retained_native_row_ids=[])]
    assert clone.authorize(hand,**args) is None
    assert clone.last_rejection

def test_self_consistent_nick_pts_cannot_override_independent_chat_clock(corpus,source):
    hand=case(corpus);args=arguments(hand)
    assert source.authorize(hand,**args), source.last_rejection
    clone=copy.copy(source)
    clone.nick_rows=copy.deepcopy(source.nick_rows)
    clone.fresh_actors=copy.copy(source.fresh_actors)
    clone.fresh_actors.cache=copy.deepcopy(source.fresh_actors.cache)
    actor=first(hand,'NO_SHOW')['player']
    proofs=clone.fresh_actors._proofs(hand)
    proof=next(p for name,p in proofs.items()
        if name==actor or p.get('canonical_actor')==actor)
    anchor=proof['anchor']
    assert anchor['source_pts']>0
    altered=anchor['source_pts']+10**12
    anchor['source_pts']=altered
    clone.nick_rows[anchor['source_line']]['source_pts']=altered
    assert clone.authorize(hand,**args) is None
    assert clone.last_rejection

def test_empty_native_nick_png_fails_closed_without_escaping_cv2_error(corpus,source):
    hand=case(corpus);args=arguments(hand)
    assert source.authorize(hand,**args), source.last_rejection
    clone=copy.copy(source);clone.nick_rows=copy.deepcopy(source.nick_rows)
    actor=first(hand,'NO_SHOW')['player']
    anchor=source.fresh_actors.for_actor(hand,actor)['anchor']
    clone.nick_rows[anchor['source_line']]['png_base64']=''
    # cv2.imdecode(empty buffer) raises cv2.error, rather than returning None.
    assert clone.authorize(hand,**args) is None
    assert clone.last_rejection

def test_failed_frame_after_first_win_before_next_hand_still_blocks(corpus,source):
    hand=case(corpus);args=arguments(hand)
    assert source.authorize(hand,**args), source.last_rejection
    clone=copy.copy(source)
    table,session,_=hand_key(hand)
    frame=max(r['frame_id'] for m in hand['messages'] for r in m['source_rows'])+1
    clone.frame_failures=[*source.frame_failures,dict(table_id=table,session_id=session,
        frame=frame,reason='ValueError:INJECTED_NATIVE_SUFFIX_FAILURE',
        retained_native_row_ids=[])]
    assert clone.authorize(hand,**args) is None
    assert clone.last_rejection

def final_gate_control():
    row={'stable_row_id':'native-control-row'}
    record={'observed_hand':{'messages':[{'source_rows':[row]}]},'fully_validated':False}
    result=dict(status='RECOVERED_BY_LOGIC_MATH',dropped_events=[],inferred_events=[],
        kept_events=[{'source':{'rows':[row]}}],pot_delta=0,payout_eligible=True,
        inferred_amounts=[],derived_final_validation={'accepted':True})
    from parser_core.full_hh_profile import reconstruction_settings
    from parser_core.poker_reconstruction_layer import reconstruction_config
    return record,result,reconstruction_settings(reconstruction_config())

def test_current_final_gate_removes_only_accepted_validation_blocker():
    from parser_core.poker_reconstruction_layer import reconstruction_output_blockers
    record,result,config=final_gate_control()
    result['derived_final_validation']['accepted']=False
    assert reconstruction_output_blockers(record,result,config)==['EXISTING_PRODUCTION_VALIDATION_UNRESOLVED']
    result['derived_final_validation']['accepted']=True
    assert reconstruction_output_blockers(record,result,config)==[]

@pytest.mark.parametrize('mutation,expected',[
    ('dropped','SOURCE_ROW_REMOVAL_NOT_AUTHORIZED'),
    ('exhausted','NO_UNIQUE_COMPLETE_EXACT_SOLUTION'),
    ('accounting','EXACT_ELIGIBLE_ACCOUNTING_NOT_PROVEN'),
    ('coverage','KEPT_SOURCE_COVERAGE_NOT_EXACT'),
    ('invented','INVENTED_ACTION_NOT_AUTHORIZED'),
])
def test_hidden_acceptance_never_bypasses_current_final_output_gate(mutation,expected):
    from parser_core.poker_reconstruction_layer import reconstruction_output_blockers
    record,result,config=final_gate_control()
    if mutation=='dropped':result['dropped_events']=[{'source':'native-control-row'}]
    elif mutation=='exhausted':result.update(status='SEARCH_EXHAUSTED',exhausted=True)
    elif mutation=='accounting':result['pot_delta']=1
    elif mutation=='coverage':result['kept_events']=[]
    elif mutation=='invented':result['inferred_events']=[{'action':'CALL'}]
    blockers=reconstruction_output_blockers(record,result,config)
    assert expected in blockers
    assert 'EXISTING_PRODUCTION_VALIDATION_UNRESOLVED' not in blockers
