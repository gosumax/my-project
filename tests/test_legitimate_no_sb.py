"""Real saved prefix/manual review controls; no acquisition or solver."""
import copy
import gzip
import json
from pathlib import Path

import pytest

from parser_core.legitimate_no_sb import LegitimateNoSBSource, RULE, prove_no_sb
from parser_core.simple_saved_policy import (LineageActorProvider, SimpleSavedPolicyProvider,
    forced_correction, hand_key, no_sb_stack_row_valid, saved_nick_observations)
from parser_core.simple_seat_who import SeatIdentityLineage
from parser_core.simple_stack_policy import corrected_start_stack
from parser_core.poker_logic_engine import validate_hand

ROOT=Path(__file__).resolve().parents[1]

def read(path):
    return [json.loads(x) for x in path.read_text('utf8').splitlines() if x.strip()]

@pytest.fixture(scope='module')
def corpus():
    return json.loads(gzip.decompress((Path(__file__).parent/'fixtures/legitimate_no_sb_saved.json.gz').read_bytes()))

@pytest.fixture(scope='module')
def source(corpus):
    native=Path(corpus['source'])
    assert native.is_dir(), 'Real saved native parents required'
    hand=corpus['record']['observed_hand']
    states=json.loads((native/'seat_forced_summary.json').read_text('utf8'))['state_receipts']
    assoc=ROOT/'runs/hh_all_hands_20261009/sitout_export_v13_saved/retained_inputs/nick_occupant_associations.jsonl'
    actors=LineageActorProvider(SeatIdentityLineage(saved_nick_observations(native),states),
        {hand_key(hand):tuple(corpus['bounds'])},[hand],read(assoc))
    result=LegitimateNoSBSource(native,actors,corpus['review_path'])
    return result

def case(corpus,source):
    return source.actor_provider.canonical_hand(copy.deepcopy(corpus['record']['observed_hand']))

def first(hand,kind):
    return next(m for m in hand['messages'] if m['action']==kind)

def provider(corpus,source):
    hand=case(corpus,source)
    proof=prove_no_sb(hand,source)
    assert proof, source.last_rejection
    identities={p['seat']:p for p in source.actor_provider.for_hand(hand)}
    rows=copy.deepcopy(corpus['seat_rows'])
    for row in rows:
        identity=identities.get(row['seat'])
        if identity is None:continue
        row.update(player=identity['canonical_actor'],WHO_status='PROVEN',identity_proof=identity)
        forced,refs=forced_correction(hand,actor=row['player'],no_sb_source=source)
        row.update(forced=forced,forced_sources=refs,START_STACK=corrected_start_stack(row['DISPLAYED_STACK'],forced),
                   no_sb_authority=copy.deepcopy(proof))
    result=SimpleSavedPolicyProvider.__new__(SimpleSavedPolicyProvider)
    result.rows={hand_key(hand):rows};result.actor_sources=source.actor_provider
    result.no_sb_source=source;result.native_source=source.source
    return result,hand

def test_real_prefix_grants_only_explicit_absence(corpus,source):
    hand=case(corpus,source);before=copy.deepcopy(hand)
    proof=prove_no_sb(hand,source)
    assert proof, source.last_rejection
    assert proof['rule']==RULE and proof['sb_status']=='NOT_POSTED'
    assert proof['sb_actor'] is None and proof['posted_sb_total']==0
    assert proof['manual_authority']=='USER_MANUAL_SOURCE_VIDEO_VERIFICATION'
    assert proof['nominal_blinds']['SB']==300 and proof['nominal_blinds']['BB']==600
    for field in ('synthetic_post_authority','occupancy_authority','deal_complete_authority',
                  'start_stack_timing_authority','who_authority','historical_seat_geometry_authority'):
        assert proof[field] is False
    assert hand==before and len(proof['prefix_source_row_ids'])==11
    assert not any(m['action']=='SMALL_BLIND' for m in hand['messages'])

@pytest.mark.parametrize('mutation',[
    'no_start','boundary_conflict','excluded_row','table','session','hand','no_deal','two_deals',
    'no_bb','no_ante','sb','late_sb','early_call','duplicate_ante','allin_bb','allin_ante',
    'unknown_prefix','actor','raw','pixel','clock','time_base','ocr_failure','source_rows',
])
def test_missing_or_conflicting_evidence_rejects(corpus,source,mutation):
    hand=case(corpus,source);bb=first(hand,'BIG_BLIND');ante=first(hand,'ANTE')
    row=ante['source_rows'][0]
    if mutation=='no_start':hand['start_observed']=False
    elif mutation=='boundary_conflict':hand['boundary_conflicts']=['CONFLICT']
    elif mutation=='excluded_row':hand['excluded_message_indices']=[0]
    elif mutation=='table':hand['table_id']='T09'
    elif mutation=='session':hand['session_id']='foreign'
    elif mutation=='hand':hand['hand_id']='foreign'
    elif mutation=='no_deal':hand['messages']=[m for m in hand['messages'] if m['action']!='DEAL']
    elif mutation=='two_deals':hand['messages'].append(copy.deepcopy(first(hand,'DEAL')))
    elif mutation=='no_bb':hand['messages'].remove(bb)
    elif mutation=='no_ante':hand['messages'].remove(ante)
    elif mutation=='sb':bb['action']='SMALL_BLIND'
    elif mutation=='late_sb':hand['messages'].append(dict(copy.deepcopy(bb),action='SMALL_BLIND'))
    elif mutation=='early_call':ante['action']='CALL'
    elif mutation=='duplicate_ante':hand['messages'].insert(4,copy.deepcopy(ante))
    elif mutation=='allin_bb':bb['all_in']=True
    elif mutation=='allin_ante':ante['all_in']=True
    elif mutation=='unknown_prefix':ante['action']='UNKNOWN'
    elif mutation=='actor':ante['player']='unproved_actor'
    elif mutation=='raw':row['raw_ocr_text']='Dealer: foreign posts Ante 70'
    elif mutation=='pixel':row['row_pixel_sha256']='0'*64
    elif mutation=='clock':row['source_pts']=-1
    elif mutation=='time_base':row['time_base']=[1,1]
    elif mutation=='ocr_failure':row['ocr_status']='ERROR'
    elif mutation=='source_rows':ante['source_rows']=[]
    assert prove_no_sb(hand,source) is None, mutation

@pytest.mark.parametrize('mutation',['no_video','bad_video','foreign_video','prefix_hash','manual_hash','schema'])
def test_review_is_bound_to_native_source(corpus,source,tmp_path,mutation):
    review=copy.deepcopy(corpus['review'])
    if mutation=='no_video':review.pop('source_video_sha256')
    elif mutation=='bad_video':review['source_video_sha256']=''
    elif mutation=='foreign_video':review['source_video_sha256']='0'*64
    elif mutation=='prefix_hash':review['prefix_sha256']='0'*64
    elif mutation=='manual_hash':review['manual_source']['sha256']='0'*64
    else:review['schema']='cached_assertion'
    path=tmp_path/'review.jsonl';path.write_text(json.dumps(review)+'\n','utf8')
    clone=copy.copy(source)
    fresh=LegitimateNoSBSource(source.source,source.actor_provider,path)
    # Reuse immutable, already verified native inventory. Review remains new.
    for k,v in vars(source).items():
        if k not in {'review_path','review_hash','review_stamp','reviews','last_rejection'}:setattr(fresh,k,v)
    if mutation=='no_video':fresh.metadata=dict(source.metadata,source_video_sha256=None)
    assert prove_no_sb(case(corpus,source),fresh) is None

def test_changed_review_and_missing_native_fail_closed(corpus,source,tmp_path):
    path=tmp_path/'review.jsonl';path.write_text(json.dumps(corpus['review'])+'\n','utf8')
    fresh=LegitimateNoSBSource(source.source,source.actor_provider,path)
    path.write_text('{}\n','utf8')
    assert prove_no_sb(case(corpus,source),fresh) is None
    missing=LegitimateNoSBSource(tmp_path,source.actor_provider,corpus['review_path'])
    assert prove_no_sb(case(corpus,source),missing) is None

@pytest.mark.parametrize('mutation',['unknown','wrong_seat','expired_interval','replacement'])
def test_stale_or_conflicting_who_rejects(corpus,source,mutation):
    hand=case(corpus,source);assert prove_no_sb(hand,source),source.last_rejection
    clone=copy.copy(source);clone.actor_provider=copy.copy(source.actor_provider)
    clone.actor_provider.cache=copy.deepcopy(source.actor_provider.cache)
    proofs=clone.actor_provider._proofs(hand)
    p=next(iter(proofs.values()))
    if mutation=='unknown':p['status']='UNKNOWN'
    elif mutation=='wrong_seat':p['seat']='seat9'
    elif mutation=='expired_interval':p['identity_interval']=[0,1]
    else:
        clone.fresh_actors=copy.copy(source.fresh_actors)
        clone.fresh_actors.cache=copy.deepcopy(source.fresh_actors.cache)
        p=next(iter(clone.fresh_actors._proofs(hand).values()))
        p['boundaries']=p.get('boundaries',[])+[dict(frame=p['anchor']['frame'],kind='REPLACE')]
    assert prove_no_sb(hand,clone) is None

def test_unknown_absence_and_fake_flags_keep_missing_blind_gate(corpus,source):
    hand=case(corpus,source);hand['no_sb_authority']={'rule':RULE}
    class Flags:
        def authorize(self,h):return hand['no_sb_authority']
    for obj in (None,Flags()):
        assert prove_no_sb(hand,obj) is None
        amounts,_=forced_correction(hand,actor='denis797',no_sb_source=obj)
        assert amounts['SB'] is None
        v=validate_hand(hand,no_sb_source=obj)
        assert v['status']=='INCOMPLETE_START' and not v.get('no_sb_authority')
    v=validate_hand(hand,no_sb_source=source)
    assert v.get('no_sb_authority') and v['status']!='INCOMPLETE_START'
    assert v['math']['pass'] and v['math']['delta']=='0'

def test_numeric_stacks_add_only_actual_forced_posts(corpus,source):
    p,hand=provider(corpus,source)
    receipts=p.for_hand(hand)
    assert len(receipts)==6
    for r in receipts:
        assert r['calculated_start_stack']==r['displayed_stack']+70+(600 if r['actor']=='denis797' else 0)
        assert r['no_sb_authority']['sb_status']=='NOT_POSTED'
        assert not any(x.get('action')=='SMALL_BLIND' for x in r['own_forced_posts'])

@pytest.mark.parametrize('mutation',['deleted_capability','null_capability','source_off','amount','seat','forced','refs'])
def test_stack_cached_zero_does_not_bypass_reproof(corpus,source,mutation):
    p,hand=provider(corpus,source)
    row=next(r for r in p.rows[hand_key(hand)] if r['player']=='denis797')
    if mutation=='deleted_capability':row.pop('no_sb_authority');p.no_sb_source=None
    elif mutation=='null_capability':row['no_sb_authority']=None
    elif mutation=='source_off':p.no_sb_source=None
    elif mutation=='amount':row['START_STACK']+=300
    elif mutation=='seat':row['seat']='seat9'
    elif mutation=='forced':row['forced']['SB']=300
    else:row['forced_sources']=[]
    assert not no_sb_stack_row_valid(row,hand,p.no_sb_source)
    assert not any(r['actor']=='denis797' for r in p.for_hand(hand))

def hh_case(corpus,source):
    p,hand=provider(corpus,source)
    hand=copy.deepcopy(corpus['candidate_hand'])
    return dict(status='COMPLETE',certified=True,blockers=[],table=hand['table_id'],
        session=hand['session_id'],hand_number=hand['hand_id'],actions=hand['messages'],
        final_pot=corpus['calculated_pot'],
        derived_final_validation=dict(candidate_hand=hand),no_sb_authority=prove_no_sb(hand,source),
        players=[dict(player=r['actor'],seat=r['seat'],START_STACK=r['calculated_start_stack'],START_STACK_status='PROVEN')
                 for r in p.for_hand(hand)])

def test_hh_uses_nominal_level_without_invented_post(corpus,source):
    from parser_core.hh_text import render_complete
    hand=hh_case(corpus,source);before=copy.deepcopy(hand)
    text=render_complete(hand,no_sb_source=source)
    assert '(300/600)' in text and 'posts small blind' not in text.lower()
    assert hand==before
    with pytest.raises(ValueError,match='LEGITIMATE_NO_SB'):render_complete(hand)

def test_source_exact_service_regrouping_preserves_capability(corpus,source):
    observed=case(corpus,source);candidate=copy.deepcopy(corpus['candidate_hand'])
    assert [r for m in observed['messages'] for r in m['source_rows']]==[r for m in candidate['messages'] for r in m['source_rows']]
    assert prove_no_sb(candidate,source)==prove_no_sb(observed,source)
    assert validate_hand(candidate,no_sb_source=source)['status']=='FULLY_VALID_HAND'

@pytest.mark.parametrize('mutation',['capability','scope','actions','bb_amount'])
def test_hh_cannot_reuse_foreign_or_tampered_authority(corpus,source,mutation):
    from parser_core.hh_text import render_complete
    hand=hh_case(corpus,source)
    if mutation=='capability':hand['no_sb_authority']['nominal_blinds']['SB']=1
    elif mutation=='scope':hand['hand_number']='foreign'
    elif mutation=='actions':hand['actions']=hand['actions'][1:]
    else:first(hand['derived_final_validation']['candidate_hand'],'BIG_BLIND')['amount']='601'
    with pytest.raises(ValueError,match='LEGITIMATE_NO_SB'):render_complete(hand,no_sb_source=source)

def test_ordinary_sb_correction_is_unchanged(corpus,source):
    hand=case(corpus,source)
    post=dict(copy.deepcopy(first(hand,'BIG_BLIND')),action='SMALL_BLIND',player='xoyt1998',amount='300')
    hand['messages'].insert(3,post)
    assert prove_no_sb(hand,source) is None
    for actor in ('xoyt1998','denis797'):
        assert forced_correction(hand,actor=actor)==forced_correction(hand,actor=actor,no_sb_source=source)

@pytest.mark.parametrize('mutation',['null_owner_other_frame','foreign_first_owner','second_header','foreign_marker'])
def test_numbered_header_ownership_exception_is_narrow(corpus,source,mutation):
    hand=case(corpus,source);assert prove_no_sb(hand,source),source.last_rejection
    clone=copy.copy(source);clone.inventory=dict(source.inventory)
    if mutation in {'null_owner_other_frame','foreign_first_owner'}:
        rid=first(hand,'ANTE')['source_row_ids'][0]
        owner=next(r for r in source.inventory['retained_row_ownership'] if r['row_id']==rid)
        changed=dict(owner,hand_id=None)
        if mutation=='foreign_first_owner':changed['first_proven_hand_id']='foreign'
        else:
            # Simulate an inventory whose confirmed numbered anchor precedes
            # this unowned physical row. Same-frame permission must not leak.
            header_id=first(hand,'HAND_HEADER')['source_row_ids'][0]
            clone.inventory['positive_headers']=[dict(r,frame=r['frame']-1) if r['row_id']==header_id else r
                                                 for r in source.inventory['positive_headers']]
        clone.inventory['retained_row_ownership']=[changed if r is owner else r for r in source.inventory['retained_row_ownership']]
    elif mutation=='second_header':hand['messages'].insert(1,copy.deepcopy(first(hand,'HAND_HEADER')))
    else:first(hand,'NEW_HAND')['hand_number']='foreign'
    assert prove_no_sb(hand,clone) is None
    if mutation in {'null_owner_other_frame','foreign_first_owner'}:
        assert clone.last_rejection=='NATIVE_PREFIX_CONTEXT_CONFLICT'
