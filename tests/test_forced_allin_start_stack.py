"""General forced-post exhaustion and native saved-source regression."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from parser_core.simple_stack_policy import forced_allin_prefix_total, FORCED_ALL_IN_STACK_RULE
from parser_core.simple_saved_policy import apply_forced_allin_stack, prove_forced_allin_start_stack, SimpleSavedPolicyProvider


def post(kind,amount,all_in=False,actor='A'):
    return dict(action=kind,amount=str(amount),all_in=all_in,player=actor)


@pytest.mark.parametrize('posts,total',[
    ([post('ANTE',137,True)],137),
    ([post('ANTE',10),post('SMALL_BLIND',40,True)],50),
    ([post('ANTE',10),post('BIG_BLIND',90,True)],100),
    ([post('SMALL_BLIND',50,True)],50),
])
def test_general_forced_exhaustion_sums_own_observed_contributions(posts,total):
    prefix=[dict(action='HAND_HEADER'),post('ANTE',99,actor='B'),*posts] if posts[0]['action']=='ANTE' else posts
    before=copy.deepcopy(prefix)
    result=forced_allin_prefix_total(prefix,'A')
    assert result['rule']==FORCED_ALL_IN_STACK_RULE and result['calculated_start_stack']==total
    assert prefix==before


@pytest.mark.parametrize('messages',[
    [post('ANTE',137)],
    [dict(action='CALL',player='A'),post('ANTE',137,True)],
    [dict(action='CHECK',player='B'),post('BIG_BLIND',137,True)],
    [dict(action='UNKNOWN_EVENT'),post('ANTE',137,True)],
    [post('ANTE',10),post('ANTE',137,True)],
    [post('ANTE',10,True),post('BIG_BLIND',137,True)],
    [post('ANTE',10,actor='B'),post('BIG_BLIND',137,True)],
    [post('SMALL_BLIND',10),post('BIG_BLIND',137,True)],
    [post('ANTE',137,True),dict(action='CALL',player='A')],
    [post('ANTE',0,True)],
    [post('ANTE','unknown',True)],
])
def test_incomplete_or_conflicting_forced_prefix_has_no_stack_authority(messages):
    assert forced_allin_prefix_total(messages,'A') is None


@pytest.fixture(scope='module')
def saved_case():
    base=Path(__file__).resolve().parents[1]/'runs/hh_all_hands_20261009'
    def read(path):return [json.loads(s) for s in path.read_text('utf8').splitlines() if s.strip()]
    hand=next(r['observed_hand'] for r in read(base/'t06_geometry_partition/ALL57/logical_hand_analysis.jsonl')
              if r['observed_hand']['hand_id']=='2878365523')
    row=next(r for r in read(base/'sitout_export_v13_saved/retained_inputs/validation_stack_results.jsonl')
             if r['hand_id']==hand['hand_id'] and r['player']=='rinatromanov19')
    source=Path(r'C:\Users\Admin\.codex\tmp\hh95_source_cache_20261008\full_v1')
    return hand,row,source


def test_saved_wrapped_forced_allin_derives_137_without_a_numeric_or_sitout(saved_case):
    hand,original,source=saved_case;row=copy.deepcopy(original);before=copy.deepcopy(hand)
    apply_forced_allin_stack(row,hand,source)
    assert row['START_STACK']==137 and row['DISPLAYED_STACK'] is None and row['first_numeric_source'] is None
    assert row['player_status'] is None and row.get('sitout_authority') is None
    receipt=row['forced_allin_stack_receipt']
    assert receipt['forced_contributions'][0]['source_row_ids']==[
        'table_6a87602fd540be48a6ed6f61:F000008','table_6a87602fd540be48a6ed6f61:F000009']
    assert receipt['displayed_stack_authority'] is False and receipt['occupancy_authority'] is False
    assert hand==before and original['START_STACK'] is None


@pytest.mark.parametrize('mutation',['no_identity','wrong_seat','foreign_hand','missing_wrap','missing_prior_row','changed_amount','changed_raw','changed_frame','prior_voluntary'])
def test_native_allin_authority_rejects_tampering_and_wrong_context(saved_case,mutation):
    original,row,source=saved_case;hand=copy.deepcopy(original);identity=copy.deepcopy(row['identity_proof'])
    if mutation=='no_identity':identity=None
    elif mutation=='wrong_seat':identity['seat']='seat6'
    elif mutation=='foreign_hand':hand['hand_id']='foreign'
    elif mutation=='missing_wrap':hand['messages'][6]['source_rows'].pop()
    elif mutation=='missing_prior_row':hand['messages'].pop(2)
    elif mutation=='changed_amount':hand['messages'][6]['amount']='999'
    elif mutation=='changed_raw':hand['messages'][6]['source_rows'][1]['raw_ocr_text']='and is All-in!'
    elif mutation=='changed_frame':hand['messages'][6]['source_rows'][1]['frame_id']+=1
    elif mutation=='prior_voluntary':hand['messages'].insert(3,dict(action='CHECK',player='other'))
    assert prove_forced_allin_start_stack(hand,row['player'],row['seat'],source,identity) is None


def test_numeric_forced_allin_conflict_is_explicit_and_numeric_source_unchanged(saved_case):
    hand,original,source=saved_case;row=copy.deepcopy(original)
    row.update(DISPLAYED_STACK=10,START_STACK=147,first_numeric_source={'raw_text':'10','confidence':.99})
    before=copy.deepcopy(row)
    apply_forced_allin_stack(row,hand,source)
    assert row['forced_allin_stack_conflict'] is True
    assert row['reason']=='NUMERIC_FORCED_ALL_IN_STACK_CONFLICT'
    for field in ('DISPLAYED_STACK','START_STACK','first_numeric_source'):assert row[field]==before[field]


def test_numeric_agreement_keeps_existing_receipt_without_fallback(saved_case):
    hand,original,source=saved_case;row=copy.deepcopy(original)
    row.update(DISPLAYED_STACK=0,START_STACK=137,first_numeric_source={'raw_text':'0','confidence':.99})
    before=copy.deepcopy(row);apply_forced_allin_stack(row,hand,source)
    assert row==before


def test_cached_forced_allin_receipt_is_reproved_by_provider(saved_case):
    hand,original,source=saved_case;row=copy.deepcopy(original);apply_forced_allin_stack(row,hand,source)
    provider=SimpleSavedPolicyProvider.__new__(SimpleSavedPolicyProvider)
    provider.native_source=source
    provider.actor_sources=SimpleNamespace(canonical_hand=lambda h:copy.deepcopy(h))
    provider.seat_rows=lambda h:[copy.deepcopy(row)]
    assert provider.for_hand(hand)[0]['calculated_start_stack']==137
    row['forced_allin_stack_receipt']['calculated_start_stack']=999
    assert provider.for_hand(hand)==[]


@pytest.mark.parametrize('field,value',[('START_STACK',999),('DISPLAYED_STACK',0),
    ('first_numeric_source',{'raw_text':'0','confidence':.99}),('WHO_status','UNKNOWN'),
    ('forced_allin_stack_receipt',None),('forced_allin_stack_receipt',{})])
def test_cached_top_level_fallback_must_match_fresh_proof(saved_case,field,value):
    hand,original,source=saved_case;row=copy.deepcopy(original);apply_forced_allin_stack(row,hand,source)
    provider=SimpleSavedPolicyProvider.__new__(SimpleSavedPolicyProvider)
    provider.native_source=source
    provider.actor_sources=SimpleNamespace(canonical_hand=lambda h:copy.deepcopy(h))
    provider.seat_rows=lambda h:[copy.deepcopy(row)]
    row[field]=value
    assert provider.for_hand(hand)==[]


def test_failed_native_chat_frame_inside_prefix_blocks_authority(saved_case,monkeypatch):
    from parser_core import simple_saved_policy as policy
    hand,row,source=saved_case
    original=policy._forced_allin_native_inventory
    def failed_inventory(versions):
        inventory=dict(original(versions))
        inventory['frame_failures']=[*inventory['frame_failures'],dict(
            table_id=row['table_id'],session_id=hand['session_id'],frame=130,reason='SOURCE_JOIN_FAILED')]
        return inventory
    monkeypatch.setattr(policy,'_forced_allin_native_inventory',failed_inventory)
    assert prove_forced_allin_start_stack(hand,row['player'],row['seat'],source,row['identity_proof']) is None
