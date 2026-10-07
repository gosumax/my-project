"""Source-preserving FULL profile reconstruction contracts."""
from parser_core.poker_reconstruction.timeline import parse_rows
import copy
import pytest
import base64
import hashlib
import json
from pathlib import Path
from parser_core.full_hh_profile import reconstruction_settings
from parser_core.poker_pipeline import analyze_timeline
from parser_core.poker_reconstruction_layer import reconstruct_production_hands
from parser_core.poker_reconstruction.models import Event, Player
from parser_core.poker_reconstruction.solver import amount_variants
from parser_core.poker_reconstruction.state_machine import State
from parser_core.poker_reconstruction.reconstruct import reconstruct_hand


def observed(texts,failed=None):
    rows=[dict(stable_row_id=f's:R{i}',row_pixel_sha256=f'{i:064x}',table_id='T01',session_id='s',
               frame_id=1,row_index=i,epoch=1,origin='NEW',raw_ocr_text=t,video_time=0)
          for i,t in enumerate(texts)]
    return analyze_timeline(rows,ocr_status={r['stable_row_id']:'ERROR' if i==failed else 'OK'
                                           for i,r in enumerate(rows)})[0]


FOLD=['Hand #123456','Dealer: Starting a new hand','Dealer: A posts Small Blind 5',
      'Dealer: B posts Big Blind 10','Dealer: A folds','Dealer: B gets uncalled bet','Dealer: B wins 10']
CONFIG=reconstruction_settings(dict(enabled=True,max_nodes=20000,allow_logic_math_output=False))


def test_real_return_suffix_continuation_is_literal_and_adjacent_only():
    source=[dict(stable_row_id=f'r{i}',table_id='T07',session_id='s',frame_id=141,
                 row_index=i,epoch=1,source_index=i,raw_ocr_text=text)
            for i,text in enumerate(('Dealer: chicago117 gets uncalled bet','2500 back●'),1)]
    events=parse_rows(source)
    assert len(events)==1 and events[0].kind=='RETURN' and events[0].amount==2500
    assert events[0].source['rows']==source
    source[1]['frame_id']=142
    events=parse_rows(source)
    assert len(events)==2 and events[0].amount is None and events[1].kind=='UNKNOWN'


def test_unique_return_is_revalidated_without_changing_observed_hand():
    original=observed(FOLD);saved=copy.deepcopy(original)
    results,summary=reconstruct_production_hands(original,config=CONFIG)
    result=results[0]
    assert result['status']=='RECOVERED_BY_LOGIC_MATH' and result['solution_count']==1
    assert result['derived_final_validation']['accepted'] and result['output_eligible']
    candidate=result['derived_final_validation']['candidate_hand']
    returned=next(m for m in candidate['messages'] if m['action']=='UNCALLED_RETURN')
    assert returned['amount']=='5' and returned['observed_amount'] is None
    assert returned['raw_text']=='Dealer: B gets uncalled bet'
    assert original==saved and not original[0]['fully_validated']
    assert summary['strict_source_uniqueness']


def test_unique_accounting_never_bypasses_failed_ocr():
    result=reconstruct_production_hands(observed(FOLD,failed=2),config=CONFIG)[0][0]
    assert result['solution_count']==1
    assert not result['derived_final_validation']['accepted'] and not result['output_eligible']


def test_missing_call_candidates_pair_amount_with_intended_semantics():
    state=State(players={'A':Player(street_commit=5)},current_bet=20)
    variants=amount_variants(Event('CALL','A',None,'UNKNOWN'),state,[])
    assert {(e.semantics,e.amount) for e,p in variants}=={('DELTA',15),('TO',20)}


def test_equivalent_call_interpretations_have_one_full_economic_history():
    events=[Event('SB','A',5),Event('BB','B',10),Event('DEAL'),Event('RAISE','A',20,'TO',True),
            Event('CALL','B',None,'UNKNOWN',False)]
    events.extend(Event('STREET',street=street) for street in ('FLOP','TURN','RIVER'))
    events.append(Event('WIN','A',40))
    result=reconstruct_hand(events,hand_id='123',strict_source_uniqueness=True)
    assert result['raw_solution_count']==2 and result['solution_count']==1
    assert result['equivalent_interpretations'][0]['count']==2
    assert result['pot_delta']==0


def test_strict_mode_cannot_drop_an_observed_action_to_balance_pot():
    source=list(FOLD);source.insert(3,'Dealer: A posts Small Blind 5')
    result=reconstruct_production_hands(observed(source),config=CONFIG)[0][0]
    assert not result['output_eligible'] and not result['dropped_events']


def test_same_final_pot_with_different_ordered_payments_remains_ambiguous():
    events=[Event('SB','A',5),Event('BB','B',10),Event('DEAL'),Event('RAISE','A',20,'UNKNOWN'),
            Event('RAISE','B',50,'TO',True),Event('CALL','A',None,'UNKNOWN')]
    events.extend(Event('STREET',street=street) for street in ('FLOP','TURN','RIVER'))
    events.append(Event('WIN','B',100))
    result=reconstruct_hand(events,hand_id='123',strict_source_uniqueness=True)
    assert result['status']=='AMBIGUOUS' and result['solution_count']==2
    assert result['raw_solution_count']==4


def test_source_tampering_never_becomes_a_validated_candidate():
    from parser_core.reconstruction_validation import validate_reconstruction
    source=observed(FOLD)
    result=reconstruct_production_hands(source,config=CONFIG)[0][0]
    result['kept_events'][2]['source']['rows'][0]['raw_ocr_text']='Dealer: X posts Small Blind 5'
    assert validate_reconstruction(source[0]['observed_hand'],result)['reason']=='RECONSTRUCTION_SOURCE_FIELDS_CHANGED'


def test_hh_text_uses_validated_paid_call_and_preserves_original_text():
    from parser_core.hh_export import action_text
    action=dict(action='CALL',player='Alice',amount='300',raw_text='Dealer: Alice calls 600',
                amount_proof=dict(chosen_interpretation='TO',prior_street_contribution=300))
    assert action_text(action)=='Dealer: Alice calls 300'
    assert action['raw_text']=='Dealer: Alice calls 600'


def test_real_physical_return_and_show_wraps_preserve_every_source():
    from parser_core.poker_hand_assembly import assemble_messages
    from parser_core.poker_reconstruction.assembly import candidate_assemblies
    fixture=json.loads((Path(__file__).parent/'fixtures/full_hh_literal_wraps_smoke19.json').read_text('utf-8'))
    for row in fixture['physical'].values():
        assert hashlib.sha256(base64.b64decode(row['row_pixels_base64'])).hexdigest()==row['row_pixel_sha256']
    for group in fixture['groups']:
        saved=copy.deepcopy(group)
        messages=assemble_messages(group,require_pixel_cards=True,require_adjacency=True,
                                   pixel_sources={r['stable_row_id']:r for r in group})
        assert len(messages)==1
        assert messages[0]['action'] in {'UNCALLED_RETURN','SHOW'}
        assert messages[0]['source_rows']==saved
        if messages[0]['action']=='SHOW':assert len(messages[0]['cards'])==2
        else:assert messages[0]['amount'] in {'2500','24414'}
        events=parse_rows(group)
        assert len(events)==1 and events[0].kind in {'RETURN','SHOW'}
        indexed=[dict(r,source_index=i) for i,r in enumerate(group)]
        assemblies,exhausted,graph=candidate_assemblies(indexed)
        assert not exhausted and any(len(a)==1 and a[0].kind in {'RETURN','SHOW'} for a in assemblies)
        assert group==saved


def test_literal_wrap_does_not_cross_source_context_and_card_hash_is_required():
    from parser_core.poker_hand_assembly import observed_wrapped_continuation
    fixture=json.loads((Path(__file__).parent/'fixtures/full_hh_literal_wraps_smoke19.json').read_text('utf-8'))
    for group in fixture['groups']:
        assert observed_wrapped_continuation(group[:-1],group[-1])
        changed=copy.deepcopy(group[-1]);changed['epoch']+=1
        assert not observed_wrapped_continuation(group[:-1],changed)
        if len(group)==3:
            changed=copy.deepcopy(group[-1]);changed['row_pixel_sha256']='0'*64
            assert not observed_wrapped_continuation(group[:-1],changed)


def test_real_first_physical_starts_are_not_demoted_and_raw_is_unchanged():
    from parser_core.hand_lifecycle import classify_lifecycle, gate_first_context
    records=json.loads((Path(__file__).parent/'fixtures/full_hh_first_physical_starts_smoke19.json').read_text('utf-8'))
    frozen=copy.deepcopy([r['observed_hand'] for r in records])
    classify_lifecycle(records)
    for r in records:
        assert r['lifecycle']['status']=='COMPLETE'
        gate_first_context(r)
        assert r['lifecycle']['status']=='COMPLETE'
        assert len(r['lifecycle']['first_context_start_proof']['source_row_ids'])==3
    assert [r['observed_hand'] for r in records]==frozen


@pytest.mark.parametrize('change',['sidecar','birth','frame_zero','epoch','gap','number','ocr'])
def test_first_header_without_matching_physical_birth_remains_partial(change):
    from parser_core.hand_lifecycle import classify_lifecycle, gate_first_context
    r=json.loads((Path(__file__).parent/'fixtures/full_hh_first_physical_starts_smoke19.json').read_text('utf-8'))[1]
    classify_lifecycle([r])
    source=r['observed_hand']['messages'][0]['source_rows'][0]
    if change=='sidecar':source['source_kind']='SIDECAR_HEADER_RECEIPT'
    if change=='birth':source.pop('admission_ref')
    if change=='frame_zero':source['frame_id']=0
    if change=='epoch':source['epoch']+=1
    if change=='gap':source['row_index']+=1
    if change=='number':r['observed_hand']['messages'][1]['raw_text']='Dealer: Starting a new hand #2878369999'
    if change=='ocr':source['ocr_status']='ERROR'
    gate_first_context(r)
    assert r['lifecycle']['status']=='START_PARTIAL'


@pytest.mark.parametrize('texts,kind,amount',[
    (['Dealer: ubivashka88888 posts Big','Blind 5000'],'BIG_BLIND','5000'),
    (['Dealer: rinatromanov19 posts Ante 137','and is All-in'],'ANTE','137'),
    (['Dealer: krismoltisanta wins 27 440','from side pot #1 with Flush, J high'],'WIN','27440')])
def test_literal_blind_ante_allin_and_numbered_sidepot_wrap(texts,kind,amount):
    from parser_core.poker_hand_assembly import assemble_messages
    source=[dict(stable_row_id=f's:R{i}',table_id='T01',session_id='s',frame_id=150,
                 row_index=i,epoch=1,source_index=i,raw_ocr_text=text) for i,text in enumerate(texts)]
    message=assemble_messages(source,require_adjacency=True)
    assert len(message)==1 and message[0]['action']==kind and message[0]['amount']==amount
    if kind=='ANTE':assert message[0]['all_in']
    assert len(parse_rows(source))==1


def test_verified_native_card_group_has_no_competing_orphan_interpretation():
    from parser_core.poker_reconstruction.assembly import candidate_assemblies
    fixture=json.loads((Path(__file__).parent/'fixtures/full_hh_literal_wraps_smoke19.json').read_text('utf-8'))
    for group in fixture['groups'][2:]:
        rows=[dict(r,source_index=i) for i,r in enumerate(group)]
        paths,exhausted,_=candidate_assemblies(rows)
        assert not exhausted and len(paths)==1 and len(paths[0])==1
        assert paths[0][0].source['rows']==rows
