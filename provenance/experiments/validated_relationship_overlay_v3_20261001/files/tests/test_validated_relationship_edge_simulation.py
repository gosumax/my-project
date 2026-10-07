import copy
import hashlib

import pytest

from experiments.validated_relationship_edge_simulation import simulate
from parser_core.edge_gate import classify_edges
from experiments import run_validated_relationship_edge_simulation as runner


def source(*texts):
    return [dict(table_id='T01', session_id='s', sequence_id=i+1,
        stable_row_id=f's:R{i}', frame_id=i, row_index=0, epoch=0,
        raw_ocr_text=text, status='RAW_UNREVIEWED', notes_error='',origin='TRACKED',
        row_pixel_sha256=hashlib.sha256(text.encode()).hexdigest()) for i,text in enumerate(texts)]


def link(rows,a,b,label='VISUALLY_SAME'):
    old,new=rows[a],rows[b]
    return dict(pair_id=f'{a}->{b}',case_id=f'V{a}-{b}',visual_label=label,
        relationship='VALIDATED_SAME_PHYSICAL_OCCURRENCE',table_id='T01',session_id='s',
        old_track_id=old['stable_row_id'],new_track_id=new['stable_row_id'],
        old_first_frame=old['frame_id'],new_first_frame=new['frame_id'],
        old_last_frame=old['frame_id'],new_last_frame=new['frame_id'],
        old_pixel_hash=old['row_pixel_sha256'],new_pixel_hash=new['row_pixel_sha256'],
        old_source_pixel_hash=old['row_pixel_sha256'],new_source_pixel_hash=new['row_pixel_sha256'],
        old_source_ocr_ok=True,new_source_ocr_ok=True)


def test_tail_ownership_without_admission_or_duplicate_event_suppression():
    rows=source('A folds','Starting a new hand #2','A folds','B wins 10')
    before=copy.deepcopy(rows)
    result=simulate(rows,[link(rows,0,2)])
    a=result.attribution[2]
    assert a['ownership_evidence']=='PREVIOUS_HAND_TAIL_SUPPORTED'
    assert a['relationship_path_used']==['0->2']
    assert a['physical_first_seen_frame']==0
    assert a['duplicate_poker_event_suppressed'] is False
    assert result.rows==classify_edges(rows).rows and rows==before


@pytest.mark.parametrize('label',['VISUALLY_DIFFERENT','VISUALLY_UNCERTAIN','AMBIGUOUS',None])
def test_negative_labels_never_enter_lookup(label):
    rows=source('A folds','A folds')
    result=simulate(rows,[link(rows,0,1,label)])
    assert result.relationships==[] and result.chains==[]
    assert all(a['ownership_evidence']=='NO_VALIDATED_CONTINUITY' for a in result.attribution)


def test_equal_text_without_validated_pair_never_links():
    rows=source('A folds','A folds','A folds')
    result=simulate(rows,[link(rows,0,1)])
    assert result.attribution[2]['continuity_component_pair_ids']==[]
    assert result.attribution[2]['physical_first_seen_frame']==2


def test_chain_requires_all_edges_and_records_exact_path():
    rows=source('A folds','A folds','A folds')
    result=simulate(rows,[link(rows,0,1),link(rows,1,2)])
    assert result.attribution[2]['relationship_path_used']==['0->1','1->2']
    assert result.chains[0]['track_ids']==['s:R0','s:R1','s:R2']
    missing=simulate(rows,[link(rows,0,1)])
    assert missing.attribution[2]['continuity_component_pair_ids']==[]


def test_chain_epoch_conflict_never_resolves_gate():
    rows=source('Starting a new hand #1','A folds','A folds','B wins 10')
    rows[2]['epoch']=1
    result=simulate(rows,[link(rows,1,2)])
    assert result.chains[0]['conflicts']==['EPOCH_DISCONTINUITY']
    assert result.attribution[2]['ownership_evidence']=='UNRESOLVED_CHAIN_CONFLICT'
    assert all(r['edge_status']=='UNRESOLVED_EDGE' for r in result.rows)
    assert 'EPOCH_DISCONTINUITY' in result.attribution[2]['reason_after']


def test_branch_is_unresolved_even_when_every_pair_is_same():
    rows=source('A folds','A folds','A folds')
    result=simulate(rows,[link(rows,0,1),link(rows,0,2)])
    assert result.chains[0]['conflicts']==['BRANCH_OR_CYCLE']
    assert result.attribution[2]['ownership_evidence']=='UNRESOLVED_CHAIN_CONFLICT'


def test_real_boundary_conflict_survives_validated_continuity():
    rows=source('Starting a new hand #1','A folds','Hand #2','A folds','B wins 10')
    result=simulate(rows,[link(rows,1,3)])
    assert result.rows==classify_edges(rows).rows
    assert all(r['edge_status']=='UNRESOLVED_EDGE' for r in result.rows)
    assert all('CONFLICTING_HAND_NUMBERS' in a['reason_after'] for a in result.attribution)
    assert result.attribution[3]['ownership_evidence']=='UNRESOLVED_BOUNDARY_OWNERSHIP'


def test_same_frame_before_boundary_uses_source_row_order():
    rows=source('A folds','Starting a new hand #2','A folds','B wins 10')
    rows[0]['frame_id']=rows[1]['frame_id']=0
    rows[0]['row_index']=1;rows[1]['row_index']=2
    result=simulate(rows,[link(rows,0,2)])
    assert result.attribution[2]['ownership_evidence']=='PREVIOUS_HAND_TAIL_SUPPORTED'


def test_missing_start_and_end_stay_partial():
    rows=source('A folds','A folds')
    result=simulate(rows,[link(rows,0,1)])
    assert all(r['edge_status']=='PARTIAL_START' for r in result.rows)
    rows=source('Starting a new hand #1','A folds','A folds')
    result=simulate(rows,[link(rows,1,2)])
    assert all(r['edge_status']=='PARTIAL_END' for r in result.rows)
    assert all(a['reason_after']==['MISSING_PROVEN_HAND_END'] for a in result.attribution)


@pytest.mark.parametrize('field,value',[('old_source_pixel_hash','wrong'),('table_id','T02'),('new_first_frame',99)])
def test_provenance_mismatch_rejected(field,value):
    rows=source('A folds','A folds')
    rel=link(rows,0,1);rel[field]=value
    with pytest.raises(ValueError,match='provenance mismatch'):
        simulate(rows,[rel])


@pytest.mark.parametrize('field,value',[('last_frame',999999),('track_id','s:swapped')])
def test_sealed_lifetime_or_endpoint_mutation_rejected(monkeypatch,field,value):
    original=runner.read
    def mutated(path):
        values=original(path)
        if path.name=='sealed_evidence.jsonl': values[0]['old'][field]=value
        return values
    monkeypatch.setattr(runner,'read',mutated)
    with pytest.raises(ValueError,match='Sealed endpoints/lifetimes differ'):
        runner.load_validated(runner.ROOT/'runs/text_first_precision_review_smoke17_20261001')


def test_frozen_candidate_hash_mismatch_rejected(monkeypatch):
    original=runner.digest
    def mutated(path):
        if 'cross_track_linker_smoke17_20261001' in str(path): return 'wrong'
        return original(path)
    monkeypatch.setattr(runner,'digest',mutated)
    with pytest.raises(ValueError,match='Frozen candidate evidence hash mismatch'):
        runner.load_validated(runner.ROOT/'runs/text_first_precision_review_smoke17_20261001')


def test_causal_endpoint_crop_and_source_crop_hashes_remain_separate():
    rows=source('A folds','A folds')
    rel=link(rows,0,1)
    rel['old_pixel_hash']=hashlib.sha256(b'different observed crop').hexdigest()
    result=simulate(rows,[rel])
    assert len(result.relationships)==1
    assert result.relationships[0]['old_pixel_hash']!=result.relationships[0]['old_source_pixel_hash']
    assert result.rows==classify_edges(rows).rows


def previous_tail_source():
    rows=source('Dealer: Starting a new hand','A posts Small Blind 5','B posts Big Blind 10',
                'Dealing cards','Dealer: Starting a new hand','A folds','B wins 10')
    for p in (0,4): rows[p].update(origin='UNRESOLVED_IDENTITY',status='UNRESOLVED')
    return rows


def test_validated_previous_tail_removes_only_unsafe_current_window_veto():
    rows=previous_tail_source();before=copy.deepcopy(rows)
    baseline=classify_edges(rows)
    assert baseline.rows[1]['edge_status']=='UNRESOLVED_EDGE'
    result=simulate(rows,[link(rows,0,4)])
    assert [r['edge_status'] for r in result.rows]==['PARTIAL_START','KEEP','KEEP','KEEP','PARTIAL_START','KEEP','KEEP']
    assert result.attribution[4]['ownership_evidence']=='PREVIOUS_HAND_TAIL_SUPPORTED'
    assert result.attribution[1]['window_continuity_proofs'][0]['relationship_paths']==[['0->4']]
    assert rows==before
    assert [{k:v for k,v in r.items() if k!='edge_status'} for r in result.rows]==before


@pytest.mark.parametrize('block',['epoch','number','second_newhand','unvalidated'])
def test_previous_tail_cannot_discharge_real_conflicts(block):
    rows=previous_tail_source()
    if block=='epoch': rows[-1]['epoch']=1
    if block=='number': rows[3]['raw_ocr_text']='Hand #1';rows[5]['raw_ocr_text']='Hand #2'
    if block=='second_newhand':
        rows[5].update(raw_ocr_text='Starting a new hand',origin='UNRESOLVED_IDENTITY',status='UNRESOLVED')
    links=[] if block=='unvalidated' else [link(rows,0,4)]
    result=simulate(rows,links)
    assert result.rows==classify_edges(rows).rows


def test_tail_owner_uncertain_cannot_be_admitted_as_duplicate_event():
    rows=source('Starting a new hand #1','A bets 5','Dealer: Starting a new hand','A posts Small Blind 5',
                'B posts Big Blind 10','Dealing cards','Dealer: Starting a new hand','A folds','B wins 10')
    for p in (2,6): rows[p].update(origin='UNRESOLVED_IDENTITY',status='UNRESOLVED')
    baseline=classify_edges(rows)
    result=simulate(rows,[link(rows,2,6)])
    assert result.rows[6]['edge_status']=='UNRESOLVED_EDGE'
    assert result.rows[3]['edge_status']=='KEEP'
    assert baseline.rows[3]['edge_status']=='UNRESOLVED_EDGE'


def test_unrelated_action_chain_conflict_blocks_tail_based_window_admission():
    rows=previous_tail_source()
    # The valid marker tail cannot hide a separate branched continuity chain.
    links=[link(rows,0,4),link(rows,3,5),link(rows,3,6)]
    result=simulate(rows,links)
    assert result.attribution[5]['chain_conflicts']==['BRANCH_OR_CYCLE']
    assert result.rows==classify_edges(rows).rows


def test_previous_tail_payout_cannot_prove_current_hand_end():
    rows=source('Dealer: Starting a new hand','B wins 10','A posts Small Blind 5',
                'B posts Big Blind 10','Dealing cards','Dealer: Starting a new hand','A calls 5','B wins 10')
    for p in (0,5): rows[p].update(origin='UNRESOLVED_IDENTITY',status='UNRESOLVED')
    result=simulate(rows,[link(rows,0,5),link(rows,1,7)])
    assert result.rows[2]['edge_status']=='PARTIAL_END'
    assert result.attribution[2]['effective_end_reason'] is None
    assert result.rows[7]['edge_status']=='PARTIAL_START'


@pytest.mark.parametrize('field,value',[('origin','TRACKED'),('status','ERROR'),
    ('status','UNREADABLE'),('notes_error','OCR_FAILURE'),('raw_ocr_text','')])
def test_previous_tail_never_discharges_nonidentity_reading_failure(field,value):
    rows=previous_tail_source();rel=link(rows,0,4)
    rows[4][field]=value
    result=simulate(rows,[rel])
    assert result.rows==classify_edges(rows).rows


def test_previous_tail_raw_ocr_failure_remains_unresolved():
    rows=previous_tail_source();rel=link(rows,0,4);rel['new_source_ocr_ok']=False
    result=simulate(rows,[rel])
    assert result.rows==classify_edges(rows).rows
