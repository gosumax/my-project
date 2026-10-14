import copy
import json
import math
from pathlib import Path

import pytest

from parser_core.poker_hand_assembly import assemble_messages, assemble_hands, make_message
from parser_core.poker_reconstruction.assembly import candidate_assemblies
from parser_core.poker_reconstruction.reconstruct import reconstruct_hand
from parser_core.poker_reconstruction.timeline import parse_rows
from parser_core.win_tail_evidence import prove_win_tail, verified_win_tail, project_win_tails, current_win_context


MULTIPLE_SOURCES=json.loads((Path(__file__).parent/'fixtures/win_tail_multiple_sources.json').read_text('utf-8'))
CROSS_EPOCH_SOURCE=json.loads((Path(__file__).parent/'fixtures/win_tail_t09_cross_epoch_source.json').read_text('utf-8'))


def test_cross_epoch_native_win_uses_numbered_hand_binding_and_preserves_reading():
    source=copy.deepcopy(CROSS_EPOCH_SOURCE);before=copy.deepcopy(source)
    rows,reading,context=source['rows'],source['reading'],source['context']
    assert reading['hand_id'] is None and len({r['epoch'] for r in context})>1
    assert prove_win_tail(rows,reading,context) is None
    proof=prove_win_tail(rows,reading,context,hand_context=True)
    assert proof and proof['hand_id']=='2878366656'
    assert proof['cards']==['7s','7d','Kh','Qs','6d'] and proof['actor']=='retre1'
    assert proof['hand_binding']['source_row_ids']==[r['stable_row_id'] for r in context]
    assert proof['reading']==reading and source==before
    rows[0]['win_best_five_proof']=proof
    assert verified_win_tail(rows,context=context)==proof
    message=make_message(rows,require_pixel_cards=True)
    assert message['cards']==[] and message['win_best_five_cards']==proof['cards']


@pytest.mark.parametrize('mutation',['no_header','context_session','missing_epoch','duplicate_source',
    'reordered_context','intervening_hand','foreign_claim','foreign_row_hand','tail_epoch','pixel_hash'])
def test_cross_epoch_hand_binding_rejects_missing_or_conflicting_source(mutation):
    source=copy.deepcopy(CROSS_EPOCH_SOURCE)
    rows,reading,context=source['rows'],source['reading'],source['context']
    if mutation=='no_header':context=context[1:]
    elif mutation=='context_session':context[1]['session_id']='foreign'
    elif mutation=='missing_epoch':context[1]['epoch']=None
    elif mutation=='duplicate_source':context[1]['stable_row_id']=context[0]['stable_row_id']
    elif mutation=='reordered_context':context[0],context[1]=context[1],context[0]
    elif mutation=='intervening_hand':context[3]['raw_ocr_text']='Hand #123'
    elif mutation=='foreign_claim':reading['hand_id']='foreign'
    elif mutation=='foreign_row_hand':rows[2]['hand_id']='foreign'
    elif mutation=='tail_epoch':rows[2]['epoch']+=1
    elif mutation=='pixel_hash':rows[2]['row_pixel_sha256']='0'*64
    assert prove_win_tail(rows,reading,context,hand_context=True) is None


def test_cross_epoch_projector_and_current_consumer_reproduce_membership():
    source=copy.deepcopy(CROSS_EPOCH_SOURCE);rows=source['context']
    for row in rows:
        row['event_type']='WIN' if row['stable_row_id']==source['rows'][0]['stable_row_id'] else 'UNKNOWN_EVENT'
    reading=source['reading'];before=copy.deepcopy(rows);reading_before=copy.deepcopy(reading)
    assert project_win_tails(rows,{reading['row_key']:reading})==1
    index=len(rows)-3;proof=rows[index]['win_best_five_proof']
    assert verified_win_tail(rows[index:],context=current_win_context(rows,index))==proof
    for row,original in zip(rows,before):assert {k:v for k,v in row.items() if k!='win_best_five_proof'}==original
    assert reading==reading_before
    # A retained receipt cannot survive changed source-hand membership.
    rows[0]['raw_ocr_text']='Hand #123'
    assert verified_win_tail(rows[index:],context=current_win_context(rows,index)) is None


def test_cross_epoch_forged_membership_receipt_fails_reproduction():
    source=copy.deepcopy(CROSS_EPOCH_SOURCE);rows=source['rows']
    proof=prove_win_tail(rows,source['reading'],source['context'],hand_context=True)
    assert proof
    proof['hand_binding']['hand_id']='foreign'
    rows[0]['win_best_five_proof']=proof
    assert verified_win_tail(rows,context=source['context']) is None


@pytest.mark.parametrize('position,field,value',[
    (0,'hand_id','foreign'),(2,'hand_id','foreign'),(2,'hand_number','foreign'),
    (2,'source_file','foreign-source'),(0,'origin','UNRESOLVED_IDENTITY'),
    (0,'status','UNRESOLVED'),(2,'canonical_conflict','SOURCE_CONFLICT'),
    (2,'canonical_duplicate_of','other-row'),(2,'row_identity_status','UNRESOLVED'),
])
def test_cross_epoch_negative_source_evidence_blocks_projection_and_retained_proof(position,field,value):
    source=copy.deepcopy(CROSS_EPOCH_SOURCE);rows=source['context']
    for row in rows:
        row['event_type']='WIN' if row['stable_row_id']==source['rows'][0]['stable_row_id'] else 'UNKNOWN_EVENT'
    reading=source['reading'];index=len(rows)-3
    assert project_win_tails(rows,{reading['row_key']:reading})==1
    assert verified_win_tail(rows[index:],context=current_win_context(rows,index))
    rows[position][field]=value
    if field=='source_file':rows[0]['source_file']='original-source'
    assert verified_win_tail(rows[index:],context=current_win_context(rows,index)) is None
    rows[index].pop('win_best_five_proof')
    assert project_win_tails(rows,{reading['row_key']:reading})==0


@pytest.mark.parametrize('source',MULTIPLE_SOURCES,ids=lambda s:s['reading']['row_key'])
def test_native_wrapped_payout_and_float_pts_reproduce_separate_authority(source):
    source=copy.deepcopy(source)
    rows,reading,context=source['rows'],source['reading'],source['context']
    proof=prove_win_tail(rows,reading,context)
    assert proof and proof['cards']==reading['cards']
    rows[0]['win_best_five_proof']=proof
    message=make_message(rows,require_pixel_cards=True)
    assert message['action']=='WIN' and message['cards']==[]
    assert message['amount']==proof['amount']
    assert message['win_best_five_cards']==proof['cards']
    # The exact floating representation is required, not a tolerance window.
    reading['video_time']=math.nextafter(reading['video_time'],math.inf)
    assert prove_win_tail(rows,reading,context) is None


@pytest.mark.parametrize('description,cards',[
    ('Straight, K-9',['8h','9h','Th','Jh','Qs']),
    ('Flush, K high',['As','Js','7s','6s','5s']),
    ('High card, K',['As','Qd','9d','8h','7d']),
    ('Pair, JJ with K kicker',['Jc','Jd','As','Kh','Qs']),
    ('Two pairs, 9988 with K kicker',['9h','9s','8d','8s','Ks']),
])
def test_literal_rank_description_conflicts_are_not_accepted(description,cards):
    from parser_core.win_tail_evidence import _description_matches
    assert not _description_matches(description,cards)


@pytest.fixture
def source():
    return json.loads((Path(__file__).parent/'fixtures/win_tail_t07_source.json').read_text('utf-8'))


def test_existing_native_reading_gains_only_winning_five_authority(source):
    rows, reading, context = source['rows'], source['reading'], source['context']
    before = copy.deepcopy(source)
    proof = prove_win_tail(rows, reading, context)
    assert proof and proof['cards']==['Qc','Qd','3h','3s','Kd']
    assert proof['hand_id']=='2878367230' and proof['evidence_kind']=='WIN_BEST_FIVE'
    assert source==before
    rows[0]['win_best_five_proof']=proof
    assert verified_win_tail(rows)==proof
    message = make_message(rows,require_pixel_cards=True)
    assert message['action']=='WIN' and message['amount']=='56250'
    assert message['cards']==[] and message['win_best_five_cards']==proof['cards']
    assert 'card_provenance' not in message
    messages = assemble_messages(rows,require_pixel_cards=True,require_adjacency=True)
    assert len(messages)==1 and messages[0]['source_rows']==rows


@pytest.mark.parametrize('mutation',[
    'foreign_hand','context_header','missing_context','row_gap','foreign_session','foreign_epoch',
    'row_pixel','crop_png','tile_png','card_conflict','duplicate_card','development','hybrid_rejection',
    'suit_conflict','suit_not_confident','suit_low_score','suit_nan','rank_indices_empty','rank_indices_foreign',
    'rank_low_score','rank_not_accepted','description_ranks','description_class','foreign_pts','missing_config',
    'malformed_reading','malformed_response','malformed_hybrid','oversized_pts','boolean_pts',
    'negative_time_base','nan_time'])
def test_conflicting_or_missing_source_withholds_proof(source,mutation):
    rows, reading, context = source['rows'],source['reading'],source['context']
    tile=reading['hybrid']['tiles'][0]
    if mutation=='foreign_hand':reading['hand_id']='OTHER_HAND'
    elif mutation=='context_header':context[0]['raw_ocr_text']='#123'
    elif mutation=='missing_context':context=None
    elif mutation=='row_gap':rows[2]['row_index']+=1
    elif mutation=='foreign_session':rows[1]['session_id']='OTHER'
    elif mutation=='foreign_epoch':rows[1]['epoch']+=1
    elif mutation=='row_pixel':rows[2]['row_pixel_sha256']='0'*64
    elif mutation=='crop_png':reading['crop_png_base64']='YWJj'
    elif mutation=='tile_png':tile['tile_png_base64']='YWJj'
    elif mutation=='card_conflict':tile['card']='Ac'
    elif mutation=='duplicate_card':reading['cards'][1]=reading['cards'][0]
    elif mutation=='development':reading['hybrid']['development_only']=True
    elif mutation=='hybrid_rejection':reading['hybrid_authoritative_rejection']=True
    elif mutation=='suit_conflict':tile['suit_status']='SUIT_CONFLICT'
    elif mutation=='suit_not_confident':tile['template_confident']=False
    elif mutation=='suit_low_score':tile['template_best_score']=.5
    elif mutation=='suit_nan':tile['template_margin']=float('nan')
    elif mutation=='rank_indices_empty':tile['rank_exact_train_indices']=[]
    elif mutation=='rank_indices_foreign':tile['rank_exact_train_indices']=[-1]
    elif mutation=='rank_low_score':reading['hybrid']['tiles'][3]['rank_ncc']['rank_best_score']=.5
    elif mutation=='rank_not_accepted':reading['hybrid']['tiles'][3]['rank_ncc']['rank_accepted']=False
    elif mutation=='description_ranks':rows[1]['raw_ocr_text']='from the main pot with Two pairs, AA44'
    elif mutation=='description_class':rows[1]['raw_ocr_text']='from the main pot with Full house, QQQ33'
    elif mutation=='foreign_pts':rows[0].update(source_pts=1,time_base=[1,15360])
    elif mutation=='missing_config':reading['raw_pixel_response']['rank_config_sha256']='0'*64
    elif mutation=='malformed_reading':reading=[]
    elif mutation=='malformed_response':reading['raw_pixel_response']=[]
    elif mutation=='malformed_hybrid':reading['hybrid']=[]
    elif mutation=='oversized_pts':reading['source_pts']=10**400
    elif mutation=='boolean_pts':reading['source_pts']=True
    elif mutation=='negative_time_base':reading['time_base_den']=-15360
    elif mutation=='nan_time':reading['video_time']=float('nan')
    assert prove_win_tail(rows,reading,context) is None


def test_projector_uses_literal_header_and_preserves_raw_rows(source):
    rows=copy.deepcopy(source['context'])
    for row in rows:
        row['event_type']='WIN' if row['stable_row_id']==source['rows'][0]['stable_row_id'] else 'UNKNOWN_EVENT'
    before=copy.deepcopy(rows)
    readings={source['reading']['row_key']:source['reading']}
    assert project_win_tails(rows,readings)==1
    for row,old in zip(rows,before):
        assert {k:v for k,v in row.items() if k!='win_best_five_proof'}==old
    readings[source['reading']['row_key']]['hand_id']='OTHER_HAND'
    assert project_win_tails(copy.deepcopy(before),readings)==0


def test_saved_t07_numeric_candidate_requires_every_win_tail_source_row(source):
    rows=[copy.deepcopy(r) for m in source['hand']['messages'] for r in m['source_rows']]
    for n,row in enumerate(rows):row['source_index']=n
    before=copy.deepcopy(rows)
    assert project_win_tails(rows,{source['reading']['row_key']:source['reading']})==1
    assemblies,exhausted,_=candidate_assemblies(rows)
    win_id=source['rows'][0]['stable_row_id']
    for events in assemblies:
        win=next(e for e in events if any(r.get('stable_row_id')==win_id for r in e.source['rows']))
        assert win.kind=='WIN' and len(win.source['rows'])==3
    result=reconstruct_hand(parse_rows(rows),table='T07',hand_id='2878367230',assemblies=assemblies,
        assembly_exhausted=exhausted,strict_source_uniqueness=True,max_nodes=20000)
    assert result['solution_count']==1 and not result['exhausted']
    assert result['status']=='RECOVERED_BY_LOGIC_MATH'
    # Removing only the new proof restores the real opaque-source barrier.
    assemblies,exhausted,_=candidate_assemblies(before)
    rejected=reconstruct_hand(parse_rows(before),table='T07',hand_id='2878367230',assemblies=assemblies,
        assembly_exhausted=exhausted,strict_source_uniqueness=True,max_nodes=20000)
    assert rejected['solution_count']==0 and not rejected['kept_events']


def test_foreign_numbered_hand_cannot_reuse_winning_proof_in_solver(source):
    rows=[copy.deepcopy(r) for m in source['hand']['messages'] for r in m['source_rows']]
    assert project_win_tails(rows,{source['reading']['row_key']:source['reading']})==1
    for row in rows:
        for field in ('raw_text','raw_ocr_text'):
            if field in row:row[field]=row[field].replace('2878367230','123')
        if row.get('hand_id')=='2878367230':row['hand_id']='123'
    for n,row in enumerate(rows):row['source_index']=n
    assemblies,exhausted,_=candidate_assemblies(rows)
    direct=reconstruct_hand(parse_rows(rows),table='T07',hand_id='123',assemblies=assemblies,
        assembly_exhausted=exhausted,strict_source_uniqueness=True,max_nodes=20000)
    assert direct['solution_count']==0 and not direct['kept_events']
    hands=assemble_hands(rows,require_pixel_cards=True,require_adjacency=True,join_show_description=True)
    hand=next(h for h in hands if h['hand_id']=='123')
    win=next(m for m in hand['messages'] if m['action']=='WIN')
    assert win['win_best_five_conflict']=='NUMBERED_HAND_CONTEXT_CONFLICT'
    rows=[r for m in hand['messages'] for r in m['source_rows']]
    for n,row in enumerate(rows):row['source_index']=n
    assemblies,exhausted,_=candidate_assemblies(rows)
    result=reconstruct_hand(parse_rows(rows),table='T07',hand_id='123',assemblies=assemblies,
        assembly_exhausted=exhausted,strict_source_uniqueness=True,max_nodes=20000)
    assert result['solution_count']==0 and not result['kept_events']
