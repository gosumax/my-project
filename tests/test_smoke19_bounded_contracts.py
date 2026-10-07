import copy
import json
import pytest
from parser_core.poker_hand_assembly import assemble_hands, assemble_messages
from parser_core.poker_pipeline import analyze_timeline
from parser_core.poker_reconstruction_layer import write_reconstruction
from parser_core.seat_forced_binding import parse_forced_snapshot
from parser_core.poker_reconstruction.timeline import parse_rows
from parser_core.poker_reconstruction.assembly import candidate_assemblies
from parser_core.hand_lifecycle import classify_lifecycle, write_lifecycle, post_excel_contexts
from parser_core.poker_math_engine import write_start_stack_math


def rows(*texts, session='s'):
    return [dict(stable_row_id=f'{session}:R{i}', row_id=f'{session}:R{i}',
                 table_id='T01', session_id=session, frame_id=10, row_index=i,
                 epoch=1, raw_ocr_text=t, source_index=i+1, video_time=0,
                 row_pixel_sha256='a'*64, ocr_status='OK') for i,t in enumerate(texts)]


def lifecycle_hands(rr, **kwargs):
    hands=assemble_hands(rr,**kwargs)
    records=[dict(segment_id=str(i),observed_hand=h,validation={'status':'DIAGNOSTIC'}) for i,h in enumerate(hands)]
    frozen=copy.deepcopy(hands)
    classify_lifecycle(records)
    assert hands==frozen  # post-event metadata never rewrites observed assembly
    return [dict(r['observed_hand'],lifecycle_status=r['lifecycle']['status'],lifecycle_reason=r['lifecycle']['reason']) for r in records]


@pytest.mark.parametrize('text', ['Dealer: Dealing Flop', 'Dealer: Dealing River',
                                  'Dealer: A calls 300', 'Dealer: A wins 300'])
def test_video_starts_inside_hand_then_proven_start(text):
    rr=rows(text, 'Hand #2878365444', 'Dealer: B checks')
    hands=lifecycle_hands(rr, require_adjacency=True)
    assert [h['lifecycle_status'] for h in hands] == ['START_PARTIAL','END_PARTIAL']
    assert hands[1]['start_observed'] and not hands[1]['end_observed']


@pytest.mark.parametrize('terminal', ['Dealer: A wins 300', 'Dealer: Hand end'])
def test_proven_start_and_registered_terminal_complete(terminal):
    h=lifecycle_hands(rows('Hand #2878365444', terminal))[0]
    assert h['lifecycle_status']=='COMPLETE'


def test_showdown_without_result_does_not_weaken_terminal_contract():
    assert lifecycle_hands(rows('Hand #2878365444','Dealer: Showdown'))[0]['lifecycle_status']=='END_PARTIAL'


def test_next_start_never_supplies_previous_end():
    hh=lifecycle_hands(rows('Hand #2878365444','Dealer: A checks','Hand #2878365445','Dealer: B wins 300'))
    assert [h['lifecycle_status'] for h in hh]==['END_PARTIAL','COMPLETE']
    assert not hh[0]['end_observed']


def test_replacement_context_starts_partial():
    hh=lifecycle_hands(rows('Hand #2878365444','Dealer: A checks')+
                      rows('Dealer: Dealing River','Dealer: A wins 300',session='replacement'))
    assert [h['lifecycle_status'] for h in hh]==['END_PARTIAL','START_PARTIAL']


def test_partial_output_retained_and_solver_skipped(tmp_path, monkeypatch):
    rr=rows('Dealer: Dealing Flop','Hand #2878365444','Dealer: A checks')
    records,_=analyze_timeline(rr,ocr_status={r['stable_row_id']:'OK' for r in rr})
    complete,_,summary=classify_lifecycle(records)
    write_lifecycle(tmp_path,records,summary)
    def forbidden(*a,**kw): raise AssertionError('partial reached solver')
    monkeypatch.setattr('parser_core.poker_reconstruction_layer.reconstruct_hand',forbidden)
    write_reconstruction(tmp_path,complete)
    assert (tmp_path/'poker_reconstruction.jsonl').read_text()==''
    diag=[json.loads(x) for x in (tmp_path/'partial_hand_diagnostics.jsonl').read_text().splitlines()]
    assert [d['status'] for d in diag]==['START_PARTIAL','END_PARTIAL']
    assert sum(len(m['source_rows']) for d in diag for m in d['observed_hand']['messages'])==len(rr)
    assert all(d['evidence_refs'] and d['first_source_frame']==10 for d in diag)


def forced():
    texts=['Hand #2878365444']+[f'Dealer: p{i} posts Ante 100' for i in range(8)]
    texts+=['Dealer: p0 posts Small','Blind 300','Dealer: p1 posts Big Blind 600','Dealer: Dealing cards']
    return [dict(r,index=r['row_index'],text=r['raw_ocr_text'],score=.99,
                 source_ref=r['row_id'],source_frame=10) for r in rows(*texts)]


def test_split_sb_exact_post_all_paths_and_raw_provenance():
    rr=forced(); frozen=copy.deepcopy(rr)
    result,block=parse_forced_snapshot(rr,'2878365444',8)
    assert block is None
    sb=[p for p in result['posts'] if p['role']=='SB']
    assert len(sb)==1 and sb[0]['amount']==300 and len(sb[0]['source_refs'])==2
    assert sb[0]['source_rows']==rr[9:11] and rr==frozen
    pair=rr[9:11]
    message=assemble_messages(pair,require_adjacency=True)
    assert len(message)==1 and message[0]['action']=='SMALL_BLIND' and message[0]['amount']=='300'
    event=parse_rows(pair)
    assert len(event)==1 and event[0].kind=='SB' and event[0].amount==300
    paths,_,_=candidate_assemblies(pair)
    assert any(len(p)==1 and p[0].kind=='SB' and p[0].amount==300 for p in paths)


@pytest.mark.parametrize('change', ['index','frame','actor','session','hand','boundary','ambiguous','big','confidence'])
def test_split_sb_fail_closed(change):
    rr=forced()
    if change=='index': rr[10]['index']=rr[10]['row_index']=12
    if change=='frame': rr[10]['source_frame']=rr[10]['frame_id']=11
    if change=='actor': rr[10]['text']=rr[10]['raw_ocr_text']='Dealer: other Blind 300'
    if change=='session': rr[10]['session_id']='other'
    if change=='hand': rr[10]['hand_id']='other'
    if change=='boundary': rr.insert(10,dict(rr[10],index=9.5,text='Hand #2878365445'))
    if change=='ambiguous': rr[10]['text']=rr[10]['raw_ocr_text']='Blind 300 or 600'
    if change=='big': rr[9]['text']=rr[9]['raw_ocr_text']='Dealer: p0 posts Big'
    if change=='confidence': rr[10]['score']=.9399
    assert parse_forced_snapshot(rr,'2878365444',8)[0] is None


def test_normal_one_row_sb_unchanged():
    rr=forced();rr[9]['text']='Dealer: p0 posts Small Blind 300';rr.pop(10)
    result,block=parse_forced_snapshot(rr,'2878365444',8)
    assert block is None and len([p for p in result['posts'] if p['role']=='SB'])==1


@pytest.mark.parametrize('score,accepted',[(.9446198,True),(.9400,True),(.9399,False)])
def test_only_split_small_blind_tail_has_local_094_threshold(score,accepted):
    rr=forced();rr[10]['score']=score
    result,block=parse_forced_snapshot(rr,'2878365444',8)
    assert (result is not None)==accepted
    if accepted:
        assert next(p for p in result['posts'] if p['role']=='SB')['amount']==300
    else: assert block=='FORCED_ROW_CONFIDENCE_UNKNOWN'


@pytest.mark.parametrize('kind',['first','one_line','big','numeric_wrap','ante'])
def test_other_forced_grammar_thresholds_unchanged(kind):
    rr=forced()
    if kind=='first': rr[9]['score']=.9446198
    if kind=='one_line':
        rr[9]['text']='Dealer: p0 posts Small Blind 300';rr[9]['score']=.9446198;rr.pop(10)
    if kind=='big': rr[11]['score']=.9446198
    if kind=='numeric_wrap':
        rr[9]['text']=rr[9]['raw_ocr_text']='Dealer: p0 posts Small Blind'
        rr[10]['text']=rr[10]['raw_ocr_text']='300';rr[10]['score']=.9446198
    if kind=='ante': rr[1]['score']=.9446198
    assert parse_forced_snapshot(rr,'2878365444',8)[0] is None


def test_malformed_header_cannot_prove_start():
    h=lifecycle_hands(rows('Dealer: Hand #123 bad OCR','Dealer: A wins 300'))[0]
    assert h['lifecycle_status']=='UNKNOWN_UNPROVEN'


def test_unknown_header_fences_terminal_same_hand():
    hh=lifecycle_hands(rows('Hand #2878365444','Dealer: A checks','Hand #UNREADABLE','Dealer: B wins 600'))
    assert [h['lifecycle_status'] for h in hh]==['END_PARTIAL']
    assert hh[0]['lifecycle_reason']=='UNKNOWN_HAND_BOUNDARY'


@pytest.mark.parametrize('barrier',['Hand #2878365444','Hand #UNREADABLE'])
def test_older_or_repeated_after_unknown_header_cannot_start_normal_hand(barrier):
    rr=rows('Hand #2878365445','Dealer: A checks',barrier,'Hand #2878365445','Dealer: B wins 600')
    hh=lifecycle_hands(rr)
    assert not any(h['lifecycle_status']=='COMPLETE' for h in hh)
    assert hh[0]['lifecycle_status']=='END_PARTIAL'


def test_post_event_layer_preserves_all_excel_inputs_and_only_complete_math(tmp_path, monkeypatch):
    rr=rows('Dealer: Dealing Flop','Hand #2878365444','Dealer: A checks',
            'Hand #2878365445','Dealer: B wins 600')
    records,_=analyze_timeline(rr,ocr_status={r['stable_row_id']:'OK' for r in rr})
    raw=copy.deepcopy([r['observed_hand'] for r in records]);before=copy.deepcopy(rr)
    complete,eligible,summary=classify_lifecycle(records)
    assert rr==before and [r['observed_hand'] for r in records]==raw
    assert len(records)==3 and len(complete)==1 and summary['excel_policy']=='ALL_OBSERVED_EVENTS_AND_HANDS'
    assert sum(len(m['source_rows']) for r in records for m in r['observed_hand']['messages'])==len(rr)
    facts=[dict(table=1,session='s',hand_id=n) for n in ['2878365444','2878365445']]
    (tmp_path/'start_stack_facts.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in facts),'utf-8')
    called=[]
    def calculate(f):
        called.append(f['hand_id']);return dict(f,status='UNKNOWN')
    monkeypatch.setattr('parser_core.poker_math_engine.calculate_start_stack',calculate)
    result=write_start_stack_math(tmp_path,eligible_contexts=eligible)
    assert called==['2878365445'] and result['excluded_partial_or_unproven_facts']==1
    assert '2878365444' in (tmp_path/'start_stack_math_excluded_diagnostics.jsonl').read_text()


def sidecar_headers(tmp_path, specs):
    state=[];ocr=[]
    for frame,number,status in specs:
        base=dict(table_id=1,session_id='s',source_frame=frame,source_pts=frame,source_span_index=0)
        ocr.append(dict(base,status='ADMITTED',pixel_sha256='b'*64,time_base=[1,1000],row_bounds=[0,16]))
        ocr.append(dict(base,status=status,pixel_sha256='b'*64,
                        ocr_response=dict(status='OK',raw_text=f'Hand #{number}',score=.99)))
        if status=='HAND_HEADER_CONFIRMED':state.append(dict(base,status=status,hand_id=number))
    for name,rs in [('hand_state_receipts.jsonl',state),('hand_header_ocr.jsonl',ocr)]:
        (tmp_path/name).write_text(''.join(json.dumps(r)+'\n' for r in rs),'utf-8')


def test_sidecar_contexts_only_after_excel_preserve_original_and_first_context_partial(tmp_path):
    rr=rows('Dealer: A checks','Dealer: B wins 600')
    rr[0]['frame_id']=2;rr[1]['frame_id']=11
    observed,_=analyze_timeline(rr,ocr_status={r['stable_row_id']:'OK' for r in rr})
    frozen=copy.deepcopy(observed)
    sidecar_headers(tmp_path,[(0,'2878365444','HAND_HEADER_CONFIRMED'),(10,'2878365445','HAND_HEADER_CONFIRMED')])
    logical,complete,eligible,summary=post_excel_contexts(tmp_path,observed)
    assert observed==frozen and all(r['observed_hand']['hand_id'] is None for r in observed)
    assert [r['lifecycle']['status'] for r in logical]==['START_PARTIAL','COMPLETE']
    assert eligible=={(1,'s','2878365445')} and len(complete)==1
    sources=[r for m in complete[0]['observed_hand']['messages'] for r in m['source_rows']]
    header=next(r for r in sources if r.get('source_kind')=='SIDECAR_HEADER_RECEIPT')
    assert header['raw_ocr_text']=='Hand #2878365445' and header['source_pts']==10
    assert header['header_provenance']['state_line']==2 and header['row_pixel_sha256']=='b'*64
    assert summary['raw_excel_event_rows']==2 and summary['sidecar_source_rows']==2


def test_unconfirmed_sidecar_header_cannot_lend_terminal_to_previous_hand(tmp_path):
    rr=rows('Dealer: A checks','Dealer: B wins 600');rr[0]['frame_id']=12;rr[1]['frame_id']=21
    observed,_=analyze_timeline(rr,ocr_status={r['stable_row_id']:'OK' for r in rr})
    sidecar_headers(tmp_path,[(0,'2878365444','HAND_HEADER_CONFIRMED'),(10,'2878365445','HAND_HEADER_CONFIRMED'),
                              (20,'UNREADABLE','HAND_HEADER_UNKNOWN')])
    logical,complete,eligible,_=post_excel_contexts(tmp_path,observed)
    assert not complete and not eligible
    assert logical[-1]['lifecycle']['status']=='END_PARTIAL'


def test_missing_admitted_header_completion_is_unknown_barrier(tmp_path):
    rr=rows('Dealer: A checks','Dealer: B wins 600');rr[0]['frame_id']=12;rr[1]['frame_id']=21
    observed,_=analyze_timeline(rr,ocr_status={r['stable_row_id']:'OK' for r in rr})
    sidecar_headers(tmp_path,[(0,'2878365444','HAND_HEADER_CONFIRMED'),(10,'2878365445','HAND_HEADER_CONFIRMED')])
    with (tmp_path/'hand_header_ocr.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(table_id=1,session_id='s',source_frame=20,source_pts=20,
            source_span_index=0,status='ADMITTED',pixel_sha256='b'*64,time_base=[1,1000]))+'\n')
    logical,complete,eligible,summary=post_excel_contexts(tmp_path,observed)
    assert not complete and not eligible and len(summary['unresolved_header_refs'])==1
    assert logical[-1]['lifecycle']['status']=='END_PARTIAL'


@pytest.mark.parametrize('status',['STALE_HEADER_IGNORED','STALE_SESSION_IGNORED'])
def test_provider_stale_callbacks_do_not_create_new_header_barriers(tmp_path,status):
    rr=rows('Dealer: B wins 600');rr[0]['frame_id']=21
    observed,_=analyze_timeline(rr,ocr_status={r['stable_row_id']:'OK' for r in rr})
    sidecar_headers(tmp_path,[(0,'2878365444','HAND_HEADER_CONFIRMED'),(10,'2878365445','HAND_HEADER_CONFIRMED'),
                              (15,'2878365445',status)])
    logical,complete,eligible,summary=post_excel_contexts(tmp_path,observed)
    assert len(complete)==1 and eligible=={(1,'s','2878365445')}
    assert summary['unresolved_header_refs']==[]


@pytest.mark.parametrize('mismatch',['pts','hash','session'])
def test_sidecar_source_join_mismatch_cannot_prove_normal_start(tmp_path,mismatch):
    rr=rows('Dealer: B wins 600');rr[0]['frame_id']=11
    observed,_=analyze_timeline(rr,ocr_status={r['stable_row_id']:'OK' for r in rr})
    sidecar_headers(tmp_path,[(0,'2878365444','HAND_HEADER_CONFIRMED'),(10,'2878365445','HAND_HEADER_CONFIRMED')])
    path=tmp_path/('hand_header_ocr.jsonl' if mismatch=='hash' else 'hand_state_receipts.jsonl')
    rs=[json.loads(x) for x in path.read_text().splitlines()]
    target=rs[-1]
    if mismatch=='pts':target['source_pts']=99
    if mismatch=='session':target['session_id']='wrong'
    if mismatch=='hash':target['pixel_sha256']='c'*64
    path.write_text(''.join(json.dumps(r)+'\n' for r in rs),'utf-8')
    _,complete,eligible,_=post_excel_contexts(tmp_path,observed)
    assert not complete and not eligible
