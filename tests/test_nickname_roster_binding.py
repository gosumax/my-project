import copy
import json
from pathlib import Path

import pytest

from parser_core.nickname_roster_binding import (prove_roster_binding, binding_receipt,
                                                 verify_binding_receipt, _unique_matching)


@pytest.fixture(params=['roster_prefix_smoke19.json','roster_prefix_smoke19_live_v4.json'])
def native(request):
    saved = json.loads((Path(__file__).parent/'fixtures'/request.param).read_text('utf-8'))
    c = saved['context']
    raw = [r for r in saved['input_rows'] if r['kind'] == 'CHAT']
    forced = [dict(r, text=r['ocr_response']['raw_text'].strip(), score=r['ocr_response']['score']) for r in raw]
    return dict(context=[c['table'], c['session'], c['hand_id']],
        snapshot=saved['forced_snapshot'],
        deal=next(r for r in saved['deal_receipts'] if r.get('status') == 'DEAL_COMPLETE'),
        forced_rows=forced, nick_rows=[r for r in saved['input_rows'] if r['kind'] == 'NICK'],
        state_receipts=saved['state_receipts'], unresolved_events=saved['unresolved_state_events'],
        hand_triggers=saved['triggers'], capture_frames=64, confidence=.95)


def test_native_truncated_seat_has_unique_complete_source_roster(native):
    before = copy.deepcopy(native)
    proof = prove_roster_binding(native)
    assert proof['status'] == 'PROVEN', proof
    assert proof['matching']['seat2'] == 'stoleyourblind'
    assert proof['binding_anchor_frame'] == 286
    assert len(proof['matching']) == 8
    assert native == before
    posts = [p for p in native['snapshot']['posts'] if p['actor'] == 'stoleyourblind']
    receipt = binding_receipt(proof, 'stoleyourblind', posts)
    assert receipt['label_evidence'][0]['constraint'] == dict(kind='PREFIX', text='stoleyourbl')
    assert verify_binding_receipt(receipt, native['snapshot'])
    assert prove_roster_binding(proof['source_inputs']) == proof


@pytest.mark.parametrize('case', ['bare_ellipsis','prefix_conflict','case_conflict','low_confidence',
                                'triple_gap','hash','pts','context','epoch','seat_change',
                                'unresolved_event','missing_ante','duplicate_ante','cardinality'])
def test_native_proof_cannot_bridge_unknown_or_contradictory_sources(native, case):
    # Limit to the first triple so an altered first source cannot be substituted
    # by an unrelated later triple in this missing-evidence case.
    native['nick_rows'] = [r for r in native['nick_rows'] if r['source_frame'] <= 286]
    selected = [r for r in native['nick_rows'] if r['seat'] == 'seat2' and 284 <= r['source_frame'] <= 286]
    if case in {'bare_ellipsis','prefix_conflict','case_conflict'}:
        for r in selected:
            r['ocr_response']['raw_text'] = {'bare_ellipsis':'...', 'prefix_conflict':'Alice...',
                                           'case_conflict':'STOLEYOURBL...'}[case]
    elif case == 'low_confidence': selected[0]['ocr_response']['score'] = .90
    elif case == 'triple_gap': native['nick_rows'].remove(selected[1])
    elif case == 'hash': selected[0]['pixel_sha256'] = '0'*64
    elif case == 'pts': selected[1]['source_pts'] = selected[0]['source_pts']
    elif case == 'context': selected[0]['hand_id'] = 'another-hand'
    elif case == 'epoch': native['state_receipts'][1]['epoch'] = None
    elif case == 'seat_change':
        native['state_receipts'].append(dict(native['state_receipts'][1], kind='EMPTY', frame=280))
    elif case == 'unresolved_event':
        native['unresolved_events'].append(dict(table=native['context'][0], session=native['context'][1], frame=275))
    elif case == 'missing_ante': native['snapshot']['posts'].pop(0)
    elif case == 'duplicate_ante': native['snapshot']['posts'].append(copy.deepcopy(native['snapshot']['posts'][0]))
    elif case == 'cardinality': native['deal']['occupied_seats'].pop()
    assert prove_roster_binding(native)['status'] == 'UNKNOWN'


def test_constraint_intersection_cannot_ignore_a_conflicting_exact_name(native):
    # A high-confidence exact different name in the same seat must constrain
    # the entire graph even if the later prefix gives a unique assignment.
    selected = [r for r in native['nick_rows'] if r['seat'] == 'seat2' and 284 <= r['source_frame'] <= 286]
    for i,r in enumerate(selected):
        earlier = copy.deepcopy(r)
        earlier['source_frame'] = 270+i
        earlier['source_pts'] = (270+i)*512
        earlier['ocr_response']['raw_text'] = 'twixspb'
        native['nick_rows'] = [x for x in native['nick_rows'] if not (x['seat'] == 'seat2' and x['source_frame'] == 270+i)]
        native['nick_rows'].append(earlier)
    assert prove_roster_binding(native)['status'] == 'UNKNOWN'


def test_unique_matching_requires_every_seat_and_more_than_roster_complement():
    assert len(_unique_matching({'a':{'Alice','Bob'}, 'b':{'Alice','Bob'}})) == 2
    assert not _unique_matching({'a':set(), 'b':{'Bob'}})
    assert _unique_matching({'a':{'Alice','Bob'}, 'b':{'Bob'}}) == [{'b':'Bob','a':'Alice'}]


@pytest.mark.parametrize('case', ['mid_hand_join','wrong_fixed_roi','pts_before_forced',
                                'forced_score','forced_hash','prefix_gap','forced_pts','forced_time_base',
                                'forced_after_labels','forced_before_deal','hand_start_after_deal'])
def test_review_counterexamples_remain_unknown(native, case):
    if case == 'mid_hand_join':
        source = native['state_receipts'][1]
        native['state_receipts'].extend([dict(source,kind='EMPTY',frame=240,epoch=None),
                                        dict(source,kind='SIT_IN',frame=250,actor='stoleyourblind')])
    elif case == 'wrong_fixed_roi':
        for r in native['nick_rows']:
            if r['seat'] == 'seat2': r['roi'] = [r['roi'][0]+1,r['roi'][1],r['roi'][2]+1,r['roi'][3]]
    elif case == 'pts_before_forced':
        for r in native['nick_rows']: r['source_pts'] -= 100000
    elif case == 'forced_score': native['forced_rows'][0]['score'] = .5
    elif case == 'forced_hash': native['forced_rows'][0]['pixel_sha256'] = '0'*64
    elif case == 'forced_pts': native['forced_rows'][0]['source_pts'] += 512
    elif case == 'forced_time_base': native['forced_rows'][0]['time_base'] = [1,30000]
    elif case == 'forced_after_labels':
        for r in native['forced_rows']:r['source_pts']+=100000
        for p in native['snapshot']['posts']:p['source_pts']+=100000
    elif case == 'forced_before_deal':
        for r in native['forced_rows']:r['source_frame']=native['deal']['source_frame']-1
        for p in native['snapshot']['posts']:p['source_frame']=native['deal']['source_frame']-1
    elif case == 'hand_start_after_deal':
        for t in native['hand_triggers']:t['source_frame']=native['deal']['source_frame']+1
    elif case == 'prefix_gap':
        for r in native['forced_rows']:
            if r['index'] > native['snapshot']['header_index']: r['index'] += 1
        for p in native['snapshot']['posts']: p['source_index'] += 1
    assert prove_roster_binding(native)['status'] == 'UNKNOWN'


@pytest.mark.parametrize('field,value', [('actor','another-actor'),('seat','seat1'),
                                        ('epoch','other-epoch'),('binding_anchor_frame',300)])
def test_receipt_reproducer_rejects_changed_binding(native, field, value):
    proof = prove_roster_binding(native)
    posts = [p for p in native['snapshot']['posts'] if p['actor'] == 'stoleyourblind']
    receipt = binding_receipt(proof, 'stoleyourblind', posts)
    receipt[field] = value
    assert not verify_binding_receipt(receipt, native['snapshot'])


@pytest.mark.parametrize('tamper', [None,'matching','rule_downgrade','rule_downgrade_without_proof'])
def test_canonical_consumer_reproduces_roster_binding_before_admission(native, tamper):
    from parser_core.canonical_events import join_forced_posts
    proof = prove_roster_binding(native)
    actor = 'stoleyourblind'
    posts = [p for p in native['snapshot']['posts'] if p['actor'] == actor]
    receipt = binding_receipt(proof,actor,posts)
    original_rule=receipt['binding_rule']
    if tamper == 'matching': receipt['roster_proof']['matching']['seat2'] = 'ragna_r'
    elif tamper:
        receipt['binding_rule']='FIRST_EXACT_LABEL_HISTORICAL_EPOCH'
        if tamper == 'rule_downgrade_without_proof': receipt.pop('roster_proof')
    table,session,hand = native['context']
    header = next(r for r in native['forced_rows'] if r['index'] == native['snapshot']['header_index'])
    timeline = [dict(table_id=f'T{table:02d}',session_id=session,frame_id=header['source_frame'],
        row_index=header['index'],stable_row_id='native-fixture-header',raw_ocr_text=header['text'],
        video_time=header['source_pts']*header['time_base'][0]/header['time_base'][1],epoch=1,status='RAW_UNREVIEWED')]
    context = dict(table=table,session=session,hand_id=hand)
    summary = dict(forced_snapshots=[dict(native['snapshot'],**context)],
        historical_binding_receipts=[receipt],
        bound_posts=[dict(p,**context,seat=receipt['seat'],epoch=receipt['epoch'],binding_rule=original_rule) for p in posts])
    frozen = copy.deepcopy(timeline)
    result,report = join_forced_posts(timeline,summary)
    admitted = [r for r in result if r.get('player') == actor and r.get('canonical_event_id')]
    assert timeline == frozen
    if tamper:
        assert not admitted
    else:
        assert len(admitted) == 2
        assert {r['action_type'] for r in admitted} == {'ANTE','BIG_BLIND'}
        assert all(r['seat'] == 'seat2' for r in admitted)


@pytest.mark.parametrize('enabled', [False,True])
def test_live_binder_finish_uses_source_roster_only_when_enabled(native, tmp_path, monkeypatch, enabled):
    import base64
    import threading
    from types import SimpleNamespace
    import cv2
    import numpy as np
    import parser_core.seat_forced_binding as module
    from parser_core.parallel_ocr import row_key

    monkeypatch.setattr(module,'ParallelRowOCR',lambda *a,**kw:SimpleNamespace(finish=lambda:None,errors=[]))
    key=tuple(native['context'])
    flags=dict(stack_confidence=.95,capture_frames=64)
    readings=(tmp_path/'start_stack_readings.jsonl').open('w',encoding='utf-8')
    sidecar=SimpleNamespace(lock=threading.RLock(),flags=flags,triggers=native['hand_triggers'],
        windows={key:SimpleNamespace(deal=native['deal'])},logs={'start_stack_readings':readings})
    binder=module.SeatForcedBinding(Path(__file__).resolve().parents[1],tmp_path,sidecar,
        SimpleNamespace(states={}),{},roster_prefix_binding=enabled)
    binder.snapshot_sizes[key]=len(native['forced_rows'])
    binder.snapshot_occupied[key]=len(native['deal']['occupied_seats'])
    for event in native['state_receipts']:
        binder.machine.observe(event['table'],event['session'],event['seat'],event['kind'],event['frame'],
            **{k:event[k] for k in ('actor','hand_id','evidence_status','source_ref','source_index') if k in event})
    for r in native['forced_rows']+native['nick_rows']:
        if r.get('pre_deal'):continue
        pixels=cv2.imdecode(np.frombuffer(base64.b64decode(r['png_base64']),np.uint8),cv2.IMREAD_GRAYSCALE)
        index=int(r['ocr_response']['item_id'].split(':')[-1])
        row=binder.row(key[0],key[1],r['source_frame'],r['source_pts'],r['time_base'],index,
            r['roi'][1] if r['kind']=='NICK' else r['row_bounds'][0],pixels,
            'NICK_ONLY' if r['kind']=='NICK' else 'FORCED_SNAPSHOT')
        meta={k:v for k,v in r.items() if k not in {'png_base64','pixel_sha256','ocr_response','text','score'}}
        binder.jobs[row_key(row)]=(key,meta)
        binder.completed(row,copy.deepcopy(r['ocr_response']))
    binder.finish()
    readings.close()
    result=json.loads((tmp_path/'seat_forced_summary.json').read_text('utf-8'))
    bound=[p for p in result['bound_posts'] if p['actor']=='stoleyourblind']
    assert len(bound)==(2 if enabled else 0)
    if enabled:
        receipt=next(r for r in result['historical_binding_receipts'] if r['actor']=='stoleyourblind')
        assert verify_binding_receipt(receipt,result['forced_snapshots'][0])
        assert result['roster_binding_proofs'][0]['status']=='PROVEN'
    else:
        assert 'roster_binding_proofs' not in result
