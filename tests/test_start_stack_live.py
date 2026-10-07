"""Source-bound live connection tests; fake OCR never supplies occupancy/forced proof."""
import base64
import hashlib
import json
import threading
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from parser_core.dealing_start_stack import CleanStackWindow
from parser_core.start_stack_live import StartStackLiveConnection


class FakeOCR:
    def __init__(self, factory, **kwargs):
        self.observer = kwargs['result_observer']
        self.batches = []
        self.errors = []
        self.before_submit = None

    def submit_batch(self, batch):
        if self.before_submit:
            self.before_submit()
        self.batches.append(batch)

    def finish(self):
        pass


@pytest.fixture
def connection(tmp_path, monkeypatch):
    monkeypatch.setattr('parser_core.start_stack_live.ParallelRowOCR', FakeOCR)
    monkeypatch.setattr('parser_core.start_stack_live.cv2.matchTemplate', lambda *a: np.array([[.99]], np.float32))
    pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    model = {'threshold': .95, 'consecutive_frames': 3, 'templates': [
        {'slot': 1, 'seat': f'seat{i}', 'pixels_base64': base64.b64encode(pixels.tobytes()).decode(),
         'template_shape': list(pixels.shape), 'template_bgr_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()}
        for i in range(1, 9)]}
    (tmp_path/'models').mkdir()
    (tmp_path/'models/start_stack_table_cards.json').write_text(json.dumps(model), 'utf-8')
    key = (1, 'S', '12345678')
    boxes = {f'seat{i}': {'shown_cards': [0, 0, 2, 2], 'stack': [0, 0, 2, 2]} for i in range(1, 9)}
    sidecar = SimpleNamespace(lock=threading.RLock(), current={1: {'session_id': 'S', 'hand_id': key[2],
        'source_frame': 9, 'source_position': [9, 0], 'source_pts': 9*512, 'time_base': [1, 15360]}},
        active_sessions={1: 'S'}, windows={key: CleanStackWindow(.95)},
        hand_state_gate=SimpleNamespace(failed_headers=defaultdict(set)),
        outstanding=defaultdict(set), header_fences={}, histories=defaultdict(list),
        flags={'capture_frames': 64, 'stack_confidence': .95},
        profile={'slots': {'slot_01': {'seats': boxes}}}, deal_calls=[], reading_calls=[])

    def deal(k, r):
        sidecar.deal_calls.append((k, r))
        return sidecar.windows[k].confirm_deal(r)

    def reading(k, r):
        sidecar.reading_calls.append((k, r.copy()))
        return sidecar.windows[k].reading(r['seat'], r['source_frame'], r['raw_text'], r['confidence'],
            r['forced_bets'], r['occupancy_epoch'], r['forced_bets_proven'], r['occupancy_proven'])

    # Fixture represents a confirmed active hand, including its immutable anchor.
    sidecar.histories[(1, 'S')].append(dict(sidecar.current[1]))
    sidecar.header_fences[(1, 'S')] = dict(hand_id=key[2], source_position=[9, 0], unknown=False)
    sidecar.confirm_deal_complete = deal
    sidecar.stack_reading = reading
    obj = StartStackLiveConnection(tmp_path, tmp_path, sidecar, {})
    obj.test_key = key
    yield obj
    obj.finish()


def observe(obj, frame, session='S'):
    obj.observe_frame(1, session, frame, frame*512, [1, 15360], np.full((2, 2, 3), 80, np.uint8), None)


def deal(obj):
    for f in (10, 11, 12):
        observe(obj, f)


def complete(obj, batch, text='1 000', confidence=.99, status='OK'):
    for row in batch:
        obj.completed(row, {'status': status, 'raw_text': text, 'score': confidence})


def test_all_eight_cards_need_three_consecutive_frames_before_numeric_jobs(connection):
    obj = connection
    observe(obj, 10)
    observe(obj, 11)
    assert not obj.sidecar.deal_calls and not obj.ocr.batches
    observe(obj, 12)
    assert len(obj.sidecar.deal_calls) == 1 and not obj.ocr.batches
    receipt = obj.sidecar.deal_calls[0][1]
    assert receipt['card_stable_frames'] == [10, 11, 12]
    assert len(receipt['occupied_seats']) == 8
    observe(obj, 13)
    assert len(obj.ocr.batches[0]) == 8


def test_missing_seat_template_remains_unknown(connection):
    connection.templates.pop((1, 'seat8'))
    deal(connection)
    assert not connection.sidecar.deal_calls and not connection.ocr.batches


def test_one_unmatched_empty_seat_does_not_supply_complete_mask(connection, monkeypatch):
    calls = Counter()
    def match(*args):
        calls['n'] += 1
        return np.array([[.1 if calls['n'] % 8 == 0 else .99]], np.float32)
    monkeypatch.setattr('parser_core.start_stack_live.cv2.matchTemplate', match)
    deal(connection)
    assert not connection.sidecar.deal_calls and not connection.ocr.batches


def test_card_source_gap_resets_run(connection):
    for f in (10, 11, 13, 14):
        observe(connection, f)
    assert not connection.sidecar.deal_calls
    observe(connection, 15)
    assert connection.sidecar.deal_calls[0][1]['card_stable_frames'] == [13, 14, 15]


def test_actual_frame_numeric_triples_never_supply_forced_or_epoch_proof(connection):
    deal(connection)
    for f in (13, 14, 15):
        observe(connection, f)
        complete(connection, connection.ocr.batches[-1])
    state = connection.states[connection.test_key]
    assert len(state['numeric_triples']) == 8
    assert all(r['frames'] == [13, 14, 15] for r in state['numeric_triples'].values())
    assert len(connection.sidecar.reading_calls) == 24
    assert all(not r['occupancy_proven'] and not r['forced_bets_proven'] and r['occupancy_epoch'] is None
               for _, r in connection.sidecar.reading_calls)
    assert not connection.sidecar.windows[connection.test_key].results
    observe(connection, 16)
    assert len(connection.ocr.batches) == 3


@pytest.mark.parametrize('text,confidence,status', [('1000', .94, 'OK'), ('ALL-IN', .99, 'OK'), ('1000', .99, 'ERROR')])
def test_invalid_ocr_breaks_numeric_run(connection, text, confidence, status):
    deal(connection)
    for f in (13, 14, 15):
        observe(connection, f)
        complete(connection, connection.ocr.batches[-1], text if f == 14 else '1000', confidence if f == 14 else .99, status if f == 14 else 'OK')
    assert not connection.states[connection.test_key]['numeric_triples']


def test_numeric_source_frame_gap_does_not_become_triple(connection):
    deal(connection)
    for f in (13, 15, 16):
        observe(connection, f)
        complete(connection, connection.ocr.batches[-1])
    assert not connection.states[connection.test_key]['numeric_triples']


@pytest.mark.parametrize('kind', ['CALL', 'BET', 'RAISE', 'ALL_IN'])
def test_money_action_does_not_close_canonical_stack_lane(connection, kind):
    deal(connection)
    for f in (13, 14, 15):
        observe(connection, f)
    window = connection.sidecar.windows[connection.test_key]
    window.action(kind, 15)
    for b in connection.ocr.batches:
        complete(connection, b)
    assert len(connection.states[connection.test_key]['numeric_triples']) == 8
    observe(connection, 16)
    assert len(connection.ocr.batches) == 3
    assert not window.results


def test_late_action_invalidates_existing_numeric_triples_in_final_evidence(connection):
    deal(connection)
    for f in (13, 14, 15):
        observe(connection, f)
        complete(connection, connection.ocr.batches[-1])
    assert len(connection.states[connection.test_key]['numeric_triples']) == 8
    connection.sidecar.windows[connection.test_key].action('CALL', 14)
    connection.finish()
    result = json.loads((connection.output/'start_stack_live_summary.json').read_text('utf-8'))
    assert len(result['hands'][0]['numeric_triples']) == 8 and result['start_stack_pass'] == 0


@pytest.mark.parametrize('kind', ['CHECK', 'FOLD'])
def test_nonmoney_actions_keep_capture_open(connection, kind):
    deal(connection)
    connection.sidecar.windows[connection.test_key].action(kind, 13)
    observe(connection, 13)
    assert len(connection.ocr.batches) == 1


def test_queue_submission_is_outside_provider_and_callback_locks(connection):
    deal(connection)
    def check():
        assert not connection.lock._is_owned()
        assert not connection.sidecar.lock._is_owned()
    connection.ocr.before_submit = check
    observe(connection, 13)


def test_wrong_current_session_cannot_submit_jobs(connection):
    deal(connection)
    observe(connection, 13, session='OTHER')
    assert not connection.ocr.batches


def test_old_session_callbacks_do_not_publish_numeric_proof(connection):
    deal(connection)
    for f in (13, 14, 15):
        observe(connection, f)
    connection.sidecar.current[1] = {'session_id': 'NEW', 'hand_id': '12345679'}
    connection.sidecar.active_sessions[1] = 'NEW'
    for b in connection.ocr.batches:
        complete(connection, b)
    assert not connection.states[connection.test_key]['numeric_triples']
    assert not connection.sidecar.windows[connection.test_key].results


def test_pending_async_new_header_prevents_old_hand_table_confirmation(connection):
    observe(connection, 10)
    observe(connection, 11)
    connection.sidecar.outstanding[(1, 'S')].add((12, 1000000))
    observe(connection, 12)
    assert not connection.sidecar.deal_calls
    connection.sidecar.outstanding[(1, 'S')].clear()
    observe(connection, 13)
    observe(connection, 14)
    assert connection.sidecar.deal_calls[0][1]['card_stable_frames'] == [10, 11, 12]
    assert connection.sidecar.deal_calls[0][1]['source_frame'] == 12


def test_closed_window_before_table_cannot_confirm_deal(connection):
    connection.sidecar.windows[connection.test_key].action('CALL', 10)
    deal(connection)
    assert not connection.sidecar.deal_calls and not connection.ocr.batches


def test_late_newer_header_invalidates_jobs_from_wrong_source_hand(connection):
    deal(connection)
    for f in (13, 14, 15):
        observe(connection, f)
    connection.sidecar.histories[(1, 'S')].append({'hand_id': '12345679', 'source_frame': 13})
    for batch in connection.ocr.batches:
        complete(connection, batch)
    assert not connection.states[connection.test_key]['numeric_triples']
    assert connection.counts['STACK_SOURCE_CONTEXT_UNKNOWN'] == 24


def test_header_learned_after_numeric_callbacks_invalidates_final_triple(connection):
    deal(connection)
    for f in (13, 14, 15):
        observe(connection, f)
        complete(connection, connection.ocr.batches[-1])
    assert len(connection.states[connection.test_key]['numeric_triples']) == 8
    connection.sidecar.histories[(1, 'S')].append({'hand_id': '12345679', 'source_frame': 14})
    connection.finish()
    result = json.loads((connection.output/'start_stack_live_summary.json').read_text('utf-8'))
    assert result['hands'][0]['numeric_triples'] == {}
    assert result['start_stack_pass'] == 0


def test_header_learned_after_table_confirmation_invalidates_final_deal(connection):
    deal(connection)
    assert connection.sidecar.windows[connection.test_key].deal is not None
    connection.sidecar.histories[(1, 'S')].append({'hand_id': '12345679', 'source_frame': 11})
    connection.finish()
    result = json.loads((connection.output/'start_stack_live_summary.json').read_text('utf-8'))
    assert result['hands'][0]['deal_complete_frame'] is None
    assert connection.sidecar.windows[connection.test_key].deal is None
    receipts = [json.loads(x) for x in (connection.output/'start_stack_live_receipts.jsonl').read_text('utf-8').splitlines()]
    assert any(r.get('blocker') == 'LATE_HEADER_SOURCE_CONTRADICTION' for r in receipts)


def test_sources_before_current_header_cannot_confirm_deal(connection):
    connection.sidecar.current[1]['source_frame'] = 20
    deal(connection)
    assert not connection.sidecar.deal_calls


def test_connection_ocr_errors_preserved_in_summary_and_runner_output(connection):
    connection.ocr.errors.append({'stage': 'STACK_OCR', 'error': 'test worker failed'})
    connection.finish()
    result = json.loads((connection.output/'start_stack_live_summary.json').read_text('utf-8'))
    assert result['errors'] == connection.ocr.errors
    root = Path(__file__).resolve().parents[1]
    source = (root/'scripts/run_smoke14_ram.py').read_text('utf-8')
    assert 'errors.extend(start_stack_live.ocr.errors)' in source


def test_default_off_runner_cannot_instantiate_connection():
    root = Path(__file__).resolve().parents[1]
    flags = json.loads((root/'configs/dealing_start_stack.json').read_text('utf-8'))
    assert flags['enable_start_stack_capture'] is False and flags['enable_dealing_cards_hand_state_trigger'] is False
    source = (root/'scripts/run_smoke14_ram.py').read_text('utf-8')
    position = source.index('from parser_core.start_stack_live import StartStackLiveConnection')
    assert "if sidecar_flags['enable_start_stack_capture']:" in source[position-150:position]


def test_first_numeric_visibility_after_overlay_fixes_first_attempt(connection):
    deal(connection)
    for f in (13,14,15,16):
        observe(connection,f)
        complete(connection,connection.ocr.batches[-1], 'SITOUT' if f==13 else '1000')
    r=connection.states[connection.test_key]['canonical_attempts']['seat1']
    assert r['first_numeric_visible_frame']==14 and r['frames']==[14,15,16]
    assert r['ocr_values']==['1000']*3 and r['status']=='PASS'
    assert all(e['source_pts']==e['source_frame']*512 and e['pixel_sha256'] and e['ocr_response'] and e['png_base64'] for e in r['evidence'])


def test_async_later_triple_cannot_preempt_pending_earlier_source(connection):
    deal(connection)
    for f in range(13,19):observe(connection,f)
    batches=connection.ocr.batches
    for b in batches[3:]:complete(connection,b,'2000')
    assert not connection.states[connection.test_key]['numeric_triples']
    for b in reversed(batches[:3]):complete(connection,b,'1000')
    r=connection.states[connection.test_key]['numeric_triples']['seat1']
    assert r['frames']==[13,14,15] and r['displayed_stack']=='1000'
    complete(connection,batches[0],'9999')
    assert connection.states[connection.test_key]['numeric_triples']['seat1']==r
    assert connection.counts['DUPLICATE_STACK_CALLBACK']==8


def test_mismatch_first_attempt_never_replaced_by_prequeued_beautiful_triple(connection):
    deal(connection)
    for f in range(13,19):observe(connection,f)
    for i,b in enumerate(connection.ocr.batches):complete(connection,b,'1000' if i==0 else '2000')
    state=connection.states[connection.test_key]
    assert not state['numeric_triples']
    assert state['canonical_attempts']['seat1']['frames']==[13,14,15]
    assert state['canonical_attempts']['seat1']['blocker']=='FIRST_NUMERIC_TRIPLE_NOT_STABLE'


def test_gap_in_first_attempt_is_unknown_and_later_triple_cannot_replace(connection):
    deal(connection)
    for f in (13,15,16,17):
        observe(connection,f)
        if connection.ocr.batches:complete(connection,connection.ocr.batches[-1])
    state=connection.states[connection.test_key]
    assert not state['numeric_triples']
    assert state['canonical_attempts']['seat1']['blocker']=='FIRST_NUMERIC_TRIPLE_SOURCE_GAP'


def test_unqualified_predecessor_ocr_does_not_fix_numeric_visibility(connection):
    deal(connection)
    for f in range(13,19):observe(connection,f)
    for i,b in enumerate(connection.ocr.batches):complete(connection,b,'1000',status='ERROR' if i==0 else 'OK')
    state=connection.states[connection.test_key]
    assert state['numeric_triples']['seat1']['frames']==[14,15,16]
    assert state['numeric_triples']['seat1']['first_numeric_visible_frame']==14


def test_missing_first_source_after_deal_cannot_promote_later_snapshot(connection):
    deal(connection)
    for f in (14,15,16):
        observe(connection,f)
        complete(connection,connection.ocr.batches[-1])
    assert connection.states[connection.test_key]['canonical_attempts']['seat1']['blocker']=='FIRST_NUMERIC_SOURCE_COVERAGE_UNKNOWN'


def test_late_header_invalidates_canonical_attempt_as_well_as_numeric(connection):
    deal(connection)
    for f in (13,14,15):
        observe(connection,f)
        complete(connection,connection.ocr.batches[-1])
    connection.sidecar.histories[(1,'S')].append({'hand_id':'12345679','source_frame':14})
    connection.finish()
    state=connection.states[connection.test_key]
    assert not state['numeric_triples']
    assert state['canonical_attempts']['seat1']['status']=='UNKNOWN'
    assert state['canonical_attempts']['seat1']['blocker']=='LATE_HEADER_SOURCE_CONTRADICTION'


def test_low_confidence_ante_overlay_waits_for_normal_numeric(connection):
    deal(connection)
    for f in (13,14,15,16):
        observe(connection,f)
        complete(connection,connection.ocr.batches[-1],'ANIE' if f==13 else '1000', .82 if f==13 else .99)
    assert connection.states[connection.test_key]['numeric_triples']['seat1']['frames']==[14,15,16]


def test_same_hand_late_header_restores_exact_table_deal_source_frame(connection):
    obj=connection
    observe(obj,10)
    obj.sidecar.outstanding[(1,'S')].add((11,1000000))
    for f in (11,12,13):observe(obj,f)
    assert not obj.sidecar.deal_calls and not obj.ocr.batches
    obj.sidecar.outstanding[(1,'S')].clear()
    observe(obj,14)
    r=obj.sidecar.deal_calls[0][1]
    assert r['source_frame']==12 and r['card_stable_frames']==[10,11,12]
    assert r['source_pts']==12*512
    assert len(obj.sidecar.deal_calls)==1
    assert [b[0].frame_id for b in obj.ocr.batches]==[13,14]
    assert not obj.pending_tables


def test_unknown_header_does_not_release_saved_table_sources(connection):
    obj=connection
    observe(obj,10)
    obj.sidecar.outstanding[(1,'S')].add((11,1000000))
    observe(obj,11);observe(obj,12)
    obj.sidecar.outstanding[(1,'S')].clear()
    obj.sidecar.hand_state_gate.failed_headers[(1,'S')].add((11,0))
    observe(obj,13)
    assert not obj.sidecar.deal_calls and not obj.pending_tables


def test_new_hand_discards_old_table_buffer(connection):
    obj=connection
    observe(obj,10)
    obj.sidecar.outstanding[(1,'S')].add((11,1000000))
    observe(obj,11);observe(obj,12)
    obj.sidecar.outstanding[(1,'S')].clear()
    new=(1,'S','12345679')
    obj.sidecar.current[1]=dict(obj.sidecar.current[1],hand_id=new[2],source_frame=11,source_position=[11,0])
    obj.sidecar.windows[new]=CleanStackWindow(.95)
    observe(obj,13)
    assert not obj.sidecar.deal_calls and not obj.pending_tables
    assert obj.counts['PENDING_CONTEXT_DISCARDED']==1


def test_card_threshold_and_source_gap_unchanged_during_table_replay(connection,monkeypatch):
    obj=connection
    observe(obj,10)
    obj.sidecar.outstanding[(1,'S')].add((11,1000000))
    observe(obj,11);observe(obj,13)
    obj.sidecar.outstanding[(1,'S')].clear()
    observe(obj,14)
    assert not obj.sidecar.deal_calls
    observe(obj,15)
    assert obj.sidecar.deal_calls[0][1]['card_stable_frames']==[13,14,15]


def test_pending_header_two_sources_same_hand_preserves_buffered_stack_evidence(connection):
    obj=connection
    deal(obj)
    obj.sidecar.outstanding[(1,'S')].add((13,1000000))
    observe(obj,13);observe(obj,14)
    assert not obj.pending_tables and not obj.ocr.batches
    assert len(obj.pending_stacks[obj.test_key])==2
    assert all(evidence['seat1'].shape==(2,2) for source,evidence in obj.pending_stacks[obj.test_key])
    obj.sidecar.outstanding[(1,'S')].clear()
    observe(obj,15)
    assert not obj.pending_stacks
    assert [b[0].frame_id for b in obj.ocr.batches]==[13,14,15]
    for b in obj.ocr.batches:complete(obj,b)
    r=obj.states[obj.test_key]['numeric_triples']['seat1']
    assert r['frames']==[13,14,15] and r['first_numeric_visible_frame']==13
    assert [e['source_pts'] for e in r['evidence']]==[13*512,14*512,15*512]
    assert all(e['pixel_sha256'] for e in r['evidence'])


def test_pending_table_expiration_never_invents_consecutive_sources(connection):
    obj=connection
    obj.pending_buffer_frames=2
    observe(obj,10)
    obj.sidecar.outstanding[(1,'S')].add((11,1000000))
    for f in (11,12,13):observe(obj,f)
    obj.sidecar.outstanding[(1,'S')].clear()
    observe(obj,14)
    assert obj.counts['TABLE_PENDING_BUFFER_EXPIRED']==1
    assert obj.sidecar.deal_calls[0][1]['card_stable_frames']==[12,13,14]


def test_new_hand_discards_pending_stack_roi_without_ocr(connection):
    obj=connection;deal(obj)
    obj.sidecar.outstanding[(1,'S')].add((13,1000000))
    observe(obj,13);observe(obj,14)
    obj.sidecar.outstanding[(1,'S')].clear()
    new=(1,'S','12345679')
    obj.sidecar.current[1]=dict(obj.sidecar.current[1],hand_id=new[2],source_frame=13,source_position=[13,0])
    obj.sidecar.windows[new]=CleanStackWindow(.95)
    observe(obj,15)
    assert not obj.pending_stacks and not obj.ocr.batches


def test_unknown_header_discards_pending_stack_roi_without_ocr(connection):
    obj=connection;deal(obj)
    obj.sidecar.outstanding[(1,'S')].add((13,1000000))
    observe(obj,13);observe(obj,14)
    obj.sidecar.outstanding[(1,'S')].clear()
    obj.sidecar.hand_state_gate.failed_headers[(1,'S')].add((13,0))
    observe(obj,15)
    assert not obj.pending_stacks and not obj.ocr.batches


def test_pending_stack_ram_bound_is_five_sources_and_expired_coverage_unknown(connection):
    connection.pending_buffer_frames = 5  # Frozen old-cap loss remains fail closed.
    obj=connection;deal(obj)
    obj.sidecar.outstanding[(1,'S')].add((13,1000000))
    for f in range(13,20):observe(obj,f)
    assert len(obj.pending_stacks[obj.test_key])==5 and not obj.ocr.batches
    obj.sidecar.outstanding[(1,'S')].clear()
    observe(obj,20)
    assert len(obj.ocr.batches)==6
    for b in obj.ocr.batches:complete(obj,b)
    assert obj.states[obj.test_key]['canonical_attempts']['seat1']['blocker']=='FIRST_NUMERIC_SOURCE_COVERAGE_UNKNOWN'


def test_buffered_table_roi_uses_original_full_source_hash_without_full_frame_copy(connection):
    obj=connection
    obj.sidecar.outstanding[(1,'S')].add((10,1000000))
    for f in (10,11,12):observe(obj,f)
    pending=obj.pending_tables[obj.test_key]
    assert all(source[5] is None for source,evidence in pending['sources'])
    digest=hashlib.sha256(np.full((2,2,3),80,np.uint8).tobytes()).hexdigest()
    assert all(evidence['table_pixel_sha256']==digest for source,evidence in pending['sources'])
    obj.sidecar.outstanding[(1,'S')].clear()
    observe(obj,13)
    assert obj.sidecar.deal_calls[0][1]['table_pixel_sha256']==digest
    assert obj.sidecar.deal_calls[0][1]['source_frame']==12


def test_same_hand_header_resolved_at_eof_releases_stack_buffer(connection):
    obj=connection;deal(obj)
    obj.sidecar.outstanding[(1,'S')].add((13,1000000))
    observe(obj,13);observe(obj,14)
    assert not obj.ocr.batches
    obj.sidecar.outstanding[(1,'S')].clear()
    obj.finish()
    assert [b[0].frame_id for b in obj.ocr.batches]==[13,14]
    assert not obj.pending_stacks and not obj.pending_tables


def test_unresolved_eof_clears_ram_without_publishing_stack_sources(connection):
    obj=connection;deal(obj)
    obj.sidecar.outstanding[(1,'S')].add((13,1000000))
    observe(obj,13);observe(obj,14)
    obj.finish()
    assert not obj.ocr.batches and not obj.pending_stacks and not obj.pending_tables


@pytest.mark.parametrize('delay',[2,5,19,34])
def test_measured_pending_delay_preserves_first_source_order_hash_pts_and_no_png(connection, delay):
    obj=connection;deal(obj)
    obj.sidecar.outstanding[(1,'S')].add((13,1000000))
    for f in range(13,13+delay):observe(obj,f)
    assert len(obj.pending_stacks[obj.test_key])==delay and not obj.ocr.batches
    assert obj.counts['PEAK_BUFFERED_ROI_BYTES']==delay*8*4
    obj.sidecar.outstanding[(1,'S')].clear()
    observe(obj,13+delay)
    assert [b[0].frame_id for b in obj.ocr.batches]==list(range(13,14+delay))
    for batch in obj.ocr.batches:complete(obj,batch)
    attempt=obj.states[obj.test_key]['canonical_attempts']['seat1']
    assert attempt['status']=='PASS' and attempt['frames']==[13,14,15]
    assert [r['source_pts'] for r in attempt['evidence']]==[13*512,14*512,15*512]
    digest=hashlib.sha256(np.full((2,2),80,np.uint8).tobytes()).hexdigest()
    assert all(r['pixel_sha256']==digest for r in attempt['evidence'])
    assert not list(obj.output.rglob('*.png'))


def test_new_48_source_bound_enforced_and_loss_stays_unknown(connection):
    obj=connection;deal(obj)
    obj.sidecar.outstanding[(1,'S')].add((13,1000000))
    for f in range(13,63):observe(obj,f)
    assert len(obj.pending_stacks[obj.test_key])==48
    assert obj.counts['STACK_PENDING_BUFFER_EXPIRED']==2
    obj.sidecar.outstanding[(1,'S')].clear()
    observe(obj,63)
    for batch in obj.ocr.batches:complete(obj,batch)
    assert obj.states[obj.test_key]['canonical_attempts']['seat1']['blocker']=='FIRST_NUMERIC_SOURCE_COVERAGE_UNKNOWN'


def retain_26_27_then_current_28(obj):
    """Capture actual sources before a later receipt of the same hand."""
    deal(obj)
    obj.sidecar.outstanding[(1, 'S')].add((26, 1000000))
    observe(obj, 26)
    observe(obj, 27)
    saved = [(source, {seat: pixels.copy() for seat, pixels in evidence.items()})
             for source, evidence in obj.pending_stacks[obj.test_key]]
    obj.sidecar.current[1].update(source_frame=28, source_position=[28, 0], source_pts=28*512)
    obj.sidecar.header_fences[(1, 'S')].update(source_position=[28, 0])
    obj.sidecar.outstanding[(1, 'S')].clear()
    return saved


def test_retained_same_hand_26_27_replay_uses_anchor_preserves_exact_sources(connection):
    obj=connection
    saved=retain_26_27_then_current_28(obj)
    batch=obj._replay_pending(1, 'S', 29)
    assert [r.frame_id for r in batch]==[26]*8+[27]*8
    assert obj.sidecar.current[1]['source_frame']==28
    assert obj.sidecar.histories[(1, 'S')][0]['source_frame']==9
    for row in batch:
        source, crops=next(item for item in saved if item[0][2]==row.frame_id)
        _,seat,_=obj.jobs[f'{row.table_id}:{row.frame_id}:{row.source_span_index}']
        assert row.source_pts==source[3] and [row.time_base_num,row.time_base_den]==list(source[4])
        assert np.array_equal(row.pixels,crops[seat])
        assert row.pixel_sha256==hashlib.sha256(crops[seat].tobytes()).hexdigest()
    assert obj.counts.get('STACK_PENDING_BUFFER_EXPIRED',0)==0
    assert not list(obj.output.rglob('*.png'))


@pytest.mark.parametrize('frame',[26,27])
def test_fresh_stale_source_still_rejected_after_same_hand_header(connection,frame):
    obj=connection;deal(obj)
    obj.sidecar.current[1].update(source_frame=28,source_position=[28,0],source_pts=28*512)
    observe(obj,frame)
    assert not obj.ocr.batches and not obj.jobs


@pytest.mark.parametrize('failure',['history','unknown_fence','wrong_fence','active_session','failed_header','conflicting_history','anchor_frame','anchor_pts'])
def test_retained_replay_context_proof_fail_closed(connection,failure):
    obj=connection;retain_26_27_then_current_28(obj)
    if failure=='history':obj.sidecar.histories[(1,'S')].clear()
    if failure=='unknown_fence':obj.sidecar.header_fences[(1,'S')]['unknown']=True
    if failure=='wrong_fence':obj.sidecar.header_fences[(1,'S')]['hand_id']='12345679'
    if failure=='active_session':obj.sidecar.active_sessions[1]='OTHER'
    if failure=='failed_header':obj.sidecar.hand_state_gate.failed_headers[(1,'S')].add((20,0))
    if failure=='conflicting_history':obj.sidecar.histories[(1,'S')].append(dict(obj.sidecar.current[1],hand_id='12345679',source_frame=20))
    if failure=='anchor_frame':obj.sidecar.histories[(1,'S')][0].update(source_frame=28,source_position=[28,0])
    if failure=='anchor_pts':obj.sidecar.histories[(1,'S')][0]['source_pts']=28*512
    assert obj._replay_pending(1,'S',29)==[]
    assert not obj.jobs and not obj.pending_stacks


@pytest.mark.parametrize('change',['hand','session'])
def test_retained_replay_different_context_discards(connection,change):
    obj=connection;retain_26_27_then_current_28(obj)
    if change=='hand':obj.sidecar.current[1]['hand_id']='12345679'
    else:obj.sidecar.current[1]['session_id']='OTHER';obj.sidecar.active_sessions[1]='OTHER'
    assert obj._replay_pending(1,'S',29)==[]
    assert not obj.pending_stacks and not obj.jobs


@pytest.mark.parametrize('kind',['table','stack'])
def test_direct_evidence_without_retained_permit_cannot_bypass_current(connection,kind):
    obj=connection;retain_26_27_then_current_28(obj)
    source,crops=obj.pending_stacks[obj.test_key][0]
    kwargs={'stack_evidence':crops} if kind=='stack' else {'table_evidence':{'stacks':crops,'cards':{}}}
    assert not obj._observe_frame(*source,**kwargs)
    assert not obj.jobs


@pytest.mark.parametrize('control',['budget','money_closed','pending_again'])
def test_retained_replay_keeps_window_and_budget_controls(connection,control):
    obj=connection;retain_26_27_then_current_28(obj)
    if control=='budget':obj.sidecar.flags['capture_frames']=1
    if control=='money_closed':obj.sidecar.windows[obj.test_key].action('CALL',26)
    if control=='pending_again':obj.sidecar.outstanding[(1,'S')].add((26,1000000))
    batch=obj._replay_pending(1,'S',29)
    if control=='money_closed':
        # Existing API admits diagnostic OCR after closure but never stack proof.
        assert len(batch)==16 and obj.sidecar.windows[obj.test_key].closed==26
        complete(obj,batch)
        assert not obj.sidecar.windows[obj.test_key].results
    else:
        assert batch==[] and not obj.jobs


def test_retained_permit_with_wrong_evidence_identity_fails_closed(connection):
    obj=connection;retain_26_27_then_current_28(obj)
    source,evidence=obj.pending_stacks[obj.test_key][0]
    copy_evidence={k:v.copy() for k,v in evidence.items()}
    assert not obj._observe_frame(*source,stack_evidence=copy_evidence,
        _retained_source=(obj.test_key,source,evidence))
    assert not obj.jobs
