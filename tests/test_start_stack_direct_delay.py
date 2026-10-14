"""Bounded source-only tests using authentic Smoke19 pixels; no OCR runtime."""
import copy
import base64
import importlib.util
import json
from pathlib import Path

import pytest
import cv2
import numpy as np

from parser_core.start_stack_live import first_visible_stack_triple, delay_source_guard
from parser_core.start_stack_live import StartStackLiveConnection
from parser_core.start_stack_visibility import first_visible_proof
from parser_core.start_stack_visibility import VisibilitySource
from parser_core.first_stack_ocr import direct_numeric_policy

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('native_live_fixture_d20', ROOT/'tests/test_start_stack_live.py')
_native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_native)
connection = _native.connection
OCR = ('ocr_response', 'raw_text', 'confidence', 'ocr_status')
MARKERS = ('source_state', 'source_only', 'ocr_submitted', 'direct_ocr_delay_frames', 'deal_complete_frame')


def retained(seat='seat4'):
    payload = json.loads((ROOT/'tests/fixtures/first_visible_stack_smoke19.json').read_text('utf-8'))
    c = next(c for c in payload['contexts'] if c['seat'] == seat)
    return c, {r['source_frame']: r for r in c['readings']}


def skipped(readings, deal=257):
    result = copy.deepcopy(readings)
    for frame, r in result.items():
        if deal < frame <= deal + 20:
            for key in OCR:
                r.pop(key, None)
            r.update(source_state='OCR_SKIPPED_BY_DELAY', source_only=True,
                     ocr_submitted=False, direct_ocr_delay_frames=20, deal_complete_frame=deal)
    return result


def triple(readings, delay=20):
    return first_visible_stack_triple(readings, .85, final=True, start_frame=258,
        allow_scaled=True, numeric_policy=direct_numeric_policy(), direct_ocr_delay_frames=delay)


@pytest.mark.parametrize('seat', ['seat1', 'seat2', 'seat4', 'seat5'])
def test_native_proof_all_authority_and_pixel_phases_preserved(seat):
    c, original = retained(seat)
    candidate = skipped(original)
    frames = c['fact']['frames']
    before = first_visible_proof(original, frames, start_frame=258)
    after = first_visible_proof(candidate, frames, start_frame=258)
    assert {k:v for k,v in before.items() if k != 'source_evidence'} == {
        k:v for k,v in after.items() if k != 'source_evidence'}
    def sources(proof):
        return [{k:v for k,v in r.items() if k not in OCR + MARKERS}
                for r in proof.get('source_evidence', [])]
    assert sources(before) == sources(after)
    assert len([r for r in candidate.values() if r.get('source_only')]) == 20
    assert all(not any(k in r for k in OCR) for r in candidate.values() if r.get('source_only'))
    assert triple(original, 0) == triple(candidate)
    # This authentic fixture has no selected numeric inside the skipped prefix.
    assert not frames or frames[0] > 277


def test_skipped_is_not_pending_but_submitted_pending_fences_numeric():
    _, original = retained()
    candidate = skipped(original)
    assert triple(candidate)['status'] == 'PASS'
    candidate[278] = None
    assert triple(candidate) is None


@pytest.mark.parametrize('corruption', ['missing', 'none', 'hash', 'clock', 'context', 'fake_ocr', 'wrong_delay', 'outside_prefix'])
def test_skip_requires_authentic_source_and_cannot_supply_fake_ocr(corruption):
    _, original = retained()
    candidate = skipped(original)
    if corruption == 'missing':
        del candidate[266]
    elif corruption == 'none':
        candidate[266] = None
    elif corruption == 'hash':
        candidate[266]['pixel_sha256'] = '0' * 64
    elif corruption == 'clock':
        candidate[266]['source_pts'] = candidate[265]['source_pts']
    elif corruption == 'context':
        candidate[266]['hand_id'] = 'other-hand'
    elif corruption == 'fake_ocr':
        candidate[266]['ocr_status'] = 'NOT_READ'
    elif corruption == 'wrong_delay':
        candidate[266]['direct_ocr_delay_frames'] = 19
    else:
        candidate[278] = copy.deepcopy(candidate[277])
        candidate[278]['source_frame'] = 278
    actual = triple(candidate)
    if corruption == 'none':
        assert actual is None
    else:
        assert actual['status'] == 'UNKNOWN'
        assert actual['blocker'] == ('FIRST_NUMERIC_SOURCE_COVERAGE_UNKNOWN' if corruption == 'missing' else 'DELAY_SOURCE_PROOF_INVALID')


def test_default_zero_rejects_d20_source_only_receipts():
    _, original = retained()
    assert triple(skipped(original), 0)['blocker'] == 'DELAY_SOURCE_PROOF_INVALID'


def test_selected_numeric_pixels_cannot_be_erased_to_claim_first_visibility():
    c, original = retained()
    # Test double: deliberately malicious prefix containing authentic numeric
    # pixels copied from the selected frame. Native pixel proof must reject it.
    candidate = skipped(original)
    selected = copy.deepcopy(original[284])
    candidate[266]['visibility_source'] = selected['visibility_source']
    candidate[266]['pixel_sha256'] = selected['pixel_sha256']
    candidate[266]['png_base64'] = selected['png_base64']
    assert first_visible_proof(candidate, c['fact']['frames'], start_frame=258)['status'] == 'UNKNOWN'


def live_one_seat(connection):
    """Test-double transport/context, authentic retained RGB stack patches."""
    obj = connection
    _native.deal(obj)
    _, readings = retained()
    reference = readings[283]
    boxes = obj.sidecar.profile['slots']['slot_01']['seats']
    # Retain the fixture's actual table/seat ROI; native verifier reloads its
    # own immutable ROI config and must not be replaced by a verification fake.
    old_key = obj.test_key
    new_key = (5, old_key[1], old_key[2])
    obj.sidecar.profile['slots']['slot_05'] = {'seats': {'seat4': dict(boxes['seat1'], stack=reference['roi'])}}
    obj.sidecar.current[5] = obj.sidecar.current.pop(1)
    obj.sidecar.active_sessions[5] = obj.sidecar.active_sessions.pop(1)
    obj.sidecar.histories[(5,'S')] = obj.sidecar.histories.pop((1,'S'))
    obj.sidecar.header_fences[(5,'S')] = obj.sidecar.header_fences.pop((1,'S'))
    obj.sidecar.windows[new_key] = obj.sidecar.windows.pop(old_key)
    obj.states[new_key] = obj.states.pop(old_key)
    obj.test_key = new_key
    obj.sidecar.windows[new_key].deal['occupied_seats'] = ['seat4']
    obj.visibility_source = VisibilitySource()
    obj.direct_ocr_delay_frames = 20
    def observe_pixels(frame, retained_frame):
        reading = readings[retained_frame]
        crop = cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(
            reading['visibility_source']['source_rgb_png_base64']), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        rgb = np.zeros((404,847,3), np.uint8)
        x,y,X,Y = reference['roi']
        rgb[y:Y,x:X] = crop
        obj.observe_frame(5,'S',frame,frame*512,[1,15360],rgb,None)
        return crop
    return obj, observe_pixels


@pytest.mark.parametrize('prefix,unexpected', [('blank',284), ('overlay',284), ('blank',258)])
def test_actual_unexplained_pixels_restore_true_sources_in_order_once(connection, prefix, unexpected):
    obj, observe_pixels = live_one_seat(connection)
    observe_pixels(13,283 if prefix == 'blank' else 258)
    state = obj.states[obj.test_key]
    assert state['source_states']['seat4'][13] == 'OCR_SKIPPED_BY_DELAY'
    assert state['source_readings']['seat4'][13] is not None
    assert not obj.jobs and not obj.ocr.batches
    old_row = state['delayed_rows']['seat4'][0][0]
    # Authentic numeric, or forced overlay reappearing after blank: neither can
    # be positively explained as an uninterrupted old overlay/fade prefix.
    observe_pixels(14,unexpected)
    rows = [row for batch in obj.ocr.batches for row in batch]
    assert [r.frame_id for r in rows] == [13,14]
    assert rows[0] is old_row
    assert all(state['source_readings']['seat4'][f] is None for f in (13,14))
    assert all(state['source_states']['seat4'][f] == 'OCR_SUBMITTED_PENDING' for f in (13,14))
    assert state['delay_disabled'] == {'seat4'}
    assert obj.counts['STACK_DELAY_BACKFILLED'] == 1
    assert obj.counts['STACK_DELAY_FALLBACK_SEATS'] == 1
    assert first_visible_stack_triple(state['source_readings']['seat4'],.95,start_frame=13,direct_ocr_delay_frames=20) is None
    observe_pixels(15,283)  # Even blank is ordinary OCR after fallback.
    assert [r.frame_id for batch in obj.ocr.batches for r in batch] == [13,14,15]
    assert len(obj.jobs) == 3
    # Explicit test-double responses only; callbacks must remove pending states.
    for row in [r for batch in obj.ocr.batches for r in batch]:
        obj.completed(row, {'status':'EMPTY','raw_text':'','score':0.})
    assert not obj.jobs
    assert set(state['source_states']['seat4'].values()) == {'OCR_COMPLETED'}
    obj.completed(old_row, {'status':'EMPTY','raw_text':'','score':0.})
    assert obj.counts['DUPLICATE_STACK_CALLBACK'] == 1


def test_eof_only_skipped_source_creates_no_pending_or_fake_attempt(connection):
    obj, observe_pixels = live_one_seat(connection)
    observe_pixels(13,283)
    obj.finish()
    summary = json.loads((obj.output/'start_stack_live_summary.json').read_text('utf-8'))
    assert summary['pending_jobs'] == summary['submitted_pending_sources'] == 0
    assert not summary['errors']
    assert not summary['hands'][0]['canonical_attempts']
    assert not obj.ocr.batches


def test_eof_retains_genuine_submitted_pending_after_fallback(connection):
    obj, observe_pixels = live_one_seat(connection)
    observe_pixels(13,283)
    observe_pixels(14,284)
    obj.finish()  # Explicit FakeOCR transport supplies no callbacks.
    summary = json.loads((obj.output/'start_stack_live_summary.json').read_text('utf-8'))
    assert summary['pending_jobs'] == summary['submitted_pending_sources'] == 2
    assert not summary['hands'][0]['canonical_attempts']
    assert summary['hands'][0]['source_states']['seat4'] == {'13':'OCR_SUBMITTED_PENDING','14':'OCR_SUBMITTED_PENDING'}


def test_mixed_positive_overlay_labels_fail_closed():
    _, readings = retained()
    projection = VisibilitySource().verify(readings[258])
    class MixedLabels:
        """Explicit test double: same pixels, conflicting positive labels."""
        def __init__(self): self.calls = 0
        def identify(self, pixels):
            self.calls += 1
            return {'label': 'A' if self.calls == 1 else 'B'}
    state = {'patterns': MixedLabels()}
    assert delay_source_guard(projection,state) is not None
    assert delay_source_guard(projection,state) is None


def test_genuine_missing_source_is_not_policy_skipped_coverage():
    """Source existence remains mandatory even when adjacent frames are skipped."""
    _, original = retained()
    candidate = skipped(original)
    assert candidate[266]['source_state'] == 'OCR_SKIPPED_BY_DELAY'
    assert candidate[266]['pixel_sha256'] == original[266]['pixel_sha256']
    assert triple(candidate)['status'] == 'PASS'
    del candidate[266]
    missing = triple(candidate)
    assert missing['status'] == 'UNKNOWN'
    assert missing['blocker'] == 'FIRST_NUMERIC_SOURCE_COVERAGE_UNKNOWN'


@pytest.mark.parametrize('invalid_delay', [True, False, -1, 65, 1.5, '20'])
def test_invalid_config_delay_is_rejected_before_ocr_submission(connection, tmp_path, invalid_delay):
    """Fixture's real capture budget is64; policy cannot outlive that budget."""
    assert connection.sidecar.flags['capture_frames'] == 64
    output = tmp_path / ('invalid_delay_' + repr(invalid_delay).replace("'", ''))
    output.mkdir()
    with pytest.raises((ValueError, TypeError)):
        StartStackLiveConnection(connection.root, output, connection.sidecar, {},
                                 direct_ocr_delay_frames=invalid_delay)
    assert not connection.ocr.batches


def test_default_constructor_keeps_legacy_d0_submission(connection):
    assert connection.direct_ocr_delay_frames == 0
    _native.deal(connection)
    _native.observe(connection, 13)
    assert len(connection.ocr.batches) == 1
    assert len(connection.ocr.batches[0]) == 8
    assert {row.frame_id for row in connection.ocr.batches[0]} == {13}
    assert not connection.counts['STACK_OCR_SKIPPED_BY_DELAY']


def test_d20_requires_real_first_visible_contract(connection, tmp_path):
    rejected = tmp_path / 'delay_without_source_contract'
    rejected.mkdir()
    with pytest.raises(ValueError):
        StartStackLiveConnection(connection.root, rejected, connection.sidecar, {},
                                 direct_ocr_delay_frames=20)
    accepted = tmp_path / 'delay_with_source_contract'
    accepted.mkdir()
    obj = StartStackLiveConnection(connection.root, accepted, connection.sidecar, {},
                                   first_visible_contract=True, direct_ocr_delay_frames=20)
    try:
        assert obj.direct_ocr_delay_frames == 20
        assert obj.visibility_source is not None
        assert not obj.jobs and not obj.ocr.batches
    finally:
        obj.finish()


@pytest.mark.parametrize('seat', ['seat1', 'seat2', 'seat4', 'seat5'])
def test_omitted_delay_matches_explicit_legacy_zero(seat):
    _, readings = retained(seat)
    kwargs = dict(final=True, start_frame=258, allow_scaled=True,
                  numeric_policy=direct_numeric_policy())
    implicit = first_visible_stack_triple(readings, .85, **kwargs)
    explicit = first_visible_stack_triple(readings, .85,
                                          direct_ocr_delay_frames=0, **kwargs)
    assert implicit == explicit
