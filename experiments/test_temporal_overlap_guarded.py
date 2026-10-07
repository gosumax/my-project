import cv2
import numpy as np
import pytest
from experiments import temporal_overlap_dedup as baseline
from experiments.temporal_overlap_guarded import Observation, Reducer, guarded_metrics
from experiments.temporal_overlap_native import native_signature, native_distance


def text_pixels(text):
    pixels = np.zeros((16, 228), np.uint8)
    cv2.putText(pixels, text, (1, 11), cv2.FONT_HERSHEY_SIMPLEX, .3, 220, 1)
    return pixels


@pytest.mark.parametrize('old_digit,new_digit', [(a, b) for a in '0123456789' for b in '0123456789' if a != b])
def test_single_amount_digit_is_not_duplicate(old_digit, new_digit):
    old = native_signature(text_pixels('lifeforkaifff bets 44 05' + old_digit))
    new = native_signature(text_pixels('lifeforkaifff bets 44 05' + new_digit))
    assert guarded_metrics(old, new) is None


@pytest.mark.parametrize('edge', [0, -1])
def test_active_edge_difference_cannot_be_discarded(edge):
    old = native_signature(text_pixels('same action'))
    new = old.copy()
    new[edge, 20:50] = 220
    assert guarded_metrics(old, new) is None


def obs(rid, text, frame, index, inherited=False, origin='UNRESOLVED_IDENTITY'):
    pixels = text_pixels(text or 'unreadable pixels')
    v = baseline.View(rid, native_signature(pixels), str(hash(pixels.tobytes())),
                      index * 20, index, inherited, origin, text)
    return Observation(v, frame, pixels)


def test_empty_ocr_cannot_create_alias(monkeypatch):
    monkeypatch.setattr(baseline, 'canonical_row_distance', native_distance)
    reducer = Reducer()
    reducer.process(1, 0, [obs('a', '', 0, 0), obs('b', 'anchor b', 0, 1), obs('c', 'anchor c', 0, 2)])
    reducer.process(1, 1, [obs('new', '', 1, 0), obs('b', 'anchor b', 1, 1, True), obs('c', 'anchor c', 1, 2, True)])
    assert not reducer.removals and not reducer.aliases


def test_alias_chain_gallery_is_prior_only_and_tables_isolated(monkeypatch):
    monkeypatch.setattr(baseline, 'canonical_row_distance', native_distance)
    reducer = Reducer(gallery=True)
    reducer.process(1, 0, [obs('a', 'action', 0, 0), obs('b', 'anchor b', 0, 1), obs('c', 'anchor c', 0, 2)])
    reducer.process(2, 1, [obs('other:a', 'action', 1, 0), obs('other:b', 'anchor b', 1, 1), obs('other:c', 'anchor c', 1, 2)])
    reducer.process(1, 2, [obs('dup1', 'action', 2, 0), obs('b', 'anchor b', 2, 1, True), obs('c', 'anchor c', 2, 2, True)])
    reducer.process(1, 3, [obs('dup2', 'action', 3, 0), obs('b', 'anchor b', 3, 1, True), obs('c', 'anchor c', 3, 2, True)])
    assert reducer.resolve('dup1') == reducer.resolve('dup2') == 'a'
    assert len(reducer.removals) == 2
    assert all(r['matched_previous_observation']['frame_id'] < r['frame_id'] for r in reducer.removals)
    assert [v.row_id for v in reducer.histories[2]] == ['other:a', 'other:b', 'other:c']


def test_future_and_self_gallery_cannot_confirm(monkeypatch):
    monkeypatch.setattr(baseline, 'canonical_row_distance', native_distance)
    reducer = Reducer(gallery=True)
    reducer.process(1, 0, [obs('a', 'action', 0, 0), obs('b', 'anchor b', 0, 1), obs('c', 'anchor c', 0, 2)])
    reducer.gallery['a'] = {'future': obs('a', 'action', 2, 0, True)}
    reducer.process(1, 1, [obs('dup', 'action', 1, 0), obs('b', 'anchor b', 1, 1, True), obs('c', 'anchor c', 1, 2, True)])
    assert not reducer.removals


def test_identical_action_in_later_hand_with_new_anchors_is_kept(monkeypatch):
    monkeypatch.setattr(baseline, 'canonical_row_distance', native_distance)
    reducer = Reducer(gallery=True)
    reducer.process(1, 0, [obs('old:a', 'same action', 0, 0),
                          obs('old:b', 'anchor b', 0, 1), obs('old:c', 'anchor c', 0, 2)])
    # New poker context introduces new identities even for identical text/pixels.
    reducer.process(1, 10, [obs('new:a', 'same action', 10, 0),
                           obs('new:b', 'anchor b', 10, 1), obs('new:c', 'anchor c', 10, 2)])
    assert not reducer.removals
    assert reducer.resolve('new:a') == 'new:a'
    assert [v.row_id for v in reducer.histories[1]] == ['old:a', 'old:b', 'old:c', 'new:a', 'new:b', 'new:c']
