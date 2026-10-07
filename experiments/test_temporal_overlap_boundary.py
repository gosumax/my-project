import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from experiments import temporal_overlap_dedup as baseline
from experiments.temporal_overlap_native import native_signature, native_distance
from experiments.temporal_overlap_guarded import Observation
from experiments.temporal_overlap_boundary import BoundaryReducer, Context, hand_number


def observation(rid, text, frame, index, inherited=False, pixels=None):
    if pixels is None:
        pixels = np.zeros((16, 228), np.uint8)
        cv2.putText(pixels, text, (1, 11), cv2.FONT_HERSHEY_SIMPLEX, .3, 220, 1)
    return Observation(baseline.View(rid, native_signature(pixels),
        hashlib.sha256(pixels.tobytes()).hexdigest(), index * 20, index,
        inherited, 'UNRESOLVED_IDENTITY', text), frame, pixels)


@pytest.fixture
def reducer(monkeypatch):
    monkeypatch.setattr(baseline, 'canonical_row_distance', native_distance)
    return BoundaryReducer({'1': 'S', '2': 'T'})


def seed(r, text='Dealer: PlayerA posts Ante 600'):
    r.process(1, 0, [observation('S:m', 'Hand #123456', 0, 0),
        observation('S:a', text, 0, 1), observation('S:b', 'PlayerB posts Ante 600', 0, 2),
        observation('S:c', 'PlayerC posts Ante 600', 0, 3)])


def replay(r, text, marker='123456', frame=10):
    r.process(1, frame, [observation('S:m' if marker == '123456' else 'S:m2', 'Hand #' + marker, frame, 0, marker == '123456'),
        observation('S:new', text, frame, 1),
        observation('S:b', 'PlayerB posts Ante 600', frame, 2, True),
        observation('S:c', 'PlayerC posts Ante 600', frame, 3, True)])


@pytest.mark.parametrize('text', ['Dealer: PlayerA posts Ante 600',
    'Dealer: PlayerA posts Small Blind 12500', 'Dealer: PlayerA posts Big Blind 25000',
    'Dealer: Dealing cards', 'Dealer: PlayerA raises to 50000'])
def test_same_hand_scroll_duplicate_still_removed(reducer, text):
    seed(reducer, text)
    replay(reducer, text)
    assert [r['old_row_id'] for r in reducer.removals] == ['S:new']
    assert reducer.removals[0]['hand_id'] == reducer.removals[0]['previous_hand_id'] == '123456'


@pytest.mark.parametrize('text', ['Dealer: PlayerA posts Ante 600',
    'Dealer: PlayerA posts Small Blind 12500', 'Dealer: PlayerA posts Big Blind 25000',
    'Dealer: Dealing cards'])
def test_new_hand_identical_pixels_and_old_inherited_anchors_kept(reducer, text):
    seed(reducer, text)
    replay(reducer, text, marker='123457')
    assert not reducer.removals
    assert reducer.birth['S:new'].hand == '123457'
    assert reducer.birth['S:b'].hand == '123456'
    assert all('S:b' not in c.gallery for ctx, c in reducer.cores.items() if ctx.hand == '123457')


def test_same_hand_starting_duplicate_with_number_pair_removed(reducer):
    first = [observation('S:m', 'Hand #123456', 0, 0),
        observation('S:a', 'Dealer: Starting a new hand', 0, 1),
        observation('S:n', '#123456', 0, 2), observation('S:b', 'ante B', 0, 3)]
    reducer.process(1, 0, first)
    reducer.process(1, 2, [observation('S:m', 'Hand #123456', 2, 0, True),
        observation('S:new', 'Dealer: Starting a new hand', 2, 1),
        observation('S:n', '#123456', 2, 2, True), observation('S:b', 'ante B', 2, 3, True)])
    assert [r['old_row_id'] for r in reducer.removals] == ['S:new']


def test_starting_and_ante_after_new_number_keep(reducer):
    reducer.process(1, 0, [observation('S:m', 'Hand #123456', 0, 0),
        observation('S:start', 'Dealer: Starting a new hand', 0, 1),
        observation('S:n', '#123456', 0, 2), observation('S:b', 'ante B', 0, 3), observation('S:c', 'ante C', 0, 4)])
    reducer.process(1, 10, [observation('S:m2', 'Hand #123457', 10, 0),
        observation('S:start2', 'Dealer: Starting a new hand', 10, 1),
        observation('S:n2', '#123457', 10, 2), observation('S:newb', 'ante B', 10, 3),
        observation('S:c', 'ante C', 10, 4, True)])
    assert not reducer.removals
    assert reducer.birth['S:start2'].hand == reducer.birth['S:newb'].hand == '123457'


def test_markerless_old_anchors_cannot_confer_context_on_new_ante(reducer):
    seed(reducer)
    reducer.process(1, 100, [observation('S:new', 'Dealer: PlayerA posts Ante 600', 100, 1),
        observation('S:b', 'PlayerB posts Ante 600', 100, 2, True),
        observation('S:c', 'PlayerC posts Ante 600', 100, 3, True)])
    assert not reducer.removals
    assert reducer.birth['S:new'] is None


@pytest.mark.parametrize('boundary', ['Dealer: Starting a new hand', 'Hand #????', '#12345x'])
def test_ambiguous_boundary_keeps_unresolved(reducer, boundary):
    seed(reducer)
    reducer.process(1, 10, [observation('S:unclear', boundary, 10, 0),
        observation('S:new', 'Dealer: PlayerA posts Ante 600', 10, 1),
        observation('S:b', 'PlayerB posts Ante 600', 10, 2, True),
        observation('S:c', 'PlayerC posts Ante 600', 10, 3, True)])
    assert not reducer.removals
    assert reducer.birth['S:new'] is None


def test_stale_inherited_marker_ocr_does_not_hide_changed_digit(reducer):
    seed(reducer)
    changed = observation('S:changed', 'Hand #123457', 10, 0).pixels
    reducer.process(1, 10, [observation('S:m', 'Hand #123456', 10, 0, True, changed),
        observation('S:new', 'Dealer: PlayerA posts Ante 600', 10, 1),
        observation('S:b', 'PlayerB posts Ante 600', 10, 2, True),
        observation('S:c', 'PlayerC posts Ante 600', 10, 3, True)])
    assert not reducer.removals and reducer.birth['S:new'] is None


@pytest.mark.parametrize('marker_text', ['Hand#123456', 'Hand  #123456', 'Dealer: #123456'])
def test_rejected_marker_any_valid_format_is_a_barrier(reducer, marker_text):
    reducer.process(1, 0, [observation('S:m', 'Hand #123456', 0, 0),
        observation('S:n', marker_text, 0, 1), observation('S:a', 'ante A', 0, 2),
        observation('S:b', 'ante B', 0, 3), observation('S:c', 'ante C', 0, 4)])
    changed = observation('S:changed', 'Hand #123457', 10, 1).pixels
    reducer.process(1, 10, [observation('S:m', 'Hand #123456', 10, 0, True),
        observation('S:n', marker_text, 10, 1, True, changed),
        observation('S:new', 'ante A', 10, 2), observation('S:b', 'ante B', 10, 3, True),
        observation('S:c', 'ante C', 10, 4, True)])
    assert not reducer.removals and reducer.birth['S:new'] is None


def test_conflict_barrier_does_not_join_two_anchors(reducer):
    seed(reducer)
    reducer.birth['S:foreign'] = Context(1, 'S', '123455')
    reducer.process(1, 10, [observation('S:m', 'Hand #123456', 10, 0, True),
        observation('S:foreign', 'foreign context', 10, 1, True),
        observation('S:new', 'Dealer: PlayerA posts Ante 600', 10, 2),
        observation('S:b', 'PlayerB posts Ante 600', 10, 3, True)])
    assert not reducer.removals
    assert reducer.barriers > 0


def test_decreasing_scroll_marker_cannot_roll_live_hand_back(reducer):
    seed(reducer)
    replay(reducer, 'Dealer: PlayerA posts Ante 600', marker='123457')
    replay(reducer, 'Dealer: PlayerA posts Ante 600', frame=20)
    assert reducer.active[1].hand == '123457'
    assert not reducer.removals


def test_sessions_are_isolated_and_wrong_session_rejected(reducer):
    seed(reducer)
    reducer.process(2, 10, [observation('T:m', 'Hand #123456', 10, 0),
        observation('T:a', 'Dealer: PlayerA posts Ante 600', 10, 1)])
    assert not reducer.removals and reducer.birth['T:a'].session == 'T'
    with pytest.raises(ValueError, match='session'):
        reducer.process(2, 11, [observation('S:b', 'same', 11, 0, True)])


def test_number_parser_cannot_treat_side_pot_as_hand():
    assert hand_number('side pot #123456') is None
    assert hand_number('Hand #123456') == hand_number('#123456') == '123456'


@pytest.mark.parametrize('case_index', range(4))
def test_recorded_smoke17_losses_and_smoke18_positive(monkeypatch, case_index):
    from experiments.boundary_regression_replay import replay_case
    monkeypatch.setattr(baseline, 'canonical_row_distance', native_distance)
    fixture = Path(__file__).parent / 'fixtures/boundary_smoke17_20261001_v2/regression_fixtures.jsonl'
    cases = [json.loads(line) for line in fixture.read_text('utf-8').splitlines()]
    assert len(cases) == 4
    assert replay_case(cases[case_index])['pass']
