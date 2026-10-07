import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from parser_core.temporal_overlap import (
    View, native_signature, native_distance, Observation,
    BoundaryReducer, Context, hand_number, TemporalObservations,
)



def observation(rid, text, frame, index, inherited=False, pixels=None):
    if pixels is None:
        pixels = np.zeros((16, 228), np.uint8)
        cv2.putText(pixels, text, (1, 11), cv2.FONT_HERSHEY_SIMPLEX, .3, 220, 1)
    return Observation(View(rid, native_signature(pixels),
        hashlib.sha256(pixels.tobytes()).hexdigest(), index * 20, index,
        inherited, 'UNRESOLVED_IDENTITY', text), frame, pixels)


@pytest.fixture
def reducer(monkeypatch):
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


def test_unreadable_marker_after_old_visible_marker_is_context_barrier(reducer):
    reducer.process(1, 0, [observation('S:m', 'Hand #123456', 0, 0),
        observation('S:unknown', 'some line', 0, 1),
        observation('S:a', 'ante A', 0, 2), observation('S:b', 'ante B', 0, 3),
        observation('S:c', 'ante C', 0, 4)])
    changed_marker = observation('S:changed', 'Hand #123457', 1, 1).pixels
    reducer.process(1, 1, [observation('S:m', 'Hand #123456', 1, 0, True),
        observation('S:unreadable', '', 1, 1, pixels=changed_marker),
        observation('S:new', 'ante A', 1, 2), observation('S:b', 'ante B', 1, 3, True),
        observation('S:c', 'ante C', 1, 4, True)])
    assert not reducer.removals and reducer.birth['S:new'] is None


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


def test_visible_block_contract_rejects_repeated_frame_and_duplicate_span(reducer):
    seed(reducer)
    with pytest.raises(ValueError, match='frame'):
        reducer.process(1, 0, [observation('S:z', 'text', 0, 0)])
    with pytest.raises(ValueError, match='span'):
        reducer.process(1, 1, [observation('S:z', 'text', 1, 0),
                              observation('S:y', 'other', 1, 0)])


@pytest.mark.parametrize('ocr_error', [False, True])
def test_ram_collector_filters_before_downstream_without_mutating_raw_rows(ocr_error):
    records = TemporalObservations()
    first = [observation('S:m', 'Hand #123456', 0, 0),
             observation('S:a', 'ante A', 0, 1),
             observation('S:b', 'ante B', 0, 2),
             observation('S:c', 'ante C', 0, 3)]
    later = [observation('S:m', 'Hand #123456', 1, 0, True),
             observation('S:new', 'ante A', 1, 1),
             observation('S:b', 'ante B', 1, 2, True),
             observation('S:c', 'ante C', 1, 3, True)]
    timeline = []
    for obs in first + later:
        v = obs.view
        records.record(1, obs.frame_id, v.row_id, v.index, v.y1, v.origin,
                       obs.pixels, v.inherited)
        if not v.inherited:
            timeline.append({'stable_row_id': v.row_id, 'raw_ocr_text': v.text,
                             'status': 'UNRESOLVED', 'video_time': obs.frame_id / 30})
    before = json.dumps(timeline)
    statuses = {x['stable_row_id']: 'RAW_UNREVIEWED' for x in timeline}
    if ocr_error:
        statuses['S:new'] = 'ERROR'
    retained, r = records.reduce(timeline, {'1': 'S'}, ocr_status=statuses)
    assert json.dumps(timeline) == before
    assert [x['stable_row_id'] for x in retained] == ['S:m', 'S:a', 'S:b', 'S:c'] + (['S:new'] if ocr_error else [])
    assert len(r.removals) == (0 if ocr_error else 1) and len(r.decisions) == len(timeline)
    assert all(x['status'] == 'UNRESOLVED' for x in retained)
    assert records.compressed_bytes > 0 and records.records == []
    assert r.cores == {} and not r.marker_templates
    if r.removals:
        assert r.removals[0]['hand_marker_evidence']


@pytest.mark.parametrize('case_index', range(4))
def test_recorded_smoke17_losses_and_smoke18_positive(monkeypatch, case_index):
    import base64
    fixture = Path('experiments/fixtures/boundary_smoke17_20261001_v2/regression_fixtures.jsonl')
    cases = [json.loads(line) for line in fixture.read_text('utf-8').splitlines()]
    case = cases[case_index]
    def unpack(row):
        pixels = np.frombuffer(base64.b64decode(row['pixels']), np.uint8).reshape(row['shape']).copy()
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == row['sha256']
        return Observation(View(row['row_id'], native_signature(pixels), row['sha256'],
            row['y1'], row['index'], row['inherited'], row['origin'], row['text']), row['frame_id'], pixels)
    r = BoundaryReducer(case['sessions'])
    old = [unpack(row) for row in case['old_block']]
    r.process(case['table'], max(o.frame_id for o in old), old)
    ctx = Context(case['table'], case['sessions'][str(case['table'])], case['previous_hand'])
    core = r.cores[ctx]
    for entry in case['gallery']:
        obs = unpack(entry['observation'])
        assert obs.frame_id < case['source_frame']
        rid = entry['matched_id']
        r.birth.setdefault(obs.view.row_id, ctx)
        core.gallery[rid][obs.view.pixel_hash] = obs
        if obs.view.row_id != rid:
            core.aliases[obs.view.row_id] = rid
    r.process(case['table'], case['source_frame'], [unpack(row) for row in case['current_block']])
    removed = {x['old_row_id'] for x in r.removals} & set(case['expected_row_ids'])
    if case['kind'] == 'PROVEN_DUPLICATE':
        assert removed == set(case['expected_row_ids'])
    else:
        assert not removed
