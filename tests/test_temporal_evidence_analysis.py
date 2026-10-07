"""Regression contracts for the offline, lossless temporal evidence sidecar."""
import copy
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from experiments.capture_temporal_evidence import lifetime, source_ms, timestamp_text
from experiments.analyze_temporal_evidence import classify, enrich, order, source_window, conflicting_marker, leading_prefix
from parser_core.jev_poker_resolver import hypothesis_options
import parser_core.ram_first_stage as capture


def track(frame=10, ms=1000, origin='NEW_CANDIDATE', index=4, **extra):
    return dict(first_seen_frame=frame, first_seen_ms=ms, first_seen_row_index=index,
        origin=origin, left_censored=False, visibility_gap_count=0, **extra)


def boundary(frame=10, ms=1000, index=2):
    return dict(frame_id=frame, timestamp_ms=ms, row_index=index)


def test_real_pts_vfr_not_frame_over_fps():
    assert source_ms(5103, 1, 10000) == 510.3
    assert source_ms(None, 1, 30) is None
    assert timestamp_text(511433) == '00:08:31.433'


def test_adjacent_rows_reuse_existing_track_and_unchanged_roi(monkeypatch):
    images = []
    for i in range(4):
        rgb = np.zeros((160, 226, 3), dtype=np.uint8)
        for j, y in enumerate((8, 38, 68)):
            rgb[y:y+8, 12:112] = 100+j*40
            rgb[y+2:y+6, 25+j*7:39+j*7] = 230
        if i >= 2:
            rgb[150, 225] = 1
        images.append(rgb)
    class Frame:
        width = 226
        height = 160
        time_base = Fraction(1, 30)
        def __init__(self, i): self.pts = i
        def to_ndarray(self, format): return images[self.pts].copy()
    class Source:
        streams = SimpleNamespace(video=[SimpleNamespace(time_base=Fraction(1, 30))])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def decode(self, stream): return iter(Frame(i) for i in range(4))
    layout = SimpleNamespace(source_width=226, source_height=160,
        slots=[SimpleNamespace(slot_id=1, table_box=(0, 0, 226, 160), chat_box=(0, 0, 226, 160))])
    monkeypatch.setattr(capture, 'load_fixed_layout', lambda *args: layout)
    monkeypatch.setattr(capture.av, 'open', lambda *args: Source())
    states = {}
    def observe(m, pixels):
        states.setdefault(('s', m['frame_id']), []).append(dict(track_id=m['stable_row_id'],
            table_id='T01', row_index=m['source_span_index'], origin=m['origin']))
    stage = capture.FirstStage(Path('fake'), {'1': 's'}, 4, root=Path('.'), identity_observer=observe)
    emitted = list(stage.rows())
    for r in emitted:
        states.setdefault(('s', r.frame_id), []).append(dict(track_id=r.stable_row_id,
            table_id='T01', row_index=r.source_span_index, origin=r.origin))
    assert len(emitted) == 2
    assert {v['track_id'] for v in states[('s', 2)]} == {r.stable_row_id for r in emitted}
    times = [{'timestamp_ms': source_ms(i, 1, 30)} for i in range(4)]
    result = lifetime(states, {'s': [0, 2]}, times, 4)
    assert all(t['observation_count'] == 4 and t['visible_spans'] == [[0, 3]] for t in result.values())


def test_repeated_text_after_gap_remains_separate_ids():
    # IDs come from the production visual tracker, never OCR text equality.
    states = {('s', 0): [dict(track_id='a', table_id='T01', row_index=1, origin='NEW_CANDIDATE')],
              ('s', 10): [dict(track_id='b', table_id='T01', row_index=1, origin='NEW_CANDIDATE')]}
    times = [{'timestamp_ms': i*1000} for i in range(12)]
    result = lifetime(states, {'s': [0, 1, 10]}, times, 12)
    assert set(result) == {'a', 'b'}
    assert result['a']['last_seen_frame'] == 0
    assert result['b']['first_seen_ms'] == 10000


def test_previous_track_persists_after_new_hand():
    assert classify(track(frame=5, ms=500), boundary(), boundary(20, 2000))[0] == 'PREVIOUS_HAND_TAIL'


def test_delayed_ocr_uses_old_source_time():
    t = track(frame=5, ms=500)
    t['ocr_completion_ms'] = 3000
    assert classify(t, boundary(), boundary(20, 2000))[0] == 'PREVIOUS_HAND_TAIL'


def test_new_track_after_next_start():
    assert classify(track(frame=21, ms=2100), boundary(), boundary(20, 2000))[0] == 'NEXT_HAND_HEAD'


def test_completion_order_does_not_control_source_chronology():
    a = dict(boundary(20, 2000), completion_ms=100)
    b = dict(boundary(10, 1000), completion_ms=5000)
    assert sorted([a, b], key=order) == [b, a]


def test_enrichment_preserves_original_fields_pixels_ids_and_timestamp():
    rid = 's:R000001'
    row = dict(stable_row_id=rid, frame_id=1, row_index=2, session_id='s', table_id='T01',
        row_pixel_sha256='abc', video_time=.125, source_roi_change_id='s:1:chat',
        raw_ocr_text='raw  text', normalized_text='raw  text', confidence=.77)
    physical = dict(row_id=rid, frame_id=1, source_span_index=2, table_session_id='s',
        row_pixel_sha256='abc', source_pts=1, time_base_num=1, time_base_den=8,
        row_key='1:1:2', y1=25, row_pixels_base64='AAAA')
    t = dict(track(frame=1, ms=125), last_seen_ms=250, observation_count=2,
        explicit_view_count=1, first_seen_kind='FIRST_VISIBLE', right_censored=False)
    before = copy.deepcopy((row, physical, t))
    enriched = enrich([row], [physical], {rid: t})
    assert (row, physical, t) == before
    assert all(enriched[0][k] == v for k, v in row.items())
    assert enriched[0]['timestamp_ms'] == 125 and enriched[0]['source_pts'] == 1
    assert enriched[0]['stable_row_id'] == rid and enriched[0]['row_key'] == '1:1:2'
    assert 'track_id' not in enriched[0] and 'row_id' not in enriched[0]
    assert physical['row_pixels_base64'] == 'AAAA'


def test_unresolved_identity_or_censored_current_is_not_upgraded():
    assert classify(track(origin='UNRESOLVED_IDENTITY'), boundary(), boundary(20, 2000))[0] == 'AMBIGUOUS'
    t = track(); t['left_censored'] = True
    assert classify(t, boundary(), boundary(20, 2000))[0] == 'AMBIGUOUS'
    assert classify(track(), boundary(), None)[0] == 'AMBIGUOUS'


def test_same_timestamp_uses_source_spatial_order():
    assert classify(track(index=1), boundary(index=2), boundary(20, 2000))[0] == 'AMBIGUOUS'
    assert classify(track(index=3), boundary(index=2), boundary(20, 2000))[0] == 'CURRENT_HAND'


def test_current_header_before_blinds_is_not_previous_tail():
    t = track(frame=3699, ms=123300, index=3, origin='UNRESOLVED_IDENTITY')
    assert classify(t, boundary(3699, 123300, 13), boundary(4000, 133333))[0] == 'AMBIGUOUS'
    assert classify(track(frame=8, ms=800), boundary(), boundary(20, 2000), candidate_prefix=True)[0] == 'AMBIGUOUS'


def test_unresolved_or_gapped_old_track_not_certified_previous():
    assert classify(track(frame=5, ms=500, origin='UNRESOLVED_IDENTITY'), boundary(), boundary(20, 2000))[0] == 'AMBIGUOUS'
    t = track(frame=5, ms=500); t['visibility_gap_count'] = 1
    assert classify(t, boundary(), boundary(20, 2000))[0] == 'AMBIGUOUS'


def test_empty_changed_viewport_ends_lifetime():
    states = {('s', 0): [dict(track_id='a', table_id='T01', row_index=1, origin='NEW_CANDIDATE')]}
    times = [{'timestamp_ms': i*100} for i in range(5)]
    t = lifetime(states, {'s': [0, 2]}, times, 5)['a']
    assert t['visible_spans'] == [[0, 1]] and t['last_seen_ms'] == 100


def test_history_reappearance_retains_gap_without_inventing_visibility():
    view = dict(track_id='a', table_id='T01', row_index=1, origin='NEW_CANDIDATE')
    states = {('s', 0): [view], ('s', 4): [view]}
    times = [{'timestamp_ms': i*100} for i in range(6)]
    t = lifetime(states, {'s': [0, 2, 4]}, times, 6)['a']
    assert t['visible_spans'] == [[0, 1], [4, 5]]
    assert t['observation_count'] == 4 and t['visibility_gap_count'] == 1


def test_sameframe_missing_window_excludes_later_rows():
    rows = [dict(frame_id=5608, row_index=i, timestamp_ms=186933.333, session_id='s') for i in range(2, 7)]
    result = source_window(rows, 's', rows[0], rows[1])
    assert result == rows[:2]
    assert source_window(rows, 'other', rows[0], rows[1]) == []


def test_existing_resolver_duplicate_options_are_replayed_without_api():
    messages = [dict(raw_text='Dealer: a folds', source_row_ids=[rid], amount=None, action='FOLD')
                for rid in ['r1', 'r2']]
    hand = {'messages': messages}
    before = copy.deepcopy(hand)
    option = hypothesis_options(hand, [{'message_index': 1}])[0]
    assert option['decision'] == 'LIKELY_STALE_DUPLICATE'
    assert option['target_rows'] == ['r2'] and option['evidence_rows'] == ['r1']
    assert hand == before


def test_other_table_or_session_header_cannot_block_current_interval():
    current = dict(boundary(12841, 428033), table_id='T01', session_id='s1')
    start, following = boundary(12841, 428033), boundary(14037, 467900)
    other = dict(boundary(13933, 464433), table_id='T03', session_id='s3')
    members = {'header': {'hand_id': 'number', 'start_row_id': 'header'}}
    assert not conflicting_marker(current, start, following, members, {'header': other}, [])
    other['table_id'] = 'T01'
    assert not conflicting_marker(current, start, following, members, {'header': other}, [])
    other['session_id'] = 's1'
    assert conflicting_marker(current, start, following, members, {'header': other}, [])


def test_header_of_unique_confirmed_next_candidate_not_a_hidden_boundary():
    current = dict(boundary(100, 3333), table_id='T06', session_id='s6')
    start, following = dict(boundary(100, 3333), row_id='start'), dict(boundary(3699, 123300, 13), row_id='blinds')
    header = dict(boundary(3699, 123300, 3), table_id='T06', session_id='s6')
    members = {'header': {'hand_id': 'number', 'start_row_id': 'header'},
               'blinds': {'hand_id': 'number', 'start_row_id': 'header'},
               'start': {'hand_id': 'old', 'start_row_id': 'start'}}
    rows = {'header': header, 'start': current, 'blinds': dict(following, table_id='T06', session_id='s6')}
    assert not conflicting_marker(current, start, following, members, rows, [start, following])
    members['blinds']['start_row_id'] = 'different_header'
    assert conflicting_marker(current, start, following, members, rows, [start, following])


def test_future_unknown_marker_does_not_negate_earlier_visibility():
    assert classify(track(frame=5, ms=500), boundary(), boundary(20, 2000), blocked=True)[0] == 'PREVIOUS_HAND_TAIL'
    assert classify(track(), boundary(), boundary(20, 2000), blocked=True)[0] == 'AMBIGUOUS'


def test_unnumbered_interior_candidate_is_not_leading_prefix():
    assert not leading_prefix({'hand_id': None, 'start_row_id': 'R000247'}, 'R000001')
    assert leading_prefix({'hand_id': None, 'start_row_id': 'R000001'}, 'R000001')
    assert not leading_prefix({'hand_id': '123', 'start_row_id': 'R000001'}, 'R000001')
