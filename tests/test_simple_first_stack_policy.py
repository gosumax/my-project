import pytest
from experiments.simple_first_stack_policy import add_forced, first_numeric, parse_numeric


def row(frame=10, raw="171 250", **changes):
    return dict(table_id=2, session_id="current", seat="seat4", frame=frame, raw_text=raw, **changes)


def select(rows):
    return first_numeric(rows, table_id=2, session_id="current", seat="seat4", hand_start=10, hand_end=50)


@pytest.mark.parametrize("raw,value", [("171250", 171250), ("171 250", 171250),
    ("171\u00a0250", 171250), ("0", 0), (" 42 ", 42), ("Sit out", None),
    ("Ante 625", None), ("", None), (None, None), ("12 34", None)])
def test_parse_raw_integer_without_ocr_qualification(raw, value):
    assert parse_numeric(raw) == value


@pytest.mark.parametrize("former_blocker", ["WHO_UNKNOWN", "FIRST_VISIBLE", "FADE", "OVERLAY",
    "COVERAGE_UNKNOWN", "FORCED_ROLE_CONFLICT", "SQL_OBSERVATION_CONTEXT_CONFLICT",
    "INDEPENDENT_DIRECT_EARLIER_ACCEPTED_NUMBER", "PROVIDER_CONFLICT", "SOURCE_UNQUALIFIED"])
def test_former_blockers_and_low_confidence_cannot_block_numeric(former_blocker):
    result = select([row(10, "170 000", confidence=0, status=former_blocker, WHO="UNKNOWN"), row(11)])
    assert result["displayed_stack"] == 170000
    assert result["frame"] == 10


def test_first_numeric_is_source_order_not_input_order_and_needs_one_frame():
    assert select([row(13, "900"), row(10, "Ante"), row(12, "100"), row(11, "80")])["displayed_stack"] == 80


def test_current_hand_route_and_bounds():
    rows = [row(9, "1"), dict(row(10, "2"), seat="seat3"),
            dict(row(10, "3"), session_id="previous"), dict(row(10, "4"), table_id=1),
            row(51, "5"), row(50, "0")]
    assert select(rows)["displayed_stack"] == 0


def test_no_number_means_no_displayed_stack():
    assert select([row(10, "SB"), row(11, ""), row(51)]) is None


def test_same_frame_conflict_is_deterministic_and_never_rejects():
    assert select([row(10, "999", source_order=5), row(10, "20", source_order=1)])["displayed_stack"] == 20


def test_forced_correction_separate_from_displayed_and_who():
    first = select([row(10, "171250", WHO="UNKNOWN")])
    assert first["displayed_stack"] == 171250
    assert add_forced(first["displayed_stack"], {"ANTE": None, "SB": 0, "BB": 0}) is None
    assert first["displayed_stack"] == 171250
    assert add_forced(first["displayed_stack"], {"ANTE": 625, "SB": 0, "BB": 0}) == 171875
    assert add_forced(0, {"ANTE": 70, "SB": 350, "BB": 0}) == 420


def test_saved_scaled_ocr_is_read_from_native_nested_receipt_without_gates():
    from scripts.replay_saved_simple_first_stack import saved_scaled_response
    response = dict(raw_text="30", score=.01, status="UNKNOWN", item_id="8:146:1")
    raw = dict(ocr_response=dict(same_source_scale_evidence=dict(status="CONFLICT", response=response)))
    assert saved_scaled_response(raw) == response
    assert select([row(10, "Fade"), row(10, saved_scaled_response(raw)["raw_text"], source_order=1)])["displayed_stack"] == 30


def test_seat_forced_partial_amount_survives_unknown_who():
    from scripts.replay_saved_simple_first_stack import forced_for_seat
    posts = [dict(table=1, session="current", hand_id="1", seat="seat7", role="ANTE", amount=70, source_refs=["a"]),
             dict(table=1, session="current", hand_id="1", seat="seat1", role="BB", amount=600, source_refs=["b"])]
    forced, status, sources = forced_for_seat(posts, (1, "current", "1"), "seat7")
    assert forced == {"ANTE": 70, "SB": None, "BB": 0}
    assert status == "FORCED_CORRECTION_UNKNOWN"
    assert sources[0]["amount"] == 70
    assert add_forced(171250, forced) is None


def test_seat_forced_full_correction_does_not_need_player_name():
    from scripts.replay_saved_simple_first_stack import forced_for_seat
    posts = [dict(table=1, session="current", hand_id="1", seat="seat7", role="ANTE", amount=70, source_refs=["a"]),
             dict(table=1, session="current", hand_id="1", seat="seat1", role="BB", amount=600, source_refs=["b"]),
             dict(table=1, session="current", hand_id="1", seat="seat2", role="SB", amount=300, source_refs=["c"])]
    forced, status, _ = forced_for_seat(posts, (1, "current", "1"), "seat7")
    assert status == "KNOWN_SAVED_SEAT_POSTS"
    assert add_forced(171250, forced) == 171320
