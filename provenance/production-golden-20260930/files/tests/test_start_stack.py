from decimal import Decimal
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from parser_core.start_stack import (ChatAction, HandStackTracker, StackReading,
                                     classify_stack_text, contribution_before_stack)
from parser_core.start_stack_capture import AsyncStackCapture


def reading(seat, frame, text, *, table="T01", context="session1", score=.99):
    return StackReading(table, context, seat, frame, frame * 100, frame / 30,
                        f"hash{seat}{frame}", f"crop{seat}{frame}.png", text, score)


def tracker(role="NORMAL", seats=("seat1",), *, partial=False):
    hand = HandStackTracker("T01", "session1", "H1", 10, partial)
    for seat in seats:
        hand.bind_seat(seat, f"player_{seat}", role, identity_proven=True)
    return hand


@pytest.mark.parametrize("role,overlay,visible,forced,expected", [
    ("NORMAL", "ANTE", 198250, [("ANTE", 1750)], 200000),
    ("SB", "SB", 191250, [("ANTE", 1750), ("SMALL_BLIND", 7000)], 200000),
    ("BB", "BB", 184250, [("ANTE", 1750), ("BIG_BLIND", 14000)], 200000),
])
def test_forced_bet_transition(role, overlay, visible, forced, expected):
    hand = tracker(role)
    hand.observe(reading("seat1", 11, overlay))
    hand.observe(reading("seat1", 12, str(visible)))
    hand.observe(reading("seat1", 13, "1"))
    actions = [ChatAction("player_seat1", event, amount, True) for event, amount in forced]
    result = hand.result("seat1", actions, chat_complete=True, timing_proven=True)
    assert result["status"] == "CERTIFIED"
    assert Decimal(result["start_stack"]) == expected
    assert result["stack_frame"] == 12


def test_seats_transition_independently():
    hand = tracker(seats=("seat1", "seat2"))
    hand.observe(reading("seat1", 11, "ANTE"))
    hand.observe(reading("seat2", 11, "ANTE"))
    hand.observe(reading("seat1", 12, "198250"))
    hand.observe(reading("seat2", 13, "198250"))
    for seat, frame in (("seat1", 12), ("seat2", 13)):
        result = hand.result(seat, [ChatAction(f"player_{seat}", "ANTE", 1750, True)],
                             chat_complete=True, timing_proven=True)
        assert result["stack_frame"] == frame
        assert result["start_stack"] == "200000"


def test_fast_action_and_raise_to_accounting():
    hand = tracker("BB")
    hand.observe(reading("seat1", 11, "BB"))
    hand.observe(reading("seat1", 18, "148250"))
    actions = [ChatAction("player_seat1", "ANTE", 1750, True),
               ChatAction("player_seat1", "BIG_BLIND", 14000, True),
               ChatAction("player_seat1", "RAISE_TO", 50000, True)]
    total, reason = contribution_before_stack(actions, "player_seat1")
    assert reason is None and total == 51750
    result = hand.result("seat1", actions, chat_complete=True, timing_proven=True)
    assert result["start_stack"] == "200000"


def test_ambiguous_action_timing_is_unknown():
    hand = tracker()
    hand.observe(reading("seat1", 11, "ANTE"))
    hand.observe(reading("seat1", 12, "190000"))
    actions = [ChatAction("player_seat1", "ANTE", 1750, True),
               ChatAction("player_seat1", "CALL", 8250, None)]
    assert hand.result("seat1", actions, chat_complete=True,
                       timing_proven=True)["reason"] == "CONTRIBUTION_AMBIGUOUS"


def test_unreadable_identity_partial_and_context_reset():
    hand = tracker()
    hand.observe(reading("seat1", 11, "ANTE"))
    hand.observe(reading("seat1", 12, "???"))
    assert hand.result("seat1", [], chat_complete=True,
                       timing_proven=True)["status"] == "UNKNOWN"
    assert hand.result("seat1", [], chat_complete=True,
                       timing_proven=True)["reason"] == "STACK_OCR_FAILED"
    other = HandStackTracker("T01", "session2", "H2", 20, False)
    other.bind_seat("seat1", None, None, identity_proven=False)
    assert other.result("seat1", [], chat_complete=True,
                        timing_proven=True)["reason"] == "PLAYER_IDENTITY_UNKNOWN"
    with pytest.raises(ValueError):
        other.observe(reading("seat1", 21, "ANTE"))
    partial = tracker(partial=True)
    assert partial.result("seat1", [], chat_complete=True,
                          timing_proven=True)["reason"] == "PARTIAL_START"


def test_first_numeric_without_overlay_is_post_forced_bets():
    hand = tracker("SB")
    hand.observe(reading("seat1", 11, "191250"))
    hand.observe(reading("seat1", 12, "ANTE"))
    result = hand.result("seat1", [
        ChatAction("player_seat1", "ANTE", 1750, None),
        ChatAction("player_seat1", "SMALL_BLIND", 7000, None)],
        chat_complete=True, timing_proven=True)
    assert result["status"] == "CERTIFIED"
    assert result["start_stack"] == "200000"
    assert result["stack_frame"] == 11


def test_bb_forced_posts_added_without_frame_order_and_ante_can_be_absent():
    hand = tracker("BB")
    hand.observe(reading("seat1", 11, "97750"))
    result = hand.result("seat1", [
        ChatAction("player_seat1", "BIG_BLIND", 2000, None),
        ChatAction("player_seat1", "ANTE", 250, None)],
        chat_complete=True, timing_proven=True)
    assert result["start_stack"] == "100000"
    no_ante = HandStackTracker("T01", "session1", "H2", 10, False)
    no_ante.bind_seat("seat1", "player_seat1", "BB",
                      identity_proven=True, ante_policy="ABSENT")
    no_ante.observe(reading("seat1", 11, "98000"))
    result = no_ante.result("seat1", [ChatAction("player_seat1", "BIG_BLIND", 2000, None)],
                           chat_complete=True, timing_proven=True)
    assert result["start_stack"] == "100000"


def test_low_confidence_overlay_does_not_veto_numeric_and_proven_forced_post():
    hand = tracker()
    hand.observe(reading("seat1", 11, "ANTE", score=.1))
    hand.observe(reading("seat1", 12, "198250"))
    result = hand.result("seat1", [ChatAction("player_seat1", "ANTE", 1750, True)],
                         chat_complete=True, timing_proven=True)
    assert result["status"] == "CERTIFIED"
    assert result["start_stack"] == "200000"


def test_earlier_frame_rejected_before_queue(tmp_path: Path):
    hand = tracker()
    capture = AsyncStackCapture(hand, {"seat1": (0, 0, 2, 2)},
                                lambda _: ("ANTE", .99), crop_dir=tmp_path)
    with pytest.raises(ValueError):
        capture.offer_table("T01", "session1", 9, 9, .3,
                            np.zeros((2, 2, 3), dtype=np.uint8))
    capture.finish()


def test_exact_numeric_parser_rejects_action_text():
    assert classify_stack_text("1 750") == ("STACK", Decimal(1750))
    assert classify_stack_text("CALL 50000")[0] == "UNREADABLE"


def test_worker_order_and_bounded_nonblocking_queue(tmp_path: Path):
    gate = threading.Event()
    def slow(_pixels):
        gate.wait(2)
        return "ANTE", .99
    hand = tracker()
    capture = AsyncStackCapture(hand, {"seat1": (0, 0, 2, 2)}, slow,
                                crop_dir=tmp_path, queue_size=1)
    for frame in range(11, 20):
        capture.offer_table("T01", "session1", frame, frame, frame / 30,
                            np.full((2, 2, 3), frame, dtype=np.uint8))
    assert capture.max_queue <= 1
    assert "seat1" in capture.dropped
    gate.set()
    capture.finish()
    assert hand.result("seat1", [], chat_complete=True,
                       timing_proven=True)["status"] == "UNKNOWN"


def test_recognizer_failure_is_isolated(tmp_path: Path):
    def failing(_pixels):
        raise RuntimeError("model unavailable")
    hand = tracker()
    capture = AsyncStackCapture(hand, {"seat1": (0, 0, 2, 2)}, failing,
                                crop_dir=tmp_path, queue_size=2)
    capture.offer_table("T01", "session1", 11, 11, .3,
                        np.zeros((2, 2, 3), dtype=np.uint8))
    results = capture.finish()
    assert results[0].recognition_status == "ERROR"
    assert hand.result("seat1", [], chat_complete=True,
                       timing_proven=True)["status"] == "UNKNOWN"


def test_out_of_order_worker_completion_keeps_visual_order(tmp_path: Path):
    def recognizer(pixels):
        if int(pixels[0, 0, 0]) == 1:
            time.sleep(.05)
            return "ANTE", .99
        return "198250", .99
    hand = tracker()
    capture = AsyncStackCapture(hand, {"seat1": (0, 0, 2, 2)}, recognizer,
                                crop_dir=tmp_path, queue_size=2, workers=2)
    for frame, value in ((11, 1), (12, 2)):
        capture.offer_table("T01", "session1", frame, frame, frame / 30,
                            np.full((2, 2, 3), value, dtype=np.uint8))
    capture.finish()
    result = hand.result("seat1", [ChatAction("player_seat1", "ANTE", 1750, True)],
                         chat_complete=True, timing_proven=True)
    assert result["status"] == "CERTIFIED"
    assert result["stack_frame"] == 12
