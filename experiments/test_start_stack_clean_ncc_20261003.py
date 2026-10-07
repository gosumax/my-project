"""Boundary regressions and synthetic forced-only clean-window integration."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import start_stack_clean_window_20261003 as clean


def event(action, frame, player="opponent", amount="999"):
    return dict(action=action, frame_id=frame, player=player, amount=amount,
                source_row_ids=[f"row-{frame}"], table="T01", hand_id="h1")


@pytest.mark.parametrize("action", ["CALL", "BET", "RAISE", "ALL_IN"])
def test_first_global_money_closes_despite_fold_check(action):
    events = [event("CHECK", 11), event("FOLD", 12), event(action, 13),
              event("RAISE", 14, player="seat-player")]
    close, first = clean.boundary(events, 10, 100)
    assert close == 13
    assert first["action"] == action and first["player"] == "opponent"


def test_no_money_fold_check_stay_open_until_settlement():
    assert clean.boundary([event("FOLD", 11), event("CHECK", 12)], 10, 100) == (100, None)
    assert clean.boundary([event("CHECK", 11), event("WIN", 30)], 10, 100) == (30, None)
    assert clean.boundary([event("UNCALLED_RETURN", 25), event("CALL", 40)], 10, 100)[0] == 25


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    """Exercise run() with local synthetic pixels, no OCR process or real journal."""
    configs = tmp_path / "configs"
    configs.mkdir()
    profile = {"slots": {"slot_01": {"seats": {"seat1": {"stack": [0, 0, 2, 2]}}}}}
    profile_path = configs / "layout_847x404.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    (configs / "ocr_tiny_local.json").write_text(json.dumps(dict(
        python="unused", model_dir="unused", model_name="unused",
        weights_sha256="unused", device="cpu")), encoding="utf-8")
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    crop_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
    ocr = dict(item_id=crop_hash, raw_text="100", score=.99, status="OK")
    probes = [dict(table="T01", hand_id="h1", seat="seat1", kind="stack",
                   frame=f, crop_bgr_sha256=crop_hash, ocr=ocr,
                   table_path=f"synthetic-frame-{f}") for f in [2, 3, 4]]
    data = {
        "selected_hands.jsonl": [dict(table="T01", hand_id="h1", session="session1", start=0, end=100)],
        "visual_reviews.jsonl": [dict(table="T01", hand_id="h1", deal_complete_frame=1,
            occupied_seats=["seat1"], visually_read_nicknames={"seat1": "player"})],
        "forced_visual_evidence.jsonl": [dict(table="T01", hand_id="h1", status="CONFIRMED",
            evidence_frame=1, forced_by_seat={"seat1": dict(ante="2", sb="3", bb="7")})],
        "numeric_nick_probes.jsonl": probes,
    }

    class Database:
        def execute(self, *args):
            return self

        def fetchone(self):
            return (json.dumps(dict(slot_id=1, profile_sha256=clean.digest(profile_path))),)

        def close(self):
            pass

    class NoOCR:
        def __init__(self, *args):
            pass

        def read_batch(self, *args):
            pytest.fail("Existing synthetic OCR should be reused")

        def close(self):
            pass

    monkeypatch.setattr(clean, "ROOT", tmp_path)
    monkeypatch.setattr(clean, "database", Database)
    monkeypatch.setattr(clean, "TinyChatClient", NoOCR)
    monkeypatch.setattr(clean, "table", lambda db, session, f: (pixels, dict(frame=f, table_path=f"synthetic-frame-{f}")))
    monkeypatch.setattr(clean, "read_jsonl", lambda p: data.get(Path(p).name, []))

    def run(close):
        data["events.jsonl"] = [event("FOLD", 2), event("CHECK", 3), event("CALL", close)]
        summary = clean.run(tmp_path)
        records = [json.loads(s) for s in (tmp_path / "clean_forced_window/start_stack_results.jsonl").read_text(encoding="utf-8").splitlines()]
        return summary, records[0]

    return run


def test_confirmation_on_first_money_frame_is_rejected(synthetic_run):
    summary, record = synthetic_run(4)
    assert record["status"] == "NO_STACK_BEFORE_FIRST_ACTION"
    assert record["snapshot_frame"] is None
    assert summary["counts"]["START_STACK_RECOVERED"] == 0


def test_pre_gate_call_closes_window_and_prevents_forced_only_recovery(synthetic_run):
    assert clean.boundary([event("CALL", 0), event("RAISE", 5)], 1, 100)[0] == 0
    summary, record = synthetic_run(0)
    assert summary["hands_with_clean_window"] == 0
    assert record["window_close_exclusive"] == 0
    assert record["status"] == "NO_STACK_BEFORE_FIRST_ACTION"
    assert record["snapshot_frame"] is None
    assert summary["counts"]["START_STACK_RECOVERED"] == 0


def test_complete_triple_before_first_money_uses_forced_only(synthetic_run):
    summary, record = synthetic_run(5)
    assert record["status"] == "START_STACK_RECOVERED"
    assert record["source_frames"] == [2, 3, 4]
    assert record["stable_confirmed_frame"] < record["window_close_exclusive"]
    assert record["start_stack"] == "112"
    assert record["used_call_bet_raise_return_win"] is False
    assert record["future_money_events_used"] == summary["post_window_money_rows_in_calculation"] == 0
