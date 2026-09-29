from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from parser_core.annotation import FIELDS, VERSION, audit, compare, sha256, write_json
from parser_core.journal import Journal
from scripts.evaluate_visual_corpus import load_predictions
from scripts.freeze_visual_corpus import validate_splits
from scripts.build_visual_review_queue import build as build_visual_review_queue


def reference(two=False):
    def message(i):
        return {"message_id": f"m{i}", "row_ids": [f"r{i}"], "order": i,
                "completeness": "COMPLETE", "scenarios": ["identical_consecutive"],
                "fields": {f: {"state": "KNOWN" if f in {"actor", "action"} else "NONE",
                               "value": "player" if f == "actor" else "FOLD" if f == "action" else None}
                           for f in FIELDS}}
    return {"units": [{"unit_id": "u", "review_status": "VISUALLY_REVIEWED",
                       "reviewer": "visual-reviewer", "review_time": "2026-09-28T00:00:00Z",
                       "coverage_complete": True,
                       "rows": [{"row_id": f"r{i}", "visual_text": {"state": "KNOWN", "value": "player сбрасывает"},
                                 "anchors": [{"evidence_id": "e", "bbox": [0, i*20, 100, i*20+10]}]}
                                for i in range(2 if two else 1)],
                       "messages": [message(i) for i in range(2 if two else 1)]}]}


def prediction(i=0, status="CANDIDATE"):
    return {"unit_id": "u", "event_id": f"p{i}", "status": status, "order_key": str(i),
            "fields": {"actor": "player", "action": "FOLD", "amount": None, "role": None, "cards": None},
            "fragments": [{"anchors": [{"evidence_id": "e", "bbox": [0, i*20, 100, i*20+12]}]}]}


def test_exact_candidate_is_not_accepted():
    result = compare(reference(), [prediction()])
    assert result["metrics"]["exact_event"] == {"numerator": 1, "denominator": 1, "rate": 1}
    assert result["counts"]["accepted_exact_event"] == 0
    assert result["metrics"]["false_accept"]["rate"] is None


def test_identical_text_on_distinct_rows_keeps_two_events():
    result = compare(reference(two=True), [prediction(0), prediction(1)])
    assert result["counts"]["exact_event"] == 2
    assert result["counts"]["reference"] == 2
    assert result["counts"]["order_errors"] == 0


def test_missing_and_unresolved_are_not_success():
    p = prediction(status="UNRESOLVED")
    p["fields"]["action"] = "UNKNOWN_EVENT"
    result = compare(reference(two=True), [p])
    assert result["counts"]["missing"] == 1
    assert result["metrics"]["exact_event"]["numerator"] == 0
    assert result["metrics"]["unresolved_predictions"]["rate"] == 1


def test_duplicate_candidates_do_not_get_greedy_credit():
    a, b = prediction(status="VERIFIED"), prediction(status="VERIFIED")
    b["event_id"] = "other"
    result = compare(reference(), [a, b])
    assert result["metrics"]["exact_event"]["numerator"] == 0
    assert result["counts"]["excess_predictions_for_message"] == 1
    assert result["metrics"]["false_accept"] == {"numerator": 2, "denominator": 2, "rate": 1}


def test_merge_false_accept_counted_once_in_each_scenario():
    p = prediction(status="VERIFIED")
    p["fragments"].extend(prediction(1)["fragments"])
    result = compare(reference(two=True), [p])
    assert result["counts"]["boundary_error"] == 2
    assert result["counts"]["false_accept"] == 1
    assert result["scenarios"]["identical_consecutive"]["metrics"]["false_accept"]["rate"] == 1


def test_wrong_order_detected_from_observation_keys():
    a, b = prediction(0), prediction(1)
    a["order_key"], b["order_key"] = b["order_key"], a["order_key"]
    result = compare(reference(two=True), [a, b])
    assert result["counts"]["order_errors"] == 1


def test_unknown_reference_is_not_positive_and_blocks_acceptance():
    labels = reference()
    labels["units"][0]["messages"][0]["fields"]["actor"] = {"state": "UNKNOWN", "value": None}
    result = compare(labels, [prediction(status="VERIFIED")])
    assert result["counts"]["exact_event_denominator"] == 0
    assert result["counts"]["false_accept"] == 1
    assert result["counts"]["exact_actor_denominator"] == 0


def test_unreviewed_unit_is_not_quality_measurement():
    labels = reference()
    labels["units"][0]["review_status"] = "UNREVIEWED"
    result = compare(labels, [prediction(status="VERIFIED")])
    assert result["quality_status"] == "NOT_MEASURED"
    assert result["metrics"]["exact_event"]["rate"] is None


def test_false_extra_is_reported_without_text_matching():
    p = prediction(1, "VERIFIED")
    result = compare(reference(), [p])
    assert result["counts"]["missing"] == 1
    assert result["counts"]["extra_predictions"] == 1
    assert result["counts"]["false_accept"] == 1


def test_amount_is_exact_without_normalizing_reference():
    labels = reference()
    labels["units"][0]["messages"][0]["fields"]["amount"] = {"state": "KNOWN", "value": "6000"}
    p = prediction()
    p["fields"]["amount"] = "6 000"
    assert compare(labels, [p])["counts"]["exact_amount"] == 0


def corpus(tmp_path):
    Image.new("L", (100, 40)).save(tmp_path / "chat.png")
    write_json(tmp_path / "manifest.json", {"evidence": [{"evidence_id": "e", "local": "chat.png",
        "sha256": sha256(tmp_path / "chat.png"), "size": [100, 40], "roi_type": "chat"}],
        "units": [{"unit_id": "u", "evidence_ids": ["e"]}],
        "independent_test": False, "independence_basis": "DEV"})
    digest = sha256(tmp_path / "manifest.json")
    (tmp_path / "manifest.sha256").write_text(digest)
    labels = reference()
    labels.update({"schema_version": VERSION, "manifest_sha256": digest})
    return labels


def test_audit_hashes_box_and_review_attestation(tmp_path):
    labels = corpus(tmp_path)
    assert audit(tmp_path, labels)["valid"]
    labels["units"][0]["rows"][0]["anchors"][0]["bbox"][2] = 101
    assert not audit(tmp_path, labels)["valid"]
    labels = corpus(tmp_path)
    labels["units"][0]["reviewer"] = None
    assert not audit(tmp_path, labels)["valid"]


def test_audit_rejects_changed_png_and_manifest(tmp_path):
    labels = corpus(tmp_path)
    Image.new("L", (100, 40), 255).save(tmp_path / "chat.png")
    assert any(x.startswith("PNG_HASH_MISMATCH") for x in audit(tmp_path, labels)["errors"])
    (tmp_path / "manifest.json").write_text((tmp_path / "manifest.json").read_text() + " ")
    assert "FROZEN_MANIFEST_CHANGED" in audit(tmp_path, labels)["errors"]


def test_split_leakage_and_prior_use_fail_closed():
    base = {"sha256": "a", "session_group": "session", "split": "dev"}
    with pytest.raises(ValueError, match="leakage"):
        validate_splits([base, {**base, "sha256": "b", "split": "validation"}], set())
    holdout = {**base, "split": "test", "prior_use": "UNUSED_AT_FREEZE",
               "independence_reviewer": "r", "independence_review_time": "time"}
    with pytest.raises(ValueError, match="prior use"):
        validate_splits([holdout], {"a"})
    assert validate_splits([holdout], set())
    assert validate_splits([{**holdout, "prior_use": "RAW_ONLY_REVIEW_TOOL_QA"}], set())
    with pytest.raises(ValueError, match="prior use"):
        validate_splits([{**holdout, "prior_use": "PARSER_RULE_TUNING"}], set())


def test_visual_review_queue_selects_changes_without_ocr_answers(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    evidence = []
    for frame, color in enumerate((0, 0, 255)):
        for kind in ("chat", "table"):
            name = f"{frame}_{kind}.png"
            Image.new("L", (20, 20), color if kind == "chat" else 127).save(frozen / name)
            evidence.append({"evidence_id": name, "frame_id": frame, "roi_type": kind,
                             "source_pts": frame, "time_base": [1, 30],
                             "local": name, "sha256": sha256(frozen / name)})
    write_json(frozen / "manifest.json", {"evidence": evidence, "units": [{
        "unit_id": "u", "table_session_id": "t",
        "table_session": {"geometry_json": '{"slot_id": 1}'},
        "evidence_ids": [x["evidence_id"] for x in evidence]}]})
    (frozen / "manifest.sha256").write_text(sha256(frozen / "manifest.json"))
    result = build_visual_review_queue(frozen, tmp_path / "queue")
    assert result["selected_changed_frames"] == 2
    assert result["raw_chat_frames"] == 3
    summary = json.loads((tmp_path / "queue" / "queue_summary.json").read_text(encoding="utf-8"))
    assert summary["frame_ids_contiguous_all_units"] is True
    text = (tmp_path / "queue" / "changed_chat_frames.csv").read_text(encoding="utf-8-sig")
    assert "FIRST_FRAME" in text and "CHAT_PNG_CHANGED" in text
    assert "UNREVIEWED" in text and "FOLD" not in text


def test_adapter_selects_latest_requested_normalizer_not_historical_max(tmp_path):
    journal = Journal(tmp_path / "journal.sqlite3")
    journal.add_source("s", "video", "h", 1)
    journal.add_session("t", "s", "layout", {})
    journal.add_frame("s", 1, 10, 1, 1000, 100, 40)
    Image.new("L", (100, 40)).save(tmp_path / "chat.png")
    Image.new("L", (100, 12)).save(tmp_path / "row.png")
    journal.add_observation("e", "s", 1, "t", "chat", [0, 0, 100, 40], "chat.png", sha256(tmp_path / "chat.png"), "RAW")
    journal.add_observation("o", "s", 1, "t", "CHAT_ROW", [0, 0, 100, 12], "row.png", sha256(tmp_path / "row.png"), "RAW")
    db = journal.db
    db.execute("INSERT INTO physical_rows VALUES ('r','t','epoch',10,10,'RAW',NULL)")
    db.execute("INSERT INTO row_observations VALUES ('r','o',0)")
    db.execute("INSERT INTO messages VALUES ('m','t','CANDIDATE','KNOWN','KNOWN',NULL,'assembly')")
    db.execute("INSERT INTO message_fragments VALUES ('m',0,'r',NULL)")
    for rev, version in ((1, "requested"), (2, "historical_other")):
        db.execute("""INSERT INTO event_revisions VALUES
            ('event',?,'t','m',NULL,'FOLD','player',NULL,NULL,NULL,NULL,'[]','01',10,'UNKNOWN','CANDIDATE',NULL,?,'{}')""", (rev, version))
    db.commit()
    journal.close()
    manifest = {"units": [{"unit_id": "u", "evidence_ids": ["e"]}],
                "evidence": [{"evidence_id": "e", "roi_type": "chat", "source_sha256": "h",
                              "table_session_id": "t", "frame_id": 1, "bbox": [0,0,100,40],
                              "size": [100,40], "sha256": sha256(tmp_path / "chat.png")}]}
    before = sha256(tmp_path / "journal.sqlite3")
    predictions, _ = load_predictions(tmp_path, manifest, "assembly", "requested")
    assert predictions[0]["revision"] == 1
    assert predictions[0]["fields"]["cards"] is None
    assert sha256(tmp_path / "journal.sqlite3") == before
    with pytest.raises(ValueError, match="no events"):
        load_predictions(tmp_path, manifest, "wrong", "requested")
