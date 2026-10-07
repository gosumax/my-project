import json

from scripts.report_full_start_stack import build


def test_full_report_reserves_temporal_unknown_for_voluntary_action(tmp_path):
    timeline = tmp_path / "timeline.jsonl"
    rows = [
        (10, 0, "#123", "HAND_ID"),
        (11, 1, "Dealer: Vasya posts Ante 100", "UNKNOWN_EVENT"),
        (30, 0, "#124", "HAND_ID"),
        (31, 1, "Dealer: Vasya posts Ante 100", "UNKNOWN_EVENT"),
        (35, 2, "Dealer: Vasya raises to 500", "UNKNOWN_EVENT"),
    ]
    timeline.write_text("".join(json.dumps({
        "table_id": "T01", "frame_id": frame, "row_index": index,
        "raw_ocr_text": text, "event_type": event,
        "source_roi_change_id": f"table:{frame}:chat"}) + "\n"
        for frame, index, text, event in rows), "utf-8")
    observer = tmp_path / "start_stack_observations.jsonl"
    seen = [{"table_id": "T01", "seat": "seat1", "frame_index": frame,
             "roi_type": roi, "table_crop_sha256": f"table{frame}",
             "visual_kind": kind, "confidence": .99,
             "raw_recognition": text}
            for frame in (20, 40)
            for roi, kind, text in (("nick", "NICKNAME", "Vasya"),
                                    ("stack", "STACK", "19700"))]
    observer.write_text("".join(json.dumps(item) + "\n" for item in seen), "utf-8")
    (tmp_path / "start_stack_observer_summary.json").write_text(json.dumps({
        "status": "RAW_EVIDENCE_ONLY", "metrics": {"dropped_candidates": 0}}), "utf-8")
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps({"hands": []}), "utf-8")
    result = build(timeline, observer, proof)
    assert result["hands"][0]["seats"][0]["reason"] == "EVIDENCE_REVIEW_MISSING"
    assert result["hands"][1]["seats"][0]["reason"] == "TEMPORAL_ORDER_UNKNOWN"
    assert result["temporal_details"] == {"VOLUNTARY_RAISE_TO_BEFORE_FIRST_STACK": 1}
