import hashlib
import json
import sqlite3

from PIL import Image

from parser_core.visual_crops import crop_saved_table


def test_derived_crop_keeps_exact_table_frame_and_pts(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    original = baseline / "table.png"
    Image.new("RGB", (20, 20), (10, 20, 30)).save(original)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    with sqlite3.connect(baseline / "journal.sqlite3") as db:
        db.execute("CREATE TABLE observations (crop_path TEXT, crop_sha256 TEXT, bbox_json TEXT, "
                   "source_id TEXT, frame_id INTEGER, table_session_id TEXT, roi_type TEXT)")
        db.execute("CREATE TABLE frames (source_id TEXT, frame_id INTEGER, source_pts INTEGER, "
                   "time_base_num INTEGER, time_base_den INTEGER)")
        db.execute("CREATE TABLE sources (source_id TEXT, sha256 TEXT)")
        db.execute("INSERT INTO sources VALUES ('source', 'video-hash')")
        db.execute("INSERT INTO frames VALUES ('source', 42, 21504, 1, 15360)")
        db.execute("INSERT INTO observations VALUES (?,?,?,?,?,?,?)",
                   ("table.png", digest, "[0,0,20,20]", "source", 42, "table1", "table"))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"slots": {"slot_01": {"global": {
        "board_flop": [2, 3, 12, 11]}}}}), encoding="utf-8")
    result = crop_saved_table(baseline, profile, tmp_path / "derived.png",
                              session_id="table1", table_id=1, frame_id=42,
                              kind="BOARD", street="FLOP")
    assert result["frame_id"] == 42
    assert result["source_pts"] == 21504
    assert result["crop_box_local"] == [2, 3, 12, 11]
    assert result["source_table_sha256"] == digest
    assert Image.open(result["crop_path"]).size == (10, 8)
