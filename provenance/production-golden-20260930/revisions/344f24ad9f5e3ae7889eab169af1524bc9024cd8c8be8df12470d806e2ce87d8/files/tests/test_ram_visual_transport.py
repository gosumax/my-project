import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from parser_core.board_roundtrip import associate_board_crops, board_triggers
from parser_core.board_tiles import read_board_tiles, read_board_tiles_bytes
from parser_core.card_hybrid import CardHybrid
from parser_core.visual_cards import associate_visual_evidence
from parser_core.visual_crops import crop_ram_board, snapshot_ram_board, crop_saved_table
from parser_core.visual_tile_queue import VisualTileQueue
from parser_core.ram_first_stage import FirstStage
from parser_core.vl_chat_client import VLChatClient
from scripts.vl_chat_worker import request_image_bytes
from scripts.run_smoke14_ram import source_identity

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reader():
    return CardHybrid(ROOT / "configs/card_hybrid_local.json", ROOT / "configs/suit_backup_local.json")


@pytest.fixture(scope="module")
def card_record():
    return json.loads((ROOT / "runs/card_train_smoke16_test_20260930/current_production_rank_probe/raw_responses.jsonl")
                      .read_text("utf-8").splitlines()[0])


def normalized_reading(result):
    return {**result, "tiles": [{k: v for k, v in tile.items()
                                 if k not in {"tile_png", "tile_png_base64"}}
                                for tile in result["tiles"]]}


@pytest.mark.parametrize("raw_text", [None, "A", "A A A A A"])
def test_card_hybrid_ram_preserves_file_reading_exactly(reader, card_record, tmp_path, raw_text):
    path = Path(card_record["chat_row_png"])
    raw = path.read_bytes()
    text = raw_text if raw_text is not None else card_record["vl_response"]["raw_text"]
    old = reader.read(path, card_record["chat_row_sha256"], text, tmp_path / "old")
    new = reader.read_bytes(raw, card_record["chat_row_sha256"], text)
    assert normalized_reading(old) == normalized_reading(new)
    for tile in new["tiles"]:
        assert tile["tile_png"] is None
        data = base64.b64decode(tile["tile_png_base64"], validate=True)
        assert hashlib.sha256(data).hexdigest() == tile["tile_sha256"]
        with Image.open(io.BytesIO(data)) as image:
            assert hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() == tile["tile_pixel_sha256"]
    assert read_board_tiles(path, card_record["chat_row_sha256"], text) == read_board_tiles_bytes(
        raw, card_record["chat_row_sha256"], text)


def test_real_18px_bright_span_compares_native_suit_pixels(reader):
    record = json.loads((ROOT / "tests/fixtures/card_hybrid_unsupported_18px.json").read_text("utf-8"))
    raw = base64.b64decode(record["crop_png_base64"], validate=True)
    result = reader.read_bytes(raw, record["crop_sha256"], record["vl_raw_text"])
    assert len(result["vl_rank_tokens"]) == len(result["tiles"]) == 5
    first = result["tiles"][0]
    assert first["detected_bright_box_xyxy_exclusive"] == [14, 0, 32, 16]
    assert first["box_xyxy_exclusive"] == [14, 0, 32, 16]
    assert first["tile_geometry_source"] == "BRIGHT_SPAN"
    assert first["ncc_query_box_xyxy_exclusive"] == [12, 0, 32, 16]
    assert first["ncc_geometry_source"] == "SUIT_COMPONENT_NATIVE_ROW_ROI"
    assert first["suit_source"] == "template_backup"
    assert first["size_wh"] == [18, 16]
    assert first["template_suit"] == "c" and first["template_confident"] is True
    assert first["card"] == "Jc"
    assert first["template_best_score"] >= .98 and first["template_margin"] >= .03
    with Image.open(io.BytesIO(raw)) as row:
        pixels = np.asarray(row.convert("RGB"))
    for tile in result["tiles"]:
        x1, y1, x2, y2 = tile["box_xyxy_exclusive"]
        data = base64.b64decode(tile["tile_png_base64"], validate=True)
        assert hashlib.sha256(data).hexdigest() == tile["tile_sha256"]
        with Image.open(io.BytesIO(data)) as image:
            np.testing.assert_array_equal(np.asarray(image), pixels[y1:y2, x1:x2])


def test_unsupported_geometry_cannot_be_replaced_by_strict_vl_cards():
    timeline = [{"table_id": "T09", "session_id": "s", "epoch": 0, "frame_id": 0,
                 "row_index": 4, "raw_ocr_text": "Dealer: Dealing River"},
                {"table_id": "T09", "session_id": "s", "epoch": 0, "frame_id": 0,
                 "row_index": 5, "raw_ocr_text": "cards"}]
    reading = {"reason": "UNSUPPORTED_TILE_GEOMETRY", "vl_status": "OK",
               "vl_raw_text": "Jh Td 8d Kd 8s"}
    associate_visual_evidence(timeline, {"9:0:5": reading})
    assert reading["reason"] == "UNSUPPORTED_TILE_GEOMETRY"
    assert reading["status"] == "UNRESOLVED"
    assert reading["trigger_row_key"] == "9:0:4"
    assert "cards" not in reading


def test_ram_queue_never_uses_path_or_writes_png(tmp_path):
    pixels = np.zeros((16, 226), dtype=np.uint8)
    pixels[:, :50] = 255
    row = SimpleNamespace(table_id=2, frame_id=9, source_span_index=3, session_id="session-2",
        video_time=.3, source_pts=4608, time_base_num=1, time_base_den=15360,
        pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(), pixels=pixels,
        stable_row_id="session-2:R000123", epoch=4)
    class Client:
        def read_bytes(self, data, sha, trim):
            assert trim == 12
            assert hashlib.sha256(data).hexdigest() == sha
            with Image.open(io.BytesIO(data)) as image:
                np.testing.assert_array_equal(np.asarray(image), pixels)
            return {"status": "ERROR", "raw_text": "partial", "source_sha256": sha}
        def read(self, *args, **kwargs):
            pytest.fail("filesystem used as VL transport")
        def close(self):
            pass
    pool = VisualTileQueue(lambda _: Client(), tmp_path, image_transport="ram")
    assert pool.submit(row)
    pool.finish()
    result = pool.results["2:9:3"]
    assert result["row_id"] == row.stable_row_id
    assert (result["frame_id"], result["table_session_id"], result["epoch"]) == (9, "session-2", 4)
    assert result["crop_path"] is None
    assert result["status"] == "UNRESOLVED" and result["vl_raw_text"] == "partial"
    assert hashlib.sha256(base64.b64decode(result["crop_png_base64"])).hexdigest() == result["crop_sha256"]
    assert not list(tmp_path.rglob("*.png"))


def test_queue_propagates_unsupported_geometry_over_primary_reading(tmp_path, monkeypatch):
    pixels = np.zeros((16, 226), dtype=np.uint8)
    pixels[:, 10:30] = 255
    pixels[:, 33:53] = 255
    row = SimpleNamespace(table_id=9, frame_id=0, source_span_index=5, session_id="s",
        video_time=0, source_pts=0, time_base_num=1, time_base_den=15360,
        pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(), pixels=pixels,
        stable_row_id="s:R5", epoch=0)
    class Client:
        def read_bytes(self, data, sha, trim):
            return {"status": "OK", "raw_text": "Jc Td", "source_sha256": sha}
        def close(self):
            pass
    monkeypatch.setattr("parser_core.visual_tile_queue.read_board_tiles_bytes", lambda *args:
        SimpleNamespace(cards=("J♣", "T♦"), template_sha256="template"))
    pool = VisualTileQueue(lambda _: Client(), tmp_path, image_transport="ram")
    pool.hybrid = SimpleNamespace(suit_backup=SimpleNamespace(mode="BACKUP"),
        read_bytes=lambda *args, **kwargs: {"reason": "UNSUPPORTED_TILE_GEOMETRY", "tiles": []})
    assert pool.submit(row)
    pool.finish()
    reading = pool.results["9:0:5"]
    assert reading["reason"] == "UNSUPPORTED_TILE_GEOMETRY"
    assert reading["status"] == "UNRESOLVED" and reading["cards"] is None


def test_worker_ram_transport_validates_exact_hash_and_rejects_ambiguous_transport():
    data = b"exact bytes"
    request = {"png_base64": base64.b64encode(data).decode(), "source_sha256": hashlib.sha256(data).hexdigest()}
    assert request_image_bytes(request) == data
    with pytest.raises(ValueError, match="SHA256"):
        request_image_bytes({**request, "source_sha256": "0" * 64})
    with pytest.raises(ValueError, match="exactly one"):
        request_image_bytes({**request, "path": "must not be opened"})


def test_client_ram_request_contains_no_path_and_rejects_foreign_response():
    data = b"bytes"
    sha = hashlib.sha256(data).hexdigest()
    client = VLChatClient(Path("unused"), Path("unused"), "unused", Path("unused"))
    sink = io.StringIO()
    client.process = SimpleNamespace(stdin=sink, stdout=True)
    client._next_line = lambda _: json.dumps({"type": "result", "request_id": "1", "source_sha256": sha})
    client.read_bytes(data, sha, 12)
    request = json.loads(sink.getvalue())
    assert "path" not in request
    assert base64.b64decode(request["png_base64"]) == data
    client._next_line = lambda _: json.dumps({"type": "result", "request_id": "2", "source_sha256": "foreign"})
    with pytest.raises(RuntimeError, match="SHA"):
        client.read_bytes(data, sha, 12)


@pytest.mark.parametrize("street", ["FLOP", "TURN", "RIVER"])
def test_ram_board_snapshot_matches_exact_table_pixels_and_survives_buffer_reuse(street):
    table = np.arange(30*50*3, dtype=np.uint8).reshape(30, 50, 3)
    original = table.copy()
    slot = {"global": {"board_flop": [3, 5, 17, 20], "board_turn": [20, 5, 25, 20],
                       "board_river": [28, 5, 33, 20]}}
    snapshot = snapshot_ram_board(table, slot, frame_id=19, table_session_id="s", table_id=1)
    table[:] = 0
    data, record = crop_ram_board(snapshot, street)
    right = {"FLOP": 17, "TURN": 25, "RIVER": 33}[street]
    with Image.open(io.BytesIO(data)) as image:
        np.testing.assert_array_equal(np.asarray(image), original[5:20, 3:right])
    assert record["frame_id"] == 19 and record["table_session_id"] == "s"
    assert record["source_table_pixel_sha256"] == hashlib.sha256(original.tobytes()).hexdigest()
    snapshot["pixels"][0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="changed"):
        crop_ram_board(snapshot, street)


@pytest.mark.parametrize("damage", [None, "foreign", "pixel_sha", "response_sha"])
def test_board_ram_roundtrip_keeps_fail_closed_identity(damage):
    rows = [{"table_id": "T01", "session_id": "s", "epoch": 1, "frame_id": 1,
             "row_index": 1, "raw_ocr_text": "#123", "event_type": "HAND_ID"},
            {"table_id": "T01", "session_id": "s", "epoch": 1, "frame_id": 2,
             "row_index": 1, "raw_ocr_text": "Dealer: Dealing Flop", "event_type": "UNKNOWN_EVENT"}]
    trigger = board_triggers(rows)[0]
    buf = io.BytesIO(); image = Image.new("RGB", (15, 10), (12, 23, 34)); image.save(buf, format="PNG")
    data = buf.getvalue(); sha = hashlib.sha256(data).hexdigest(); pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
    record = {**trigger, "table_session_id": "s", "trigger_row_key": trigger["row_key"],
              "image_transport": "ram", "crop_path": None, "crop_png_base64": base64.b64encode(data).decode(),
              "crop_sha256": sha, "board_sha": pixel_sha, "vl_worker_id": "board",
              "raw_vl_response": {"status": "OK", "request_id": "1", "source_sha256": sha,
                                  "board_sha": pixel_sha, "board_evidence_id": trigger["board_evidence_id"],
                                  "raw_text": "Ah 7c 2s"}}
    if damage == "foreign": record["table_session_id"] = "foreign"
    if damage == "pixel_sha": record["board_sha"] = "0" * 64
    if damage == "response_sha": record["raw_vl_response"]["source_sha256"] = "0" * 64
    counts = associate_board_crops(rows, [record])
    assert counts == ({"accepted": 1} if damage is None else {
        {"foreign": "FOREIGN_BOARD_EVIDENCE", "pixel_sha": "BOARD_SHA_MISMATCH",
         "response_sha": "BOARD_RESPONSE_SHA_MISMATCH"}[damage]: 1})


def test_source_sessions_equal_historical_capture_sessions():
    baseline = Path(r"C:/ParserData/runs/smoke14_first_stage_old_500_20260929/summary.json")
    if not baseline.exists(): pytest.skip("historical baseline unavailable")
    metadata, sessions = source_identity(ROOT / "video/smoke_14.mp4", ROOT)
    old = json.loads(baseline.read_text("utf-8"))
    assert sessions == old["table_sessions"]
    assert metadata["source_video_sha256"] == old["source_sha256"]


def test_first_stage_uses_selected_board_profile(tmp_path):
    profile = json.loads((ROOT / "configs/layout_847x404.json").read_text("utf-8"))
    for slot in profile["slots"].values():
        for field in ("board_flop", "board_turn", "board_river"):
            slot["global"][field][0] += 3
            slot["global"][field][2] += 3
    path = tmp_path / "board-profile.json"
    path.write_text(json.dumps(profile), "utf-8")
    stage = FirstStage(ROOT / "video/smoke_14.mp4", {str(i): f"s{i}" for i in range(1,10)},
                       1, root=ROOT, retain_board=True, board_profile=path)
    rows = list(stage.rows())
    assert rows and stage.board_snapshots
    for snapshot in stage.board_snapshots.values():
        slot = profile["slots"][f"slot_{snapshot['table_id']:02d}"]
        assert snapshot["street_boxes"]["FLOP"] == slot["global"]["board_flop"]


def test_ram_board_crops_match_saved_baseline_exactly(tmp_path):
    baseline = Path(r"C:/ParserData/runs/smoke14_first_stage_old_500_20260929")
    if not baseline.exists(): pytest.skip("historical baseline unavailable")
    sessions = json.loads((baseline / "summary.json").read_text("utf-8"))["table_sessions"]
    profile = ROOT / "configs/layout_847x404.json"
    stage = FirstStage(ROOT / "video/smoke_14.mp4", sessions, 1, root=ROOT, retain_board=True)
    assert list(stage.rows()) and stage.board_snapshots
    for snapshot in stage.board_snapshots.values():
        for street in ("FLOP", "TURN", "RIVER"):
            data, ram = crop_ram_board(snapshot, street)
            path = tmp_path / f"{snapshot['table_id']}_{street}.png"
            saved = crop_saved_table(baseline, profile, path, session_id=snapshot["table_session_id"],
                table_id=snapshot["table_id"], frame_id=snapshot["frame_id"], kind="BOARD", street=street)
            assert data == path.read_bytes()
            assert ram["crop_sha256"] == saved["crop_sha256"]
            assert ram["board_sha"] == saved["board_sha"]
            assert ram["source_pts"] == saved["source_pts"]
