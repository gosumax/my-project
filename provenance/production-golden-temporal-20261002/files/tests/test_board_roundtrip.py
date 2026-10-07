import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from PIL import Image

from parser_core.board_roundtrip import associate_board_crops, board_triggers
from parser_core.visual_cards import project_board_events
from parser_core.visual_crops import exact_pixel_sha256


def timeline(streets=("FLOP",), *, hand=True):
    rows = []
    if hand:
        rows.append({"table_id": "T01", "session_id": "s", "epoch": 1,
                     "frame_id": 1, "row_index": 1, "raw_ocr_text": "#123",
                     "event_type": "HAND_ID"})
    for frame, street in enumerate(streets, 2):
        rows.append({"table_id": "T01", "session_id": "s", "epoch": 1,
                     "frame_id": frame, "row_index": 1,
                     "raw_ocr_text": f"Dealer: Dealing {street.title()}",
                     "event_type": "UNKNOWN_EVENT"})
    return rows


def evidence(tmp_path, trigger, cards, *, color=(10, 20, 30), request_id="1"):
    path = tmp_path / f"{trigger['frame_id']}_{trigger['street']}.png"
    Image.new("RGB", (18, 9), color).save(path)
    data = path.read_bytes()
    file_sha = hashlib.sha256(data).hexdigest()
    with Image.open(path) as image:
        board_sha = exact_pixel_sha256(image)
    return {**trigger, "crop_type": "BOARD", "table_session_id": trigger["session_id"],
            "trigger_row_key": trigger["row_key"], "crop_path": str(path),
            "crop_sha256": file_sha, "board_sha": board_sha, "vl_worker_id": "board",
            "raw_vl_response": {"status": "OK", "request_id": request_id,
                                "source_sha256": file_sha, "board_sha": board_sha,
                                "board_evidence_id": trigger["board_evidence_id"],
                                "raw_text": cards}}


def test_exact_board_sha_roundtrip_projects_to_trigger_without_neighbor(tmp_path):
    rows = timeline()
    record = evidence(tmp_path, board_triggers(rows)[0], "Ah 7c 2s")
    assert associate_board_crops(rows, [record]) == {"accepted": 1}
    assert record["identity_status"] == "SHA_ROUNDTRIP_VERIFIED"
    assert project_board_events(rows, {"BOARD:" + record["board_evidence_id"]: record}) == 1
    assert rows[-1]["event_type"] == "BOARD"
    assert rows[-1]["cards"] == ["Ah", "7c", "2s"]
    assert rows[-1]["board_sha"] == record["board_sha"]
    assert rows[-1]["board_evidence_id"] == record["board_evidence_id"]


def test_same_sha_on_different_frames_keeps_distinct_event_ids(tmp_path):
    rows = timeline(("FLOP", "TURN"))
    triggers = board_triggers(rows)
    first = evidence(tmp_path, triggers[0], "Ah 7c 2s", request_id="1")
    second = evidence(tmp_path, triggers[1], "Ah 7c 2s Jd", request_id="2")
    assert first["board_sha"] == second["board_sha"]
    assert first["board_evidence_id"] != second["board_evidence_id"]
    assert associate_board_crops(rows, [first, second]) == {"accepted": 2}
    assert project_board_events(rows, {"a": first, "b": second}) == 2
    assert [row["event_type"] for row in rows[-2:]] == ["BOARD", "BOARD"]


@pytest.mark.parametrize("damage,reason", [
    ("foreign", "FOREIGN_BOARD_EVIDENCE"),
    ("pixel_sha", "BOARD_SHA_MISMATCH"),
    ("response_sha", "BOARD_RESPONSE_SHA_MISMATCH"),
])
def test_foreign_or_wrong_sha_never_attaches(tmp_path, damage, reason):
    rows = timeline()
    record = evidence(tmp_path, board_triggers(rows)[0], "Ah 7c 2s")
    if damage == "foreign":
        record["frame_id"] = 99
    elif damage == "pixel_sha":
        record["board_sha"] = "0" * 64
    else:
        record["raw_vl_response"]["board_sha"] = "0" * 64
    assert associate_board_crops(rows, [record]) == {reason: 1}
    assert project_board_events(rows, {"x": record}) == 0


def test_duplicate_replayed_response_is_rejected(tmp_path):
    rows = timeline(("FLOP", "TURN"))
    triggers = board_triggers(rows)
    records = [evidence(tmp_path, triggers[0], "Ah 7c 2s"),
               evidence(tmp_path, triggers[1], "Ah 7c 2s Jd")]
    assert associate_board_crops(rows, records) == {
        "accepted": 1, "DUPLICATE_BOARD_VL_RESPONSE": 1}


def test_duplicate_evidence_id_rejects_both_records(tmp_path):
    rows = timeline()
    trigger = board_triggers(rows)[0]
    first = evidence(tmp_path, trigger, "Ah 7c 2s", request_id="1")
    second = evidence(tmp_path, trigger, "Kh 7c 2s", request_id="2")
    assert associate_board_crops(rows, [first, second]) == {"DUPLICATE_BOARD_EVIDENCE_ID": 2}
    assert project_board_events(rows, {"a": first, "b": second}) == 0


def test_legacy_response_without_echoed_sha_is_not_proven(tmp_path):
    rows = timeline()
    record = evidence(tmp_path, board_triggers(rows)[0], "Ah 7c 2s")
    record["raw_vl_response"].pop("board_sha")
    record["raw_vl_response"].pop("source_sha256")
    record["raw_vl_response"].pop("board_evidence_id")
    assert associate_board_crops(rows, [record]) == {"BOARD_RESPONSE_SHA_MISMATCH": 1}
    assert project_board_events(rows, {"x": record}) == 0


def test_same_pixel_response_swapped_between_frames_is_rejected(tmp_path):
    rows = timeline(("FLOP", "TURN"))
    triggers = board_triggers(rows)
    first = evidence(tmp_path, triggers[0], "Ah 7c 2s", request_id="1")
    second = evidence(tmp_path, triggers[1], "Ah 7c 2s Jd", request_id="2")
    first["raw_vl_response"], second["raw_vl_response"] = (
        second["raw_vl_response"], first["raw_vl_response"])
    assert first["board_sha"] == second["board_sha"]
    assert associate_board_crops(rows, [first, second]) == {"BOARD_RESPONSE_SHA_MISMATCH": 2}


def test_no_hand_id_and_unreadable_cards_remain_unresolved(tmp_path):
    rows = timeline(hand=False)
    record = evidence(tmp_path, board_triggers(rows)[0], "Ah 7c 2s")
    assert associate_board_crops(rows, [record]) == {"HAND_ID_NOT_VISIBLE": 1}
    assert project_board_events(rows, {"x": record}) == 0
    rows = timeline()
    bad = evidence(tmp_path, board_triggers(rows)[0], "A 7 2")
    assert associate_board_crops(rows, [bad]) == {"BOARD_CARDS_UNREADABLE": 1}


def test_new_hand_boundary_clears_old_hand_id(tmp_path):
    rows = timeline()
    rows.insert(1, {"table_id": "T01", "session_id": "s", "epoch": 1,
                    "frame_id": 1, "row_index": 2,
                    "raw_ocr_text": "Dealer: начинается новая рука",
                    "event_type": "NEW_HAND"})
    trigger = board_triggers(rows)[0]
    assert trigger["hand_id"] is None
    record = evidence(tmp_path, trigger, "Ah 7c 2s")
    assert associate_board_crops(rows, [record]) == {"HAND_ID_NOT_VISIBLE": 1}


def test_street_continuity_and_conflict(tmp_path):
    rows = timeline(("FLOP", "TURN", "RIVER"))
    triggers = board_triggers(rows)
    records = [evidence(tmp_path, trigger, cards, request_id=str(index))
               for index, (trigger, cards) in enumerate(zip(triggers,
                   ("Ah 7c 2s", "Ah 7c 2s Jd", "Ah 7c 2s Jd 9h")), 1)]
    assert associate_board_crops(rows, list(reversed(records))) == {"accepted": 3}
    assert project_board_events(rows, {str(i): r for i, r in enumerate(records)}) == 3
    rows = timeline(("FLOP", "TURN"))
    triggers = board_triggers(rows)
    wrong = [evidence(tmp_path, triggers[0], "Ah 7c 2s", request_id="1"),
             evidence(tmp_path, triggers[1], "Kh 7c 2s Jd", request_id="2")]
    assert associate_board_crops(rows, wrong) == {
        "accepted": 1, "CONFLICT_BOARD_CONTINUITY": 1}
    assert project_board_events(rows, {"a": wrong[0], "b": wrong[1]}) == 1


def test_accepted_board_is_present_in_excel_cells(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node runtime is unavailable")
    rows = timeline()
    record = evidence(tmp_path, board_triggers(rows)[0], "Ah 7c 2s")
    associate_board_crops(rows, [record])
    assert project_board_events(rows, {"board": record}) == 1
    source = tmp_path / "timeline.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")
    output = tmp_path / "timeline.xlsx"
    script = Path(__file__).resolve().parents[1] / "scripts/build_smoke14_ram_workbook.mjs"
    subprocess.run([node, str(script), str(source), str(output)], check=True,
                   timeout=90, capture_output=True, text=True)
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(output) as book:
        shared = ET.fromstring(book.read("xl/sharedStrings.xml"))
        strings = ["".join(node.itertext()) for node in shared.findall("x:si", ns)]
        for sheet in ("xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"):
            root = ET.fromstring(book.read(sheet))
            values = {}
            for cell in root.findall(".//x:sheetData/x:row/x:c", ns):
                value = cell.find("x:v", ns)
                if value is not None:
                    values[cell.attrib["r"]] = (strings[int(value.text)]
                        if cell.attrib.get("t") == "s" else value.text)
            assert values["I3"] == "BOARD"
            assert values["S3"] == "Ah 7c 2s"
            assert values["AA3"] == record["board_evidence_id"]
            assert values["AB3"] == record["board_sha"]
