from parser_core.visual_cards import (associate_visual_evidence, board_status,
                                      parse_cards, project_board_events,
                                      visual_event)


def test_visual_routing_only_explicit_card_events():
    assert visual_event("Dealer: Dealing Flop") == ("BOARD", "FLOP")
    assert visual_event("Dealer: Dealing Turn") == ("BOARD", "TURN")
    assert visual_event("Dealer: Dealing River") == ("BOARD", "RIVER")
    assert visual_event("Dealer: player123 shows") == ("SHOWDOWN", None)
    assert visual_event("Dealer: player123 folds") is None
    assert visual_event("Dealer: player123 doesn't show cards") is None


def test_card_reading_and_board_continuity_abstain_on_bad_evidence():
    flop = parse_cards("Ah 7c 2s", 3)
    turn = parse_cards("Ah 7c 2s Jd", 4)
    river = parse_cards("Ah 7c 2s Jd 9h", 5)
    assert (flop, turn, river) == (["Ah", "7c", "2s"],
                                   ["Ah", "7c", "2s", "Jd"],
                                   ["Ah", "7c", "2s", "Jd", "9h"])
    assert board_status("FLOP", flop, None) == "RAW_UNREVIEWED"
    assert board_status("TURN", turn, flop) == "RAW_UNREVIEWED"
    assert board_status("RIVER", river, turn) == "RAW_UNREVIEWED"
    assert board_status("TURN", ["Ks", "7c", "2s", "Jd"], flop) == "CONFLICT_BOARD_CONTINUITY"
    assert board_status("TURN", turn, None) == "RAW_UNREVIEWED"
    assert parse_cards("maybe Ah 7c 2s", 3) is None
    assert parse_cards("Ah 7c Ah", 3) is None
    assert parse_cards("Ks Qs", 2) == ["Ks", "Qs"]


def test_board_text_formats_and_fail_closed_values():
    assert parse_cards("J♠ 9♠ 6♣", 3) == ["Js", "9s", "6c"]
    assert parse_cards("J ♠ 9 ♠ 6 ♣", 3) == ["Js", "9s", "6c"]
    assert parse_cards("Js 9s 6c", 3) == ["Js", "9s", "6c"]
    assert parse_cards("10♥ Q♦ A♣", 3) == ["Th", "Qd", "Ac"]
    assert parse_cards("Board: J♠, 9♠, 6♣", 3) == ["Js", "9s", "6c"]
    assert parse_cards("flop = J♠, 9♠, 6♣", 3) == ["Js", "9s", "6c"]
    assert parse_cards("turn = J♠, 9♠, 6♣", 3, "FLOP") is None
    assert parse_cards("jS 9S 6C", 3) == ["Js", "9s", "6c"]
    assert parse_cards("J♠ 9♠ 6♣ 4♥", 3) is None
    assert parse_cards("J♠ 9♠ 6♣ 4♥", 4) == ["Js", "9s", "6c", "4h"]
    assert parse_cards("J♠ 9♠ 6♣ 4♥ 2♦", 5) == ["Js", "9s", "6c", "4h", "2d"]
    assert parse_cards("J♠ 9♠ J♠", 3) is None
    assert parse_cards("1♠ 9♠ 6♣", 3) is None
    assert parse_cards("Jx 9♠ 6♣", 3) is None
    assert parse_cards("player Js 9s 6c", 3) is None
    assert board_status("TURN", ["Qs", "9s", "6c", "4h"], ["Js", "9s", "6c"]) == "CONFLICT_BOARD_CONTINUITY"
    assert board_status("RIVER", ["Js", "9s", "6c", "4h", "2d"],
                        ["Js", "9s", "6c", "5h"]) == "CONFLICT_BOARD_CONTINUITY"


def test_vl_raw_board_promotes_to_existing_timeline_row():
    rows = [
        {"table_id": "T01", "session_id": "s", "epoch": 1, "frame_id": 1,
         "row_index": 1, "source_roi_change_id": "s:1:chat", "raw_ocr_text": "#123",
         "event_type": "HAND_ID"},
        {"table_id": "T01", "session_id": "s", "epoch": 1, "frame_id": 2,
         "row_index": 1, "source_roi_change_id": "s:2:chat",
         "raw_ocr_text": "Dealer: Dealing Flop", "event_type": "UNKNOWN_EVENT"},
        {"table_id": "T01", "session_id": "s", "epoch": 1, "frame_id": 2,
         "row_index": 2, "source_roi_change_id": "s:2:chat",
         "raw_ocr_text": "tile", "event_type": "UNKNOWN_EVENT"},
    ]
    readings = {"1:2:2": {"table_session_id": "s", "vl_status": "OK",
                            "vl_raw_text": "J♠ 9♠ 6♣", "cards": None,
                            "crop_path": "crop.png", "crop_sha256": "sha",
                            "status": "UNRESOLVED"}}
    associate_visual_evidence(rows, readings)
    assert readings["1:2:2"]["cards"] == ["Js", "9s", "6c"]
    assert readings["1:2:2"]["status"] == "RAW_UNREVIEWED"
    # A chat tile is useful corroboration, but it is not BOARD crop identity.
    assert project_board_events(rows, readings) == 0
    assert len(rows) == 3
    assert rows[2]["event_type"] == "UNKNOWN_EVENT"


def test_card_tiles_follow_stable_timeline_not_vl_completion_order():
    rows = []
    for frame, index, text, kind in [(1, 1, "#123", "HAND_ID"),
                                     (2, 1, "Dealer: Dealing Flop", "UNKNOWN_EVENT"),
                                     (2, 2, "tile", "UNKNOWN_EVENT"),
                                     (3, 1, "Dealer: Dealing Turn", "UNKNOWN_EVENT"),
                                     (3, 2, "tile", "UNKNOWN_EVENT"),
                                     (4, 1, "Dealer: Dealing River", "UNKNOWN_EVENT"),
                                     (4, 2, "tile", "UNKNOWN_EVENT"),
                                     (5, 1, "Dealer: player123 shows", "UNKNOWN_EVENT"),
                                     (5, 2, "tile", "UNKNOWN_EVENT")]:
        rows.append({"table_id": "T01", "session_id": "session", "frame_id": frame,
                     "row_index": index, "raw_ocr_text": text, "event_type": kind})
    readings = {"1:5:2": {"cards": ["Ks", "Qs"], "status": "RAW_UNREVIEWED"},
                "1:4:2": {"cards": ["Ah", "7c", "2s", "Jd", "9h"], "status": "RAW_UNREVIEWED"},
                "1:3:2": {"cards": ["Ah", "7c", "2s", "Jd"], "status": "RAW_UNREVIEWED"},
                "1:2:2": {"cards": ["Ah", "7c", "2s"], "status": "RAW_UNREVIEWED"}}
    associate_visual_evidence(rows, readings)
    assert [readings[f"1:{frame}:2"]["status"] for frame in (2, 3, 4, 5)] == [
        "RAW_UNREVIEWED"] * 4
    assert [readings[f"1:{frame}:2"]["type"] for frame in (2, 3, 4, 5)] == [
        "BOARD", "BOARD", "BOARD", "SHOWDOWN"]
    assert readings["1:5:2"]["player"] == "player123"
    assert all(item["hand_id"] == "123" for item in readings.values())


def test_new_session_without_hand_id_cannot_inherit_old_hand():
    rows = [
        {"table_id": "T01", "session_id": "old", "epoch": 0, "frame_id": 1,
         "row_index": 1, "raw_ocr_text": "#123", "event_type": "HAND_ID"},
        {"table_id": "T01", "session_id": "new", "epoch": 0, "frame_id": 2,
         "row_index": 1, "raw_ocr_text": "Dealer: Dealing Flop", "event_type": "UNKNOWN_EVENT"},
        {"table_id": "T01", "session_id": "new", "epoch": 0, "frame_id": 2,
         "row_index": 2, "raw_ocr_text": "tile", "event_type": "UNKNOWN_EVENT"},
    ]
    readings = {"1:2:2": {"cards": ["Ah", "7c", "2s"], "status": "RAW_UNREVIEWED"}}
    associate_visual_evidence(rows, readings)
    assert readings["1:2:2"]["hand_id"] is None
    assert readings["1:2:2"]["status"] == "UNRESOLVED"
    assert readings["1:2:2"]["reason"] == "HAND_ID_NOT_VISIBLE"


def test_wrong_card_count_does_not_consume_following_chat_tile():
    rows = [
        {"table_id": "T01", "session_id": "session", "frame_id": 1,
         "row_index": 1, "raw_ocr_text": "#123", "event_type": "HAND_ID"},
        {"table_id": "T01", "session_id": "session", "frame_id": 2,
         "row_index": 1, "raw_ocr_text": "Dealer: Dealing Flop", "event_type": "UNKNOWN_EVENT"},
        {"table_id": "T01", "session_id": "session", "frame_id": 2,
         "row_index": 2, "raw_ocr_text": "tile", "event_type": "UNKNOWN_EVENT"},
        {"table_id": "T01", "session_id": "session", "frame_id": 2,
         "row_index": 3, "raw_ocr_text": "tile", "event_type": "UNKNOWN_EVENT"},
    ]
    readings = {"1:2:2": {"cards": ["Kh", "7c"], "status": "RAW_UNREVIEWED"},
                "1:2:3": {"cards": ["Ah", "7c", "2s"], "status": "RAW_UNREVIEWED"}}
    associate_visual_evidence(rows, readings)
    assert readings["1:2:2"]["status"] == "UNRESOLVED"
    assert readings["1:2:3"]["status"] == "RAW_UNREVIEWED"
    assert readings["1:2:3"]["type"] == "BOARD"


def test_card_tile_from_later_frame_does_not_attach_to_board():
    rows = [
        {"table_id": "T01", "session_id": "session", "epoch": 0, "frame_id": 1,
         "row_index": 1, "raw_ocr_text": "#123", "event_type": "HAND_ID"},
        {"table_id": "T01", "session_id": "session", "epoch": 0, "frame_id": 2,
         "row_index": 1, "raw_ocr_text": "Dealer: Dealing Flop", "event_type": "UNKNOWN_EVENT"},
        {"table_id": "T01", "session_id": "session", "epoch": 0, "frame_id": 3,
         "row_index": 2, "raw_ocr_text": "tile", "event_type": "UNKNOWN_EVENT"},
    ]
    readings = {"1:3:2": {"cards": ["Ah", "7c", "2s"], "status": "RAW_UNREVIEWED"}}
    associate_visual_evidence(rows, readings)
    assert readings["1:3:2"]["status"] == "UNRESOLVED"
    assert readings["1:3:2"]["reason"] == "NO_NEARBY_CARD_EVENT"
    assert "MISSING:1:2:1" not in readings


def test_suit_conflict_keeps_trigger_identity_without_missing_duplicate():
    rows = [
        {"table_id": "T01", "session_id": "session", "epoch": 0, "frame_id": 1,
         "row_index": 1, "raw_ocr_text": "#123", "event_type": "HAND_ID"},
        {"table_id": "T01", "session_id": "session", "epoch": 0, "frame_id": 2,
         "row_index": 1, "raw_ocr_text": "Dealer: player123 shows", "event_type": "UNKNOWN_EVENT"},
        {"table_id": "T01", "session_id": "session", "epoch": 0, "frame_id": 2,
         "row_index": 2, "raw_ocr_text": "tile", "event_type": "UNKNOWN_EVENT"},
    ]
    readings = {"1:2:2": {"cards": None, "status": "UNRESOLVED", "reason": "SUIT_CONFLICT",
                          "crop_sha256": "sha", "raw_vl_response": {"raw_text": "Qd 7c"}}}
    associate_visual_evidence(rows, readings)
    item = readings["1:2:2"]
    assert (item["hand_id"], item["type"], item["player"], item["trigger_row_key"]) == (
        "123", "SHOWDOWN", "player123", "1:2:1")
    assert item["status"] == "UNRESOLVED"
    assert item["reason"] == "SUIT_CONFLICT"
    assert "MISSING:1:2:1" not in readings
