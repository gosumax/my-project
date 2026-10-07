from parser_core.stack_temporal import hand_windows, parse_actions, prove_hand


def row(frame, index, text, event="UNKNOWN_EVENT"):
    return {"table_id": "T01", "frame_id": frame, "row_index": index,
            "raw_ocr_text": text, "event_type": event,
            "source_roi_change_id": f"session:{frame}:chat"}


def crop(frame, roi, text, score=.99, seat="seat1", kind=None):
    return {"table_id": "T01", "table_context": "session", "seat": seat,
            "roi_type": roi, "frame_index": frame, "source_pts": frame * 100,
            "timestamp": frame / 30, "table_crop_sha256": f"table{frame}",
            "crop_sha256": f"crop{frame}{roi}", "crop_path": f"{frame}_{roi}.png",
            "raw_recognition": text, "confidence": score,
            "visual_kind": kind or ("NICKNAME" if roi == "nick" else text)}


def fixture():
    rows = [row(10, 1, "#123", "HAND_ID"),
            row(11, 2, "Dealer: Vasya posts Ante 100"),
            row(12, 3, "Dealer: Vasya posts Small Blind"),
            row(12, 4, "200"), row(13, 5, "Dealer: Dealing cards"),
            row(30, 6, "Dealer: Vasya folds")]
    window = hand_windows(rows)[0]
    crops = [crop(13, "stack", "SB"),
             crop(20, "nick", "Vasya"),
             crop(20, "stack", "19700", kind="STACK")]
    review = {"stack_frame_pot_visible": 300,
        "forced_post_chat_visually_checked": True,
        "no_other_action_or_bet_chips_visible_through_stack": True,
        "reviewed_rows_override_unresolved_tracking": True,
        "seats": {"seat1": {"overlay_frame": 13,
        "reviewed_player_actions_before_stack": [
            {"event_type": "ANTE", "amount": "100", "source_ref": "session:11:chat:2"},
            {"event_type": "SMALL_BLIND", "amount": "200", "source_ref": "session:12:chat:3"}],
        "overlay_table_sha256": "table13", "stack_frame": 20,
        "stack_table_sha256": "table20"}}}
    return window, crops, review


def test_real_shape_sb_temporal_accounting():
    window, crops, review = fixture()
    proof = prove_hand(window, crops, review)
    player = proof["players"][0]
    assert player["status"] == "CERTIFIED"
    assert player["start_stack"] == "20000"
    assert [x["event_type"] for x in player["chat_events_before_stack"]] == [
        "ANTE", "SMALL_BLIND"]


def test_no_overlay_required_when_first_numeric_stack_and_forced_posts_are_reviewed():
    window, crops, review = fixture()
    crops = [item for item in crops if item["raw_recognition"] != "SB"]
    review["seats"]["seat1"].pop("overlay_frame")
    review["seats"]["seat1"].pop("overlay_table_sha256")
    proof = prove_hand(window, crops, review)
    assert proof["players"][0]["status"] == "CERTIFIED"
    assert proof["players"][0]["start_stack"] == "20000"
    assert proof["players"][0]["forced_bets_already_posted"][0]["event_type"] == "ANTE"


def test_absent_ante_requires_explicit_tournament_policy():
    window, crops, review = fixture()
    window["rows"] = [item for item in window["rows"] if item["row_index"] != 2]
    review["stack_frame_pot_visible"] = 200
    review["seats"]["seat1"]["reviewed_player_actions_before_stack"] = [
        {"event_type": "SMALL_BLIND", "amount": "200", "source_ref": "session:12:chat:3"}]
    assert prove_hand(window, crops, review)["players"][0]["status"] == "UNKNOWN"
    review["ante_policy"] = "ABSENT"
    proof = prove_hand(window, crops, review)
    assert proof["players"][0]["status"] == "CERTIFIED"
    assert proof["players"][0]["start_stack"] == "19900"


def test_normal_player_in_explicit_no_ante_hand_keeps_visible_start_stack():
    window, crops, review = fixture()
    window["rows"] = [row(10, 1, "#123", "HAND_ID"),
                      row(12, 2, "Dealer: Petya posts Small Blind 200"),
                      row(12, 3, "Dealer: Kolya posts Big Blind 400")]
    review["ante_policy"] = "ABSENT"
    review["stack_frame_pot_visible"] = 600
    review["seats"]["seat1"]["reviewed_player_actions_before_stack"] = []
    proof = prove_hand(window, crops, review)
    assert proof["players"][0]["status"] == "CERTIFIED"
    assert proof["players"][0]["start_stack"] == "19700"


def test_later_forced_row_in_broad_hand_window_is_not_added_to_first_stack():
    window, crops, review = fixture()
    window["rows"].insert(5, row(25, 7, "Dealer: Vasya posts Ante 100"))
    proof = prove_hand(window, crops, review)
    assert proof["players"][0]["status"] == "CERTIFIED"
    assert proof["players"][0]["start_stack"] == "20000"


def test_same_frame_forced_post_is_already_in_first_visible_stack():
    window, crops, review = fixture()
    window["rows"][1]["frame_id"] = 20
    proof = prove_hand(window, crops, review)
    assert proof["players"][0]["status"] == "CERTIFIED"
    assert proof["players"][0]["start_stack"] == "20000"


def test_same_frame_voluntary_action_remains_unknown():
    window, crops, review = fixture()
    window["rows"].insert(5, row(20, 7, "Dealer: Vasya calls 100"))
    proof = prove_hand(window, crops, review)
    assert proof["players"][0]["status"] == "UNKNOWN"
    assert proof["players"][0]["reason"] == "VOLUNTARY_ACTION_TIMING_UNKNOWN"


def test_wrong_review_hash_and_unreadable_nick_fail_closed():
    window, crops, review = fixture()
    review["seats"]["seat1"]["stack_table_sha256"] = "wrong"
    assert prove_hand(window, crops, review)["players"][0]["reason"] == "STACK_FRAME_REVIEW_MISMATCH"
    crops[1]["confidence"] = .2
    assert prove_hand(window, crops, review)["players"][0]["reason"] == "NICKNAME_OCR_UNKNOWN"


def test_unknown_debit_before_stack_blocks_certification():
    window, crops, review = fixture()
    window["rows"].insert(5, row(15, 7, "Dealer: Vasya returns 50"))
    assert prove_hand(window, crops, review)["players"][0]["reason"] == "ACTION_ACCOUNTING_UNKNOWN"


def test_pot_or_visual_review_missing_cannot_certify():
    window, crops, review = fixture()
    review["stack_frame_pot_visible"] = 400
    assert prove_hand(window, crops, review)["players"][0]["reason"] == "ACTION_ACCOUNTING_UNKNOWN"
    review["stack_frame_pot_visible"] = 300
    review["reviewed_rows_override_unresolved_tracking"] = False
    assert prove_hand(window, crops, review)["players"][0]["reason"] == "EVIDENCE_REVIEW_MISSING"


def test_parse_raise_to_and_wrapped_blind():
    actions, unresolved = parse_actions([
        row(1, 1, "Dealer: Vasya posts Big Blind"),
        row(1, 2, "500"),
        row(4, 3, "Dealer: Vasya raises to 1 500")])
    assert unresolved == []
    assert [(x["event_type"], x["amount"]) for x in actions] == [
        ("BIG_BLIND", "500"), ("RAISE_TO", "1500")]


def test_same_pot_with_wrong_player_attribution_fails_closed():
    window, crops, review = fixture()
    window["rows"][1]["raw_ocr_text"] = "Dealer: Petya posts Ante 100"
    proof = prove_hand(window, crops, review)
    assert proof["players"][0]["status"] == "UNKNOWN"
    assert proof["players"][0]["reason"] == "ACTION_ACCOUNTING_UNKNOWN"


def test_supplement_requires_exact_frame_and_table_hash_pair():
    window, crops, review = fixture()
    review["visual_chat_supplement"] = [{
        "frame_index": 13, "table_sha256": "table20", "player": "Petya",
        "event_type": "ANTE", "amount": "100",
        "raw_visible_text": "Dealer: Petya posts Ante 100"}]
    import pytest
    with pytest.raises(ValueError, match="Visual chat supplement"):
        prove_hand(window, crops, review)
