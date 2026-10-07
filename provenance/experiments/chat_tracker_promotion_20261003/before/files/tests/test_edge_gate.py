import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from parser_core.edge_gate import classify_edges, rows_for_downstream
from parser_core.edge_workbook import read_workbook, append_authored_edge_column


def rows(*texts, table="T01", session="s", offset=0):
    return [dict(table_id=table, session_id=session, sequence_id=i+1+offset,
                 stable_row_id=f"{table}:{session}:{i+offset}", frame_id=i+offset,
                 row_index=0, epoch=0, raw_ocr_text=text, event_type="ORIGINAL",
                 status="RAW_UNREVIEWED", notes_error="", origin="TRACKED",
                 row_pixel_sha256=hashlib.sha256(text.encode()).hexdigest())
            for i, text in enumerate(texts)]


def status(source):
    before = copy.deepcopy(source)
    result = classify_edges(source)
    assert source == before
    assert len(source) == len(result.rows)
    assert [{k: v for k, v in r.items() if k != "edge_status"} for r in result.rows] == before
    admitted = rows_for_downstream(result.rows)
    assert admitted == [r for r in result.rows if r["edge_status"] == "KEEP"]
    return [r["edge_status"] for r in result.rows]


def test_a_partial_start():
    assert status(rows("A bets 5", "Dealing Flop", "Dealing River", "A wins 5",
                       "Starting a new hand #1", "A bets 5", "Starting a new hand #2")) == (
                           ["PARTIAL_START"]*4+["KEEP"]*2+["PARTIAL_END"])


def test_b_clean_start():
    assert status(rows("Starting a new hand #1", "A wins 10")) == ["KEEP"]*2


def test_c_middle_hands_next_start_needs_no_end():
    assert status(rows("Starting a new hand #1", "A bets 5", "Starting a new hand #2",
                       "A bets 5", "Starting a new hand #3")) == ["KEEP"]*4+["PARTIAL_END"]


def test_d_partial_end():
    assert status(rows("Starting a new hand", "A bets 5", "Dealing Flop", "A bets 10")) == ["PARTIAL_END"]*4


def test_e_fold_end_with_observed_payout():
    assert status(rows("Starting a new hand", "A posts Small Blind 5", "B posts Big Blind 10",
                       "A folds", "B gets uncalled bet 5 back", "B wins 10")) == ["KEEP"]*6


def test_f_showdown_award():
    assert status(rows("Starting a new hand", "A shows Ah Ac", "B shows Kh Kc", "A wins 20")) == ["KEEP"]*4


def test_g_independent_interleaved_tables():
    a = rows("A bets 5", "Starting a new hand", "A wins 10", table="T01")
    b = rows("Starting a new hand", "B bets 5", "Dealing Flop", table="T02")
    assert status([a[0], b[0], a[1], b[1], a[2], b[2]]) == [
        "PARTIAL_START", "PARTIAL_END", "KEEP", "PARTIAL_END", "KEEP", "PARTIAL_END"]


def test_h_session_replacement():
    assert status(rows("Starting a new hand", "A bets 5", session="old")+
                  rows("A wins 5", "Starting a new hand", "A wins 10", session="new")) == [
                      "PARTIAL_END", "PARTIAL_END", "PARTIAL_START", "KEEP", "KEEP"]


def test_i_only_fragment():
    source = rows("A bets 5", "Dealing River", "A wins 5")
    result = classify_edges(source)
    assert status(source) == ["PARTIAL_START"]*3
    assert rows_for_downstream(result.rows) == []


def test_j_one_unfinished_started_hand():
    result = classify_edges(rows("Starting a new hand", "A bets 5"))
    assert [r["edge_status"] for r in result.rows] == ["PARTIAL_END"]*2
    assert rows_for_downstream(result.rows) == []


@pytest.mark.parametrize("texts", [
    ["Hand #1", "#2", "A wins 10"],
    ["A posts Small Blind 5", "A bets 10"],
    ["B posts Big Blind 10", "Dealing cards"],
    ["A posts Small Blind 5", "unknown gap", "B posts Big Blind 10", "Dealing cards"],
    ["A posts Small Blind ???", "B posts Big Blind ???", "Dealing cards"],
])
def test_weak_signals_never_start(texts):
    assert status(rows(*texts)) == ["PARTIAL_START"]*len(texts)


@pytest.mark.parametrize("ante", [False, True])
def test_forced_start_pattern(ante):
    texts = (["C posts Ante 1"] if ante else []) + ["A posts Small Blind 5", "B posts Big Blind 10", "Dealing cards", "A wins 10"]
    assert status(rows(*texts)) == ["KEEP"]*len(texts)


def test_explicit_start_info_and_blinds_is_one_unfinished_hand():
    assert status(rows("Starting a new hand #10", "Alice sits in", "A posts Small Blind 5",
                       "B posts Big Blind 10", "Dealing cards")) == ["PARTIAL_END"]*5


def test_duplicate_number_start_is_unresolved_not_closure():
    assert status(rows("Starting a new hand #10", "A bets 5", "Starting a new hand #10")) == ["UNRESOLVED_EDGE"]*3


def test_late_number_on_repeated_unnumbered_start_is_unresolved():
    assert status(rows("Starting a new hand #10", "A bets 5", "Starting a new hand",
                       "A posts Small Blind 5", "B posts Big Blind 10", "Dealing cards", "#10")) == ["UNRESOLVED_EDGE"]*7


@pytest.mark.parametrize("tail", ["A bets 5", "???", "#20", "Dealing Flop"])
def test_award_cannot_close_unknown_or_active_tail(tail):
    assert status(rows("Starting a new hand", "A wins 10", tail)) != ["KEEP"]*3


def test_verified_hand_end_cannot_close_later_betting():
    source = rows("Starting a new hand", "Hand end", "A bets 5")
    source[1]["status"] = "VERIFIED"
    assert status(source) == ["PARTIAL_END"]*3


def test_show_or_fold_alone_not_completion():
    for tail in ("A folds", "A shows Ah Ac"):
        assert status(rows("Starting a new hand", tail)) == ["PARTIAL_END"]*2


def test_header_wrap_is_kept_with_start():
    assert status(rows("Hand #123", "Dealer: Starting a new hand", "#123", "A wins 10")) == ["KEEP"]*4


def test_identity_uncertainty_and_epoch_gap_fail_closed():
    source = rows("Starting a new hand", "A bets 5", "Starting a new hand #2", "A wins 10")
    source[2]["origin"] = "UNRESOLVED_IDENTITY"
    assert status(source) == ["UNRESOLVED_EDGE"]*4
    source = rows("Starting a new hand", "A bets 5", "A wins 10")
    source[1]["epoch"] = 1
    assert status(source) == ["UNRESOLVED_EDGE"]*3


def test_duplicate_identity_or_bad_order_fails_closed():
    source = rows("Starting a new hand", "A wins 10")
    source[1]["stable_row_id"] = source[0]["stable_row_id"]
    assert status(source) == ["UNRESOLVED_EDGE"]*2
    assert status(list(reversed(rows("Starting a new hand", "A wins 10")))) == ["UNRESOLVED_EDGE"]*2


def test_downstream_requires_classification_and_detaches_rows():
    with pytest.raises(ValueError):
        rows_for_downstream(rows("A bets 10"))
    classified = classify_edges(rows("Starting a new hand", "A wins 10")).rows
    admitted = rows_for_downstream(classified)
    admitted[0]["raw_ocr_text"] = "changed downstream"
    assert classified[0]["raw_ocr_text"] == "Starting a new hand"


def test_workbook_append_preserves_original_cells_and_native_parts(tmp_path):
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    def workbook(path, data):
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("xl/workbook.xml", '<workbook xmlns="'+ns+'" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="ALL_EVENTS" sheetId="1" r:id="r1"/></sheets></workbook>')
            z.writestr("xl/_rels/workbook.xml.rels", '<Relationships><Relationship Id="r1" Target="worksheets/sheet1.xml"/></Relationships>')
            z.writestr("xl/worksheets/sheet1.xml", '<x:worksheet xmlns:x="'+ns+'"><x:sheetData>'+data+'</x:sheetData></x:worksheet>')
            z.writestr("native-evidence.bin", b"untouched bytes\x00\xff")
    source, donor, final = (tmp_path/name for name in ("source.xlsx", "donor.xlsx", "final.xlsx"))
    workbook(source, '<x:row r="1"><x:c r="A1" t="inlineStr"><x:is><x:t>row_id</x:t></x:is></x:c></x:row><x:row r="2"><x:c r="A2" t="inlineStr"><x:is><x:t>literal=source</x:t></x:is></x:c></x:row>')
    workbook(donor, '<x:row r="1"><x:c r="A1" t="inlineStr"><x:is><x:t>edge_status</x:t></x:is></x:c></x:row><x:row r="2"><x:c r="A2" t="inlineStr"><x:is><x:t>KEEP</x:t></x:is></x:c></x:row>')
    before = source.read_bytes()
    parity = append_authored_edge_column(source, donor, final, {"ALL_EVENTS": {"values": ["KEEP"]}})
    assert source.read_bytes() == before
    assert read_workbook(final)["ALL_EVENTS"]["rows"] == [{"row_id": "literal=source", "edge_status": "KEEP"}]
    assert parity["sheets"]["ALL_EVENTS"]["original_cells_unchanged"]
    with pytest.raises(FileExistsError):
        append_authored_edge_column(source, donor, source, {"ALL_EVENTS": {"values": ["KEEP"]}})


def test_replay_filters_before_reconstruction_and_keeps_full_evidence(tmp_path, monkeypatch):
    from experiments import replay_poker_logic_math_jev as replay
    source = rows("A bets 5", "Starting a new hand #1", "A wins 10", "Starting a new hand #2")
    input_path = tmp_path/"source.jsonl"
    input_path.write_text("\n".join(json.dumps(r) for r in source), encoding="utf-8")
    before = input_path.read_bytes()
    output = tmp_path/"replay"
    received = []
    def reconstruction_spy(admitted):
        received.extend(admitted)
        raise RuntimeError("STOP_AFTER_ADMISSION_TEST")
    monkeypatch.setattr(replay, "assemble_hands", reconstruction_spy)
    monkeypatch.setattr(replay, "registered_hashes", lambda: {})
    monkeypatch.setattr(sys, "argv", ["replay", "--input", str(input_path), "--output", str(output)])
    with pytest.raises(RuntimeError, match="STOP_AFTER_ADMISSION_TEST"):
        replay.main()
    assert [r["raw_ocr_text"] for r in received] == ["Starting a new hand #1", "A wins 10"]
    assert all(r["edge_status"] == "KEEP" for r in received)
    assert len((output/"edge_evidence.jsonl").read_text("utf-8").splitlines()) == len(source)
    assert input_path.read_bytes() == before


def test_number_timing_audit_does_not_call_next_header_a_late_read():
    from scripts.audit_edge_hand_numbers import audit
    result = audit(rows("Starting a new hand", "A posts Small Blind 5", "B posts Big Blind 10",
                        "A bets 5", "Hand #2", "Starting a new hand", "A posts Small Blind 5"))
    assert result["transitions"][0]["first_number_candidate"] is None
    assert result["transitions"][1]["first_number_candidate"] == "2"
    assert result["transitions"][1]["number_before_first_forced_message"] is True
    assert result["number_after_forced"] == 0


def test_number_timing_audit_records_actual_late_marker_without_enabling_gate():
    from scripts.audit_edge_hand_numbers import audit
    result = audit(rows("Starting a new hand", "A posts Small Blind 5", "#123", "A bets 5"))
    assert result["number_after_forced"] == 1
    assert result["number_as_standalone_boundary"] is False
