from collections import Counter
from pathlib import Path
import json
import pytest
from parser_core.poker_reconstruction.timeline import read_timeline, parse_rows
from parser_core.poker_reconstruction.reconstruct import reconstruct_rows

INPUT = Path(__file__).resolve().parents[1] / "runs/production_rtf1_smoke17_full_candidate_20261002/ordered_timeline.xlsx"
EXPECTED = [("T05", "2876720900", 2933000, "murmosenok"),
            ("T05", "2876724645", 2217600, "kurastar"),
            ("T09", "2876720864", 1328344, "bagzik"),
            ("T09", "2876726210", 1030200, "brodyga856"),
            ("T09", "2876728620", 4538374, "kisunya777")]
SPEC = json.loads((Path(__file__).parent / "fixtures/poker_reconstruction_expected.json").read_text(encoding="utf-8"))
NUMBERED = sorted({e.source["parsed_hand_id"] for e in parse_rows(read_timeline(INPUT)) if e.kind == "START_HAND"}) if INPUT.exists() else []


@pytest.fixture(scope="module")
def dataset():
    if not INPUT.exists():
        pytest.skip("Frozen user workbook unavailable")
    rows = read_timeline(INPUT)
    return rows, reconstruct_rows(rows)


def test_workbook_source_coverage_and_boundaries(dataset):
    rows, hands = dataset
    indices = [row["source_index"] for h in hands for e in h["raw_events"] for row in e["source"]["rows"]]
    assert Counter(indices) == Counter(row["source_index"] for row in rows)
    assert sum(h["hand_id"] is not None for h in hands) == 189
    assert sum(h["hand_id"] is None for h in hands) == 7


def test_all_numbered_hand_invariants(dataset):
    _, hands = dataset
    for hand in hands:
        if hand["status"] in {"AUTO_RECOVERED", "RECOVERED_BY_LOGIC_MATH"}:
            assert hand["pot_delta"] == 0
            assert hand["payout_eligible"]
            assert hand["solution_count"] == 1
        else:
            assert hand["decision_reason"]
            assert not hand["dropped_events"]


@pytest.mark.parametrize("table,number,pot,winner", EXPECTED)
def test_control_acceptance(dataset, table, number, pot, winner):
    _, hands = dataset
    hand = next(h for h in hands if h["table"] == table and h["hand_id"] == number)
    assert hand["status"] in {"AUTO_RECOVERED", "RECOVERED_BY_LOGIC_MATH"}, hand["decision_reason"]
    assert hand["calculated_pot"] == hand["observed_payout"] == pot
    assert hand["players"][winner]["payout"] == pot


def test_physical_continuation_and_raw_preservation():
    rows = [{"frame_id": 1, "row_index": 0, "raw_ocr_text": "Dealer: a posts Big Blind", "source_index": 1},
            {"frame_id": 1, "row_index": 1, "raw_ocr_text": "80", "source_index": 2}]
    events = parse_rows(rows)
    assert len(events) == 1 and events[0].amount == 80
    assert events[0].source["rows"] == rows
    rows[1]["frame_id"] = 2
    assert len(parse_rows(rows)) == 2


@pytest.mark.parametrize("number", NUMBERED)
def test_each_numbered_acceptance(dataset, number):
    _, hands = dataset
    hand = next(h for h in hands if h["hand_id"] == number)
    if number in SPEC["truncated_hands"]:
        assert hand["status"] == "TRUNCATED_END", hand["decision_reason"]
    else:
        assert hand["status"] in {"AUTO_RECOVERED", "RECOVERED_BY_LOGIC_MATH"}, hand["decision_reason"]
        assert hand["pot_delta"] == 0
