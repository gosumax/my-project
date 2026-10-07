"""Report full Smoke14 stack coverage without promoting unreviewed OCR to proof."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.stack_temporal import FORCED, hand_windows, parse_actions


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build(timeline_path: Path, observer_path: Path, proof_path: Path) -> dict:
    windows = hand_windows(jsonl(timeline_path))
    observations = jsonl(observer_path)
    observer_summary = json.loads((observer_path.parent / "start_stack_observer_summary.json").read_text("utf-8"))
    proof = json.loads(proof_path.read_text("utf-8"))
    certified = {(hand["table_id"], hand["hand_id"], player["seat"]): player
                 for hand in proof["hands"] for player in hand["players"]
                 if player["status"] == "CERTIFIED"}
    by_table = defaultdict(list)
    for observation in observations:
        by_table[observation["table_id"]].append(observation)
    reasons = Counter()
    statuses = Counter()
    seats_with_observation = set()
    seats_with_nick_and_stack = set()
    output_hands = []
    for window in windows:
        table_id, hand_id = window["table_id"], window["hand_number"]
        actions, unparsed = parse_actions(window["rows"])
        within = [item for item in by_table[table_id]
                  if item["frame_index"] >= window["start_frame"] and
                  (window["end_frame"] is None or item["frame_index"] < window["end_frame"])]
        per_seat = defaultdict(list)
        for item in within:
            per_seat[item["seat"]].append(item)
            seats_with_observation.add((table_id, item["seat"]))
        results = []
        for number in range(1, 9):
            seat = f"seat{number}"
            seen = per_seat[seat]
            numeric = [item for item in seen if item["roi_type"] == "stack" and
                       item.get("visual_kind") == "STACK" and item.get("confidence", 0) >= .90]
            first = numeric[0] if numeric else None
            nickname = next((item for item in seen if first is not None and
                item["roi_type"] == "nick" and item["frame_index"] == first["frame_index"] and
                item["table_crop_sha256"] == first["table_crop_sha256"] and
                item.get("visual_kind") == "NICKNAME" and item.get("confidence", 0) >= .90), None)
            paired = nickname is not None
            own_voluntary = [action for action in actions if nickname is not None and
                action["player"] == nickname["raw_recognition"] and
                action["event_type"] not in FORCED and
                action["frame_index"] <= first["frame_index"]]
            own_forced = [action for action in actions if nickname is not None and
                action["player"] == nickname["raw_recognition"] and
                action["event_type"] in FORCED and
                action["frame_index"] <= first["frame_index"]]
            if paired:
                seats_with_nick_and_stack.add((table_id, seat))
            checked = certified.get((table_id, hand_id, seat))
            detail = None
            if checked:
                status, reason = "CERTIFIED", None
            elif window["start_frame"] == 0:
                status, reason = "PARTIAL_START", "HAND_BOUNDARY_UNKNOWN"
            elif not seen:
                status, reason = "UNKNOWN", "STACK_OBSERVATION_MISSING"
            elif not numeric:
                status, reason = "UNKNOWN", "STACK_OCR_UNKNOWN"
            elif not paired:
                status, reason = "UNKNOWN", "NICKNAME_OCR_UNKNOWN"
            elif own_voluntary:
                status, reason = "UNKNOWN", "TEMPORAL_ORDER_UNKNOWN"
                action = own_voluntary[0]
                suffix = "SAME_FRAME" if action["frame_index"] == first["frame_index"] else "BEFORE_FIRST_STACK"
                detail = f"VOLUNTARY_{action['event_type']}_{suffix}"
            elif any(row["frame_id"] <= first["frame_index"] for row in unparsed):
                status, reason = "UNKNOWN", "ACTION_ACCOUNTING_UNKNOWN"
            else:
                if not own_forced:
                    status, reason = "UNKNOWN", "FORCED_BET_EVIDENCE_UNKNOWN"
                elif any(action["event_type"] == "ANTE" for action in actions if
                         action["event_type"] in FORCED and
                         action["frame_index"] <= first["frame_index"]) and not any(
                         action["event_type"] == "ANTE" for action in own_forced):
                    status, reason = "UNKNOWN", "FORCED_BET_EVIDENCE_UNKNOWN"
                else:
                    status, reason = "UNKNOWN", "EVIDENCE_REVIEW_MISSING"
            statuses[status] += 1
            if reason:
                reasons[reason] += 1
            results.append({"seat": seat, "status": status, "reason": reason,
                            "temporal_detail": detail,
                            "nickname": nickname["raw_recognition"] if nickname else None,
                            "first_valid_stack_frame": first["frame_index"] if first else None,
                            "observations": len(seen), "numeric_stack_observations": len(numeric),
                            "same_frame_nick_stack": paired,
                            "start_stack": checked["start_stack"] if checked else None})
        output_hands.append({"table_id": table_id, "hand_id": hand_id,
                             "start_frame": window["start_frame"], "seats": results})
    return {"schema": "pokerdom.start-stack-full-coverage.v1",
            "classification_unit": "seat ROI per observed HAND_ID; this is not a player count",
            "players_count": None,
            "configured_table_seat_rois": len({x["table_id"] for x in output_hands}) * 8,
            "observer_evidence_status": observer_summary["status"],
            "dropped_candidates": observer_summary["metrics"].get("dropped_candidates", 0),
            "unknown_reason_counts_are_complete": observer_summary["status"] == "RAW_EVIDENCE_ONLY",
            "unique_hand_ids": len({x["hand_id"] for x in output_hands}),
            "hand_windows": len(output_hands),
            "tables": len({x["table_id"] for x in output_hands}),
            "stack_observations": sum(x["roi_type"] == "stack" for x in observations),
            "nickname_observations": sum(x["roi_type"] == "nick" for x in observations),
            "observed_table_seat_pairs": len(seats_with_observation),
            "table_seat_pairs_with_same_frame_nick_stack": len(seats_with_nick_and_stack),
            "seat_hand_opportunities": len(output_hands) * 8,
            "statuses": dict(statuses), "unknown_reasons": dict(reasons),
            "temporal_details": dict(Counter(seat["temporal_detail"] for hand in output_hands
                for seat in hand["seats"] if seat["temporal_detail"])),
            "observer_summary": observer_summary, "hands": output_hands}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.timeline, args.observer, args.proof)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({key: value for key, value in report.items()
        if key not in {"hands", "observer_summary"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
