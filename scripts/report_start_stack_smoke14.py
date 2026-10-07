"""Conservative START STACK coverage from the Smoke14 ordered raw timeline.

HAND_ID rows identify candidates, not certified complete hands. A player count
cannot be inferred from eight configured seat ROIs without seat occupancy and
seat-to-player evidence. This report therefore never invents player stacks.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def build_report(timeline: Path, stack_summary: Path | None = None) -> dict:
    by_table = defaultdict(list)
    rows = 0
    with timeline.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            rows += 1
            if item.get("event_type") == "HAND_ID":
                text = (item.get("raw_ocr_text") or "").strip()
                if text.startswith("#") and text[1:].isdigit():
                    by_table[item["table_id"]].append({"hand_number": text[1:],
                        "first_observed_frame": item["frame_id"],
                        "first_observed_timestamp": item.get("video_time"),
                        "source_roi_change_id": item.get("source_roi_change_id")})
    hands = []
    for table_id, candidates in sorted(by_table.items()):
        seen = set()
        for index, candidate in enumerate(candidates):
            key = candidate["hand_number"]
            if key in seen:
                continue
            seen.add(key)
            partial = index == 0 and candidate["first_observed_frame"] == 0
            reason = "PARTIAL_START" if partial else "PLAYER_IDENTITY_UNKNOWN"
            hands.append({"table_id": table_id, **candidate,
                "hand_start_status": "PARTIAL_START" if partial else "UNVERIFIED_BOUNDARY",
                "start_stack_status": "PARTIAL" if partial else "UNKNOWN",
                "reason": reason, "certified_players": 0})
    counts = Counter(item["start_stack_status"] for item in hands)
    reasons = Counter(item["reason"] for item in hands)
    stack = json.loads(stack_summary.read_text("utf-8")) if stack_summary else None
    return {"schema": "pokerdom.start-stack-coverage-diagnostic.v1",
        "source_timeline": str(timeline.resolve()), "source_stack_observer": str(stack_summary.resolve()) if stack_summary else None,
        "timeline_rows": rows, "tables_with_hand_ids": len(by_table),
        "unique_hand_ids": len({item["hand_number"] for item in hands}),
        "hand_candidates": len(hands), "eligible_complete_hands": None,
        "hand_start_stack_certified": counts["CERTIFIED"],
        "hand_start_stack_partial": counts["PARTIAL"],
        "hand_start_stack_unknown": counts["UNKNOWN"],
        "player_count": None, "player_start_stack_certified": 0,
        "player_start_stack_partial": None, "player_start_stack_unknown": None,
        "unknown_reasons": dict(reasons),
        "global_blockers": ["PLAYER_IDENTITY_UNKNOWN", "ROLE_UNKNOWN",
            "CONTRIBUTION_AMBIGUOUS", "HAND_BOUNDARIES_UNVERIFIED"],
        "not_measured_on_full_video": ["STACK_NEVER_VISIBLE", "STACK_OCR_FAILED",
            "TABLE_RESET", "SEAT_OCCUPANCY", "PLAYER_COUNT"],
        "observer_metrics": stack.get("metrics") if stack else None,
        "observer_status": stack.get("status") if stack else "NOT_RUN",
        "hands": hands, "verdict": "START_STACK_PIPELINE_FAIL",
        "reason": "NO_PROVEN_SEAT_PLAYER_BINDING_OR_FRAME_ALIGNED_CHAT_ACCOUNTING"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--stack-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.timeline, args.stack_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()
