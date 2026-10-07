"""Saved-row timing audit only; findings never enable hand-number-only gating."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.poker_hand_assembly import assemble_messages


def audit(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["table_id"], row["session_id"])].append(row)
    transitions = []
    for (table, session), source in groups.items():
        messages = assemble_messages(source)
        starts = [i for i, m in enumerate(messages) if m["action"] == "NEW_HAND"]
        seen = Counter()
        for n, start in enumerate(starts):
            end = starts[n+1] if n+1 < len(starts) else len(messages)
            if n+1 < len(starts) and end > start and messages[end-1]["action"] == "HAND_HEADER":
                end -= 1
            window = messages[start:end]
            # A header immediately before explicit start is metadata of that
            # transition, not a late number of the preceding hand.
            candidates = [(i+start, m) for i, m in enumerate(window) if m["hand_number"]]
            if start and messages[start-1]["action"] == "HAND_HEADER":
                candidates.insert(0, (start-1, messages[start-1]))
            number = candidates[0][1]["hand_number"] if candidates else None
            if number:
                seen[number] += 1
            forced = {action: next(((i+start, m) for i, m in enumerate(window) if m["action"] == action), None)
                      for action in ["ANTE", "SMALL_BLIND", "BIG_BLIND", "DEAL"]}
            first_forced_index = min((m[0] for m in forced.values() if m), default=None)
            transitions.append({"table_id": table, "session_id": session,
                "explicit_start_frame": messages[start]["frame_id"],
                "explicit_start_row_ids": messages[start]["source_row_ids"],
                "first_number_candidate": number,
                "first_number_frame": candidates[0][1]["frame_id"] if candidates else None,
                "first_number_row_ids": candidates[0][1]["source_row_ids"] if candidates else [],
                "first_number_raw_text": candidates[0][1]["raw_text"] if candidates else None,
                "number_before_first_forced_message": (candidates[0][0] <= first_forced_index
                    if candidates and first_forced_index is not None else None),
                "number_first_observed_after_forced_message": (candidates[0][0] > first_forced_index
                    if candidates and first_forced_index is not None else None),
                "candidate_number_repeated_across_explicit_starts": bool(number and seen[number] > 1),
                "number_candidates_in_window": sorted({m["hand_number"] for _, m in candidates}),
                "first_forced_frames": {action: m[1]["frame_id"] if m else None for action, m in forced.items()}})
    contexts = []
    for (table, session), source in groups.items():
        for number in sorted({r["hand_id"] for r in source if r.get("hand_id")}):
            first = next(r for r in source if r.get("hand_id") == number)
            marker = next((r for r in source if r.get("raw_ocr_text", "").strip() == "#"+str(number)), None)
            contexts.append({"table_id": table, "session_id": session, "number_candidate": number,
                             "first_hand_id_frame": first["frame_id"], "first_hand_id_event": first["event_type"],
                             "raw_number_frame": marker["frame_id"] if marker else None,
                             "context_after_marker": first["frame_id"] > marker["frame_id"] if marker else None})
    return {"explicit_transitions_audited": len(transitions),
            "number_before_forced": sum(t["number_before_first_forced_message"] is True for t in transitions),
            "number_after_forced": sum(t["number_first_observed_after_forced_message"] is True for t in transitions),
            "number_timing_unresolved": sum(t["number_before_first_forced_message"] is None for t in transitions),
            "number_repeat_across_starts": sum(t["candidate_number_repeated_across_explicit_starts"] for t in transitions),
            "context_after_raw_marker": sum(c["context_after_marker"] is True for c in contexts),
            "number_as_standalone_boundary": False, "ocr_number_error_rate": "NOT_MEASURED_NO_INDEPENDENT_GT",
            "transitions": transitions, "context_first_appearances": contexts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Audit output must be fresh")
    data = args.input.read_bytes()
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    result = audit(rows)
    result["source_sha256"] = hashlib.sha256(data).hexdigest()
    result["audit_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"transitions", "context_first_appearances"}}))


if __name__ == "__main__":
    main()
