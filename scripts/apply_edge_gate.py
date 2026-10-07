"""Offline edge gate for a saved production workbook; no OCR/video/model runs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.edge_gate import classify_edges, rows_for_downstream, VERSION
from parser_core.edge_workbook import read_workbook, append_authored_edge_column
from parser_core.poker_hand_assembly import assemble_messages


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hand_number_audit(rows):
    """Diagnostic only: number timing never enables a boundary rule."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["table_id"], row["session_id"])].append(row)
    observations = []
    context_late = []
    for (table, session), source in groups.items():
        messages = assemble_messages(source)
        prior_start = None
        seen = Counter()
        for index, message in enumerate(messages):
            if message["action"] == "NEW_HAND":
                prior_start = index
            number = message["hand_number"]
            if not number:
                continue
            seen[number] += 1
            next_forced = next((m for m in messages[index+1:]
                                if m["action"] in {"NEW_HAND", "SMALL_BLIND", "BIG_BLIND", "ANTE"}), None)
            since_start = messages[prior_start+1:index] if prior_start is not None else []
            observations.append({"table_id": table, "session_id": session,
                "number_candidate": number, "frame": message["frame_id"],
                "row_ids": message["source_row_ids"], "raw_text": message["raw_text"],
                "marker_action": message["action"], "repeat_occurrence": seen[number],
                "prior_explicit_start_frame": messages[prior_start]["frame_id"] if prior_start is not None else None,
                "actions_since_prior_start": [m["action"] for m in since_start],
                "next_forced_or_start": {"action": next_forced["action"], "frame": next_forced["frame_id"]} if next_forced else None,
                "observed_after_hand_activity": any(m["action"] in {"ANTE", "SMALL_BLIND", "BIG_BLIND", "DEAL", "BET", "CALL", "RAISE", "FLOP", "TURN", "RIVER"} for m in since_start)})
        for number in {r.get("hand_id") for r in source if r.get("hand_id")}:
            first = next(r for r in source if r.get("hand_id") == number)
            marker = next((r for r in source if re.fullmatch(r"\s*#\s*"+re.escape(str(number))+r"\s*", r.get("raw_ocr_text", ""))), None)
            context_late.append({"table_id": table, "session_id": session, "number_candidate": number,
                                 "first_hand_id_frame": first["frame_id"],
                                 "first_hand_id_event": first["event_type"],
                                 "raw_number_frame": marker["frame_id"] if marker else None})
    return {"number_alone_is_boundary": False,
            "reason": "No hand_number column; raw OCR markers may be initial/duplicate/late; hand_id is card context, not onset. OCR correctness lacks independent ground truth.",
            "number_marker_observations": len(observations),
            "repeat_observations": sum(o["repeat_occurrence"] > 1 for o in observations),
            "after_activity_observations": sum(o["observed_after_hand_activity"] for o in observations),
            "ocr_error_rate": "NOT_MEASURED_NO_INDEPENDENT_GT",
            "observations": observations, "context_first_appearances": context_late}


def prepare(workbook, timeline):
    sheets = read_workbook(workbook)
    source = [json.loads(line) for line in timeline.read_text("utf-8").splitlines() if line.strip()]
    excel = sheets["ALL_EVENTS"]
    if len(excel["rows"]) != len(source):
        raise ValueError("Workbook / JSONL row count mismatch")
    for number, (row, original) in enumerate(zip(excel["rows"], source), 2):
        for key in excel["headers"]:
            expected = " ".join(original.get(key) or []) if key == "cards" else original.get(key)
            if expected == "":
                expected = None  # Source builder exports empty strings as empty cells.
            if row[key] != expected:
                raise ValueError(f"Workbook/JSONL mismatch at ALL_EVENTS row {number}: {key}")
    result = classify_edges(source)
    downstream = rows_for_downstream(result.rows)
    by_sequence = {row["sequence_id"]: row for row in result.rows}
    if len(by_sequence) != len(source):
        raise ValueError("Duplicate sequence_id")
    specs = {}
    for name, sheet in sheets.items():
        if "sequence_id" not in sheet["headers"]:
            continue  # CARD_AUDIT retains its independent evidence schema.
        if "edge_status" in sheet["headers"]:
            raise ValueError("Input already annotated; use original evidence")
        values = []
        for row in sheet["rows"]:
            canonical = by_sequence[row["sequence_id"]]
            if any(row[key] != excel["rows"][canonical["sequence_id"]-1][key] for key in sheet["headers"]):
                raise ValueError(f"Table sheet differs from ALL_EVENTS: {name}")
            values.append(canonical["edge_status"])
        specs[name] = {"values": values}
    # Exact original JSON data equality, including raw IDs/cards/amounts and nested evidence.
    if any({k: v for k, v in new.items() if k != "edge_status"} != old
           for old, new in zip(source, result.rows)):
        raise ValueError("Source row mutated")
    report = {"version": VERSION, "source_workbook": str(workbook.resolve()),
              "source_workbook_sha256": digest(workbook), "source_timeline_sha256": digest(timeline),
              "total_rows_before": len(source), "total_rows_after": len(result.rows),
              "source_values_and_order_unchanged": True,
              "keep_rows": len(downstream), "excluded_rows": len(source)-len(downstream),
              "sessions": result.sessions, "hand_number_audit": hand_number_audit(source),
              "production_code_changed": False, "production_downstream_exists": False,
              "implementation_sha256": {str(path.relative_to(ROOT)): digest(path) for path in [
                  ROOT/"parser_core/edge_gate.py", ROOT/"parser_core/edge_workbook.py",
                  ROOT/"parser_core/poker_hand_assembly.py", ROOT/"parser_core/poker_card_engine.py",
                  ROOT/"parser_core/normalize_chat.py", ROOT/"scripts/apply_edge_gate.py",
                  ROOT/"scripts/author_edge_column.mjs"]},
              "verdict": "EDGE_BOUNDARY_GATE_NOT_PROVEN",
              "limitations": ["Offline gate uses the existing experimental message parser, not production event promotion.",
                              "No independent visual/OCR ground truth for boundary certification.",
                              "KEEP proves observed edges only; internal gaps and poker/math validity remain downstream checks."]}
    return result, downstream, specs, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--node-modules", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be fresh; source evidence must not be overwritten")
    began = time.perf_counter()
    result, downstream, specs, report = prepare(args.workbook, args.timeline)
    args.output.mkdir(parents=True)
    write_json(args.output/"edge_column_spec.json", specs)
    for filename, records in [("ordered_timeline.edge.jsonl", result.rows), ("downstream_keep.jsonl", downstream)]:
        with (args.output/filename).open("x", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
    # Keep dependency junction and builder confined to this fresh output.
    builder = args.output/"author_edge_column.mjs"
    shutil.copyfile(ROOT/"scripts/author_edge_column.mjs", builder)
    junction = (args.output/"node_modules").resolve()
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(args.node_modules.resolve())], check=True, capture_output=True)
    else:
        junction.symlink_to(args.node_modules.resolve(), target_is_directory=True)
    donor = args.output/"edge_column.donor.xlsx"
    subprocess.run([str(args.node), str(builder), str(args.output/"edge_column_spec.json"), str(donor)], check=True)
    report["workbook_parity"] = append_authored_edge_column(args.workbook, donor, args.output/"ordered_timeline.xlsx", specs)
    report["input_workbook_unchanged"] = digest(args.workbook) == report["source_workbook_sha256"]
    report["input_timeline_unchanged"] = digest(args.timeline) == report["source_timeline_sha256"]
    if not report["input_workbook_unchanged"] or not report["input_timeline_unchanged"]:
        raise ValueError("Input evidence changed during classification")
    report["wall_seconds"] = time.perf_counter()-began
    if any(digest(ROOT/path) != sha for path, sha in report["implementation_sha256"].items()):
        raise ValueError("Gate implementation changed during classification")
    write_json(args.output/"edge_report.json", report)
    print(json.dumps({k: report[k] for k in ["total_rows_before", "total_rows_after", "keep_rows", "excluded_rows", "wall_seconds", "verdict"]}))


if __name__ == "__main__":
    main()
