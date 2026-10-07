"""Offline reconstruction CLI; does not launch or modify the production pipeline."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.poker_reconstruction.reconstruct import reconstruct_rows
from parser_core.poker_reconstruction.timeline import read_timeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-nodes", type=int, default=20000)
    parser.add_argument("--start-stack-oracle", action="store_true")
    parser.add_argument("--start-stacks", help="JSON mapping table:hand_id -> player -> exact integer stack")
    parser.add_argument("--expectations", help="Optional regression-only expected results JSON; never supplied to solver")
    parser.add_argument("--identity-evidence", help="Independent seat/continuity evidence JSON keyed table:hand_id")
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        parser.error("Output must be a fresh directory to preserve previous evidence")
    stacks = json.loads(Path(args.start_stacks).read_text(encoding="utf-8")) if args.start_stacks else None
    if stacks and any(type(value) is not int or value < 0 for hand in stacks.values() for value in hand.values()):
        parser.error("Stacks must be nonnegative exact integers")
    start = time.perf_counter()
    rows = read_timeline(args.input)
    identities = json.loads(Path(args.identity_evidence).read_text(encoding="utf-8")) if args.identity_evidence else None
    hands = reconstruct_rows(rows, max_nodes=args.max_nodes, identity_evidence=identities)
    recovered = {"AUTO_RECOVERED", "RECOVERED_BY_LOGIC_MATH"}
    counts = Counter(h["status"] for h in hands)
    numbered = sum(h["hand_id"] is not None for h in hands)
    summary = {"input": str(Path(args.input).resolve()),
               "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
               "source_rows": len(rows), "segments": len(hands), "numbered_hands": numbered,
               "counts": dict(counts), "fully_recovered": sum(counts[s] for s in recovered),
               "pot_exact": sum(h["status"] in recovered and h["pot_delta"] == 0 for h in hands),
               "pot_mismatch": sum(h["pot_delta"] is not None and h["pot_delta"] != 0 for h in hands),
               "accounting_unresolved": sum(h["pot_delta"] is None for h in hands),
               "production_code_changed": False, "elapsed_seconds": time.perf_counter() - start,
               "verification_scope": "Observed CHAT betting state and exact aggregate payouts; no independent seat-order/card-rank certification",
               "verdict": "NOT_PROVEN"}
    if args.start_stack_oracle:
        if stacks:
            oracle = reconstruct_rows(rows, max_nodes=args.max_nodes, start_stacks=stacks, identity_evidence=identities)
            after = sum(h["status"] in recovered for h in oracle)
            summary["start_stack_oracle"] = {"mode": "ACTUAL_EXACT_STACKS", "base": summary["fully_recovered"],
                                             "with_stacks": after, "additional": after - summary["fully_recovered"]}
        else:
            summary["start_stack_oracle"] = {"mode": "CAPABILITY_ANALYSIS_NO_STACKS_INVENTED",
                                             "base": summary["fully_recovered"], "with_stacks": None,
                                             "additional": None,
                                             "reason": "No exact stack observations supplied; numeric counterfactual is not measured",
                                             "boundary_truncated_cannot_recover": sum(counts[s] for s in ("TRUNCATED_START", "TRUNCATED_END"))}
    expectation_diff = []
    if args.expectations:
        expected = json.loads(Path(args.expectations).read_text(encoding="utf-8"))
        controls = []
        for control in expected["controls"]:
            hand = next((h for h in hands if h["table"] == control["table"] and h["hand_id"] == control["hand_id"]), None)
            passed = bool(hand and hand["status"] in recovered and hand["calculated_pot"] == control["pot"]
                          and hand["players"].get(control["winner"], {}).get("payout") == control["pot"])
            controls.append(dict(control, result="PASS" if passed else "FAIL", actual_status=hand["status"] if hand else "MISSING"))
        summary["regression_controls"] = controls
        for hand in hands:
            expect = "TRUNCATED_START" if hand["hand_id"] is None else "TRUNCATED_END" if hand["hand_id"] in expected["truncated_hands"] else "FULLY_RECOVERED"
            passed = hand["status"] in recovered if expect == "FULLY_RECOVERED" else hand["status"] == expect
            if not passed:
                expectation_diff.append({"table": hand["table"], "hand_id": hand["hand_id"], "expected": expect,
                                         "actual_status": hand["status"], "reason": hand["decision_reason"],
                                         "raw_evidence": hand["raw_events"], "accounting": hand["candidate_ledgers"],
                                         "conflicts": hand["failures"]})
        summary["expected_result_diff_count"] = len(expectation_diff)
        summary["verdict"] = "POKER_RECONSTRUCTION_PROVEN" if (not expectation_diff and
            numbered == expected["numbered_hands"] and summary["fully_recovered"] == expected["fully_recovered"]
            and all(c["result"] == "PASS" for c in controls)) else "NOT_PROVEN"
    out.mkdir(parents=True, exist_ok=True)
    def write_jsonl(name, records):
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    write_jsonl("reconstructed_hands.jsonl", hands)
    write_jsonl("reconstruction_failures.jsonl", [h for h in hands if h["status"] not in recovered])
    if args.expectations:
        write_jsonl("expected_result_diff.jsonl", expectation_diff)
    decisions = []
    for hand in hands:
        keep = {r["source_index"] for e in hand["kept_events"] for r in e["source"].get("rows", [])}
        drops = {r["source_index"]: d["reason"] for d in hand["dropped_events"] for r in d["event"]["source"].get("rows", [])}
        for event in hand["raw_events"]:
            index = event["source"]["source_index"]
            for source_row in event["source"]["rows"]:
                decisions.append({"table": hand["table"], "hand_id": hand["hand_id"], "event": event,
                              "source_row": source_row,
                              "decision": "KEEP" if source_row["source_index"] in keep else "DROP_STALE" if source_row["source_index"] in drops else "UNRESOLVED",
                              "reason": drops.get(source_row["source_index"], hand["decision_reason"])})
    write_jsonl("regression_diagnostic_traces.jsonl", [h for h in hands if args.expectations and
                 any(h["hand_id"] == c["hand_id"] and h["table"] == c["table"] for c in expected["controls"])])
    write_jsonl("reconstruction_decisions.jsonl", decisions)
    (out / "reconstruction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Offline poker reconstruction", "", "Production is unchanged. No certified HH publication.", "",
             "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", "",
             "Input columns are retained in each raw source row. Source order is preserved per table/session.",
             "The solver branches only on repeated/phase-stale evidence and explicit amount ambiguity.",
             "A budget limit is AMBIGUOUS. No missing action or blind is invented to balance a pot.", "",
             "## Unrecovered hands and general remediation", ""]
    for hand in hands:
        if hand["status"] in recovered:
            continue
        rules = Counter(f["rule"] for f in hand["failures"])
        blockers = [e for e in hand["raw_events"] if e["kind"] == "UNKNOWN" or
                    (e["amount"] is None and e["kind"] in {"ANTE", "SB", "BB", "CALL", "BET", "RAISE", "RETURN", "WIN"})]
        lines.extend([f"### {hand['table']} #{hand['hand_id']} — {hand['status']}",
                      hand["decision_reason"], f"Rules encountered: {dict(rules)}.",
                      f"Last viable prefix: {hand['processed_event_count']} of {hand['raw_event_count']} events.",
                      "First engine blocker: " + json.dumps(hand.get("first_engine_blocker"), ensure_ascii=False),
                      "Unresolved source rows: " + json.dumps([{"source_index": e["source"]["source_index"], "frame": e["source"]["frame"], "text": e["source"]["raw_text"]} for e in blockers], ensure_ascii=False),
                      "Candidate prefix accounting (not a complete hand): " + json.dumps(hand["candidate_ledgers"], ensure_ascii=False),
                      "Raw evidence, candidate accounting and precise offending rows are in reconstruction_failures.jsonl.",
                      "General remediation: resolve observed continuations/amount semantics or source identity; "
                      "boundary omissions require additional recorded observations. Pot equality alone cannot supply actions.", ""])
    (out / "RECONSTRUCTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
