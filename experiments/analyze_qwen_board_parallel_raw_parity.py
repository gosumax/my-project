"""Make raw-message equality independent of parsed-card reference validity.

No API calls or response repair. Preserve the originally recorded summary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.benchmark_qwen_board_parallel_requests import is_success


def raw_equal(actual, reference, field):
    return bool(actual and reference and actual["http_status"] == 200
                and reference["http_status"] == 200 and isinstance(reference.get(field), str)
                and actual.get(field) == reference[field])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    run = args.run
    read = lambda name: json.loads((run / name).read_text(encoding="utf-8"))
    save = lambda name, value: (run / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = read("summary.json")
    if not (run / "summary_recorded.json").exists():
        (run / "summary_recorded.json").write_bytes((run / "summary.json").read_bytes())
    known = read("manifest.json")["known_crop_sha"]
    for entry in summary["levels"]:
        concurrency = entry["concurrency"]
        outcomes = read(f"c{concurrency}_outcomes.json")
        for mode, key, base in (("RAW", "first", "baseline_raw.json"),
                                ("EFFECTIVE", "effective", "baseline_effective.json")):
            refs = read(base)
            records = outcomes[key]
            stats = entry[mode]
            full_matches = sum(raw_equal(r, refs.get(r["crop_sha"]), "raw_response") for r in records if r)
            card_matches = sum(raw_equal(r, refs.get(r["crop_sha"]), "raw_text") for r in records if r)
            stats["raw_parity"]["matches"] = full_matches
            stats["raw_card_text_parity"]["matches"] = card_matches
            stats["accuracy_safe"] = (stats["successful"] == stats["planned"]
                and stats["parsed_parity"]["matches"] == stats["planned"] and full_matches == stats["planned"])
            known_records = [r for r in records if r and r["crop_sha"] == known]
            entry["known_crop"][mode]["raw_matches"] = sum(raw_equal(r, refs.get(known), "raw_response") for r in known_records)
            entry["known_crop"][mode]["raw_card_text_matches"] = sum(raw_equal(r, refs.get(known), "raw_text") for r in known_records)
            diff = []
            for r in records:
                if not r:
                    continue
                ref = refs.get(r["crop_sha"])
                parsed_match = bool(is_success(ref) and ref.get("parsed_cards") is not None
                    and is_success(r) and r["parsed_cards"] == ref["parsed_cards"])
                raw_match = raw_equal(r, ref, "raw_response")
                if not parsed_match or not raw_match:
                    diff.append({"request_id": r["request_id"], "crop_id": r["crop_id"], "crop_sha": r["crop_sha"],
                        "baseline_valid": bool(is_success(ref) and ref.get("parsed_cards") is not None),
                        "baseline_raw_response": ref.get("raw_response") if ref else None, "actual_raw_response": r["raw_response"],
                        "baseline_raw_text": ref.get("raw_text") if ref else None, "actual_raw_text": r["raw_text"],
                        "baseline_cards": ref.get("parsed_cards") if ref else None, "actual_cards": r["parsed_cards"],
                        "parsed_match": parsed_match, "raw_match": raw_match})
            save(f"c{concurrency}_{mode.lower()}_parity_independent.json", diff)
            # Preserve the original parity files; report reads this corrected file.
    safe = [e for e in summary["levels"] if e["concurrency"] > 1 and e["RAW"]["accuracy_safe"] and e["EFFECTIVE"]["accuracy_safe"]]
    summary["best_safe_concurrency"] = max(safe, key=lambda e:e["RAW"]["images_s"])["concurrency"] if safe else None
    summary["production_safe"] = summary["best_safe_concurrency"] is not None
    summary["metric_corrections"] = {"raw_parity": "HTTP200 assistant text equality independent of parsed-card validity; format/identity errors remain separately counted",
        "original_summary": "summary_recorded.json", "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save("summary.json", summary)
    print(json.dumps({"raw_parity_corrected": True, "levels": len(summary["levels"]),
                      "best_safe_concurrency": summary["best_safe_concurrency"]}))


if __name__ == "__main__":
    main()
