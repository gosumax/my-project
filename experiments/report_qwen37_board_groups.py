"""Render saved hosted multi-image metrics without issuing API requests."""
import argparse
import collections
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    run = args.run
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    calls = [json.loads(s) for s in (run / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    usage = [c.get("usage") or {} for c in calls]
    aggregates = {"requests": len(calls), "http_statuses": dict(collections.Counter(str(c["http_status"]) for c in calls)),
                  "cost_rub_reported": sum(u.get("cost_rub", 0) for u in usage),
                  "cost_missing_requests": sum("cost_rub" not in u for u in usage),
                  "completion_tokens_reported": sum(u.get("completion_tokens", 0) for u in usage),
                  "reasoning_tokens_reported": sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for u in usage),
                  "finish_reasons": dict(collections.Counter(str(c["finish_reason"]) for c in calls)),
                  "returned_models": sorted({str(c.get("model")) for c in calls})}
    rows = summary["results"]
    complete = [r for r in rows if r["complete"]]
    aggregates["fastest_batch"] = max(complete, key=lambda r: r["images_s"])["batch"] if complete else None
    aggregates["observed_wall_ratios"] = [{"batch": r["batch"], "attempted_images_s": summary["images"]/r["wall_s"],
                                          "b1_wall_ratio": rows[0]["wall_s"]/r["wall_s"],
                                          "valid_speedup": r["speedup"]} for r in rows]
    aggregates["safe_acceleration_batches"] = [r["batch"] for r in rows if r.get("safe_acceleration")]
    (run / "aggregate.json").write_text(json.dumps(aggregates, ensure_ascii=False, indent=2), encoding="utf-8")
    text = ["# Qwen3.7 Flash: BOARD multi-image experiment", "",
            f"Run status: {summary['status']}. Dataset: {summary['images']} unique BOARD crops.", "",
            "Hosted shared-context multi-image requests; no local GPU tensor batching. Production unchanged.", "",
            "| Group | API wall s | Images/s | ms/image | Speedup | Exact parity | Parsed | GPU peak/free | OOM |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in rows:
        def f(key):
            return f"{r[key]:.3f}" if r.get(key) is not None else "N/A"
        speed = f"{r['speedup']:.3f}x" if r.get("speedup") is not None else "N/A"
        text.append(f"| {r['batch']} | {f('wall_s')} | {f('images_s')} | {f('ms_image')} | {speed} | {r['parity_matches']}/{r['parity_total']} | {r['parsed_count']}/{r['parity_total']} | N/A | N/A |")
    text += ["", f"Baseline repeat: {summary.get('baseline_repeat')}", "",
             "Order checks (against same group size in original order):"]
    for r in summary.get("order_checks", []):
        text.append(f"- Group {r['batch']}: {r['matches']}/{r['images']} exact matches; complete={r['complete']}.")
    text += ["", f"Fastest completed group: {aggregates['fastest_batch']}.",
             f"Groups passing baseline stability, main parity and order checks: {aggregates['safe_acceleration_batches']}.", "",
             f"All phases: {aggregates['requests']} requests; reported cost {aggregates['cost_rub_reported']:.4f} RUB; {aggregates['cost_missing_requests']} missing cost records.",
             f"Output tokens: {aggregates['completion_tokens_reported']}; reasoning tokens: {aggregates['reasoning_tokens_reported']}.",
             f"Finish reasons: {aggregates['finish_reasons']}", "",
             "## Concrete differences"]
    if not summary.get("baseline_valid"):
        text += ["", "Baseline is incomplete. Valid speedup is unavailable; observed wall ratios below include attempted requests with contract errors:"]
        for r in aggregates["observed_wall_ratios"]:
            text.append(f"- Group {r['batch']}: {r['attempted_images_s']:.3f} attempted images/s, {r['b1_wall_ratio']:.3f}x wall ratio (not a safe speedup).")
    crops = {r["id"]: r for r in manifest["selected"]}
    for r in rows[1:]:
        diffs = json.loads((run / f"b{r['batch']}_parity.json").read_text(encoding="utf-8"))
        if diffs:
            d = diffs[0]
            text += ["", f"Group {r['batch']}, crop `{d['id']}`:",
                     f"Path: `{crops[d['id']]['crop_path']}`", "```json", json.dumps(d, ensure_ascii=False, indent=2), "```"]
    text += ["", "## Limits", "",
             "- Wall is the sum of sequential HTTP round trips, including upload/download; disk recording is reported separately in summary.json.",
             "- Model weights, precision, CUDA, server resize, persistence, VRAM, server OOM and EOS token IDs are unobservable via this API.",
             "- Exact per-image parity compares the decoded JSON raw_text field without stripping, parsed cards, reading status and response finish_reason. Entire response bodies are saved; per-crop tokens are unavailable.",
             "- JSON extraction prompt and 2048-token cap are fixed across sizes. The Qwen path is new and differs from the previous local VL OCR: path. Full SHA ids consume output tokens.",
             "- Images share context; this experiment measures grouping, not independent tensor batching or /batches asynchronous jobs.",
             "- With 80 inputs, full reversal preserves group membership for 4/8/16/20 and reverses within-group positions and request schedule. No same-size original-order repeat was run to isolate order causality from hosted nondeterminism.",
             "- Parsed results are RAW_UNREVIEWED, not visual accuracy labels or certified hand histories.",
             "- One timed dataset pass per size, several full groups each; order checks are separate. Network/cache/provider load and run order may affect timing; no timing confidence interval is claimed.",
             "- Reproducibility includes request bodies, original PNG copies, script/parser/input SHA and raw responses. Hosted model weights cannot be pinned.", ""]
    (run / "report.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps(aggregates, ensure_ascii=False))


if __name__ == "__main__":
    main()
