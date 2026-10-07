"""Read-only aggregation/rendering of isolated single-image concurrency results."""
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
    all_usage = [c["token_usage"] for c in calls if isinstance(c.get("token_usage"), dict)]
    aggregate = {"attempts_including_warmup": len(calls),
                 "request_ids": len({c["request_id"] for c in calls}),
                 "attempt_ids_unique": len({c["attempt_id"] for c in calls}) == len(calls),
                 "image_count_violations": sum(c["image_count"] != 1 for c in calls),
                 "cost_rub_reported": sum(u.get("cost_rub", 0) for u in all_usage),
                 "usage_missing_attempts": len(calls) - len(all_usage),
                 "completion_tokens_reported": sum(u.get("completion_tokens", 0) for u in all_usage),
                 "reasoning_tokens_reported": sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for u in all_usage),
                 "http_statuses": dict(collections.Counter(str(c["http_status"]) for c in calls)),
                 "returned_models": sorted({str(c.get("returned_model")) for c in calls}),
                 "best_safe_concurrency": summary.get("best_safe_concurrency"),
                 "production_safe": summary.get("production_safe", False)}
    save = lambda name, value: (run / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    save("aggregate.json", aggregate)
    text = ["# Qwen BOARD: one image per independent parallel API request", "",
            f"Status: {summary['status']}. Baseline: {manifest['baseline_requests']} unique crops. Each parallel level: {manifest['parallel_requests']} logical requests (10 per original SHA).", "",
            f"BEST SAFE CONCURRENCY: {summary.get('best_safe_concurrency') or 'NONE'}", "",
            f"Production safe: {'YES' if summary.get('production_safe') else 'NO'}. Production unchanged.", "",
            f"Stop reason: {summary.get('stop_reason', 'all levels completed')}", ""]
    def fmt(value):
        return f"{value:.3f}" if isinstance(value, (int, float)) else "N/A"
    for mode in ("RAW", "EFFECTIVE"):
        text += [f"## {mode}", "",
                 "| C | Requests/planned | Successful | Failed | 429 | 5xx | Timeout | Format | Parsed parity | Raw parity | Wall s | req/s | img/s | p50 s | p95 s | p99 s | Max s | Speedup |",
                 "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for entry in summary["levels"]:
            r = entry[mode]
            p, raw = r["parsed_parity"], r["raw_parity"]
            text.append(f"| {entry['concurrency']} | {r['requests']}/{r['planned']} | {r['successful']} | {r['failed']} | {r['429']} | {r['5xx']} | {r['timeouts']} | {r['format_errors']} | {p['matches']}/{p['total']} | {raw['matches']}/{raw['total']} | {fmt(r['wall_s'])} | {fmt(r['requests_s'])} | {fmt(r['images_s'])} | {fmt(r['p50_s'])} | {fmt(r['p95_s'])} | {fmt(r['p99_s'])} | {fmt(r['max_latency_s'])} | {fmt(r['speedup'])}x |")
        text.append("")
    text += ["## Actual inflight concurrency and error details", ""]
    for entry in summary["levels"]:
        r, e = entry["RAW"], entry["EFFECTIVE"]
        text += [f"- C={entry['concurrency']}: peak inflight API attempt coroutines={entry['peak_inflight_attempts']}; RAW skipped-unsent={r['skipped_unsent']}, connection errors={r['connection_errors']}, resets={r['connection_reset']}, truncated={r['truncated']}, missing HTTP responses={r['missing_responses']}, wrong binding={r['wrong_identity']}, duplicate server IDs={r['duplicate_responses']}, wrong model={r['wrong_model']}; effective retries={e['retries']}."]
    text += ["", "## Known problem crop", "", f"Crop SHA: `{manifest['known_crop_sha']}`."]
    for entry in summary["levels"]:
        for mode in ("RAW", "EFFECTIVE"):
            r = entry["known_crop"][mode]
            text.append(f"- C={entry['concurrency']} {mode}: {r['requests']} responses, parsed matches={r['parsed_matches']}, full raw matches={r['raw_matches']}, raw card strings={r['raw_texts']}.")
    text += ["", "## Levels not measured", ""]
    for entry in summary.get("skipped_levels", []):
        text.append(f"- C={entry['concurrency']}: {entry['reason']}.")
    text += ["", "## Concrete parity differences", ""]
    crop_paths = {c["id"]: c["crop_path"] for c in manifest["selected"]}
    for entry in summary["levels"]:
        if entry["concurrency"] == 1:
            continue
        diff_path = run / f"c{entry['concurrency']}_raw_parity_independent.json"
        if not diff_path.exists():
            diff_path = run / f"c{entry['concurrency']}_raw_parity.json"
        diff = json.loads(diff_path.read_text(encoding="utf-8"))
        # Prefer a card difference with a valid baseline over JSON whitespace.
        decisive = next((d for d in diff if d["baseline_valid"] and d["baseline_cards"] != d["actual_cards"]), None)
        if decisive is None and diff:
            decisive = diff[0]
        if decisive:
            text += [f"C={entry['concurrency']}, `{crop_paths[decisive['crop_sha']]}`:", "```json", json.dumps(decisive, ensure_ascii=False, indent=2), "```", ""]
    text += ["## Measurement limits", "",
             "- Every payload has exactly one image, original PNG bytes and its SHA id; contexts are independent. Raw HTTP bodies and per-attempt request/crop/response binding are preserved.",
             "- RAW is each logical request's first attempt. EFFECTIVE is final outcome after at most two technical retries. Format, cards, binding and content failures are never retried.",
             "- Retry phase is separate after first attempts; capped at min(C,10), with shared Retry-After cooldown and bounded exponential backoff. Effective wall and latency include phase wait/retries/backoff. This is the explicit experimental client policy, not a production implementation claim.",
             "- AITunnel may retry upstream requests internally; RAW describes client-visible attempts, not unobservable provider attempts. See https://aitunnel.ru/docs/limits .",
             "- images/sec counts successful HTTP+JSON+identity+stop+model outcomes, not certified card accuracy. parsed_images/sec is separate. Speedup compares images/sec, because baseline and parallel request counts differ by user choice.",
             "- Raw parity compares available HTTP200 assistant content (including JSON formatting), independently of card parsing/JSON validity; raw_card_text_parity compares just available per-image card strings. Parsed parity requires a valid strict parse_cards baseline. Invalid baseline card references are never replaced or repaired.",
             "- Strict safe verdict requires zero failures/missing/identity errors and full parsed+raw parity for a measured C>1. Baseline self-comparison does not certify parallel safety.",
             "- First429 or three technical errors stops unsent jobs. Any visible technical failure prevents higher concurrency levels. Two successive throughput gains below10% or latency spike without throughput gain also stops escalation.",
             "- Peak inflight describes active client API attempt coroutines, not remote GPU/server execution. No local VRAM claims.",
             "- One workload pass per concurrency; ten repeats per crop are within-run parity observations. Server/model revision, internal preprocessing/cache and upstream retries cannot be pinned.", "",
             f"All attempts: {aggregate['attempts_including_warmup']}; reported cost {aggregate['cost_rub_reported']:.4f} RUB; {aggregate['usage_missing_attempts']} missing usage records.", ""]
    (run / "report.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False))


if __name__ == "__main__":
    main()
