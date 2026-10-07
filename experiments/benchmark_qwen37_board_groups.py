"""Isolated hosted-API multi-image experiment; not CUDA tensor batching.

Original PNG bytes are sent unchanged. Group size is the only variable in the
main experiment. Per-item raw strings are compared exactly, without stripping.
API usage/finish_reason belong to the whole response, not individual images.
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.visual_cards import parse_cards

PROMPT = '''Read the visible poker BOARD cards in each supplied image independently,
in left-to-right order. Each image is preceded by its id. Return exactly one JSON
object with key "items", an array of objects with exactly "id" and "raw_text".
Copy each supplied id exactly. raw_text must contain only the card sequence:
ranks 2-9,T,J,Q,K,A and suits s,h,d,c, separated by spaces (example: As Td 2c).
If any rank or suit is unreadable, raw_text must be "UNKNOWN". Do not infer cards
from other images. No explanations, markdown, or additional keys.'''
PARAMS = {"model": "qwen3.7-flash", "temperature": 0, "seed": 37,
          "max_tokens": 2048, "reasoning_effort": "low",
          "response_format": {"type": "json_object"}, "stream": False}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def select(evidence, limit):
    unique = {}
    for line in evidence.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("crop_type") == "BOARD" and row.get("crop_sha256"):
            unique.setdefault(row["crop_sha256"], row)
    # Stable coverage across both street and table, not merely early frames.
    groups = collections.defaultdict(list)
    for row in sorted(unique.values(), key=lambda r: (r["frame_id"], r["crop_sha256"])):
        groups[(str(row["table_id"]), row["street"])].append(row)
    selected = []
    while len(selected) < min(limit, len(unique)):
        for key in sorted(groups):
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop(0))
    result = []
    for row in selected:
        data = Path(row["crop_path"]).read_bytes()
        if sha(data) != row["crop_sha256"]:
            raise ValueError("Input SHA mismatch: " + row["crop_path"])
        result.append({"id": row["crop_sha256"], "crop_path": row["crop_path"],
                       "street": row["street"], "table_id": row["table_id"],
                       "frame_id": row["frame_id"], "source_row_key": row.get("row_key"),
                       "data_url": "data:image/png;base64," + base64.b64encode(data).decode()})
    return result, len(unique)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--key-env", default="AITUNNEL_API_KEY")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, available = select(args.evidence, args.limit)
    if len(rows) < 60:
        raise ValueError("Need at least 60 unique BOARD crops for multiple full groups")
    manifest = {"experiment": "HOSTED_MULTI_IMAGE_SHARED_CONTEXT",
                "not_tensor_batching": True, "production_changed": False,
                "endpoint": "https://api.aitunnel.ru/v1/chat/completions",
                "parameters": PARAMS, "prompt": PROMPT, "prompt_sha256": sha(PROMPT.encode()),
                "script_sha256": sha(Path(__file__).read_bytes()),
                "parser_sha256": sha((Path(__file__).resolve().parents[1] / "parser_core/visual_cards.py").read_bytes()),
                "evidence_path": str(args.evidence.resolve()),
                "evidence_sha256": sha(args.evidence.read_bytes()),
                "preprocessing": "original PNG bytes; no local resize/crop/re-encode",
                "unobservable": ["weights_sha", "FP16", "CUDA_backend", "server_preprocessing",
                                 "persistent_model", "GPU_memory", "EOS_token_ids", "per_crop_tokens"],
                "batch_sizes": [1, 2, 4, 8, 16, 20], "available_unique": available,
                "python": sys.version, "platform": platform.platform(),
                "warmup": "one group of each size outside measured phase",
                "retries": 0, "timeout_seconds": 90,
                "order_check": "reverse full dataset for sizes >=4",
                "baseline_repeat": "first 8 crops as single-image calls",
                "selected": [{k: v for k, v in r.items() if k != "data_url"} for r in rows]}
    save(args.output / "manifest.json", manifest)
    # Immutable PNG copies make input replay independent of the original run.
    crop_dir = args.output / "inputs"
    crop_dir.mkdir()
    for row in rows:
        (crop_dir / (row["id"] + ".png")).write_bytes(Path(row["crop_path"]).read_bytes())
    if args.check_only:
        print(json.dumps({"check": "PASS", "images": len(rows), "available": available}))
        return
    import httpx
    key = os.environ.get(args.key_env)
    if not key:
        save(args.output / "summary.json", {"status": "BLOCKED_MISSING_KEY", "key_env": args.key_env})
        raise SystemExit(2)
    baseline = {}
    summary = {"status": "RUNNING", "images": len(rows), "results": [], "order_checks": []}
    request_number = 0
    with httpx.Client(timeout=httpx.Timeout(90, connect=10),
                      headers={"Authorization": "Bearer " + key}) as client:
        def infer(group, size, phase):
            nonlocal request_number
            request_number += 1
            content = [{"type": "text", "text": PROMPT}]
            for row in group:
                content.extend([{"type": "text", "text": "id: " + row["id"]},
                                {"type": "image_url", "image_url": {"url": row["data_url"]}}])
            body = {**PARAMS, "messages": [{"role": "user", "content": content}]}
            name = f"{request_number:04d}_{phase}_b{size}"
            save(args.output / (name + "_request.json"), body)
            started = time.perf_counter()
            try:
                response = client.post(manifest["endpoint"], json=body)
                elapsed = time.perf_counter() - started
                raw_body = response.text
                (args.output / (name + "_response.txt")).write_text(raw_body, encoding="utf-8")
                obj = response.json()
                if not isinstance(obj, dict):
                    obj = {"invalid_response_root": obj}
                http_status = response.status_code
            except (httpx.HTTPError, ValueError) as exc:
                elapsed = time.perf_counter() - started
                obj = {"error": {"type": type(exc).__name__}}
                http_status = None
                save(args.output / (name + "_error.json"), obj)
            raw_text, finish, mapped, error = None, None, {}, None
            if http_status == 200:
                try:
                    choice = obj["choices"][0]
                    raw_text = choice["message"]["content"]
                    finish = choice["finish_reason"]
                    items = json.loads(raw_text)["items"]
                    if not isinstance(items, list) or len(items) != len(group):
                        raise ValueError("wrong output count")
                    for item in items:
                        if set(item) != {"id", "raw_text"} or not isinstance(item["raw_text"], str) or item["id"] in mapped:
                            raise ValueError("invalid/duplicate output item")
                        mapped[item["id"]] = item["raw_text"]
                    if set(mapped) != {r["id"] for r in group}:
                        raise ValueError("output/input id mismatch")
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    error = "OUTPUT_CONTRACT_ERROR: " + str(exc)
                    mapped = {}
            else:
                error = "HTTP_ERROR" if http_status else "TRANSPORT_ERROR"
            outputs = []
            for pos, row in enumerate(group):
                raw = mapped.get(row["id"])
                cards = parse_cards(raw, {"FLOP": 3, "TURN": 4, "RIVER": 5}[row["street"]], row["street"]) if raw is not None else None
                item = {"id": row["id"], "position": pos, "raw_text": raw, "parsed": cards,
                        "status": error or ("RAW_UNREVIEWED" if cards else "UNRESOLVED_READING"),
                        "finish_reason": finish, "token_count": None}
                outputs.append(item)
            record = {"phase": phase, "batch_size": size, "members": [r["id"] for r in group],
                      "wall_s": elapsed, "http_status": http_status, "error": error,
                      "usage": obj.get("usage"), "model": obj.get("model"),
                      "system_fingerprint": obj.get("system_fingerprint"),
                      "finish_reason": finish, "raw_output_text": raw_text, "outputs": outputs}
            with (args.output / "requests.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            return record

        def phase_run(ordered, size, phase):
            records, outputs = [], []
            started = time.perf_counter()
            for offset in range(0, len(ordered), size):
                rec = infer(ordered[offset:offset+size], size, phase)
                records.append(rec)
                outputs.extend(rec["outputs"])
                if rec["http_status"] in (401, 402, 403):
                    break
            return time.perf_counter() - started, records, outputs

        def compare(outputs, reference):
            fields = ["raw_text", "parsed", "status", "finish_reason"]
            mismatches = []
            for item in outputs:
                expected = reference.get(item["id"])
                differences = {f: {"baseline": expected.get(f) if expected else None, "actual": item[f]}
                               for f in fields if not expected or expected[f] != item[f]}
                if item["raw_text"] is None or differences:
                    mismatches.append({"id": item["id"], "differences": differences,
                                       "missing_raw": item["raw_text"] is None})
            return mismatches

        for size in manifest["batch_sizes"]:
            warm = infer(rows[:size], size, "warmup")
            if warm["http_status"] in (401, 402, 403):
                summary["status"] = "BLOCKED_API_ACCESS"
                summary["http_status"] = warm["http_status"]
                save(args.output / "summary.json", summary)
                print(json.dumps(summary), flush=True)
                raise SystemExit(3)
            wall, records, outputs = phase_run(rows, size, "measured")
            if size == 1:
                baseline = {o["id"]: o for o in outputs}
            mismatches = compare(outputs, baseline)
            save(args.output / f"b{size}_outputs.json", outputs)
            save(args.output / f"b{size}_parity.json", mismatches)
            valid = len(outputs) == len(rows) and not any(r["error"] for r in records)
            api_wall = sum(r["wall_s"] for r in records)
            base_valid = summary["results"][0]["complete"] if summary["results"] else valid
            base_wall = summary["results"][0]["wall_s"] if summary["results"] else api_wall
            token_values = [(r["usage"] or {}).get("completion_tokens") for r in records]
            token_metrics_complete = all(isinstance(v, int) for v in token_values)
            result = {"batch": size, "wall_s": api_wall, "artifact_inclusive_wall_s": wall,
                      "images_s": len(rows)/api_wall if valid else None,
                      "ms_image": api_wall*1000/len(rows) if valid else None,
                      "speedup": base_wall/api_wall if valid and base_valid else None, "complete": valid,
                      "parity_matches": len(outputs)-len(mismatches), "parity_total": len(rows),
                      "parity": valid and not mismatches, "parsed_count": sum(o["parsed"] is not None for o in outputs),
                      "api_errors": sum(r["error"] is not None for r in records),
                      "completion_tokens": sum(token_values) if token_metrics_complete else None,
                      "missing_token_metrics": sum(v is None for v in token_values),
                      "finish_reasons": dict(collections.Counter(r["finish_reason"] for r in records)),
                      "peak_allocated_mib": None, "peak_reserved_mib": None,
                      "min_free_vram_mib": None, "oom_count": None}
            summary["results"].append(result)
            print(json.dumps(result), flush=True)
            save(args.output / "summary.json", summary)
            if size == 1:
                _, repeat_records, repeat = phase_run(rows[:8], 1, "baseline_repeat")
                diff = compare(repeat, baseline)
                save(args.output / "baseline_repeat_parity.json", diff)
                summary["baseline_repeat"] = {"images": len(repeat), "mismatches": len(diff),
                                              "stable": len(repeat) == 8 and not diff and not any(r["error"] for r in repeat_records)}
            if size >= 4:
                _, order_records, reordered = phase_run(list(reversed(rows)), size, "reordered")
                same_group = {o["id"]: o for o in outputs}
                diff = compare(reordered, same_group)
                save(args.output / f"b{size}_order_parity.json", diff)
                summary["order_checks"].append({"batch": size, "images": len(reordered),
                                                "matches": len(reordered)-len(diff), "mismatches": len(diff),
                                                "complete": len(reordered) == len(rows) and not any(r["error"] for r in order_records)})
            save(args.output / "summary.json", summary)
        summary["status"] = "COMPLETED"
        summary["strict_parity_limits"] = "Per-image raw_text JSON field exact; whole-response raw is saved. Per-image tokens and EOS IDs unavailable."
        summary["baseline_valid"] = summary["results"][0]["complete"]
        summary["baseline_stable"] = summary["baseline_repeat"]["stable"]
        for result in summary["results"]:
            order = next((r for r in summary["order_checks"] if r["batch"] == result["batch"]), None)
            result["order_parity"] = (order["complete"] and order["mismatches"] == 0) if order else None
            result["safe_acceleration"] = (result["batch"] >= 4 and summary["baseline_valid"]
                                           and summary["baseline_stable"] and result["parity"]
                                           and result["order_parity"] is True)
        summary["timing"] = "wall_s=sum sequential HTTP round trips including upload/download; artifact_inclusive_wall_s also includes recording/parsing"
        save(args.output / "summary.json", summary)
        print("COMPLETED", flush=True)


if __name__ == "__main__":
    main()
