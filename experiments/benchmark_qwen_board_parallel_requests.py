"""One original BOARD PNG per independent hosted API request, isolated experiment.

Baseline covers unique crops once. Each parallel level repeats the identical
fixed dataset ten times. First attempts and bounded technical retries are
separate phases; no OCR/JSON/identity/content retries.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.benchmark_qwen37_board_groups import PROMPT, PARAMS, select, sha
from parser_core.visual_cards import parse_cards

LEVELS = [1, 10, 25, 50, 100, 200, 300, 500, 1000]
ENDPOINT = "https://api.aitunnel.ru/v1/chat/completions"
KNOWN_NAME = "000000731_016_TURN.png"


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def retry_after(value):
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return 0.0


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    left = int(position)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (position - left)


def decode_response(obj, crop):
    """Fail closed on structure, identity, truncation, and missing content."""
    result = {"raw_response": None, "raw_text": None, "parsed_cards": None,
              "finish_reason": None, "format_error": None, "wrong_identity": False,
              "server_request_id": None, "returned_model": None, "token_usage": None}
    try:
        if not isinstance(obj, dict):
            raise ValueError("response root must be object")
        result.update(server_request_id=obj.get("id"), returned_model=obj.get("model"),
                      token_usage=obj.get("usage"))
        choices = obj["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("expected exactly one choice")
        choice = choices[0]
        result["finish_reason"] = choice["finish_reason"]
        raw = choice["message"]["content"]
        if not isinstance(raw, str):
            raise ValueError("content must be text")
        result["raw_response"] = raw
        decoded = json.loads(raw)
        if not isinstance(decoded, dict) or set(decoded) != {"items"}:
            raise ValueError("expected items object")
        items = decoded["items"]
        if not isinstance(items, list) or len(items) != 1:
            raise ValueError("expected exactly one image output")
        item = items[0]
        if not isinstance(item, dict) or set(item) != {"id", "raw_text"}:
            raise ValueError("invalid output fields")
        if item["id"] != crop["id"]:
            result["wrong_identity"] = True
            raise ValueError("output crop id mismatch")
        if not isinstance(item["raw_text"], str):
            raise ValueError("raw_text must be string")
        result["raw_text"] = item["raw_text"]
        result["parsed_cards"] = parse_cards(item["raw_text"],
            {"FLOP": 3, "TURN": 4, "RIVER": 5}[crop["street"]], crop["street"])
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        result["format_error"] = str(exc)
    return result


def is_success(record):
    return (record is not None and record["http_status"] == 200
            and not record["transport_error"] and not record["format_error"]
            and not record["wrong_identity"] and not record.get("duplicate_server_id")
            and record["finish_reason"] == "stop"
            and record.get("returned_model") == PARAMS["model"])


def technical(record):
    return record["http_status"] == 429 or (record["http_status"] or 0) >= 500 or bool(record["transport_error"])


def metrics(records, planned, wall, references):
    present = [r for r in records if r is not None]
    latency = [r["latency_s"] for r in present]
    parsed_matches = raw_matches = card_raw_matches = 0
    compared = 0
    bad_refs = set()
    mismatches = []
    for r in present:
        ref = references.get(r["crop_sha"])
        ref_valid = is_success(ref) and ref.get("parsed_cards") is not None
        if not ref_valid:
            bad_refs.add(r["crop_sha"])
        parsed_match = bool(ref_valid and is_success(r) and r["parsed_cards"] == ref["parsed_cards"])
        raw_match = bool(ref_valid and is_success(r) and r["raw_response"] == ref["raw_response"])
        card_raw_match = bool(ref_valid and is_success(r) and r["raw_text"] == ref["raw_text"])
        parsed_matches += parsed_match
        raw_matches += raw_match
        card_raw_matches += card_raw_match
        compared += bool(ref_valid)
        if not parsed_match or not raw_match:
            mismatches.append({"request_id": r["request_id"], "crop_id": r["crop_id"], "crop_sha": r["crop_sha"],
                               "baseline_valid": ref_valid, "baseline_raw_response": ref.get("raw_response") if ref else None,
                               "actual_raw_response": r["raw_response"],
                               "baseline_raw_text": ref.get("raw_text") if ref else None,
                               "actual_raw_text": r["raw_text"],
                               "baseline_cards": ref.get("parsed_cards") if ref else None,
                               "actual_cards": r["parsed_cards"],
                               "parsed_match": parsed_match, "raw_match": raw_match})
    successful = sum(is_success(r) for r in present)
    result = {"planned": planned, "requests": len(present), "successful": successful,
              "failed": len(present) - successful, "skipped_unsent": planned - len(present),
              "http_errors": sum(r["http_status"] is not None and r["http_status"] != 200 for r in present),
              "429": sum(r["http_status"] == 429 for r in present),
              "5xx": sum((r["http_status"] or 0) >= 500 for r in present),
              "timeouts": sum(r["transport_error"] == "TIMEOUT" for r in present),
              "connection_errors": sum(r["transport_error"] == "CONNECTION_ERROR" for r in present),
              "connection_reset": sum(r.get("connection_reset", False) for r in present),
              "format_errors": sum(r["http_status"] == 200 and r["format_error"] is not None for r in present),
              "wrong_identity": sum(r["wrong_identity"] for r in present),
              "duplicate_responses": sum(r.get("duplicate_server_id", False) for r in present),
              "wrong_model": sum(r["http_status"] == 200 and r.get("returned_model") != PARAMS["model"] for r in present),
              "truncated": sum(r["finish_reason"] == "length" for r in present),
              "missing_responses": sum(r["http_status"] is None for r in present),
              "retries": sum(r["retry_count"] for r in present),
              "parsed_parity": {"matches": parsed_matches, "total": planned, "valid_reference_requests": compared},
              "raw_parity": {"matches": raw_matches, "total": planned},
              "raw_card_text_parity": {"matches": card_raw_matches, "total": planned},
              "baseline_invalid_unique": len(bad_refs), "wall_s": wall,
              "requests_s": len(present)/wall if wall else None,
              "images_s": successful/wall if wall else None,
              "parsed_images_s": sum(is_success(r) and r["parsed_cards"] is not None for r in present)/wall if wall else None,
              "p50_s": percentile(latency, .5), "p95_s": percentile(latency, .95),
              "p99_s": percentile(latency, .99), "max_latency_s": max(latency) if latency else None,
              "completion_tokens": sum((r["token_usage"] or {}).get("completion_tokens", 0) for r in present),
              "token_metrics_missing": sum(not isinstance(r["token_usage"], dict) or "completion_tokens" not in r["token_usage"] for r in present),
              "finish_reasons": dict(collections.Counter(str(r["finish_reason"]) for r in present))}
    result["accuracy_safe"] = (successful == planned and parsed_matches == planned and raw_matches == planned)
    return result, mismatches


class Experiment:
    def __init__(self, output, crops, payloads, key):
        self.output, self.crops, self.payloads, self.key = output, crops, payloads, key
        self.logs = (output / "requests.jsonl").open("a", encoding="utf-8", buffering=1)
        self.seen_ids = {}
        self.cooldown_until = 0.0
        self.first_origin = None
        self.active = self.peak_active = 0

    async def cooldown(self):
        while self.cooldown_until > time.perf_counter():
            await asyncio.sleep(self.cooldown_until - time.perf_counter())

    async def attempt(self, client, crop, request_id, concurrency, attempt_index):
        import httpx
        await self.cooldown()
        attempt_id = f"{request_id}_a{attempt_index}"
        started = time.perf_counter()
        start_utc = utc()
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        record = {"request_id": request_id, "attempt_id": attempt_id, "concurrency": concurrency,
                  "crop_id": crop["id"], "crop_sha": crop["id"], "crop_path": crop["crop_path"],
                  "start_time": start_utc, "finish_time": None, "latency_s": None,
                  "start_offset_s": started - self.first_origin,
                  "retry_count": attempt_index, "image_count": 1,
                  "request_body_sha256": sha(self.payloads[crop["id"]]),
                  "http_status": None, "transport_error": None, "connection_reset": False,
                  "retry_after_header": None, "retry_after_s": 0,
                  "raw_response": None, "raw_text": None, "parsed_cards": None,
                  "finish_reason": None, "format_error": None, "wrong_identity": False,
                  "server_request_id": None, "returned_model": None, "token_usage": None,
                  "duplicate_server_id": False, "response_body_path": None,
                  "transport_response_id": None}
        try:
            # An outer deadline is total attempt wall, not a per-read timeout.
            response = await asyncio.wait_for(client.post(ENDPOINT, content=self.payloads[crop["id"]],
                headers={"X-Client-Request-Id": attempt_id}), timeout=90)
            finished = time.perf_counter()
            record["http_status"] = response.status_code
            record["transport_response_id"] = response.headers.get("x-request-id")
            record["retry_after_header"] = response.headers.get("retry-after")
            record["retry_after_s"] = retry_after(record["retry_after_header"])
            if record["retry_after_s"]:
                self.cooldown_until = max(self.cooldown_until, finished + record["retry_after_s"])
            body_path = self.output / "responses" / (attempt_id + ".json")
            body_path.write_bytes(response.content)
            record["response_body_path"] = str(body_path.resolve())
            record["response_body_sha256"] = sha(response.content)
            if response.status_code == 200:
                try:
                    record.update(decode_response(response.json(), crop))
                except ValueError:
                    record["format_error"] = "HTTP body not JSON"
                server_id = record["server_request_id"]
                if isinstance(server_id, str) and server_id:
                    if server_id in self.seen_ids:
                        record["duplicate_server_id"] = True
                        record["duplicate_of_attempt"] = self.seen_ids[server_id]
                    else:
                        self.seen_ids[server_id] = attempt_id
            else:
                try:
                    obj = response.json()
                    record["api_error"] = obj.get("error") if isinstance(obj, dict) else obj
                except ValueError:
                    record["api_error"] = "NON_JSON_HTTP_ERROR"
        except (asyncio.TimeoutError, httpx.TimeoutException):
            finished = time.perf_counter()
            record["transport_error"] = "TIMEOUT"
        except httpx.HTTPError as exc:
            finished = time.perf_counter()
            record["transport_error"] = "CONNECTION_ERROR"
            record["transport_error_type"] = type(exc).__name__
            record["connection_reset"] = "reset" in str(exc).lower()
        finally:
            self.active -= 1
        record["finish_time"] = utc()
        record["latency_s"] = finished - started
        record["finish_offset_s"] = finished - self.first_origin
        self.logs.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    async def run_level(self, jobs, concurrency):
        import httpx
        self.peak_active = 0
        first = [None] * len(jobs)
        final = [None] * len(jobs)
        stop_reason = None
        technical_failures = 0
        semaphore = asyncio.Semaphore(concurrency)
        self.first_origin = time.perf_counter()
        async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=10),
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"}) as client:
            async def first_job(index, crop):
                nonlocal stop_reason, technical_failures
                async with semaphore:
                    if stop_reason:
                        return
                    await self.cooldown()
                    if stop_reason:
                        return
                    request_id = f"c{concurrency:04d}_r{index:05d}"
                    rec = await self.attempt(client, crop, request_id, concurrency, 0)
                    first[index] = final[index] = rec
                    if technical(rec):
                        technical_failures += 1
                    if rec["http_status"] == 429:
                        stop_reason = "FIRST_429_STOP_UNSENT"
                    elif rec["http_status"] in (401, 402, 403):
                        stop_reason = "API_ACCESS_ERROR"
                    elif technical_failures >= 3:
                        stop_reason = "THREE_TECHNICAL_ERRORS_STOP_UNSENT"
                    if (index+1) % 50 == 0:
                        print(json.dumps({"progress": concurrency, "finished": sum(r is not None for r in first),
                                          "planned": len(jobs), "peak_inflight": self.peak_active}), flush=True)
            await asyncio.gather(*(first_job(i, crop) for i, crop in enumerate(jobs)))
            raw_wall = time.perf_counter() - self.first_origin
            retry_sem = asyncio.Semaphore(min(concurrency, 10))
            async def retry_job(index):
                rec = first[index]
                if rec is None or not technical(rec):
                    return
                async with retry_sem:
                    for retry in (1, 2):
                        # Explicit Retry-After has priority; do not shorten it.
                        delay = max(rec["retry_after_s"], min(30, 2 ** retry + random.Random(index*37+retry).random()))
                        if delay > 300:
                            final[index] = {**rec, "retry_not_sent_reason": "RETRY_AFTER_EXCEEDS_300S_BUDGET"}
                            return
                        await asyncio.sleep(delay)
                        rec = await self.attempt(client, jobs[index], rec["request_id"], concurrency, retry)
                        final[index] = rec
                        if not technical(rec):
                            break
            await asyncio.gather(*(retry_job(i) for i in range(len(jobs))))
            effective_wall = time.perf_counter() - self.first_origin
            # EFFECTIVE latency covers the first attempt, phase barrier, retries
            # and backoff. requests.jsonl retains each attempt's HTTP latency.
            for index, rec in enumerate(final):
                if rec is not None:
                    final[index] = {**rec, "last_attempt_latency_s": rec["latency_s"],
                                    "latency_s": rec["finish_offset_s"] - first[index]["start_offset_s"],
                                    "first_start_time": first[index]["start_time"]}
        # Stop subsequent higher levels on any visible transport/provider failure.
        limit_hit = any(technical(r) or r["http_status"] in (401, 402, 403) for r in first if r)
        return first, final, raw_wall, effective_wall, self.peak_active, stop_reason, limit_hit


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    crops, available = select(args.evidence, 100000)
    known = [r for r in crops if Path(r["crop_path"]).name == KNOWN_NAME]
    if len(known) != 1:
        raise ValueError("known problem crop missing or ambiguous")
    if len(crops) * args.repeats < 1000:
        raise ValueError("parallel workload must have at least 1000 requests")
    (args.output / "inputs").mkdir()
    (args.output / "responses").mkdir()
    (args.output / "payloads").mkdir()
    payloads = {}
    for crop in crops:
        data = Path(crop["crop_path"]).read_bytes()
        if sha(data) != crop["id"]:
            raise ValueError("SHA mismatch")
        (args.output / "inputs" / (crop["id"] + ".png")).write_bytes(data)
        body = {**PARAMS, "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT}, {"type": "text", "text": "id: " + crop["id"]},
            {"type": "image_url", "image_url": {"url": crop["data_url"]}}]}]}
        assert sum(i["type"] == "image_url" for i in body["messages"][0]["content"]) == 1
        payloads[crop["id"]] = json.dumps(body, ensure_ascii=False).encode("utf-8")
        (args.output / "payloads" / (crop["id"] + ".json")).write_bytes(payloads[crop["id"]])
    schedule = crops * args.repeats
    random.Random(37).shuffle(schedule)
    manifest = {"experiment": "ONE_IMAGE_ONE_INDEPENDENT_API_REQUEST", "endpoint": ENDPOINT,
                "model_parameters": PARAMS, "prompt": PROMPT, "prompt_sha256": sha(PROMPT.encode()),
                "script_sha256": sha(Path(__file__).read_bytes()),
                "source_helper_sha256": sha((Path(__file__).parent / "benchmark_qwen37_board_groups.py").read_bytes()),
                "parser_sha256": sha((Path(__file__).resolve().parents[1] / "parser_core/visual_cards.py").read_bytes()),
                "evidence_sha256": sha(args.evidence.read_bytes()), "evidence_path": str(args.evidence.resolve()),
                "levels": LEVELS, "baseline_requests": len(crops), "parallel_requests": len(schedule),
                "unique_crops": available, "repeats": args.repeats, "schedule_seed": 37,
                "comparison": "speedup by successful images/sec; baseline has one request per unique crop",
                "known_crop_sha": known[0]["id"], "preprocessing": "original PNG bytes, no modifications",
                "retry_policy": {"max_retries": 2, "technical_only": True, "separate_phase": True,
                                 "retry_concurrency_cap": 10, "backoff": "2^retry seconds plus deterministic 0..1 jitter, capped30",
                                 "retry_after": "honored; >300s leaves failure without retry", "attempt_deadline_s": 90},
                "stop_policy": "first429 or 3technical errors stops unsent; any technical failure prevents next higher level; two successive <10% throughput gains stops escalation",
                "provider_internal_retries": "unobservable; AITunnel docs say provider429 may be retried internally",
                "warmup": "one independent known-crop request, excluded",
                "httpx": None, "production_changed": False,
                "selected": [{k: v for k,v in c.items() if k != "data_url"} for c in crops],
                "parallel_schedule": [c["id"] for c in schedule],
                "request_body_sha256": {k: sha(v) for k,v in payloads.items()}}
    save(args.output / "manifest.json", manifest)
    if args.check_only:
        print(json.dumps({"preflight": "PASS", "unique": len(crops), "parallel_requests": len(schedule),
                          "known_crop": known[0]["id"]}), flush=True)
        return
    import httpx
    manifest["httpx"] = httpx.__version__
    save(args.output / "manifest.json", manifest)
    key = os.environ.get("AITUNNEL_API_KEY")
    if not key:
        raise RuntimeError("AITUNNEL_API_KEY absent")
    experiment = Experiment(args.output, crops, payloads, key)
    summary = {"status": "RUNNING", "levels": [], "skipped_levels": [], "baseline_unique": len(crops)}
    references_raw, references_effective = {}, {}
    plateau_count = 0
    try:
        # Warmup separate request ID and no measurement/retries.
        experiment.first_origin = time.perf_counter()
        async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=10),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}) as warm_client:
            warm = await experiment.attempt(warm_client, known[0], "warmup_known", 0, 0)
        if warm["http_status"] in (401, 402, 403) or technical(warm):
            summary["status"] = "BLOCKED_WARMUP_API_OR_TRANSPORT"
            summary["warmup"] = warm
            summary["skipped_levels"] = [{"concurrency": c, "reason": "WARMUP_TECHNICAL_FAILURE"} for c in LEVELS]
            save(args.output / "summary.json", summary)
            return
        for concurrency in LEVELS:
            jobs = crops if concurrency == 1 else schedule
            first, final, raw_wall, eff_wall, peak, circuit, limit_hit = await experiment.run_level(jobs, concurrency)
            if concurrency == 1:
                references_raw = {c["id"]: r for c,r in zip(jobs,first)}
                references_effective = {c["id"]: r for c,r in zip(jobs,final)}
                save(args.output / "baseline_raw.json", references_raw)
                save(args.output / "baseline_effective.json", references_effective)
            raw_stats, raw_diff = metrics(first, len(jobs), raw_wall, references_raw)
            eff_stats, eff_diff = metrics(final, len(jobs), eff_wall, references_effective)
            for stats, mode in ((raw_stats, "RAW"), (eff_stats, "EFFECTIVE")):
                baseline_rate = summary["levels"][0][mode]["images_s"] if summary["levels"] else stats["images_s"]
                stats["speedup"] = stats["images_s"]/baseline_rate if baseline_rate else None
            entry = {"concurrency": concurrency, "peak_inflight_attempts": peak, "circuit_stop_reason": circuit,
                     "api_limit_observed": limit_hit, "RAW": raw_stats, "EFFECTIVE": eff_stats}
            entry["known_crop"] = {}
            for mode, records, refs in (("RAW", first, references_raw), ("EFFECTIVE", final, references_effective)):
                selected = [r for r in records if r and r["crop_sha"] == known[0]["id"]]
                stat, diff = metrics(selected, len(selected), 1, refs)
                entry["known_crop"][mode] = {"requests": len(selected), "raw_texts": dict(collections.Counter(str(r["raw_text"]) for r in selected)),
                    "parsed_matches": stat["parsed_parity"]["matches"], "raw_matches": stat["raw_parity"]["matches"],
                    "concurrent_other_requests": concurrency > 1}
            save(args.output / f"c{concurrency}_raw_parity.json", raw_diff)
            save(args.output / f"c{concurrency}_effective_parity.json", eff_diff)
            save(args.output / f"c{concurrency}_outcomes.json", {"first": first, "effective": final})
            summary["levels"].append(entry)
            save(args.output / "summary.json", summary)
            print(json.dumps(entry, ensure_ascii=False), flush=True)
            reason = None
            if limit_hit:
                reason = "VISIBLE_API_OR_TRANSPORT_LIMIT"
            elif len(summary["levels"]) > 2:
                previous = summary["levels"][-2]["RAW"]
                gain = raw_stats["images_s"] / previous["images_s"] if previous["images_s"] else 0
                plateau_count = plateau_count + 1 if gain < 1.1 else 0
                if plateau_count >= 2:
                    reason = "TWO_LEVEL_THROUGHPUT_PLATEAU"
                elif gain < 1.1 and raw_stats["p95_s"] > previous["p95_s"] * 3:
                    reason = "LATENCY_SPIKE_WITHOUT_THROUGHPUT_GAIN"
            if reason:
                summary["stop_reason"] = reason
                summary["skipped_levels"] = [{"concurrency": c, "reason": reason} for c in LEVELS if c > concurrency]
                break
        summary["status"] = "COMPLETED"
        safe = [e for e in summary["levels"] if e["concurrency"] > 1
                and e["RAW"]["accuracy_safe"] and e["EFFECTIVE"]["accuracy_safe"]]
        summary["best_safe_concurrency"] = max(safe, key=lambda e: e["RAW"]["images_s"])["concurrency"] if safe else None
        summary["production_safe"] = summary["best_safe_concurrency"] is not None
        summary["best_measured_throughput"] = max(summary["levels"], key=lambda e: e["RAW"]["images_s"])["concurrency"]
        save(args.output / "summary.json", summary)
        print("COMPLETED", flush=True)
    finally:
        experiment.logs.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
