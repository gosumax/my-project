"""Current PP-OCRv6 Tiny on each saved Smoke16 CARD crop, by rank.

Raw model text is never changed. Strict rank interpretation accepts only an
entire rank token or an entire rank+single-suit token; all else is unresolved.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import time
from pathlib import Path

from parser_core.tiny_chat_client import TinyChatClient

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/card_train_smoke16_test_20260930"
DEFAULT_OUT = DATA / "tiny_individual_card_rank_probe"
RANKS = "23456789TJQKA"
RANK_RE = re.compile(r"^(10|[2-9TJQKA])$", re.IGNORECASE)
CARD_RE = re.compile(r"^(10|[2-9TJQKA])([CDHS♣♦♥♠])$", re.IGNORECASE)

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

def strict_rank(raw: str):
    normalized = raw.strip().upper()
    match = RANK_RE.fullmatch(normalized)
    source = "EXACT_RANK" if match else None
    if match is None:
        match = CARD_RE.fullmatch(normalized)
        source = "FULL_CARD_TEXT" if match else None
    if match is None:
        return None, "UNRESOLVED"
    token = match.group(1)
    return ("T" if token == "10" else token), source

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Use a fresh output directory")
    source_path = DATA / "smoke16_test_manifest.jsonl"
    source = read(source_path)
    if len(source) != 225:
        raise ValueError("Expected 225 frozen individual CARD crops")
    user = json.loads((DATA / "stage2_template/user_adjudication_A_B.json").read_text(encoding="utf-8"))
    correction = {d["tile_sha256"]: d["user_visual_reading"] for d in user["decisions"] if d["decision"] == "CORRECT_GT_POST_BASELINE"}
    config_path = ROOT / "configs/ocr_tiny_local.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True)
    settings = {"status": "POST_BASELINE_TINY_SINGLE_CARD_DIAGNOSTIC",
                "source_manifest_sha256": sha(source_path.read_bytes()),
                "user_adjudication_sha256": sha((DATA / "stage2_template/user_adjudication_A_B.json").read_bytes()),
                "tiny_config_sha256": sha(config_path.read_bytes()),
                "tiny_model": cfg["model_name"], "tiny_batch_size": cfg["batch_size"],
                "preprocessing": "SAVED_CARD_PNG_AS_IS", "parser_rule": "whole rank or whole rank plus one suit only",
                "production_note": "Tiny normally receives full CHAT rows. Single CARD crops are a diagnostic input, not current production routing."}
    (args.output / "config.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    client = TinyChatClient(Path(cfg["python"]), Path(cfg["model_dir"]), cfg["model_name"],
                            cfg["weights_sha256"], cfg["device"], args.output / "tiny_worker.stderr.log")
    started = time.perf_counter()
    result = []
    try:
        for start in range(0, len(source), cfg["batch_size"]):
            batch = source[start:start + cfg["batch_size"]]
            items = []
            for index, tile in enumerate(batch, start):
                path = Path(tile["tile_png"])
                if sha(path.read_bytes()) != tile["tile_sha256"]:
                    raise ValueError(f"CARD PNG SHA changed: {path}")
                items.append({"item_id": str(index), "path": str(path),
                              "source_sha256": tile["tile_sha256"]})
            response = client.read_batch(items, cfg["batch_size"])
            for item in response["results"]:
                index = int(item["item_id"])
                tile = source[index]
                gt = correction.get(tile["tile_sha256"], tile["ground_truth"])
                rank, parse_status = strict_rank(item.get("raw_text", "")) if item["status"] == "OK" else (None, "UNRESOLVED")
                verdict = "UNRESOLVED" if rank is None else "CORRECT" if rank == gt[0] else "WRONG"
                result.append({"prediction_index": index, "tile_png": tile["tile_png"],
                               "tile_sha256": tile["tile_sha256"], "source_row_sha256": tile["source_row_sha256"],
                               "blind_test_index": tile["blind_test_index"], "card_index": tile["card_index"],
                               "ground_truth_adjudicated": gt, "rank_ground_truth": gt[0],
                               "tiny_raw_response": item, "rank_prediction": rank or "UNRESOLVED",
                               "parse_status": parse_status, "verdict": verdict})
            if len(result) % 40 == 0:
                print(json.dumps({"completed": len(result), "total": len(source), "elapsed_s": round(time.perf_counter()-started, 2)}), flush=True)
    finally:
        client.close()
    result.sort(key=lambda row: row["prediction_index"])
    if [row["prediction_index"] for row in result] != list(range(len(source))):
        raise ValueError("Tiny result order/coverage differs")
    payload = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in result) + "\n").encode("utf-8")
    with (args.output / "raw_and_rank_predictions.jsonl").open("xb") as stream:
        stream.write(payload)
    counts = collections.Counter(row["verdict"] for row in result)
    by_rank = {}
    for rank in RANKS:
        subset = [row for row in result if row["rank_ground_truth"] == rank]
        by_rank["10" if rank == "T" else rank] = {
            "total": len(subset), "correct": sum(row["verdict"] == "CORRECT" for row in subset),
            "wrong": sum(row["verdict"] == "WRONG" for row in subset),
            "unresolved": sum(row["verdict"] == "UNRESOLVED" for row in subset),
            "raw_examples": list(dict.fromkeys(row["tiny_raw_response"].get("raw_text", "") for row in subset))[:12]}
    summary = {"status": "POST_BASELINE_TINY_SINGLE_CARD_DIAGNOSTIC", "total": len(result),
               "correct": counts["CORRECT"], "wrong": counts["WRONG"], "unresolved": counts["UNRESOLVED"],
               "rank_accuracy_all_unresolved_incorrect": counts["CORRECT"]/len(result),
               "by_rank": by_rank, "raw_predictions_sha256": sha(payload),
               "elapsed_s": time.perf_counter()-started,
               "note": "No rank guessed from context; raw responses preserved. Single CARD crops are outside the normal Tiny CHAT-row route."}
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("total", "correct", "wrong", "unresolved", "rank_accuracy_all_unresolved_incorrect", "elapsed_s")}, ensure_ascii=False))

if __name__ == "__main__":
    main()
