"""Bounded current-production CHAT OCR/VL rank probe on saved Smoke16 rows.

Uses configured, unmodified workers and the same whole-row geometry/prompt as
the existing production paths. All raw responses and source hashes are kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from parser_core.board_tiles import read_board_tiles, _RANK_TOKEN
from parser_core.tiny_chat_client import TinyChatClient
from parser_core.visual_cards import parse_cards
from parser_core.vl_chat_client import VLChatClient

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/card_train_smoke16_test_20260930"
DEFAULT_OUT = DATA / "current_production_rank_probe"

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def read(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

def corrected_sequences():
    rows = read(DATA / "smoke16_blind_review_manifest.jsonl")
    reviews = read(DATA / "smoke16_visual_review_frozen.jsonl")
    if len(rows) != 60 or len(reviews) != 60:
        raise ValueError("Expected 60 frozen CHAT rows")
    mapping = {r["blind_test_index"]: r["ground_truth"].split() for r in reviews}
    user = json.loads((DATA / "stage2_template/user_adjudication_A_B.json").read_text(encoding="utf-8"))
    if user["status"] != "USER_VISUAL_ADJUDICATION_AFTER_FIRST_BLIND_RUN":
        raise ValueError("Expected user adjudication addendum")
    for decision in user["decisions"]:
        i, j = decision["blind_test_index"], decision["card_index"]
        if mapping[i][j] != decision["original_ground_truth"]:
            raise ValueError("Original GT differs from user review")
        mapping[i][j] = decision["user_visual_reading"]
    return rows, mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=60)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if not 1 <= args.max_rows <= 60 or args.output.exists():
        ap.error("Use 1..60 rows and a fresh output directory")
    rows, gt = corrected_sequences()
    rows = rows[:args.max_rows]
    tiny_cfg = json.loads((ROOT / "configs/ocr_tiny_local.json").read_text(encoding="utf-8"))
    vl_cfg = json.loads((ROOT / "configs/ocr_vl15_local.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True)
    config = {"status": "CURRENT_PRODUCTION_CONFIG_DIAGNOSTIC", "rows": len(rows),
              "source_manifest_sha256": sha((DATA / "smoke16_blind_review_manifest.jsonl").read_bytes()),
              "user_adjudication_sha256": sha((DATA / "stage2_template/user_adjudication_A_B.json").read_bytes()),
              "tiny_config_sha256": sha((ROOT / "configs/ocr_tiny_local.json").read_bytes()),
              "vl_config_sha256": sha((ROOT / "configs/ocr_vl15_local.json").read_bytes()),
              "tiny_model": tiny_cfg["model_name"], "tiny_batch_size": tiny_cfg["batch_size"],
              "vl_model_revision": vl_cfg["model_revision"], "vl_prompt": "OCR:",
              "vl_right_trim_px": 12, "lora_adapter_loaded": False,
              "source": "SAVED_SMOKE16_CHAT_ROWS_POST_BASELINE_USER_ADJUDICATED_GT",
              "note": "No video decode or production journal mutation; unmodified configured model paths."}
    (args.output / "probe_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tiny = TinyChatClient(Path(tiny_cfg["python"]), Path(tiny_cfg["model_dir"]),
                          tiny_cfg["model_name"], tiny_cfg["weights_sha256"],
                          tiny_cfg["device"], args.output / "tiny_worker.stderr.log")
    vl = VLChatClient(Path(vl_cfg["python"]), Path(vl_cfg["model_dir"]),
                      vl_cfg["weights_sha256"], args.output / "vl_worker.stderr.log")
    started = time.perf_counter()
    records = []
    try:
        tiny_by_index = {}
        for start in range(0, len(rows), tiny_cfg["batch_size"]):
            batch = rows[start:start + tiny_cfg["batch_size"]]
            items = []
            for row in batch:
                path = Path(row["chat_row_png"])
                payload = path.read_bytes()
                if sha(payload) != row["chat_row_sha256"]:
                    raise ValueError(f"CHAT PNG hash changed: {path}")
                items.append({"item_id": str(row["blind_test_index"]),
                              "path": str(path), "source_sha256": row["chat_row_sha256"]})
            result = tiny.read_batch(items, tiny_cfg["batch_size"])
            for item in result["results"]:
                tiny_by_index[int(item["item_id"])] = item
        if len(tiny_by_index) != len(rows):
            raise ValueError("Tiny OCR result count differs")
        with (args.output / "raw_responses.jsonl").open("x", encoding="utf-8") as stream:
            for sequence, row in enumerate(rows, 1):
                index = row["blind_test_index"]
                path = Path(row["chat_row_png"])
                tiny_result = tiny_by_index[index]
                vl_result = vl.read(path, row["chat_row_sha256"], 12)
                expected = gt[index]
                tiny_cards = parse_cards(tiny_result.get("raw_text", ""), len(expected)) if tiny_result.get("status") == "OK" else None
                vl_cards = parse_cards(vl_result.get("raw_text", ""), len(expected)) if vl_result.get("status") == "OK" else None
                pixel_reading = (read_board_tiles(path, row["chat_row_sha256"], vl_result.get("raw_text", ""))
                                 if vl_result.get("status") == "OK" else None)
                pixel_cards = [card.replace("10", "T") for card in pixel_reading.cards] if pixel_reading else None
                raw_ranks = _RANK_TOKEN.findall(vl_result.get("raw_text", "").upper())
                raw_ranks = ["T" if rank == "10" else rank for rank in raw_ranks]
                record = {"blind_test_index": index, "source_video_sha256": row["source_video_sha256"],
                          "source_pts": row["source_pts"], "table_id": row["table_id"],
                          "source_span_index": row["source_span_index"], "chat_row_png": str(path),
                          "chat_row_sha256": row["chat_row_sha256"], "ground_truth_adjudicated": expected,
                          "tiny_response": tiny_result, "tiny_strict_cards": tiny_cards,
                          "vl_response": vl_result, "vl_strict_cards": vl_cards,
                          "vl_rank_tokens_diagnostic": raw_ranks,
                          "board_tile_cards": pixel_cards,
                          "board_tile_rank_source": pixel_reading.rank_source if pixel_reading else None,
                          "board_tile_template_sha256": pixel_reading.template_sha256 if pixel_reading else None}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(record)
                if sequence % 10 == 0:
                    print(json.dumps({"rows_completed": sequence, "total": len(rows),
                                      "elapsed_s": round(time.perf_counter()-started, 2)}), flush=True)
    finally:
        tiny.close()
        vl.close()
    def measures(field):
        accepted = [r for r in records if r[field] is not None]
        aligned = [r for r in accepted if len(r[field]) == len(r["ground_truth_adjudicated"])]
        cards = sum(len(r["ground_truth_adjudicated"]) for r in records)
        correct = sum(a[0] == b[0] for r in aligned for a,b in zip(r[field], r["ground_truth_adjudicated"]))
        return {"accepted_rows": len(accepted), "aligned_rows": len(aligned),
                "accepted_cards": sum(len(r["ground_truth_adjudicated"]) for r in aligned),
                "correct_ranks_on_aligned": correct,
                "rank_accuracy_all_cards_unresolved_incorrect": correct/cards,
                "rank_accuracy_on_aligned_cards": correct/sum(len(r["ground_truth_adjudicated"]) for r in aligned) if aligned else None}
    summary = {"status": "DIAGNOSTIC_NOT_CERTIFIED_HH", "rows": len(records),
               "cards": sum(len(r["ground_truth_adjudicated"]) for r in records),
               "tiny_strict": measures("tiny_strict_cards"),
               "vl_strict": measures("vl_strict_cards"),
               "board_tile_production": measures("board_tile_cards"),
               "vl_rank_tokens_diagnostic": measures("vl_rank_tokens_diagnostic"),
               "vl_status_counts": {status: sum(r["vl_response"].get("status") == status for r in records)
                                    for status in sorted({r["vl_response"].get("status") for r in records})},
               "raw_responses_sha256": sha((args.output / "raw_responses.jsonl").read_bytes()),
               "elapsed_s": time.perf_counter()-started,
               "note": "Strict readers abstain unless card identities and count parse; rank-token-only measure is diagnostic, not an accepted card reading."}
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
