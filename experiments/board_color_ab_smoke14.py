"""Isolated RGB versus grayscale VL experiment on the fixed Smoke14 BOARD manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.visual_cards import parse_cards

MANIFEST = ROOT / "runs/smoke14_board_vl_batch_benchmark_20260930_v2/dataset.json"
EVIDENCE = ROOT / "runs/smoke14_board_sha_replay_v3_20260930/visual_evidence.jsonl"
TRUTH = ROOT / "runs/board_vl_quality_audit_20260930/ground_truth.json"
CONFIG = ROOT / "configs/ocr_vl15_local.json"
PROMPT = "Read the visible board cards from left to right. Return only rank and suit for each card."
BATCH = 4
TOKENS = 256


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gpu_used() -> int | None:
    try:
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
        ], text=True, timeout=5).splitlines()[0]
        return int(raw.strip())
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(MANIFEST.read_text("utf-8"))["selected"]
    truth_data = json.loads(TRUTH.read_text("utf-8"))
    truth = truth_data["cards_by_manifest_index"]
    evidence = [json.loads(line) for line in EVIDENCE.read_text("utf-8").splitlines()]
    evidence_by_sha = {row["crop_sha256"]: row for row in evidence if row.get("crop_type") == "BOARD"}
    assert len(manifest) == len(truth) == 32
    assert len(evidence_by_sha) == sum(row.get("crop_type") == "BOARD" for row in evidence)
    config = json.loads(CONFIG.read_text("utf-8"))
    model_dir = Path(config["model_dir"])
    with (model_dir / "model.safetensors").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == config["weights_sha256"]

    rows = []
    for index, (item, cards) in enumerate(zip(manifest, truth), 1):
        path = Path(item["path"])
        assert sha(path.read_bytes()) == item["sha256"]
        source = evidence_by_sha[item["sha256"]]
        assert source["frame_id"] == item["frame_id"]
        assert int(str(source["table_id"]).removeprefix("T")) == item["table_id"]
        assert source["street"] == item["street"]
        table_path = Path(source["source_table_path"])
        assert sha(table_path.read_bytes()) == source["source_table_sha256"]
        with Image.open(table_path) as opened:
            original = opened.convert("RGB").crop(tuple(source["crop_box_local"]))
        with Image.open(path) as opened:
            saved = opened.convert("RGB")
        assert original.size == saved.size and original.tobytes() == saved.tobytes()
        assert sha(saved.tobytes()) == source["board_sha"]
        assert len(cards) == {"FLOP": 3, "TURN": 4, "RIVER": 5}[item["street"]]
        rows.append({"manifest_index": index, "board_evidence_id": source["board_evidence_id"],
                     "source_sha256": item["sha256"], "source_pixel_sha256": source["board_sha"],
                     "source_table_sha256": source["source_table_sha256"],
                     "table_id": item["table_id"], "frame_id": item["frame_id"],
                     "street": item["street"], "crop_size": list(saved.size),
                     "ground_truth": cards, "ground_truth_status": truth_data["status"],
                     "source_path": str(path)})

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    assert torch.cuda.is_available()
    torch.set_num_threads(3)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.float16, attn_implementation="sdpa", local_files_only=True
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    assert {str(parameter.device) for parameter in model.parameters()} == {"cuda:0"}
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_ids.discard(None)
    pad_id = processor.tokenizer.pad_token_id

    def infer(variant: str, subset: list[dict], save: bool) -> tuple[float, list[dict], str]:
        messages = []
        for row in subset:
            with Image.open(row["source_path"]) as opened:
                original = opened.convert("RGB")
            image = original if variant == "CURRENT_RGB" else original.convert("L").convert("RGB")
            if save:
                input_path = args.output / "inputs" / variant / f"{row['manifest_index']:02d}.png"
                input_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(input_path)
                row[variant.lower() + "_input_path"] = str(input_path.resolve())
                row[variant.lower() + "_pixel_sha256"] = sha(image.tobytes())
                channel_bytes = [channel.tobytes() for channel in image.split()]
                row[variant.lower() + "_colored_pixel_count"] = sum(
                    r != g or g != b for r, g, b in zip(*channel_bytes))
            messages.append([{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": PROMPT}
            ]}])
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, padding=True,
            return_dict=True, return_tensors="pt",
            images_kwargs={"size": {"shortest_edge": processor.image_processor.min_pixels,
                                     "longest_edge": 1280 * 28 * 28}},
        ).to(model.device)
        tensor_sha = sha(inputs["pixel_values"].detach().cpu().numpy().tobytes())
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=TOKENS, do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        out = []
        for row, tokens in zip(subset, generated):
            ids = tokens.tolist()
            first_eos = next((i for i, token in enumerate(ids) if token in eos_ids), None)
            kept = ids[:first_eos + 1] if first_eos is not None else [x for x in ids if x != pad_id]
            stop = "EOS" if first_eos is not None else "TOKEN_LIMIT" if len(kept) >= TOKENS else "OTHER"
            raw = processor.decode(kept, skip_special_tokens=True).strip()
            parsed = parse_cards(raw, len(row["ground_truth"]), row["street"])
            out.append({"board_evidence_id": row["board_evidence_id"], "variant": variant,
                        "raw_text": raw, "parsed_cards": parsed, "stop_reason": stop,
                        "output_tokens": len(kept), "batch_seconds": elapsed,
                        "full_board_exact": parsed == row["ground_truth"]})
        return elapsed, out, tensor_sha

    summary = {}
    for variant in ("CURRENT_RGB", "GRAY_EXPERIMENT"):
        for _ in range(2):
            infer(variant, rows[:BATCH], False)
        torch.cuda.reset_peak_memory_stats()
        stop = threading.Event()
        samples = []

        def monitor():
            while not stop.is_set():
                samples.append(gpu_used())
                stop.wait(.25)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        results = []
        tensor_shas = []
        wall_start = time.perf_counter()
        for offset in range(0, 32, BATCH):
            elapsed, batch_results, tensor_sha = infer(variant, rows[offset:offset+BATCH], True)
            results.extend(batch_results)
            tensor_shas.append(tensor_sha)
            print(f"{variant}: {offset+BATCH}/32, batch {elapsed:.3f}s", flush=True)
        wall = time.perf_counter() - wall_start
        stop.set()
        thread.join(timeout=6)
        valid_samples = [x for x in samples if x is not None]
        (args.output / f"results_{variant}.jsonl").write_text(
            "".join(json.dumps(result, ensure_ascii=False) + "\n" for result in results), "utf-8")
        summary[variant] = {"wall_seconds": wall, "sec_per_crop": wall / 32,
                            "generation_seconds": sum(results[i]["batch_seconds"] for i in range(0, 32, BATCH)),
                            "peak_vram_mib": max(valid_samples) if valid_samples else None,
                            "peak_torch_allocated_mib": torch.cuda.max_memory_allocated() / 1048576,
                            "processor_batch_tensor_sha256": tensor_shas,
                            "valid_parse": sum(r["parsed_cards"] is not None for r in results),
                            "full_board_exact": sum(r["full_board_exact"] for r in results),
                            "stop_reasons": dict(Counter(r["stop_reason"] for r in results))}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")

    assert all(row["current_rgb_pixel_sha256"] == row["source_pixel_sha256"] for row in rows)
    assert all(row["gray_experiment_colored_pixel_count"] == 0 for row in rows)
    (args.output / "dataset.json").write_text(json.dumps({
        "source_manifest_sha256": sha(MANIFEST.read_bytes()),
        "source_evidence_sha256": sha(EVIDENCE.read_bytes()),
        "ground_truth_sha256": sha(TRUTH.read_bytes()),
        "model_revision": config["model_revision"], "weights_sha256": config["weights_sha256"],
        "prompt": PROMPT, "max_new_tokens": TOKENS, "do_sample": False,
        "use_cache": True, "batch": BATCH, "explicit_resize": None,
        "processor_min_pixels": processor.image_processor.min_pixels,
        "processor_max_pixels": 1280 * 28 * 28, "rows": rows,
    }, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()
