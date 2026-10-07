"""Isolated BOARD-VL quality probe on the fixed 32-crop Smoke14 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.visual_cards import parse_cards

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "runs/smoke14_board_vl_batch_benchmark_20260930_v2/dataset.json"
GT = ROOT / "runs/board_vl_quality_audit_20260930/ground_truth.json"
PROMPTS = {
    "A": "OCR:",
    "P": "Read the visible board cards from left to right. Return only rank and suit for each card.",
    "B": ("Return only visible community cards in rank+suit format separated by spaces. "
          "Valid ranks: 2-9,T,J,Q,K,A. Valid suits: s,h,d,c. Example: Js 9s 6c. "
          "Do not explain anything."),
    "C": ("Return only JSON with the visible community cards, left to right. "
          'Use ranks 2-9,T,J,Q,K,A and suits s,h,d,c. Example: {"cards":["Js","9s","6c"]}. '
          "Do not explain anything."),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepared(image: Image.Image, variant: str) -> Image.Image:
    image = image.convert("RGB")
    if variant == "baseline":
        return image
    if variant == "bottom_trim":
        return image.crop((0, 0, image.width, image.height - 9))
    if variant == "nearest2":
        return image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST)
    if variant == "bicubic2":
        return image.resize((image.width * 2, image.height * 2), Image.Resampling.BICUBIC)
    raise ValueError(variant)


def structured_cards(raw: str, expected: int) -> list[str] | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or set(obj) != {"cards"} or not isinstance(obj["cards"], list):
        return None
    cards = obj["cards"]
    if not all(isinstance(c, str) for c in cards):
        return None
    return parse_cards(" ".join(cards), expected)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/ocr_vl15_local.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt", choices=PROMPTS, required=True)
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--preprocess", choices=["baseline", "bottom_trim", "nearest2", "bicubic2"], default="baseline")
    p.add_argument("--batch-size", type=int, choices=[1, 4], default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(MANIFEST.read_text("utf-8"))["selected"]
    truth = json.loads(GT.read_text("utf-8"))["cards_by_manifest_index"]
    assert len(manifest) == len(truth) == 32
    config = json.loads(args.config.read_text("utf-8"))
    model_dir = Path(config["model_dir"])
    assert sha(model_dir / "model.safetensors") == config["weights_sha256"]
    for row in manifest:
        assert sha(Path(row["path"])) == row["sha256"]

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    assert torch.cuda.is_available()
    torch.set_num_threads(3)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.float16, attn_implementation="sdpa", local_files_only=True
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    assert {str(p.device) for p in model.parameters()} == {"cuda:0"}
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_ids.discard(None)
    pad_id = processor.tokenizer.pad_token_id
    records = []
    for offset in range(0, 32, args.batch_size):
        subset = manifest[offset:offset + args.batch_size]
        images = []
        messages = []
        for row in subset:
            with Image.open(row["path"]) as opened:
                im = prepared(opened, args.preprocess)
            images.append(im)
            messages.append([{"role": "user", "content": [
                {"type": "image", "image": im}, {"type": "text", "text": PROMPTS[args.prompt]}
            ]}])
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, padding=True,
            return_dict=True, return_tensors="pt",
            images_kwargs={"size": {"shortest_edge": processor.image_processor.min_pixels,
                                     "longest_edge": 1280 * 28 * 28}},
        ).to(model.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        generated = output[:, inputs["input_ids"].shape[1]:]
        grid = inputs.get("image_grid_thw")
        for j, (row, im) in enumerate(zip(subset, images)):
            ids = generated[j].tolist()
            first_eos = next((k for k, x in enumerate(ids) if x in eos_ids), None)
            if first_eos is not None:
                kept = ids[:first_eos + 1]
                stop = "EOS"
            else:
                kept = [x for x in ids if x != pad_id]
                stop = "TOKEN_LIMIT" if len(kept) >= args.max_new_tokens else "OTHER"
            raw = processor.decode(kept, skip_special_tokens=True).strip()
            expected = truth[offset + j]
            strict = parse_cards(raw, len(expected), row["street"])
            structured = structured_cards(raw, len(expected))
            normalized = strict or structured
            item = {
                "manifest_index": offset + j + 1, "sha256": row["sha256"], "path": row["path"],
                "table_id": row["table_id"], "frame_id": row["frame_id"], "street": row["street"],
                "original_size": Image.open(row["path"]).size, "model_input_size": im.size,
                "image_grid_thw": grid[j].tolist() if grid is not None else None,
                "prompt": PROMPTS[args.prompt], "preprocess": args.preprocess,
                "processor_size": {"shortest_edge": processor.image_processor.min_pixels,
                                   "longest_edge": 1280 * 28 * 28},
                "max_new_tokens": args.max_new_tokens, "output_tokens": len(kept),
                "stop_reason": stop, "raw_text": raw, "strict_cards": strict,
                "structured_cards": structured, "normalized_cards": normalized,
                "expected": expected, "full_board_exact": normalized == expected,
                "batch_seconds": elapsed, "batch_size": len(subset),
            }
            records.append(item)
            print(f"{offset+j+1:02d}/32 {raw!r} {stop} {len(kept)}t", flush=True)
    (args.output / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), "utf-8")
    (args.output / "run.json").write_text(json.dumps({
        "prompt_variant": args.prompt, "prompt": PROMPTS[args.prompt],
        "max_new_tokens": args.max_new_tokens, "preprocess": args.preprocess,
        "batch_size": args.batch_size, "model_revision": config["model_revision"],
        "weights_sha256": config["weights_sha256"], "manifest_sha256": sha(MANIFEST),
        "ground_truth_sha256": sha(GT),
        "counts": {"full_board_exact": sum(r["full_board_exact"] for r in records),
                   "strict_parse": sum(r["strict_cards"] is not None for r in records),
                   "structured_parse": sum(r["structured_cards"] is not None for r in records),
                   "token_limit": sum(r["stop_reason"] == "TOKEN_LIMIT" for r in records)},
        "generation_seconds": sum(records[i]["batch_seconds"] for i in range(0, 32, args.batch_size)),
    }, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()
