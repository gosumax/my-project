"""Bounded local CARD-tile trainability probe; does not change model weights."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

root = Path(__file__).resolve().parents[1]
config = json.loads((root / "configs/ocr_vl15_local.json").read_text())
model_dir = Path(config["model_dir"])
tile = json.loads((root / "runs/paddle_vl15_poker_sft_20260930/combined_verified_chat_card_tiles.jsonl").read_text(encoding="utf-8").splitlines()[0])
torch.set_num_threads(3)
result = {"tile_png": tile["tile_png"], "gpu": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory}
try:
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.float16, attn_implementation="sdpa", local_files_only=True).to("cuda")
    result["model_parameters"] = sum(p.numel() for p in model.parameters())
    result["load_reserved_bytes"] = torch.cuda.memory_reserved()
    result["load_allocated_bytes"] = torch.cuda.memory_allocated()
    result["linear_module_names_tail"] = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)][-25:]
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    with Image.open(tile["tile_png"]) as opened:
        img = opened.convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Read the playing card. Reply with one rank and suit code."}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", images_kwargs={"size": {"shortest_edge": processor.image_processor.min_pixels, "longest_edge": 1280 * 28 * 28}}).to("cuda")
    result["input_shapes"] = {key: list(value.shape) for key, value in inputs.items() if hasattr(value, "shape")}
    model.train()
    for param in model.parameters():
        param.requires_grad_(False)
    # The last normalization is a minimal trainable probe, not a training recipe.
    trainable = [(name, p) for name, p in model.named_parameters() if "norm" in name.lower()]
    name, param = trainable[-1]
    param.requires_grad_(True)
    result["trainable_probe_parameter"] = name
    out = model(**inputs, labels=inputs["input_ids"])
    result["loss"] = float(out.loss.detach().cpu())
    out.loss.backward()
    result["backward_ok"] = param.grad is not None
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
except Exception as exc:
    result["error_type"] = type(exc).__name__
    result["error"] = str(exc)[:1000]
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
sys.exit(0 if result.get("backward_ok") else 1)
