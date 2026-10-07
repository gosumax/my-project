"""Isolated LoRA training on verified CHAT CARD tiles; Smoke16 is never opened here."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/card_train_smoke16_test_20260930"
SPLIT = DATA / "split_v2"
PROMPT = "Read this single playing card. Reply with exactly one code: rank 2-9,T,J,Q,K,A followed by suit c,d,h,s."
CARD_IMAGE_PIXELS = 12544

class LowRankLinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.a = torch.nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)

    def forward(self, x):
        primary = self.base(x)
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x.float(), self.a), self.b)
        return primary + (delta * self.scale).to(primary.dtype)

def attach_lora(model, rank: int = 4, alpha: int = 8, last_layers: int = 4):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = model.model.language_model.layers
    names = []
    for index in range(len(blocks)-last_layers, len(blocks)):
        for proj in ("q_proj", "v_proj"):
            base = getattr(blocks[index].self_attn, proj)
            setattr(blocks[index].self_attn, proj, LowRankLinear(base, rank, alpha).to(base.weight.device))
            names.append(f"model.language_model.layers.{index}.self_attn.{proj}")
    return names

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

def make_inputs(processor, row: dict, device):
    raw = Path(row["tile_png"]).read_bytes()
    if sha(raw) != row["tile_sha256"]:
        raise ValueError("Training/validation tile SHA mismatch")
    with Image.open(row["tile_png"]) as opened:
        image = opened.convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        images_kwargs={"size": {"shortest_edge": CARD_IMAGE_PIXELS,
                                 "longest_edge": CARD_IMAGE_PIXELS}},
    )
    target = processor.tokenizer(row["ground_truth"], add_special_tokens=False, return_tensors="pt")["input_ids"]
    eos = torch.tensor([[processor.tokenizer.eos_token_id]], dtype=target.dtype)
    target = torch.cat((target, eos), dim=1)
    prefix = inputs["input_ids"].shape[1]
    inputs["input_ids"] = torch.cat((inputs["input_ids"], target), dim=1)
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(target)), dim=1)
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], torch.zeros_like(target)), dim=1)
    labels = torch.full_like(inputs["input_ids"], -100)
    labels[:, prefix:] = target
    inputs["labels"] = labels
    return inputs.to(device)

def adapter_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-train", type=int)
    ap.add_argument("--max-validation", type=int)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260930)
    random.seed(20260930)
    np.random.seed(20260930)
    torch.set_num_threads(3)
    config = json.loads((ROOT / "configs/ocr_vl15_local.json").read_text(encoding="utf-8"))
    train_path, valid_path = SPLIT / "train_manifest.jsonl", SPLIT / "validation_manifest.jsonl"
    train, valid = read_rows(train_path), read_rows(valid_path)
    if args.max_train is not None:
        train = train[:args.max_train]
    if args.max_validation is not None:
        valid = valid[:args.max_validation]
    if any(x["source_video"] == "smoke_16.mp4" for x in train + valid):
        raise ValueError("Smoke16 leakage")
    run_config = {"model_dir": config["model_dir"], "model_revision": config["model_revision"],
                  "weights_sha256": config["weights_sha256"], "train_manifest_sha256": sha(train_path.read_bytes()),
                  "validation_manifest_sha256": sha(valid_path.read_bytes()), "prompt": PROMPT,
                  "epochs": args.epochs, "learning_rate": 0.0002, "optimizer": "AdamW",
                  "seed": 20260930, "lora_rank": 4, "lora_alpha": 8,
                  "lora_targets": "last_4_language_layers_q_proj_v_proj", "batch_size": 1,
                  "gradient_accumulation": 1, "max_train": args.max_train,
                  "max_validation": args.max_validation,
                  "checkpoint_selection": "lowest_validation_token_cross_entropy_including_base"}
    run_config["card_image_pixels"] = CARD_IMAGE_PIXELS
    (args.output / "train_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    start = time.perf_counter()
    weights = Path(config["model_dir"]) / "model.safetensors"
    with weights.open("rb") as stream:
        actual_weights_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_weights_sha256 != config["weights_sha256"]:
        raise ValueError("PaddleOCR-VL base weights SHA256 mismatch")
    run_config["actual_weights_sha256"] = actual_weights_sha256
    model = AutoModelForImageTextToText.from_pretrained(config["model_dir"], dtype=torch.float16,
                                                         attn_implementation="sdpa", local_files_only=True).to("cuda")
    processor = AutoProcessor.from_pretrained(config["model_dir"], local_files_only=True, use_fast=False)
    names = attach_lora(model)
    run_config["attached_modules"] = names
    (args.output / "train_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0002, weight_decay=0.01)
    log_path = args.output / "training_log.jsonl"
    def log(item):
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(json.dumps(item, ensure_ascii=False), flush=True)
    def validate(epoch):
        model.eval()
        losses = []
        with torch.inference_mode():
            for i, row in enumerate(valid):
                inputs = make_inputs(processor, row, "cuda")
                out = model(**inputs)
                losses.append(float(out.loss.detach().cpu()))
                del inputs, out
        mean = sum(losses) / len(losses)
        log({"kind": "validation", "epoch": epoch, "rows": len(losses), "mean_token_loss": mean,
             "elapsed_s": round(time.perf_counter()-start, 2)})
        return mean
    best_loss = validate(0)
    best_epoch = 0
    torch.save(adapter_state(model), args.output / "checkpoint_base_adapter.pt")
    for epoch in range(1, args.epochs+1):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        train_losses = []
        for step, idx in enumerate(order, 1):
            optimizer.zero_grad(set_to_none=True)
            inputs = make_inputs(processor, train[idx], "cuda")
            out = model(**inputs)
            loss = out.loss
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            del inputs, out, loss
            if step % 25 == 0 or step == len(order):
                log({"kind": "train_progress", "epoch": epoch, "step": step, "steps": len(order),
                     "mean_loss_so_far": sum(train_losses)/len(train_losses),
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "elapsed_s": round(time.perf_counter()-start, 2)})
        checkpoint = args.output / f"checkpoint_epoch_{epoch}.pt"
        torch.save(adapter_state(model), checkpoint)
        val_loss = validate(epoch)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
        (args.output / "checkpoint_selection.json").write_text(json.dumps(
            {"best_epoch": best_epoch, "best_validation_token_loss": best_loss,
             "selection_rule": "minimum_validation_token_loss_including_base",
             "selected_checkpoint": str(args.output / ("checkpoint_base_adapter.pt" if best_epoch == 0 else f"checkpoint_epoch_{best_epoch}.pt")),
             "smoke16_seen": False}, indent=2) + "\n", encoding="utf-8")
    log({"kind": "finished", "best_epoch": best_epoch, "best_validation_loss": best_loss,
         "elapsed_s": round(time.perf_counter()-start, 2),
         "peak_allocated_bytes": torch.cuda.max_memory_allocated()})

if __name__ == "__main__":
    main()
