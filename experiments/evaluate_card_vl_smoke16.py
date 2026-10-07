"""One-shot blind Smoke16 CARD predictions after validation-only selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from experiments.train_card_vl_lora import attach_lora, CARD_IMAGE_PIXELS, PROMPT, DATA, SPLIT, read_rows

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-output", type=Path, default=DATA / "train_run")
    args = ap.parse_args()
    test_path = DATA / "smoke16_test_manifest.jsonl"
    freeze = json.loads((DATA / "smoke16_gt_freeze.json").read_text(encoding="utf-8"))
    if sha(test_path.read_bytes()) != freeze["test_manifest_sha256"]:
        raise ValueError("Smoke16 GT freeze changed")
    test = read_rows(test_path)
    selection_path = args.train_output / "checkpoint_selection_final.json"
    selection_raw = selection_path.read_bytes()
    selection = json.loads(selection_raw)
    config = json.loads((args.train_output / "train_config.json").read_text(encoding="utf-8"))
    if selection.get("status") != "FINAL_VALIDATION_SELECTION_FROZEN" or selection.get("smoke16_seen") is not False or config["max_train"] is not None or config["max_validation"] is not None:
        raise ValueError("Checkpoint was not selected blind on the full train/validation sets")
    if selection["completed_training_log_sha256"] != sha((args.train_output / "training_log.jsonl").read_bytes()) or selection["train_config_sha256"] != sha((args.train_output / "train_config.json").read_bytes()):
        raise ValueError("Completed training evidence changed")
    if config["prompt"] != PROMPT or config["card_image_pixels"] != CARD_IMAGE_PIXELS:
        raise ValueError("Prompt or preprocessing changed")
    if config["train_manifest_sha256"] != sha((SPLIT / "train_manifest.jsonl").read_bytes()) or config["validation_manifest_sha256"] != sha((SPLIT / "validation_manifest.jsonl").read_bytes()):
        raise ValueError("Train/validation manifest changed")
    train_val_pixels = {row["tile_pixel_sha256"] for name in ("train", "validation") for row in read_rows(SPLIT / f"{name}_manifest.jsonl")}
    if any(row["tile_pixel_sha256"] in train_val_pixels or Path(row["source_video"]).name != "smoke_16.mp4" for row in test):
        raise ValueError("TEST pixel/source leakage")
    predictions_path = DATA / "smoke16_blind_predictions.jsonl"
    if predictions_path.exists():
        raise FileExistsError("Blind predictions already frozen; refusing second run")
    checkpoint = Path(selection["selected_checkpoint"])
    checkpoint_sha = sha(checkpoint.read_bytes())
    if checkpoint_sha != selection["selected_checkpoint_sha256"] or selection["all_checkpoints"][str(selection["best_epoch"])]["sha256"] != checkpoint_sha:
        raise ValueError("Selected adapter bytes differ from validation-pinned checkpoint")
    weights = Path(config["model_dir"]) / "model.safetensors"
    with weights.open("rb") as stream:
        actual_weights_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_weights_sha != config["actual_weights_sha256"]:
        raise ValueError("Base model weights changed")
    torch.set_num_threads(3)
    model = AutoModelForImageTextToText.from_pretrained(config["model_dir"], dtype=torch.float16,
                                                         attn_implementation="sdpa", local_files_only=True).to("cuda")
    processor = AutoProcessor.from_pretrained(config["model_dir"], local_files_only=True, use_fast=False)
    attach_lora(model, rank=config["lora_rank"], alpha=config["lora_alpha"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != set(trainable):
        raise ValueError("Adapter keys do not match")
    for name, parameter in trainable.items():
        parameter.data.copy_(state[name].to(parameter.device))
    model.eval()
    claim_path = DATA / "smoke16_inference_claim.json"
    claim = {"status": "ONE_SHOT_CLAIMED_BEFORE_FIRST_TEST_INFERENCE",
             "test_manifest_sha256": sha(test_path.read_bytes()),
             "selection_sha256": sha(selection_raw), "checkpoint_sha256": checkpoint_sha}
    with claim_path.open("x", encoding="utf-8") as stream:
        json.dump(claim, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    started = time.perf_counter()
    predictions = []
    for index, row in enumerate(test):
        raw = Path(row["tile_png"]).read_bytes()
        if sha(raw) != row["tile_sha256"]:
            raise ValueError("TEST PNG SHA changed")
        with Image.open(row["tile_png"]) as opened:
            image = opened.convert("RGB")
        if sha(image.tobytes()) != row["tile_pixel_sha256"]:
            raise ValueError("TEST pixels changed")
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                    {"type": "text", "text": PROMPT}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            images_kwargs={"size": {"shortest_edge": CARD_IMAGE_PIXELS,
                                     "longest_edge": CARD_IMAGE_PIXELS}},
        ).to("cuda")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True)
        token_ids = output[0][inputs["input_ids"].shape[1]:].tolist()
        raw_text = processor.decode(token_ids, skip_special_tokens=True).strip()
        generation_status = "TOKEN_LIMIT" if len(token_ids) >= 8 else "EMPTY" if not raw_text else "OK"
        predicted = raw_text if re.fullmatch(r"[2-9TJQKA][cdhs]", raw_text) else "UNRESOLVED"
        verdict = "UNRESOLVED" if predicted == "UNRESOLVED" else "CORRECT" if predicted == row["ground_truth"] else "WRONG"
        predictions.append({**row, "prediction": predicted, "raw_response": raw_text,
                            "generated_token_ids": token_ids, "generation_status": generation_status,
                            "verdict": verdict,
                            "selected_checkpoint_sha256": checkpoint_sha,
                            "prompt": PROMPT, "prediction_index": index})
        del inputs, output
        if (index+1) % 25 == 0:
            print(json.dumps({"predicted": index+1, "total": len(test), "elapsed_s": round(time.perf_counter()-started, 2)}), flush=True)
    payload = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in predictions) + "\n").encode("utf-8")
    with predictions_path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {"status": "FIRST_BLIND_PREDICTIONS_FROZEN", "predictions_sha256": sha(payload),
               "test_manifest_sha256": sha(test_path.read_bytes()), "checkpoint_selection_sha256": sha(selection_raw),
               "selected_checkpoint_sha256": checkpoint_sha, "count": len(predictions),
               "elapsed_s": time.perf_counter()-started}
    with (DATA / "smoke16_blind_prediction_receipt.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
