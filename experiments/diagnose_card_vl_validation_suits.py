"""Post-baseline generation on VALIDATION only to diagnose suit bias."""
import collections
import hashlib
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from experiments.train_card_vl_lora import attach_lora, CARD_IMAGE_PIXELS, DATA, PROMPT, SPLIT, read_rows

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main():
    dest = DATA / "postbaseline_validation_predictions.jsonl"
    if dest.exists():
        raise FileExistsError("Validation diagnostic already exists")
    config = json.loads((DATA / "train_run/train_config.json").read_text(encoding="utf-8"))
    selection = json.loads((DATA / "train_run/checkpoint_selection_final.json").read_text(encoding="utf-8"))
    baseline = json.loads((DATA / "smoke16_blind_prediction_receipt.json").read_text(encoding="utf-8"))
    if baseline["status"] != "FIRST_BLIND_PREDICTIONS_FROZEN":
        raise ValueError("Smoke16 baseline must precede this diagnostic")
    checkpoint = Path(selection["selected_checkpoint"])
    if sha(checkpoint.read_bytes()) != selection["selected_checkpoint_sha256"]:
        raise ValueError("Selected adapter changed")
    rows = read_rows(SPLIT / "validation_manifest.jsonl")
    torch.set_num_threads(3)
    model = AutoModelForImageTextToText.from_pretrained(config["model_dir"], dtype=torch.float16,
                                                         attn_implementation="sdpa", local_files_only=True).to("cuda")
    processor = AutoProcessor.from_pretrained(config["model_dir"], local_files_only=True, use_fast=False)
    attach_lora(model, rank=config["lora_rank"], alpha=config["lora_alpha"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.data.copy_(state[name].to(parameter.device))
    model.eval()
    start = time.perf_counter()
    predictions = []
    for i, row in enumerate(rows):
        with Image.open(row["tile_png"]) as opened:
            image = opened.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt",
                                               images_kwargs={"size": {"shortest_edge": CARD_IMAGE_PIXELS,
                                                                        "longest_edge": CARD_IMAGE_PIXELS}}).to("cuda")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True)
        raw = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred = raw if re.fullmatch(r"[2-9TJQKA][cdhs]", raw) else "UNRESOLVED"
        predictions.append({"ground_truth": row["ground_truth"], "prediction": pred, "raw_response": raw,
                            "source_video": row["source_video"], "tile_sha256": row["tile_sha256"],
                            "checkpoint_sha256": selection["selected_checkpoint_sha256"]})
        if (i+1) % 50 == 0:
            print(json.dumps({"done": i+1, "total": len(rows), "elapsed_s": time.perf_counter()-start}), flush=True)
    payload = ("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in predictions) + "\n").encode("utf-8")
    with dest.open("xb") as stream:
        stream.write(payload)
    correct = sum(x["prediction"] == x["ground_truth"] for x in predictions)
    confusion = collections.Counter((x["ground_truth"][1], x["prediction"][1] if x["prediction"] != "UNRESOLVED" else "U") for x in predictions)
    summary = {"status": "POST_BASELINE_VALIDATION_DIAGNOSTIC", "count": len(predictions),
               "correct": correct, "exact_accuracy": correct/len(predictions),
               "suit_confusion": {f"{a}->{b}": n for (a,b),n in sorted(confusion.items())},
               "predictions_sha256": sha(payload), "elapsed_s": time.perf_counter()-start}
    (DATA / "postbaseline_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
