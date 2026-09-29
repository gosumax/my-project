"""Persistent, offline PaddleOCR-VL-1.5 row reader (run in its own venv)."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    startup_started = time.perf_counter()
    model_dir = args.model_dir.resolve(strict=True)
    weights = model_dir / "model.safetensors"
    with weights.open("rb") as stream:
        actual_weights_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_weights_sha256 != args.weights_sha256:
        raise ValueError("PaddleOCR-VL weights SHA256 mismatch")
    print(f"weights_sha256_verified_s={time.perf_counter() - startup_started:.3f}",
          file=sys.stderr, flush=True)

    import torch
    from PIL import Image
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              StoppingCriteria, StoppingCriteriaList)

    class Deadline(StoppingCriteria):
        def __init__(self, seconds: int):
            self.end = time.perf_counter() + seconds
            self.hit = False

        def __call__(self, input_ids, scores, **kwargs):
            self.hit = time.perf_counter() > self.end
            return self.hit

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for PaddleOCR-VL")
    torch.set_num_threads(3)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.float16, attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    print(f"model_ready_s={time.perf_counter() - startup_started:.3f}",
          file=sys.stderr, flush=True)
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    emit({"type": "ready", "model_dir": str(model_dir)})
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            path = Path(request["path"]).resolve(strict=True)
            source = path.read_bytes()
            if hashlib.sha256(source).hexdigest() != request["source_sha256"]:
                raise ValueError("Source crop SHA256 mismatch")
            with Image.open(io.BytesIO(source)) as opened:
                image = opened.convert("RGB")
            trim = int(request.get("right_trim_px", 12))
            if not 0 <= trim < image.width:
                raise ValueError("Invalid right-edge trim")
            image = image.crop((0, 0, image.width - trim, image.height))
            start = time.perf_counter()
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": "OCR:"},
            ]}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
                images_kwargs={"size": {
                    "shortest_edge": processor.image_processor.min_pixels,
                    "longest_edge": 1280 * 28 * 28,
                }},
            ).to(model.device)
            deadline = Deadline(120)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, use_cache=True,
                    stopping_criteria=StoppingCriteriaList([deadline]),
                )
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            text = processor.decode(generated, skip_special_tokens=True).strip()
            status = ("TIME_LIMIT" if deadline.hit else
                      "TOKEN_LIMIT" if len(generated) >= args.max_new_tokens else
                      "OK" if text else "EMPTY")
            emit({"type": "result", "request_id": request["request_id"],
                  "status": status, "raw_text": text,
                  "elapsed_s": time.perf_counter() - start,
                  "new_tokens": len(generated)})
            del inputs, outputs, generated
        except Exception:
            emit({"type": "result", "request_id": request.get("request_id"),
                  "status": "ERROR", "raw_text": "", "error": traceback.format_exc()[-4000:]})


if __name__ == "__main__":
    main()
