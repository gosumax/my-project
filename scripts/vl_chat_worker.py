"""Persistent, offline PaddleOCR-VL-1.5 row reader (run in its own venv)."""

from __future__ import annotations

import argparse
import base64
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


def request_image_bytes(request: dict) -> bytes:
    if ("path" in request) == ("png_base64" in request):
        raise ValueError("VL request must contain exactly one image transport")
    if "png_base64" in request:
        source = base64.b64decode(request["png_base64"], validate=True)
    else:
        source = Path(request["path"]).resolve(strict=True).read_bytes()
    if hashlib.sha256(source).hexdigest() != request["source_sha256"]:
        raise ValueError("Source crop SHA256 mismatch")
    return source


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
            self.memory_hit = False
            self.free_min = torch.cuda.mem_get_info()[0]

        def __call__(self, input_ids, scores, **kwargs):
            self.hit = time.perf_counter() > self.end
            self.free_min = min(self.free_min, torch.cuda.mem_get_info()[0])
            self.memory_hit = self.free_min < 768*1048576
            return self.hit or self.memory_hit

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
    emit({"type": "ready", "model_dir": str(model_dir),
          "startup_s": time.perf_counter()-startup_started})

    def infer(requests):
        if not 1 <= len(requests) <= 8:
            raise ValueError("VL batch must contain 1..8 images")
        messages = []
        for request in requests:
            source = request_image_bytes(request)
            with Image.open(io.BytesIO(source)) as opened:
                image = opened.convert("RGB")
            board_sha = request.get("board_sha")
            if board_sha is not None and hashlib.sha256(image.tobytes()).hexdigest() != board_sha:
                raise ValueError("BOARD pixel SHA256 mismatch")
            trim = int(request.get("right_trim_px", 12))
            if not 0 <= trim < image.width:
                raise ValueError("Invalid right-edge trim")
            image = image.crop((0, 0, image.width-trim, image.height))
            prompt = request.get("prompt", "OCR:")
            if not isinstance(prompt, str) or not prompt or len(prompt) > 500:
                raise ValueError("Invalid VL prompt")
            messages.append([{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt}]}])
        start = time.perf_counter()
        kwargs = {"add_generation_prompt": True, "tokenize": True,
            "return_dict": True, "return_tensors": "pt", "images_kwargs": {"size": {
                "shortest_edge": processor.image_processor.min_pixels,
                "longest_edge": 1280*28*28}}}
        if len(requests) > 1:
            kwargs["padding"] = True
        inputs = processor.apply_chat_template(
            messages[0] if len(requests) == 1 else messages, **kwargs).to(model.device)
        deadline = Deadline(120)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, use_cache=True,
                stopping_criteria=StoppingCriteriaList([deadline]))
        generated = outputs[:, inputs["input_ids"].shape[-1]:]
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        duration = time.perf_counter()-start
        results = []
        for request, sequence in zip(requests, generated):
            tokens = sequence.tolist()
            if len(requests) > 1:
                end = next((i+1 for i, token in enumerate(tokens) if token in eos), len(tokens))
                tokens = tokens[:end]
            text = processor.decode(tokens, skip_special_tokens=True).strip()
            status = ("MEMORY_GUARD" if deadline.memory_hit else "TIME_LIMIT" if deadline.hit else
                      "TOKEN_LIMIT" if len(tokens) >= args.max_new_tokens else "OK" if text else "EMPTY")
            results.append({"type": "result", "request_id": request["request_id"],
                "status": status, "raw_text": text, "source_sha256": request["source_sha256"],
                "board_sha": request.get("board_sha"), "board_evidence_id": request.get("board_evidence_id"),
                "elapsed_s": duration/len(requests), "batch_elapsed_s": duration,
                "batch_size": len(requests), "new_tokens": len(tokens),
                "stop_reason": "EOS" if tokens and tokens[-1] in eos else status,
                "cuda_free_min_MiB": deadline.free_min/1048576})
        del inputs, outputs, generated
        return results

    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if request.get("type") == "batch":
                emit({"type": "batch_result", "batch_id": request["batch_id"],
                      "results": infer(request["requests"])})
            else:
                emit(infer([request])[0])
        except Exception:
            error = traceback.format_exc()[-4000:]
            if request.get("type") == "batch":
                emit({"type": "batch_result", "batch_id": request.get("batch_id"),
                    "results": [{"type": "result", "request_id": item.get("request_id"),
                        "status": "ERROR", "raw_text": "", "error": error,
                        "source_sha256": item.get("source_sha256")}
                        for item in request.get("requests", [])]})
                torch.cuda.empty_cache()
                continue
            emit({"type": "result", "request_id": request.get("request_id"),
                  "status": "ERROR", "raw_text": "", "error": error,
                  "source_sha256": request.get("source_sha256"),
                  "board_sha": request.get("board_sha"),
                  "board_evidence_id": request.get("board_evidence_id")})


if __name__ == "__main__":
    main()
