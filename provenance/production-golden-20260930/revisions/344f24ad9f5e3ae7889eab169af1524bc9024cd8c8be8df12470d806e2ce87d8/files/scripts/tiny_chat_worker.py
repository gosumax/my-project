"""Persistent single-pass PP-OCRv6 tiny recognition worker."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    startup = time.perf_counter()
    model_dir = args.model_dir.resolve(strict=True)
    weights = model_dir / "inference.pdiparams"
    with weights.open("rb") as stream:
        actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_hash != args.weights_sha256:
        raise ValueError("PP-OCRv6 tiny weights SHA256 mismatch")

    from paddleocr import TextRecognition

    model = TextRecognition(model_name=args.model_name, device=args.device)
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    emit({"type": "ready", "model_name": args.model_name,
          "model_dir": str(model_dir), "device": args.device,
          "startup_seconds": time.perf_counter() - startup})
    for line in sys.stdin:
        request: dict = {}
        try:
            request = json.loads(line)
            items = request["items"]
            batch_size = int(request["batch_size"])
            if not items or batch_size < 1 or len(items) > batch_size:
                raise ValueError("Invalid recognition batch")
            paths = []
            images = []
            ram_mode = "png_base64" in items[0]
            if any(("png_base64" in item) != ram_mode for item in items):
                raise ValueError("Mixed RAM and path batch")
            for item in items:
                if ram_mode:
                    packed = base64.b64decode(item["png_base64"], validate=True)
                    image = cv2.imdecode(np.frombuffer(packed, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("Invalid RAM PNG")
                    actual = hashlib.sha256(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).tobytes()).hexdigest()
                    if actual != item["source_sha256"]:
                        raise ValueError("RAM row pixel SHA256 mismatch")
                    images.append(image)
                else:
                    path = Path(item["path"]).resolve(strict=True)
                    with path.open("rb") as stream:
                        actual = hashlib.file_digest(stream, "sha256").hexdigest()
                    if actual != item["source_sha256"]:
                        raise ValueError(f"Source crop SHA256 mismatch: {path}")
                    paths.append(str(path))
            inference_started = time.perf_counter()
            predictions = list(model.predict(images if ram_mode else paths, batch_size=batch_size))
            inference_seconds = time.perf_counter() - inference_started
            by_path = ({str(Path(result["input_path"]).resolve()).lower(): result
                        for result in predictions} if not ram_mode else {})
            results = []
            for index, item in enumerate(items):
                prediction = (predictions[index] if index < len(predictions) else None) if ram_mode else by_path.get(str(Path(paths[index]).resolve()).lower())
                if prediction is None:
                    results.append({"item_id": item["item_id"], "status": "ERROR",
                                    "raw_text": "", "score": None,
                                    "error": "MODEL_RESULT_MISSING"})
                    continue
                raw_text = (prediction.get("rec_text") or "").strip()
                results.append({"item_id": item["item_id"],
                                "status": "OK" if raw_text else "EMPTY",
                                "raw_text": raw_text,
                                "score": float(prediction.get("rec_score", 0.0))})
            emit({"type": "result", "request_id": request["request_id"],
                  "results": results, "inference_seconds": inference_seconds})
        except Exception:
            emit({"type": "result", "request_id": request.get("request_id"),
                  "status": "ERROR", "results": [],
                  "error": traceback.format_exc()[-4000:]})


if __name__ == "__main__":
    main()
