"""Isolate PaddleX batch-max-width padding as a Tiny OCR text variable."""

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def main():
    manifest = json.loads(Path(sys.argv[1]).read_text("utf-8"))
    target_id = sys.argv[2]
    output = Path(sys.argv[3])
    by_width = defaultdict(list)
    target = None
    for item in manifest:
        if item["observation_id"] == target_id:
            target = item
        width = cv2.imread(item["path"], cv2.IMREAD_COLOR).shape[1]
        by_width[width].append(item)
    if target is None:
        raise ValueError(target_id)
    target_width = cv2.imread(target["path"]).shape[1]
    from paddleocr import TextRecognition
    model = TextRecognition(model_name="PP-OCRv6_tiny_rec", device="cpu")
    predictor = model.paddlex_predictor
    original_to_batch = predictor.pre_tfs["ToBatch"]
    captured = []

    def capture_to_batch(*, imgs):
        values = original_to_batch(imgs=imgs)
        captured.append(values[0])
        return values

    predictor.pre_tfs["ToBatch"] = capture_to_batch
    records = []
    for neighbor_width in sorted(by_width):
        if neighbor_width < target_width:
            continue
        candidates = [item for item in by_width[neighbor_width]
                      if item["observation_id"] != target_id]
        for neighbor_set in (0, 1):
            neighbors = candidates[neighbor_set * 7:(neighbor_set + 1) * 7]
            if len(neighbors) != 7:
                continue
            for position in (0, 7):
                paths = ([target["path"]] + [item["path"] for item in neighbors]
                         if position == 0 else [item["path"] for item in neighbors] + [target["path"]])
                for repeat in range(3):
                    captured.clear()
                    predictions = list(model.predict(paths, batch_size=8))
                    match = {str(Path(row["input_path"]).resolve()).lower(): row for row in predictions}
                    result = match[str(Path(target["path"]).resolve()).lower()]
                    tensor = captured[0]
                    records.append({"target_id": target_id, "target_sha256": target["sha256"],
                                    "target_width": target_width, "neighbor_width": neighbor_width,
                                    "neighbor_set": neighbor_set,
                                    "neighbor_ids": [item["observation_id"] for item in neighbors],
                                    "batch_size": 8, "position": position, "repeat": repeat,
                                    "batch_tensor_shape": list(tensor.shape),
                                    "target_tensor_sha256": hashlib.sha256(tensor[position].tobytes()).hexdigest(),
                                    "raw_text": (result.get("rec_text") or "").strip(),
                                    "score": result.get("rec_score")})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
