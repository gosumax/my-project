"""Test actual list lengths 1..8 while keeping production batch_size=8."""

import csv
import json
import os
import sys
from pathlib import Path

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def main():
    benchmark = Path(sys.argv[1]).resolve(strict=True)
    diagnosis = Path(sys.argv[2]).resolve(strict=True)
    output = Path(sys.argv[3]).resolve()
    entries = json.loads((diagnosis / "crop_metadata.json").read_text("utf-8"))
    manifest = json.loads((benchmark / "manifest.json").read_text("utf-8"))
    index = {item["observation_id"]: i for i, item in enumerate(manifest)}
    from paddleocr import TextRecognition
    model = TextRecognition(model_name="PP-OCRv6_tiny_rec", device="cpu")
    records = []
    for count, entry in enumerate(entries, 1):
        obs = entry["observation_id"]
        i = index[obs]
        neighbors = [item for j, item in enumerate(manifest) if j != i]
        for actual_length in range(1, 9):
            for position in sorted({0, actual_length - 1}):
                batch = (neighbors[:position] + [manifest[i]] +
                         neighbors[position:actual_length - 1])
                assert len(batch) == actual_length
                for repeat in range(2):
                    predictions = list(model.predict([item["path"] for item in batch], batch_size=8))
                    found = {str(Path(p["input_path"]).resolve()).lower(): p for p in predictions}
                    value = found[entry["crop_path"].lower()]
                    records.append({"observation_id": obs, "actual_batch_length": actual_length,
                                    "configured_batch_size": 8, "position": position,
                                    "repeat": repeat, "raw_text": (value.get("rec_text") or "").strip(),
                                    "score": value.get("rec_score"),
                                    "baseline_text": entry["baseline_raw_text"],
                                    "matches_baseline": (value.get("rec_text") or "").strip() == entry["baseline_raw_text"]})
        if count % 5 == 0:
            print(json.dumps({"done": count, "total": len(entries)}), flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.with_suffix(".partial.json").write_text(json.dumps(records, ensure_ascii=False), "utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"records": len(records), "crops": len(entries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
