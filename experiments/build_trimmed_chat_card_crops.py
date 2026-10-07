"""Create source-linked production-preprocessed copies of saved CHAT rows."""

import hashlib
import io
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/paddle_vl15_poker_sft_20260930"


def main():
    sources = [json.loads(s) for s in (OUT / "candidates_manifest.jsonl").read_text(encoding="utf-8").splitlines() if s]
    folder = OUT / "production_trim12_chat_crops"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sources:
        if not source["production_raw_geometry"]:
            continue
        raw = Path(source["artifact_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["image_sha256"]:
            raise ValueError(f"Raw source changed: {source['candidate_id']}")
        with Image.open(io.BytesIO(raw)) as opened:
            image = opened.convert("RGB")
        image = image.crop((0, 0, image.width - 12, image.height))
        if list(image.size) != source["production_model_input_size"]:
            raise ValueError(f"Production size mismatch: {source['candidate_id']}")
        target = folder / f"{source['candidate_id']}.png"
        image.save(target, format="PNG")
        records.append({"candidate_id": source["candidate_id"],
                        "source_png": source["artifact_path"], "source_sha256": source["image_sha256"],
                        "derived_png": str(target), "derived_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "width": image.width, "height": image.height, "right_trim_px": 12,
                        "gt_status": "UNREVIEWED"})
    path = OUT / "production_trim12_chat_crops.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"source_rows": len(sources), "derived_production_geometry": len(records),
                      "right_trim_px": 12, "manifest": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
