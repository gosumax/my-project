"""Freeze assistant visual labels before any Smoke16 model inference."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image
import numpy as np

from experiments.build_verified_chat_card_tiles import spans_like_production

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/card_train_smoke16_test_20260930"
LABELS = [
    "Jh Jd Tc 8c 4s", "Kh Jc 5h 5d", "6h 5s Ad", "3d 7d Tc", "3h Jc Jh 6s",
    "Qd 6d Qh", "3d 7d Tc 8d 4d", "Th 9s 6h 8h", "4d 5h 2c 6h Qs", "Ks 2h Td Tc 8s",
    "2s 4d 6d", "Kc Ts 9h", "Kc Ts 9h", "Kc Ts 9h 3s 8s", "Kc Ts 9h 3s",
    "Qd 2h 4d", "Qd 2h 4d 5c", "Ad 4s Qs", "2h 4h 2d 7c", "2d Tc Qh 4s Jc",
    "As 2s 4h Jh", "Kh 7s 7c", "5s 7d Qh", "6c 3c 2c", "Kd 6h 7d 2c Jh",
    "Jd Ts Kh", "9s 5h Kd", "8c 2c 5d 2d 8d", "9s 5h Kd 5c", "3h 9h 2c",
    "3h 9h 2c Kc Ts", "Qs 2c 4s 4d 7s", "4h 5d 9c", "Ad Qd Jd Td 3d", "Qd 2c Ts",
    "9d 9s 9h", "Td 2d 9h 6d", "Qc 9h 4c Js", "As 4d 3c", "5h 7s Qc",
    "9c 3h Ad 6h As", "6d 7c 8d", "6s 4s 5c 5h", "9c 3h Ad 6h As", "2s 8s Qh 5s 4h",
    "Ad 5d Kc", "Ad 5d Kc 3d", "Ad 5d Kc", "Th 9s 3h 9c", "Td 3h Tc",
    "Qs Qd 6h 8c", "Qs Qd 6h 8c", "Ac Ad 3c 2h", "3s 2s 5c", "6d 8h 7h",
    "5d Td 2h 6c", "Jc Td Kd", "5d Kc 4h", "5d Kc 4h", "5d Kc 4h",
]

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def write_once(path: Path, data: bytes) -> None:
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Frozen artifact differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

def jsonl(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in rows) + "\n").encode("utf-8")

def main() -> None:
    manifest_bytes = (OUT / "smoke16_blind_review_manifest.jsonl").read_bytes()
    rows = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines()]
    if len(rows) != len(LABELS) or [r["blind_test_index"] for r in rows] != list(range(60)):
        raise ValueError("Blind manifest/order changed")
    reviews, tiles = [], []
    for row, sequence in zip(rows, LABELS):
        cards = sequence.split()
        if any(len(card) != 2 or card[0] not in "23456789TJQKA" or card[1] not in "cdhs" for card in cards):
            raise ValueError(sequence)
        raw = Path(row["chat_row_png"]).read_bytes()
        if sha(raw) != row["chat_row_sha256"]:
            raise ValueError("Source row changed")
        with Image.open(io.BytesIO(raw)) as opened:
            original = opened.convert("RGB")
        trim = int(row["production_right_trim_px"])
        model_input = original.crop((0, 0, original.width - trim, original.height))
        spans = spans_like_production(np.asarray(model_input.convert("L")))
        frozen_spans = [tuple(s) for s in row["detected_card_spans_inclusive"]]
        if spans != frozen_spans:
            raise ValueError("Span detection changed")
        gap_ok = all(2 <= spans[i+1][0] - spans[i][1] - 1 <= 5 for i in range(len(spans)-1))
        geometry = "ACCEPTED" if gap_ok and len(spans) == len(cards) else "UNRESOLVED"
        reviews.append({"blind_test_index": row["blind_test_index"], "chat_row_sha256": row["chat_row_sha256"],
                        "status": "VISUALLY_REVIEWED_ASSISTANT", "ground_truth": sequence,
                        "card_tile_geometry_status": geometry, "source_png": row["chat_row_png"],
                        "source_video_sha256": row["source_video_sha256"], "source_pts": row["source_pts"],
                        "table_id": row["table_id"], "source_span_index": row["source_span_index"],
                        "blind_manifest_sha256": sha(manifest_bytes)})
        if geometry != "ACCEPTED":
            continue
        frame_index = row["source_pts"] // 512 if row["source_pts"] % 512 == 0 else None
        for index, ((start, end), card) in enumerate(zip(spans, cards)):
            crop = model_input.crop((start, 0, end + 1, model_input.height))
            tile_path = OUT / "smoke16_test_tiles" / f'{row["blind_test_index"]:02d}_{index:02d}.png'
            memory = io.BytesIO()
            crop.save(memory, format="PNG")
            payload = memory.getvalue()
            write_once(tile_path, payload)
            tiles.append({"blind_test_index": row["blind_test_index"], "card_index": index,
                          "ground_truth": card, "gt_status": "VISUALLY_REVIEWED_ASSISTANT_PARENT_DERIVED_BOX",
                          "tile_png": str(tile_path), "tile_sha256": sha(payload),
                          "tile_pixel_sha256": sha(crop.tobytes()), "box_xyxy_exclusive": [start, 0, end + 1, model_input.height],
                          "source_video": row["source_video"], "source_video_sha256": row["source_video_sha256"],
                          "source_pts": row["source_pts"], "frame_index_by_pts_512": frame_index,
                          "table_id": row["table_id"], "source_span_index": row["source_span_index"],
                          "source_row_png": row["chat_row_png"], "source_row_sha256": row["chat_row_sha256"],
                          "production_right_trim_px": trim,
                          "blind_manifest_sha256": sha(manifest_bytes)})
    write_once(OUT / "smoke16_visual_review_frozen.jsonl", jsonl(reviews))
    write_once(OUT / "smoke16_test_manifest.jsonl", jsonl(tiles))
    report = {"status": "FROZEN_BEFORE_MODEL_PREDICTIONS", "reviewed_rows": len(reviews),
              "tile_geometry_accepted_rows": sum(r["card_tile_geometry_status"] == "ACCEPTED" for r in reviews),
              "tile_geometry_unresolved_rows": sum(r["card_tile_geometry_status"] == "UNRESOLVED" for r in reviews),
              "test_tiles": len(tiles), "ground_truth_source": "ASSISTANT_VISUAL_READING_AUTHORIZED_BY_USER",
              "review_sha256": sha(jsonl(reviews)), "test_manifest_sha256": sha(jsonl(tiles)),
              "blind_manifest_sha256": sha(manifest_bytes)}
    write_once(OUT / "smoke16_gt_freeze.json", (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False))

if __name__ == "__main__":
    main()
