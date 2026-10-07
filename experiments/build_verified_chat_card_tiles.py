"""Derive auditable single-card crops from human-verified CHAT rows only.

This is an offline dataset builder. The span rule mirrors board_tiles_fixed_chat_v1;
it does not change the production reader or infer labels from VL output.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/paddle_vl15_poker_sft_20260930"
SPAN_RULE = "board_tiles_fixed_chat_v1_y3_bright150_width18_22_gap2_5"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def spans_like_production(gray: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive bright runs in x order, as in board_tiles._runs."""
    values = np.flatnonzero(gray[3, :min(150, gray.shape[1])] > 150)
    runs: list[tuple[int, int]] = []
    start = previous = None
    for value in values:
        x = int(value)
        if start is None:
            start = previous = x
        elif x > previous + 1:
            runs.append((start, previous))
            start = previous = x
        else:
            previous = x
    if start is not None:
        runs.append((start, previous))
    return [(a, b) for a, b in runs if 18 <= b - a + 1 <= 22]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    verified_path = OUT / "verified_chat_card_sequences.jsonl"
    trim_path = OUT / "production_trim12_chat_crops.jsonl"
    verified = read_jsonl(verified_path)
    trimmed = {r["candidate_id"]: r for r in read_jsonl(trim_path)}
    if len(trimmed) != len(read_jsonl(trim_path)) or len({r["candidate_id"] for r in verified}) != len(verified):
        raise ValueError("Duplicate candidate ID")
    folder = OUT / "verified_chat_card_tiles"
    folder.mkdir(exist_ok=True)
    tiles: list[dict] = []
    decisions: list[dict] = []
    for parent in verified:
        cid = parent["candidate_id"]
        if parent["task_kind"] != "CHAT_CARD_SEQUENCE" or parent["gt_status"] != "VERIFIED" or not parent["production_raw_geometry"]:
            raise ValueError(f"Not a verified production CHAT row: {cid}")
        source = trimmed[cid]
        if source["right_trim_px"] != 12 or source["source_sha256"] != parent["image_sha256"] or source["source_png"] != parent["artifact_path"]:
            raise ValueError(f"Source provenance mismatch: {cid}")
        raw = Path(parent["artifact_path"]).read_bytes()
        derived = Path(source["derived_png"]).read_bytes()
        if sha(raw) != parent["image_sha256"] or sha(derived) != source["derived_sha256"]:
            raise ValueError(f"Image hash mismatch: {cid}")
        with Image.open(io.BytesIO(raw)) as opened:
            original = opened.convert("RGB")
        with Image.open(io.BytesIO(derived)) as opened:
            image = opened.convert("RGB")
        if image.size != (original.width - 12, original.height) or image.tobytes() != original.crop((0, 0, original.width - 12, original.height)).tobytes():
            raise ValueError(f"Production trim pixels differ: {cid}")
        if list(image.size) != parent["production_model_input_size"] or image.height != 16:
            raise ValueError(f"Production geometry differs: {cid}")
        gray = np.asarray(image.convert("L"))
        spans = spans_like_production(gray)
        cards = parent["ground_truth"].split()
        gaps_valid = all(2 <= spans[i + 1][0] - spans[i][1] - 1 <= 5 for i in range(len(spans) - 1))
        status = "ACCEPTED" if 2 <= len(spans) <= 5 and len(spans) == len(cards) and gaps_valid else "UNRESOLVED"
        decision = {"candidate_id": cid, "status": status, "reason": None if status == "ACCEPTED" else "SPAN_COUNT_OR_GAP_MISMATCH",
                    "expected_cards": len(cards), "detected_spans": [[a, b] for a, b in spans],
                    "source_png": parent["artifact_path"], "source_sha256": parent["image_sha256"],
                    "model_input_png": source["derived_png"], "model_input_sha256": source["derived_sha256"],
                    "span_rule": SPAN_RULE}
        decisions.append(decision)
        if status != "ACCEPTED":
            continue
        for index, ((start, end), card) in enumerate(zip(spans, cards)):
            crop = image.crop((start, 0, end + 1, image.height))
            tile_path = folder / f"{cid}_{index:02d}.png"
            crop.save(tile_path, format="PNG")
            tile_bytes = tile_path.read_bytes()
            tiles.append({"task_kind": "CHAT_CARD", "gt_status": "VERIFIED_PARENT_DERIVED_BOX",
                          "ground_truth": card, "candidate_id": cid, "card_index": index,
                          "parent_ground_truth": parent["ground_truth"],
                          "box_xyxy_exclusive": [start, 0, end + 1, image.height],
                          "tile_png": str(tile_path), "tile_sha256": sha(tile_bytes),
                          "tile_pixel_sha256": sha(crop.tobytes()),
                          "source_png": parent["artifact_path"], "source_sha256": parent["image_sha256"],
                          "model_input_png": source["derived_png"], "model_input_sha256": source["derived_sha256"],
                          "source_video": parent["source_video"], "source_video_sha256": parent["source_video_sha256"],
                          "frame_id": parent["frame_id"], "table_id": parent["table_id"],
                          "row_id": parent["row_id"], "hand_id": parent["hand_id"],
                          "event_type": parent["event_type"], "street": parent["street"],
                          "span_rule": SPAN_RULE})
    write_jsonl(OUT / "verified_chat_card_tile_decisions.jsonl", decisions)
    write_jsonl(OUT / "verified_chat_card_tiles.jsonl", tiles)
    row_cards = Counter(card for row in verified for card in row["ground_truth"].split())
    tile_cards = Counter(row["ground_truth"] for row in tiles)
    classes = [rank + suit for rank in "23456789TJQKA" for suit in "cdhs"]
    coverage = {
        card: {"verified_rows": row_cards[card], "accepted_tiles": tile_cards[card],
               "unique_tile_pixels": len({row["tile_pixel_sha256"] for row in tiles if row["ground_truth"] == card}),
               "videos": dict(Counter(row["source_video"] for row in tiles if row["ground_truth"] == card))}
        for card in classes
    }
    (OUT / "chat_card_class_coverage.json").write_text(
        json.dumps({"schema": "poker.chat-card-class-coverage.v1", "classes": coverage},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"verified_parent_rows": len(verified), "accepted_parent_rows": sum(r["status"] == "ACCEPTED" for r in decisions),
               "unresolved_parent_rows": sum(r["status"] == "UNRESOLVED" for r in decisions),
               "standalone_card_tiles": len(tiles), "unique_card_classes": len({r["ground_truth"] for r in tiles}),
               "tiles_by_video": dict(Counter(r["source_video"] for r in tiles)),
               "unresolved_candidate_ids": [r["candidate_id"] for r in decisions if r["status"] == "UNRESOLVED"],
               "verified_manifest_sha256": sha(verified_path.read_bytes()),
               "production_trim_manifest_sha256": sha(trim_path.read_bytes()), "span_rule": SPAN_RULE}
    (OUT / "verified_chat_card_tiles_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
