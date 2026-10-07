"""Create source-linked Smoke15 CARD tiles and conservative visual episode groups.

Only user-verified row sequences are input. Inferred episode groups are not hand IDs.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.build_verified_chat_card_tiles import spans_like_production


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/smoke15_chat_card_sample_5s_20260930"
RULE = "same_video_table_prefix_and_time_gap_le30s_v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_png_checked(image: Image.Image, path: Path) -> str:
    packed = io.BytesIO()
    image.save(packed, format="PNG")
    data = packed.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Conflicting derived PNG: {path}")
    path.write_bytes(data)
    return sha(data)


def prefix_compatible(first: str, second: str) -> bool:
    a, b = first.split(), second.split()
    return a[:min(len(a), len(b))] == b[:min(len(a), len(b))]


def group_rows(rows: list[dict]) -> dict[str, str]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, first in enumerate(rows):
        for j in range(i + 1, len(rows)):
            second = rows[j]
            if first["source_video_sha256"] != second["source_video_sha256"]:
                continue
            if not prefix_compatible(first["ground_truth"], second["ground_truth"]):
                continue
            if any(a["table_id"] == b["table_id"] and
                   abs(float(a["decoded_time_s"]) - float(b["decoded_time_s"])) <= 30
                   for a in first["occurrences"] for b in second["occurrences"]):
                parent[find(j)] = find(i)
    groups: dict[int, list[str]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row["chat_row_sha256"])
    result = {}
    for members in groups.values():
        group_id = "visual_episode_" + sha("\n".join(sorted(members)).encode())[:16]
        for digest in members:
            result[digest] = group_id
    return result


def main() -> None:
    source_path = OUT / "verified_smoke15_chat_card_sequences.jsonl"
    review_path = OUT / "smoke15_chat_card_review_user_20260930.json"
    source = read_jsonl(source_path)
    candidates = read_jsonl(OUT / "candidates.jsonl")
    by_sha: dict[str, list[dict]] = {}
    for row in candidates:
        by_sha.setdefault(row["chat_row_sha256"], []).append(row)
    if len({r["chat_row_sha256"] for r in source}) != len(source):
        raise ValueError("Duplicate verified row SHA")
    review_sha = sha(review_path.read_bytes())
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review_rows = review.get("rows")
    if (review.get("schema") != "poker.smoke15-chat-review.v1" or
            review.get("sample_manifest_sha256") != sha((OUT / "candidates.jsonl").read_bytes()) or
            not isinstance(review_rows, list) or len(review_rows) != len(source) or
            len({r["chat_row_sha256"] for r in review_rows}) != len(review_rows)):
        raise ValueError("Preserved review does not match candidate manifest or verified rows")
    review_by_sha = {r["chat_row_sha256"]: r for r in review_rows}
    if any(r["gt_status"] != "USER_VERIFIED_SMOKE15" or r["review_export_sha256"] != review_sha for r in source):
        raise ValueError("Verified rows do not match preserved review export")
    groups = group_rows(source)
    decisions = []
    tiles = []
    for row in source:
        digest = row["chat_row_sha256"]
        decision = review_by_sha.get(digest)
        if (decision is None or decision.get("ground_truth") != row["ground_truth"] or
                decision.get("source_png") != row["source_png"]):
            raise ValueError(f"Verified label/source differs from user export: {digest}")
        occurrences = by_sha.get(digest, [])
        keys = ("sample_index", "decoded_time_s", "source_pts", "table_id", "source_span_index", "chat_row_png")
        expected_occurrences = [{k: candidate[k] for k in keys} for candidate in occurrences]
        if (not occurrences or expected_occurrences != row["occurrences"] or
                row["source_png"] not in {candidate["chat_row_png"] for candidate in occurrences} or
                any(candidate["source_video_sha256"] != row["source_video_sha256"] for candidate in occurrences)):
            raise ValueError(f"Occurrence inventory differs: {digest}")
        if sha(Path(row["source_png"]).read_bytes()) != digest:
            raise ValueError(f"Selected source PNG changed: {digest}")
        for occurrence in occurrences:
            row_png = Path(occurrence["chat_row_png"])
            roi_png = Path(occurrence["chat_roi_png"])
            if sha(row_png.read_bytes()) != digest or sha(roi_png.read_bytes()) != occurrence["chat_roi_sha256"]:
                raise ValueError(f"Source hash mismatch: {digest}")
            with Image.open(roi_png) as opened:
                roi = np.asarray(opened.convert("L"))
            with Image.open(row_png) as opened:
                pixels = np.asarray(opened.convert("L"))
            y0, y1 = occurrence["row_y0_y1"]
            if not np.array_equal(pixels, roi[y0:y1]):
                raise ValueError(f"Row/ROI pixels disagree: {digest}")
        with Image.open(row["source_png"]) as opened:
            original = opened.convert("RGB")
        model_input = original.crop((0, 0, original.width - 12, original.height))
        model_path = OUT / "production_trim12_verified" / f"{digest}.png"
        model_sha = save_png_checked(model_input, model_path)
        spans = spans_like_production(np.asarray(model_input.convert("L")))
        if [list(x) for x in spans] != row["detected_card_spans_inclusive"]:
            raise ValueError(f"Tile spans differ from verified record: {digest}")
        cards = row["ground_truth"].split()
        gap_ok = all(2 <= spans[i+1][0] - spans[i][1] - 1 <= 5 for i in range(len(spans)-1))
        status = "ACCEPTED" if len(cards) == len(spans) and gap_ok else "UNRESOLVED"
        if status != row["card_tile_geometry_status"]:
            raise ValueError(f"Tile geometry status differs: {digest}")
        decisions.append({"chat_row_sha256": digest, "ground_truth": row["ground_truth"],
                          "card_tile_geometry_status": status,
                          "reason": None if status == "ACCEPTED" else "SPAN_COUNT_OR_GAP_MISMATCH",
                          "detected_spans_inclusive": spans,
                          "source_png": row["source_png"], "model_input_png": str(model_path),
                          "model_input_sha256": model_sha, "episode_group_id": groups[digest],
                          "grouping_status": "INFERRED_VISUAL_EPISODE_NOT_HAND_ID", "grouping_rule": RULE})
        if status != "ACCEPTED":
            continue
        for index, ((start, end), card) in enumerate(zip(spans, cards)):
            crop = model_input.crop((start, 0, end + 1, model_input.height))
            tile_path = OUT / "verified_smoke15_card_tiles" / f"{digest}_{index:02d}.png"
            tile_sha = save_png_checked(crop, tile_path)
            tiles.append({"task_kind": "CHAT_CARD", "gt_status": "USER_VERIFIED_PARENT_DERIVED_BOX",
                          "ground_truth": card, "card_index": index, "parent_ground_truth": row["ground_truth"],
                          "box_xyxy_exclusive": [start, 0, end + 1, model_input.height],
                          "tile_png": str(tile_path), "tile_sha256": tile_sha,
                          "tile_pixel_sha256": sha(crop.tobytes()),
                          "source_png": row["source_png"], "source_sha256": digest,
                          "model_input_png": str(model_path), "model_input_sha256": model_sha,
                          "source_video_sha256": row["source_video_sha256"],
                          "occurrences": row["occurrences"], "episode_group_id": groups[digest],
                          "grouping_status": "INFERRED_VISUAL_EPISODE_NOT_HAND_ID", "grouping_rule": RULE,
                          "review_export_sha256": review_sha})
    write_jsonl(OUT / "verified_smoke15_tile_decisions.jsonl", decisions)
    write_jsonl(OUT / "verified_smoke15_card_tiles.jsonl", tiles)
    grouping = {"schema": "poker.smoke15.visual-episode-groups.v1", "rule": RULE,
                "status": "INFERRED_NOT_HAND_ID_OR_FINAL_SPLIT", "row_to_group": groups,
                "group_count": len(set(groups.values()))}
    (OUT / "verified_smoke15_grouping.json").write_text(json.dumps(grouping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"verified_rows": len(source), "tile_accepted_rows": sum(x["card_tile_geometry_status"] == "ACCEPTED" for x in decisions),
               "tile_unresolved_rows": sum(x["card_tile_geometry_status"] == "UNRESOLVED" for x in decisions),
               "card_tiles": len(tiles), "card_tile_classes": len({x["ground_truth"] for x in tiles}),
               "six_d_verified_rows": sum("6d" in x["ground_truth"].split() for x in source),
               "six_d_inferred_groups": len({groups[x["chat_row_sha256"]] for x in source if "6d" in x["ground_truth"].split()}),
               "inferred_episode_groups": len(set(groups.values())),
               "row_length_counts": dict(Counter(len(x["ground_truth"].split()) for x in source)),
               "source_review_sha256": review_sha, "source_verified_rows_sha256": sha(source_path.read_bytes()),
               "split_status": "NOT_ASSIGNED", "independent_test_status": "NOT_MEASURED"}
    (OUT / "verified_smoke15_tiles_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
