"""Combine verified CHAT-card evidence without assigning a training split."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "runs/paddle_vl15_poker_sft_20260930"
NEW = ROOT / "runs/smoke15_chat_card_sample_5s_20260930"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    old_rows_path = OLD / "verified_chat_card_sequences.jsonl"
    new_rows_path = NEW / "verified_smoke15_chat_card_sequences.jsonl"
    old_tiles_path = OLD / "verified_chat_card_tiles.jsonl"
    new_tiles_path = NEW / "verified_smoke15_card_tiles.jsonl"
    old_rows, new_rows = read_jsonl(old_rows_path), read_jsonl(new_rows_path)
    old_tiles, new_tiles = read_jsonl(old_tiles_path), read_jsonl(new_tiles_path)
    old_trim = {r["candidate_id"]: r for r in read_jsonl(OLD / "production_trim12_chat_crops.jsonl")}
    old_decisions = {r["candidate_id"]: r for r in read_jsonl(OLD / "verified_chat_card_tile_decisions.jsonl")}
    new_decisions = {r["chat_row_sha256"]: r for r in read_jsonl(NEW / "verified_smoke15_tile_decisions.jsonl")}
    grouping = json.loads((NEW / "verified_smoke15_grouping.json").read_text(encoding="utf-8"))
    review_sha = sha((NEW / "smoke15_chat_card_review_user_20260930.json").read_bytes())
    rows = []
    for r in old_rows:
        source = old_trim[r["candidate_id"]]
        decision = old_decisions[r["candidate_id"]]
        if sha(Path(r["artifact_path"]).read_bytes()) != r["image_sha256"] or sha(Path(source["derived_png"]).read_bytes()) != source["derived_sha256"]:
            raise ValueError(f"Old source or model-input image changed: {r['candidate_id']}")
        rows.append({"record_id": r["candidate_id"], "task_kind": "CHAT_CARD_SEQUENCE",
                     "ground_truth": r["ground_truth"], "gt_status": r["gt_status"],
                     "source_video": r["source_video"], "source_video_sha256": r["source_video_sha256"],
                     "source_png": r["artifact_path"], "source_sha256": r["image_sha256"],
                     "model_input_png": source["derived_png"], "model_input_sha256": source["derived_sha256"],
                     "right_trim_px": 12, "tile_geometry_status": decision["status"],
                     "source_hand_id": r["hand_id"], "inferred_episode_group_id": None,
                     "frame_id": r["frame_id"], "table_id": r["table_id"],
                     "split": None, "split_group_status": "NOT_ESTABLISHED"})
    for r in new_rows:
        digest = r["chat_row_sha256"]
        decision = new_decisions[digest]
        if (r["review_export_sha256"] != review_sha or
                sha(Path(r["source_png"]).read_bytes()) != digest or
                sha(Path(decision["model_input_png"]).read_bytes()) != decision["model_input_sha256"]):
            raise ValueError(f"Smoke15 source, review or model-input image changed: {digest}")
        occurrence = r["occurrences"][0]
        rows.append({"record_id": "smoke15_" + digest, "task_kind": "CHAT_CARD_SEQUENCE",
                     "ground_truth": r["ground_truth"], "gt_status": r["gt_status"],
                     "source_video": "smoke_15.mp4", "source_video_sha256": r["source_video_sha256"],
                     "source_png": r["source_png"], "source_sha256": digest,
                     "model_input_png": decision["model_input_png"],
                     "model_input_sha256": decision["model_input_sha256"], "right_trim_px": 12,
                     "tile_geometry_status": decision["card_tile_geometry_status"],
                     "source_hand_id": None, "inferred_episode_group_id": grouping["row_to_group"][digest],
                     "frame_id": None, "source_pts": occurrence["source_pts"],
                     "table_id": occurrence["table_id"], "occurrences": r["occurrences"],
                     "review_export_sha256": review_sha,
                     "split": None, "split_group_status": "INFERRED_VISUAL_EPISODE_ONLY"})
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate combined row record ID")
    tiles = []
    for t in old_tiles:
        tiles.append({"record_id": t["candidate_id"], "card_index": t["card_index"],
                      "ground_truth": t["ground_truth"], "tile_png": t["tile_png"],
                      "tile_sha256": t["tile_sha256"], "tile_pixel_sha256": t["tile_pixel_sha256"],
                      "source_video": t["source_video"], "inferred_episode_group_id": None,
                      "split": None})
    for t in new_tiles:
        tiles.append({"record_id": "smoke15_" + t["source_sha256"], "card_index": t["card_index"],
                      "ground_truth": t["ground_truth"], "tile_png": t["tile_png"],
                      "tile_sha256": t["tile_sha256"], "tile_pixel_sha256": t["tile_pixel_sha256"],
                      "source_video": "smoke_15.mp4", "inferred_episode_group_id": t["episode_group_id"],
                      "split": None})
    row_ids = {r["record_id"] for r in rows}
    if any(t["record_id"] not in row_ids or sha(Path(t["tile_png"]).read_bytes()) != t["tile_sha256"] for t in tiles):
        raise ValueError("Tile parent or PNG SHA mismatch")
    write_jsonl(OLD / "combined_verified_chat_card_rows.jsonl", rows)
    write_jsonl(OLD / "combined_verified_chat_card_tiles.jsonl", tiles)
    classes = [rank + suit for rank in "23456789TJQKA" for suit in "cdhs"]
    coverage = {}
    for card in classes:
        prior = [t for t in tiles if t["ground_truth"] == card and t["source_video"] != "smoke_15.mp4"]
        smoke15 = [t for t in tiles if t["ground_truth"] == card and t["source_video"] == "smoke_15.mp4"]
        coverage[card] = {"prior_tiles": len(prior), "smoke15_tiles": len(smoke15),
                          "combined_tiles": len(prior) + len(smoke15),
                          "unique_tile_pixels": len({t["tile_pixel_sha256"] for t in prior + smoke15}),
                          "smoke15_inferred_episode_groups": len({t["inferred_episode_group_id"] for t in smoke15}),
                          "combined_verified_row_labels": sum(card in r["ground_truth"].split() for r in rows)}
    summary = {"status": "VERIFIED_CHAT_POOL_NO_SPLIT", "rows": len(rows), "tiles": len(tiles),
               "rows_by_video": dict(Counter(r["source_video"] for r in rows)),
               "tiles_by_video": dict(Counter(t["source_video"] for t in tiles)),
               "tile_unresolved_rows": sum(r["tile_geometry_status"] == "UNRESOLVED" for r in rows),
               "classes_with_tiles": sum(v["combined_tiles"] > 0 for v in coverage.values()),
               "classes_under_10_tiles": [c for c, v in coverage.items() if v["combined_tiles"] < 10],
               "six_d": coverage["6d"], "coverage": coverage,
               "source_sha256": {"old_rows": sha(old_rows_path.read_bytes()), "new_rows": sha(new_rows_path.read_bytes()),
                                 "old_tiles": sha(old_tiles_path.read_bytes()), "new_tiles": sha(new_tiles_path.read_bytes()),
                                 "user_review": review_sha},
               "split": None, "independent_test_status": "NOT_MEASURED"}
    (OLD / "combined_chat_card_coverage.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"coverage", "source_sha256"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
