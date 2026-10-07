"""Class counts by verified source and available episode evidence for the frozen split."""
import collections
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "runs/card_train_smoke16_test_20260930"
rows = [json.loads(line) for line in (Path(__file__).resolve().parents[1] / "runs/paddle_vl15_poker_sft_20260930/combined_verified_chat_card_rows.jsonl").read_text(encoding="utf-8").splitlines()]
by_id = {row["record_id"]: row for row in rows}
counts = {"TRAIN": collections.defaultdict(collections.Counter),
          "VALIDATION": collections.defaultdict(collections.Counter),
          "QUARANTINE": collections.defaultdict(collections.Counter)}
group_counts = collections.defaultdict(collections.Counter)
for name, file in (("TRAIN", "train_manifest.jsonl"), ("VALIDATION", "validation_manifest.jsonl"),
                   ("QUARANTINE", "quarantined_validation_rows.jsonl")):
    for line in (root / "split_v2" / file).read_text(encoding="utf-8").splitlines():
        tile = json.loads(line)
        source = tile["source_video"]
        card = tile["ground_truth"]
        counts[name][source][card] += 1
        parent = by_id[tile["record_id"]]
        group = parent.get("inferred_episode_group_id")
        if group:
            group_key = f"{source}:{group}"
        elif parent.get("source_hand_id"):
            group_key = f"{source}:SOURCE_HAND_ID_UNCERTIFIED:{parent['source_hand_id']}"
        else:
            group_key = f"{source}:UNRESOLVED_EPISODE"
        group_counts[(name, group_key)][card] += 1
result = {"status": "EVIDENCE_COUNTS_ONLY", "source_counts":
          {split: {source: dict(sorted(counter.items())) for source, counter in sources.items()} for split, sources in counts.items()},
          "available_episode_group_counts":
          {f"{split}|{group}": dict(sorted(counter.items())) for (split, group), counter in group_counts.items()},
          "episode_note": "Smoke15 groups are inferred visual episodes. SOURCE_HAND_ID is not certified; absent IDs remain UNRESOLVED_EPISODE. Counts are tiles, not independent hands."}
(root / "split_v2/class_source_episode_counts.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
