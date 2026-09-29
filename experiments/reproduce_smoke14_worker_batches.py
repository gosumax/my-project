"""Run controlled saved-crop batches through the unmodified production worker."""

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.tiny_chat_client import TinyChatClient


def main():
    manifest = json.loads(Path(sys.argv[1]).read_text("utf-8"))
    output = Path(sys.argv[2]).resolve()
    config = json.loads((ROOT / "configs/ocr_tiny_local.json").read_text("utf-8"))
    by_id = {row["observation_id"]: row for row in manifest}
    neighbor_224 = next(row for row in manifest
                        if cv2.imread(row["path"]).shape[1] == 224)
    neighbor_226 = next(row for row in manifest
                        if cv2.imread(row["path"]).shape[1] == 226)
    neighbor_228 = next(row for row in manifest
                        if cv2.imread(row["path"]).shape[1] == 228)
    client = TinyChatClient(Path(config["python"]), Path(config["model_dir"]),
                            config["model_name"], config["weights_sha256"],
                            "cpu", output.parent / "controlled_worker.stderr.log")
    records = []
    try:
        for target_id in ("obs_03965350e1d8d18e4ed8ab40",
                          "obs_0d360daf65fc6dfa7dabd1d1",
                          "obs_81206a0807dff669b13efeb0"):
            target = by_id[target_id]
            for neighbor in (None, neighbor_224, neighbor_226, neighbor_228):
                batch = [target] if neighbor is None else [neighbor, target]
                items = [{"item_id": row["observation_id"], "path": row["path"],
                          "source_sha256": row["sha256"]} for row in batch]
                for repeat in range(3):
                    response = client.read_batch(items, 8)
                    result = next(row for row in response["results"] if row["item_id"] == target_id)
                    records.append({"observation_id": target_id, "crop_sha256": target["sha256"],
                                    "configured_batch_size": 8, "actual_batch_length": len(batch),
                                    "neighbor_width": None if neighbor is None else cv2.imread(neighbor["path"]).shape[1],
                                    "repeat": repeat, "raw_text": result["raw_text"],
                                    "score": result["score"]})
    finally:
        client.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
