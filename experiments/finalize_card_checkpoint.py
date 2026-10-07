"""Pin the validation-selected adapter after the complete training log is sealed."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "runs/card_train_smoke16_test_20260930/train_run"

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main():
    config = json.loads((TRAIN / "train_config.json").read_text(encoding="utf-8"))
    log_path = TRAIN / "training_log.jsonl"
    log_bytes = log_path.read_bytes()
    log = [json.loads(line) for line in log_bytes.decode("utf-8").splitlines()]
    if not log or log[-1]["kind"] != "finished" or config["max_train"] is not None or config["max_validation"] is not None:
        raise ValueError("Full training did not finish")
    epochs = config["epochs"]
    validation = [row for row in log if row["kind"] == "validation"]
    if [row["epoch"] for row in validation] != list(range(epochs + 1)) or any(row["rows"] != 318 for row in validation):
        raise ValueError("Validation history incomplete")
    expected_best = min(validation, key=lambda row: row["mean_token_loss"])
    old = json.loads((TRAIN / "checkpoint_selection.json").read_text(encoding="utf-8"))
    if old["best_epoch"] != expected_best["epoch"] or abs(old["best_validation_token_loss"] - expected_best["mean_token_loss"]) > 1e-8:
        raise ValueError("Selection differs from validation log")
    if log[-1]["best_epoch"] != expected_best["epoch"] or abs(log[-1]["best_validation_loss"] - expected_best["mean_token_loss"]) > 1e-8:
        raise ValueError("Completed training disagrees with selection")
    checkpoints = {}
    for epoch in range(epochs + 1):
        name = "checkpoint_base_adapter.pt" if epoch == 0 else f"checkpoint_epoch_{epoch}.pt"
        path = TRAIN / name
        checkpoints[str(epoch)] = {"path": str(path.resolve()), "sha256": sha(path.read_bytes())}
    selected = checkpoints[str(expected_best["epoch"])]
    result = {"status": "FINAL_VALIDATION_SELECTION_FROZEN", "smoke16_seen": False,
              "best_epoch": expected_best["epoch"],
              "best_validation_token_loss": expected_best["mean_token_loss"],
              "validation_losses": {str(row["epoch"]): row["mean_token_loss"] for row in validation},
              "selected_checkpoint": selected["path"], "selected_checkpoint_sha256": selected["sha256"],
              "all_checkpoints": checkpoints, "completed_training_log_sha256": sha(log_bytes),
              "train_config_sha256": sha((TRAIN / "train_config.json").read_bytes()),
              "selection_rule": "minimum_validation_token_loss_including_base"}
    path = TRAIN / "checkpoint_selection_final.json"
    payload = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("Final selection already frozen with different content")
    path.write_bytes(payload)
    print(json.dumps({"best_epoch": result["best_epoch"],
                      "best_validation_token_loss": result["best_validation_token_loss"],
                      "selected_checkpoint_sha256": result["selected_checkpoint_sha256"]}))

if __name__ == "__main__":
    main()
