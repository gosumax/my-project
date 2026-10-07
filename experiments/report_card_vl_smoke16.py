"""Report immutable first Smoke16 blind predictions; never rerun the model."""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/card_train_smoke16_test_20260930"
SPLIT = DATA / "split_v2"
CLASSES = [rank + suit for rank in "23456789TJQKA" for suit in "cdhs"]
RARE = ["2d", "3d", "5s", "8c", "Jh"]

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def write_once(path: Path, payload: bytes):
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"Report artifact differs: {path}")
    path.write_bytes(payload)

def main():
    raw = (DATA / "smoke16_blind_predictions.jsonl").read_bytes()
    receipt = json.loads((DATA / "smoke16_blind_prediction_receipt.json").read_text(encoding="utf-8"))
    if sha(raw) != receipt["predictions_sha256"]:
        raise ValueError("First blind predictions changed")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if len(rows) != 225 or receipt["count"] != len(rows):
        raise ValueError("Expected 225 TEST tiles")
    counts = collections.Counter(row["verdict"] for row in rows)
    n = len(rows)
    resolved = n - counts["UNRESOLVED"]
    matrix = {gt: {pred: 0 for pred in CLASSES + ["UNRESOLVED"]} for gt in CLASSES}
    per_class = {card: {"test": 0, "correct": 0, "wrong": 0, "unresolved": 0} for card in CLASSES}
    split_info = json.loads((SPLIT / "split_and_class_counts.json").read_text(encoding="utf-8"))
    train_counts = split_info["class_counts"]["TRAIN"]
    errors = []
    for row in rows:
        gt, pred, verdict = row["ground_truth"], row["prediction"], row["verdict"]
        if gt not in CLASSES or pred not in CLASSES + ["UNRESOLVED"]:
            raise ValueError("Unrecognized GT or prediction")
        matrix[gt][pred] += 1
        per_class[gt]["test"] += 1
        per_class[gt][verdict.lower()] += 1
        if verdict == "CORRECT":
            continue
        reason = ("нет точного кода карты в ответе" if pred == "UNRESOLVED" else
                  "ошибка ранга и масти" if pred[0] != gt[0] and pred[1] != gt[1] else
                  "ошибка ранга" if pred[0] != gt[0] else "ошибка масти")
        if train_counts.get(gt, 0) < 10:
            reason += f"; в TRAIN всего {train_counts.get(gt, 0)} примеров класса"
        size = list(Image.open(row["tile_png"]).size)
        errors.append({"ground_truth": gt, "prediction": pred, "raw_response": row["raw_response"],
                       "reason_observed": reason, "tile_size_wh": size,
                       "tile_png": row["tile_png"], "tile_sha256": row["tile_sha256"],
                       "frame_index_by_pts_512": row["frame_index_by_pts_512"],
                       "table_id": row["table_id"], "source_span_index": row["source_span_index"],
                       "blind_test_index": row["blind_test_index"], "card_index": row["card_index"]})
    metrics = {"status": "FIRST_BLIND_BASELINE_FROZEN", "test_tiles": n,
               "verdict_counts": {x: counts[x] for x in ("CORRECT", "WRONG", "UNRESOLVED")},
               "exact_card_accuracy_all": counts["CORRECT"] / n,
               "exact_card_accuracy_resolved": counts["CORRECT"] / resolved if resolved else None,
               "rank_accuracy_all": sum(row["prediction"] != "UNRESOLVED" and row["prediction"][0] == row["ground_truth"][0] for row in rows) / n,
               "suit_accuracy_all": sum(row["prediction"] != "UNRESOLVED" and row["prediction"][1] == row["ground_truth"][1] for row in rows) / n,
               "accuracy_denominator": "all 225 TEST cards; UNRESOLVED counted incorrect",
               "predictions_sha256": sha(raw), "checkpoint_sha256": receipt["selected_checkpoint_sha256"],
               "test_manifest_sha256": receipt["test_manifest_sha256"]}
    write_once(DATA / "smoke16_blind_metrics.json", (json.dumps(metrics, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    write_once(DATA / "smoke16_confusion_matrix.json", (json.dumps(matrix, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    write_once(DATA / "smoke16_per_class.json", (json.dumps(per_class, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    write_once(DATA / "smoke16_errors.jsonl", ("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in errors) + "\n").encode("utf-8"))
    examples = errors[:30]
    write_once(DATA / "smoke16_error_examples.jsonl", ("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in examples) + "\n").encode("utf-8"))
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(["GT\\PRED", *CLASSES, "UNRESOLVED"])
    for gt in CLASSES:
        writer.writerow([gt, *[matrix[gt][pred] for pred in CLASSES + ["UNRESOLVED"]]])
    write_once(DATA / "smoke16_confusion_matrix.csv", csv_buffer.getvalue().encode("utf-8-sig"))
    run_log = [json.loads(line) for line in (DATA / "train_run/training_log.jsonl").read_text(encoding="utf-8").splitlines()]
    selected = json.loads((DATA / "train_run/checkpoint_selection_final.json").read_text(encoding="utf-8"))
    lines = ["# CARD LoRA train → Smoke16 blind test", "", "Status: FIRST_BLIND_BASELINE_FROZEN.", "",
             "## Data and leakage", "",
             f"- TRAIN: {split_info['counts']['TRAIN']} cards from Smoke12/13; VALIDATION: {split_info['counts']['VALIDATION']} cards from Smoke14/15; TEST: {n} cards from Smoke16.",
             f"- {split_info['quarantined_validation_rows']} validation parent rows ({split_info['quarantined_validation_tiles']} tiles) were quarantined for near matches to TRAIN. Split v1 was rejected before full training.",
             f"- TEST: 60 CHAT rows visually read by the assistant with the user's authorization; 225 derived tiles; 0 unresolved boxes. This was frozen before predictions. This is a 5-second sampled subset, not all Smoke16 cards.",
             "- TEST has 54 distinct row sequences and 220 distinct tile pixel hashes; related rows are not certified independent hands. It covers 51 of 52 classes (Ah absent).",
             f"- Missing validation classes: {', '.join(split_info['missing_validation_classes'])}. Four source videos and repeated fixed-font glyphs limit generalization evidence.",
             "", "## Training and checkpoint", "",
             f"- Isolated PaddleOCR-VL-1.5 LoRA: rank 4, alpha 8, q/v projections in last four language layers, two epochs, batch 1, 12544 image pixels. Production code/model weights were not changed.",
             f"- Selected epoch {selected['best_epoch']} by minimum VALIDATION token cross-entropy {selected['best_validation_token_loss']:.6f}; Smoke16 did not participate.",
             f"- Training elapsed {next((x['elapsed_s'] for x in reversed(run_log) if x['kind']=='finished'), 'UNKNOWN')} s; peak allocated {next((x['peak_allocated_bytes'] for x in reversed(run_log) if x['kind']=='finished'), 'UNKNOWN')} bytes.",
             "", "## First blind Smoke16 result", "",
             f"- CORRECT {counts['CORRECT']}, WRONG {counts['WRONG']}, UNRESOLVED {counts['UNRESOLVED']} of {n}.",
             f"- Exact CARD accuracy {metrics['exact_card_accuracy_all']:.3%}; rank {metrics['rank_accuracy_all']:.3%}; suit {metrics['suit_accuracy_all']:.3%}. All use 225 as denominator.",
             "", "### Five sparse pooled classes", "",
             "| Class | TRAIN | VALIDATION | TEST | Correct | Wrong | Unresolved |", "|---|---:|---:|---:|---:|---:|---:|"]
    for card in RARE:
        row = per_class[card]
        lines.append(f"| {card} | {train_counts.get(card,0)} | {split_info['class_counts']['VALIDATION'].get(card,0)} | {row['test']} | {row['correct']} | {row['wrong']} | {row['unresolved']} |")
    six_d = per_class["6d"]
    lines.append(f"\n`6d`: TRAIN {train_counts.get('6d',0)}, VALIDATION {split_info['class_counts']['VALIDATION'].get('6d',0)}, TEST {six_d['test']}, correct {six_d['correct']}, wrong {six_d['wrong']}, unresolved {six_d['unresolved']}.")
    lines += ["", "### Real error examples", "", "Observed mismatch is reported; crop size or class scarcity does not establish causality.", "",
              "| # | GT → PRED | Observed reason | Frame / table / row / card |", "|---:|---|---|---|"]
    for i, row in enumerate(examples, 1):
        lines.append(f"| {i} | {row['ground_truth']} → {row['prediction']} | {row['reason_observed']} | {row['frame_index_by_pts_512']} / {row['table_id']} / {row['source_span_index']} / {row['card_index']} |")
    lines += ["", "See `smoke16_errors.jsonl` for all errors, `smoke16_per_class.json` for all 52 classes, and `smoke16_confusion_matrix.csv` for the full matrix.",
              "", "## Evidence SHA-256", "", f"- First predictions: `{metrics['predictions_sha256']}`", f"- Selected adapter: `{metrics['checkpoint_sha256']}`",
              f"- Frozen TEST manifest: `{metrics['test_manifest_sha256']}`", "",
              "## Анализ после baseline", "",
              "Отдельная диагностика перекоса мастей и сравнения с шаблонами находится в",
              "`SUIT_ERROR_ANALYSIS.md`. Она не меняет первый слепой результат выше. Прямое",
              "сравнение с микроспрайтом клиента оказалось слабым на VALIDATION; сравнение с",
              "проверенными TRAIN CHAT crops для масти и замороженный VL для ранга дали",
              "216/225 на уже раскрытом Smoke16. Это исследовательский результат второго",
              "этапа, не новая независимая оценка.", "",
              "### Пользовательская корректировка метки", "",
              "После первого прогона пользователь проверил два спорных кропа в чате:",
              "`A = Qc`, `B = Qd`. Для A исходная метка `Tc` оказалась ошибочной; B",
              "подтверждена. Отдельный повторный просмотр всех 39 исходно T/Q-карт",
              "других ошибок не выявил. Исходные GT и прогнозы оставлены без изменений.",
              "По исправленному эталону первый замороженный прогноз: **170/225 (75,6%)**;",
              "выбранный по VALIDATION гибрид VL-ранга и шаблонной масти: **224/225",
              "(99,6%)**, одна ошибка `Qd→Qc`. Последнее — диагностический результат на",
              "уже раскрытом Smoke16. Подробности: `stage2_template/FINAL_REPORT.md` и",
              "`stage2_template/user_adjudicated_metrics.json`.", ""]
    write_once(DATA / "FINAL_REPORT.md", ("\n".join(lines)).encode("utf-8"))
    print(json.dumps(metrics, ensure_ascii=False))

if __name__ == "__main__":
    main()
