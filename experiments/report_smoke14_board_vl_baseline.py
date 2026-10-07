"""Join the fixed benchmark manifest to original BOARD evidence and visual truth."""

import hashlib
import json
import sys
import csv
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.visual_cards import parse_cards

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/board_vl_quality_audit_20260930"
MANIFEST = ROOT / "runs/smoke14_board_vl_batch_benchmark_20260930_v2/dataset.json"
RESULTS = ROOT / "runs/smoke14_board_vl_batch_benchmark_20260930_v2/results_batch1.jsonl"
EVIDENCE = ROOT / "runs/smoke14_parallel_vl_full_20260929/visual_evidence.jsonl"
GT = OUT / "ground_truth.json"


def lines(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def main():
    rows = json.loads(MANIFEST.read_text("utf-8"))["selected"]
    results = lines(RESULTS)
    truth = json.loads(GT.read_text("utf-8"))["cards_by_manifest_index"]
    evidence = defaultdict(list)
    for row in lines(EVIDENCE):
        if row.get("crop_type") == "BOARD":
            evidence[row["crop_sha256"]].append(row)
    assert len(rows) == len(results) == len(truth) == 32
    loops = {3, 4, 5, 7, 9, 12, 13, 14, 16, 22, 23, 25, 30}
    output = []
    for index, (row, result, expected) in enumerate(zip(rows, results, truth), 1):
        source = evidence[row["sha256"]]
        assert len(source) == 1, (index, len(source))
        source = source[0]
        path = Path(row["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert result["sha256"] == row["sha256"]
        assert source["vl_raw_text"] == result["raw_text"]
        with Image.open(path) as im:
            size = list(im.size)
        raw = result["raw_text"]
        parsed = parse_cards(raw, len(expected), row["street"])
        assert parsed is None
        output.append({
            "manifest_index": index, "board_evidence_id": source.get("board_evidence_id"),
            "board_evidence_id_status": "NOT_RECORDED_IN_LEGACY_EVIDENCE",
            "source_row_key": source["row_key"], "table_session_id": source["table_session_id"],
            "sha256": row["sha256"], "path": str(path), "original_size": size,
            "table_id": row["table_id"], "frame_id": row["frame_id"], "street": row["street"],
            "prompt": "OCR:", "preprocess": "RGB; processor min_pixels and max_pixels as batch benchmark",
            "max_new_tokens": 256, "raw_text": raw,
            "original_worker_output_tokens": source.get("raw_vl_response", {}).get("new_tokens"),
            "original_worker_stop_status": source.get("raw_vl_response", {}).get("status"),
            "batch_output_tokens": None, "batch_stop_reason": "NOT_RECORDED",
            "strict_normalized": parsed, "ground_truth": expected,
            "ground_truth_status": "VISUALLY_REVIEWED_DIAGNOSTIC",
            "status": "RECOGNITION_INCOMPLETE",
            "primary_failure_reason": "REPETITION_LOOP" if index in loops else "RANKS_ONLY",
            "secondary_failure_reason": "RANKS_ONLY" if index in loops else None,
            "recognition_or_parser": "A_RECOGNITION_INCOMPLETE",
            "parser_rejection_reason": "MISSING_SUITS",
        })
    (OUT / "baseline32.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in output), "utf-8")
    with (OUT / "baseline32.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "manifest_index", "board_evidence_id", "board_evidence_id_status",
            "source_row_key", "sha256", "path", "original_size", "street",
            "EXPECTED", "VL_RAW", "VL_NORMALIZED", "STATUS", "primary_failure_reason",
            "original_worker_output_tokens", "original_worker_stop_status",
        ])
        writer.writeheader()
        for item in output:
            writer.writerow({
                "manifest_index": item["manifest_index"],
                "board_evidence_id": item["board_evidence_id"],
                "board_evidence_id_status": item["board_evidence_id_status"],
                "source_row_key": item["source_row_key"], "sha256": item["sha256"],
                "path": item["path"], "original_size": "x".join(map(str, item["original_size"])),
                "street": item["street"], "EXPECTED": " ".join(item["ground_truth"]),
                "VL_RAW": item["raw_text"], "VL_NORMALIZED": "",
                "STATUS": item["status"], "primary_failure_reason": item["primary_failure_reason"],
                "original_worker_output_tokens": item["original_worker_output_tokens"],
                "original_worker_stop_status": item["original_worker_stop_status"],
            })
    print(Counter(x["primary_failure_reason"] for x in output))


if __name__ == "__main__":
    main()
