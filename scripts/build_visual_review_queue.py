"""List exact chat-image changes in a frozen corpus for blind visual review."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.annotation import local_path, sha256, write_json


FIELDS = ("unit_id", "table_session_id", "frame_id", "source_pts", "time_base_num",
          "time_base_den", "chat_evidence_id", "chat_png", "chat_sha256",
          "table_evidence_id", "table_png", "table_sha256", "change_reason",
          "visual_review_status", "notes")


def build(corpus: Path, output: Path) -> dict:
    corpus, output = corpus.resolve(strict=True), output.resolve()
    if output.exists():
        raise ValueError("Choose a new review queue directory")
    manifest_path = corpus / "manifest.json"
    manifest_hash = sha256(manifest_path)
    if (corpus / "manifest.sha256").read_text(encoding="ascii").strip() != manifest_hash:
        raise ValueError("Frozen manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence = {x["evidence_id"]: x for x in manifest["evidence"]}
    entries = []
    unit_counts = []
    for unit in manifest["units"]:
        by_frame = {}
        for eid in unit["evidence_ids"]:
            item = evidence[eid]
            by_frame.setdefault(item["frame_id"], {})[item["roi_type"]] = item
        previous_sha = None
        changed = 0
        for frame_id, pair in sorted(by_frame.items()):
            if set(pair) != {"chat", "table"}:
                raise ValueError(f"Missing raw chat/table pair: {unit['unit_id']}:{frame_id}")
            chat, table = pair["chat"], pair["table"]
            if previous_sha == chat["sha256"]:
                continue
            for item in (chat, table):
                if sha256(local_path(corpus, item["local"])) != item["sha256"]:
                    raise ValueError(f"Frozen PNG hash changed: {item['evidence_id']}")
            entries.append({"unit_id": unit["unit_id"],
                "table_session_id": unit["table_session_id"], "frame_id": frame_id,
                "source_pts": chat["source_pts"], "time_base_num": chat["time_base"][0],
                "time_base_den": chat["time_base"][1],
                "chat_evidence_id": chat["evidence_id"], "chat_png": str(corpus / chat["local"]),
                "chat_sha256": chat["sha256"], "table_evidence_id": table["evidence_id"],
                "table_png": str(corpus / table["local"]), "table_sha256": table["sha256"],
                "change_reason": "FIRST_FRAME" if previous_sha is None else "CHAT_PNG_CHANGED",
                "visual_review_status": "UNREVIEWED", "notes": ""})
            previous_sha = chat["sha256"]
            changed += 1
        unit_counts.append({"unit_id": unit["unit_id"], "slot_id": json.loads(
            unit["table_session"]["geometry_json"])["slot_id"],
            "frames_in_window": len(by_frame), "selected_changed_frames": changed,
            "frame_ids_contiguous": all(b == a + 1 for a, b in zip(
                sorted(by_frame), sorted(by_frame)[1:]))})
    output.mkdir(parents=True)
    with (output / "changed_chat_frames.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(entries)
    report = {"status": "BLIND_VISUAL_REVIEW_QUEUE_UNREVIEWED",
              "selection": "FIRST_AND_EACH_CONSECUTIVE_CHAT_PNG_SHA_CHANGE",
              "manifest_sha256": manifest_hash, "corpus": str(corpus),
              "units": len(unit_counts), "raw_chat_frames": sum(x["frames_in_window"] for x in unit_counts),
              "selected_changed_frames": len(entries), "unit_counts": unit_counts,
              "frame_ids_contiguous_all_units": all(x["frame_ids_contiguous"] for x in unit_counts),
              "limitations": ["A changed PNG is not necessarily a new poker message.",
                              "Identical stored PNGs do not prove absence of sub-frame changes or hidden messages.",
                              "The CSV is a review queue, not visual ground truth or quality measurement."]}
    if not report["frame_ids_contiguous_all_units"]:
        report["limitations"].append(
            "Sparse or gapped frames cannot establish chat continuity between stored frames.")
    write_json(output / "queue_summary.json", report)
    return {key: report[key] for key in ("status", "units", "raw_chat_frames",
                                         "selected_changed_frames")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.corpus, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
