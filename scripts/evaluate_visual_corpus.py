"""Audit visual labels and compare selected journal versions without mutations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.annotation import audit, compare, local_path, sha256, valid_bbox, write_json
from scripts.freeze_visual_corpus import readonly_db


def load_predictions(run: Path, manifest: dict, assembler: str, normalizer: str) -> tuple[list, dict]:
    predictions = []
    with readonly_db(run) as db:
        # Select latest revision WITHIN requested normalizer, not latest across history.
        events = db.execute("""SELECT e.*,m.completeness,m.reason AS message_reason
            FROM event_revisions e JOIN messages m USING(message_id)
            WHERE m.assembler_version=? AND e.normalizer_version=?
              AND e.revision=(SELECT MAX(x.revision) FROM event_revisions x
                  WHERE x.event_id=e.event_id AND x.normalizer_version=e.normalizer_version)
            ORDER BY e.table_session_id,e.order_key""", (assembler, normalizer)).fetchall()
        evidence_by_key = {(x["source_sha256"], x["table_session_id"], x["frame_id"]): x
                           for x in manifest["evidence"] if x["roi_type"] == "chat"}
        frozen_by_id = {x["evidence_id"]: x for x in manifest["evidence"]}
        for obs in db.execute("SELECT * FROM observations WHERE roi_type IN ('chat','table')"):
            frozen = frozen_by_id.get(obs["observation_id"])
            if frozen and (obs["crop_sha256"] != frozen["sha256"] or
                           json.loads(obs["bbox_json"]) != frozen["bbox"] or
                           sha256(local_path(run, obs["crop_path"])) != frozen["sha256"]):
                raise ValueError("Prediction run differs from frozen raw evidence")
        sessions = {x["table_session_id"]: x["sha256"] for x in db.execute(
            "SELECT t.table_session_id,s.sha256 FROM table_sessions t JOIN sources s USING(source_id)")}
        for event in events:
            fragments = []
            for fragment in db.execute("SELECT * FROM message_fragments WHERE message_id=? ORDER BY ordinal",
                                       (event["message_id"],)):
                anchors = []
                for obs in db.execute("""SELECT o.* FROM row_observations ro JOIN observations o USING(observation_id)
                    WHERE ro.physical_row_id=? ORDER BY o.frame_id""", (fragment["physical_row_id"],)):
                    ev = evidence_by_key.get((sessions[event["table_session_id"]], event["table_session_id"], obs["frame_id"]))
                    if ev:
                        if sha256(local_path(run, obs["crop_path"])) != obs["crop_sha256"]:
                            raise ValueError("Prediction row PNG hash mismatch")
                        box = json.loads(obs["bbox_json"])
                        origin = ev["bbox"]
                        local_box = [box[0]-origin[0], box[1]-origin[1], box[2]-origin[0], box[3]-origin[1]]
                        if not valid_bbox(local_box, *ev["size"]):
                            raise ValueError("Prediction fragment outside frozen chat")
                        anchors.append({"evidence_id": ev["evidence_id"], "bbox":
                                        local_box,
                                        "observation_id": obs["observation_id"], "row_png_sha256": obs["crop_sha256"]})
                fragments.append({"physical_row_id": fragment["physical_row_id"],
                                  "recognition_id": fragment["recognition_id"], "anchors": anchors})
            for unit in manifest["units"]:
                if any(a["evidence_id"] in unit["evidence_ids"] for f in fragments for a in f["anchors"]):
                    # Full fragment list retained: clipping at window edges cannot hide a merge.
                    predictions.append({"unit_id": unit["unit_id"], "event_id": event["event_id"],
                        "revision": event["revision"], "message_id": event["message_id"],
                        "status": event["status"], "reason": event["reason"], "order_key": event["order_key"],
                        "message_completeness": event["completeness"], "fragments": fragments,
                        "fields": {"actor": event["actor"], "action": event["event_type"],
                                   "amount": event["amount_decimal"], "role": event["amount_role"],
                                   "cards": json.loads(event["cards_json"]) or None}})
        if not events:
            raise ValueError("Requested assembler/normalizer combination has no events")
        gaps = [dict(x) for x in db.execute("SELECT * FROM gaps")]
        suppressions = [dict(x) for x in db.execute("SELECT * FROM row_suppressions")]
    return predictions, {"registered_gaps": gaps, "row_suppressions": suppressions,
                         "note": "Run-wide diagnostics; gaps do not count as recovered messages."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--run", type=Path, action="append")
    parser.add_argument("--assembler")
    parser.add_argument("--normalizer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        parser.error("Choose a new output directory inside this project")
    corpus = args.corpus.resolve(strict=True)
    label_path = args.annotations or corpus / "annotations.blank.json"
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    result = audit(corpus, labels)
    output.mkdir(parents=True)
    write_json(output / "audit.json", result)
    if not result["valid"]:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    predictions, diagnostics = [], {}
    if args.run:
        if not args.assembler or not args.normalizer:
            parser.error("Explicit --assembler and --normalizer required for journal history")
        diagnostics = {}
        for selected_run in args.run:
            part, diagnostic = load_predictions(selected_run.resolve(strict=True), manifest, args.assembler, args.normalizer)
            predictions.extend(part)
            diagnostics[str(selected_run)] = diagnostic
        keys = [(p["unit_id"], p["event_id"]) for p in predictions]
        if len(set(keys)) != len(keys):
            raise ValueError("Repeated event from overlapping runs; select one version per window")
        if not predictions:
            raise ValueError("Run has no prediction evidence inside frozen windows")
    elif result["counts"].get("reviewed_units", 0):
        parser.error("Scoring reviewed labels requires a --run")
    score = compare(labels, predictions)
    score.update({"independent_test": manifest["independent_test"],
                  "annotation_sha256": sha256(label_path), "manifest_sha256": result["manifest_sha256"],
                  "assembler_version": args.assembler, "normalizer_version": args.normalizer,
                  "selected_predictions": len(predictions), "source_diagnostics": diagnostics,
                  "stage1_acceptance": "NOT_EVALUATED",
                  "protocol_minimum": {"complete_messages": {"observed": result["counts"].get("complete_messages", 0), "required": 300},
                                       "independent_sequences": "NOT_ESTABLISHED", "transitions": "NOT_SCORED", "complete_hands": "NOT_SCORED"}})
    write_json(output / "predictions.json", predictions)
    write_json(output / "comparison.json", score)
    print(json.dumps({"audit_valid": result["valid"], "quality_status": score["quality_status"],
                      "independent_test": score["independent_test"], "selected_predictions": len(predictions),
                      "reviewed_messages": result["counts"].get("reviewed_messages", 0)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
