"""Freeze raw chat/table evidence and a blank visual annotation contract."""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.annotation import VERSION, local_path, sha256, write_json


@contextmanager
def readonly_db(run: Path):
    db = sqlite3.connect((run / "journal.sqlite3").resolve(strict=True).as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def validate_splits(entries: list[dict], known_used: set[str]) -> bool:
    assignments = {}
    for entry in entries:
        split = entry["split"]
        if split not in {"dev", "train", "validation", "test_candidate", "test"}:
            raise ValueError("Invalid split")
        for key in ("sha256", "session_group"):
            value = entry.get(key)
            if not value:
                raise ValueError(f"Missing {key}: session grouping cannot be inferred from filename")
            token = (key, value)
            if token in assignments and assignments[token] != split:
                raise ValueError(f"Split leakage: {key}={value}")
            assignments[token] = split
        if split == "test" and (entry["sha256"] in known_used or
                                entry.get("prior_use") not in {
                                    "UNUSED_AT_FREEZE", "RAW_ONLY_REVIEW_TOOL_QA"} or
                                not entry.get("independence_reviewer") or
                                not entry.get("independence_review_time")):
            raise ValueError("Independent test source has prior use or lacks reviewed history")
    return bool(entries) and all(x["split"] == "test" for x in entries)


def freeze(plan: dict, output: Path) -> dict:
    if output.exists():
        raise ValueError("Choose a new corpus directory; frozen evidence is never overwritten")
    statement = plan.get("independence_document")
    if statement:
        statement_path = (ROOT / statement["path"]).resolve(strict=True)
        if (not statement_path.is_relative_to(ROOT) or
                sha256(statement_path) != statement["sha256"]):
            raise ValueError("Independence statement hash or location differs")
    entries, evidence, units = [], [], []
    known_used = set()
    # Any source with OCR in this workspace has been exposed during development.
    for path in (ROOT / "runs").glob("*/journal.sqlite3"):
        with readonly_db(path.parent) as db:
            if db.execute("SELECT COUNT(*) FROM recognitions").fetchone()[0]:
                known_used.update(x[0] for x in db.execute("SELECT sha256 FROM sources"))
    registry_path = ROOT / "annotation" / "source_registry_v2_20260928.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        known_used.update(x["sha256"] for x in registry["sources"] if x["prior_use"] not in
                          {"UNUSED_AT_FREEZE", "USER_REPORTED_NEW_UNVERIFIED"})
    copies = {}
    for selected in plan["runs"]:
        run = Path(selected["run"]).resolve(strict=True)
        with readonly_db(run) as db:
            sources = [dict(x) for x in db.execute("SELECT * FROM sources")]
            if len(sources) != 1:
                raise ValueError("Select a single-source capture run")
            source = sources[0]
            if source["state"] not in {"PARTIAL", "EOF"}:
                raise ValueError("Cannot freeze an unfinished or failed capture")
            if sha256(Path(source["path"]).resolve(strict=True)) != source["sha256"]:
                raise ValueError("Original source hash changed")
            entry = {**source, **{k: selected.get(k) for k in
                ("split", "session_group", "prior_use", "independence_reviewer", "independence_review_time")},
                     "run": str(run)}
            entries.append(entry)
            sessions = {x["table_session_id"]: dict(x) for x in db.execute("SELECT * FROM table_sessions")}
            selected_sessions = selected.get("table_session_ids") or list(sessions)
            for sid in selected_sessions:
                if sid not in sessions:
                    raise ValueError(f"Unknown session: {sid}")
                records = db.execute("""SELECT o.*,f.source_pts,f.time_base_num,f.time_base_den
                    FROM observations o JOIN frames f USING(source_id,frame_id)
                    WHERE o.table_session_id=? AND o.roi_type IN ('chat','table')
                      AND o.frame_id BETWEEN ? AND ? ORDER BY o.frame_id,o.roi_type""",
                    (sid, selected["first_frame"], selected["last_frame"])).fetchall()
                if not records or not any(x["roi_type"] == "chat" for x in records):
                    raise ValueError("Selected window has no raw chat evidence")
                expected_chat_frames = selected.get("expected_chat_frames")
                if expected_chat_frames is not None and sum(
                    x["roi_type"] == "chat" for x in records) != expected_chat_frames:
                    raise ValueError("Selected window has an incomplete chat frame count")
                ids = []
                for record in records:
                    record = dict(record)
                    eid = record["observation_id"]
                    ids.append(eid)
                    if eid in copies:
                        raise ValueError("Overlapping evidence windows; select disjoint table windows")
                    source_path = local_path(run, record["crop_path"])
                    if sha256(source_path) != record["crop_sha256"]:
                        raise ValueError(f"Capture PNG hash changed: {source_path}")
                    relative = f"evidence/{eid}.png"
                    with Image.open(source_path) as image:
                        size = list(image.size)
                    copies[eid] = (source_path, relative)
                    evidence.append({"evidence_id": eid, "source_sha256": source["sha256"],
                        "source_id": record["source_id"], "table_session_id": sid,
                        "frame_id": record["frame_id"], "source_pts": record["source_pts"],
                        "time_base": [record["time_base_num"], record["time_base_den"]],
                        "roi_type": record["roi_type"], "bbox": json.loads(record["bbox_json"]),
                        "transform": json.loads(record["transform_json"]), "size": size,
                        "sha256": record["crop_sha256"], "local": relative})
                units.append({"unit_id": f"unit_{len(units):04d}", "table_session_id": sid,
                    "source_sha256": source["sha256"], "session_group": selected["session_group"],
                    "split": selected["split"], "first_frame": selected["first_frame"],
                    "last_frame": selected["last_frame"], "evidence_ids": ids,
                    "table_session": sessions[sid]})
    independent = validate_splits(entries, known_used)
    fingerprints = {str(p.relative_to(ROOT)): sha256(p) for folder, pattern in
        (("parser_core", "*.py"), ("scripts", "*.py"), ("configs", "*.json"), ("docs", "*PROTOCOL.md"))
        for p in (ROOT / folder).glob(pattern)}
    if registry_path.exists():
        fingerprints[str(registry_path.relative_to(ROOT))] = sha256(registry_path)
    manifest = {"schema_version": VERSION, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "independent_test": independent,
        "independence_basis": "REVIEWER_DECLARED_SOURCE_AND_SESSION_HISTORY" if independent else "DEVELOPMENT_OR_UNASSESSED",
        "sources": entries, "units": units, "evidence": evidence,
        "code_config_protocol_sha256": fingerprints,
        "assets_manifest_sha256": sha256(ROOT / "provenance" / "manifest.json"),
        "sampling": "RAW_FRAME_WINDOWS_WITHOUT_OCR_SELECTION", "plan": plan}
    output.mkdir(parents=True)
    (output / "evidence").mkdir()
    for source_path, relative in copies.values():
        shutil.copy2(source_path, output / relative)
        if sha256(source_path) != sha256(output / relative):
            raise ValueError("Evidence copy differs")
    write_json(output / "manifest.json", manifest)
    digest = sha256(output / "manifest.json")
    (output / "manifest.sha256").write_text(digest + "\n", encoding="ascii")
    write_json(output / "annotations.blank.json", {"schema_version": VERSION, "manifest_sha256": digest,
        "units": [{"unit_id": x["unit_id"], "review_status": "UNREVIEWED", "reviewer": None,
                   "review_time": None, "coverage_complete": False, "rows": [], "messages": []} for x in units]})
    return {"units": len(units), "png": len(evidence), "independent_test": independent,
            "status": "FROZEN_UNREVIEWED", "manifest_sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("Output must remain in the new parser workspace")
    result = freeze(json.loads(args.plan.read_text(encoding="utf-8")), output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
