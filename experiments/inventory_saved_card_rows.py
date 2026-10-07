"""Inventory and select saved card-related CHAT_ROW images without video decoding."""

import argparse
import collections
import csv
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/paddle_vl15_poker_sft_20260930"
RUNS = [
    ROOT / "runs/smoke12_extension_dev_candidates_slot1_20260928_v1",
    ROOT / "runs/smoke12_nine_table_60_vl15_20260928_v1",
    Path(r"C:\ParserData\runs\smoke13_full_streaming_ssd_20260929_v1"),
    Path(r"C:\ParserData\runs\smoke14_full_tiny_ssd_20260929_v3"),
]
CARD_EVENTS = {"FLOP", "TURN", "RIVER", "SHOW", "SHOWDOWN"}
CARD_RE = re.compile(r"(?<![A-Za-z0-9])(?:10|[2-9TJQKA])\s*[♣♦♥♠cdhs](?![A-Za-z])", re.I)
SIGNAL_RE = re.compile(r"[♣♦♥♠]|\b(?:flop|turn|river|shows?|cards?)\b|флоп|т[её]рн|ривер|показ", re.I)
SUITS = str.maketrans("♣♦♥♠", "cdhs")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def perceptual_hash(path):
    with Image.open(path) as opened:
        image = opened.convert("L")
        width, height = image.size
        pixels = list(image.resize((9, 8)).get_flattened_data())
    value = sum((pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
                for y in range(8) for x in range(8))
    return f"{value:016x}", width, height


def proposed(text):
    cards = []
    for match in CARD_RE.finditer(text or ""):
        token = match.group().replace(" ", "").upper().translate(SUITS)
        rank, suit = token[:-1], token[-1].lower()
        cards.append(("T" if rank == "10" else rank) + suit)
    if len(cards) in {1, 2, 3, 4, 5} and len(set(cards)) == len(cards):
        return " ".join(cards)
    return None


def one_run(run, verify_hashes=True):
    db_path = run / "journal.sqlite3"
    with sqlite3.connect(db_path) as db:
        source = db.execute("SELECT source_id, sha256, path FROM sources LIMIT 1").fetchone()
        stats = db.execute("SELECT roi_type, COUNT(*), COUNT(DISTINCT crop_sha256) FROM observations "
                           "WHERE roi_type IN ('chat','CHAT_ROW') GROUP BY roi_type").fetchall()
        session_slots = {sid: json.loads(geo)["slot_id"] for sid, geo in
                         db.execute("SELECT table_session_id, geometry_json FROM table_sessions")}
        event_by_rec = collections.defaultdict(list)
        event_path = run / "events_diagnostics.csv"
        if event_path.is_file():
            with event_path.open(encoding="utf-8-sig", newline="") as stream:
                for event in csv.DictReader(stream):
                    if event["event_type"] not in CARD_EVENTS:
                        continue
                    info = json.loads(event["evidence_json"] or "{}")
                    for rid in info.get("recognition_ids", []):
                        event_by_rec[rid].append((event["event_id"], event["event_type"]))
        # OCR is a discovery signal only. Use recorded images and recognition IDs.
        query = """SELECT r.recognition_id, r.raw_text, r.revision,
                    o.observation_id, o.crop_path, o.crop_sha256, o.frame_id,
                    o.table_session_id, ro.physical_row_id
                    FROM recognitions r JOIN observations o ON o.observation_id=r.observation_id
                    LEFT JOIN row_observations ro ON ro.observation_id=o.observation_id
                    WHERE o.roi_type='CHAT_ROW' AND r.raw_text IS NOT NULL"""
        terms = ("%♣%", "%♦%", "%♥%", "%♠%", "%flop%", "%turn%", "%river%",
                 "%show%", "%card%", "%Флоп%", "%флоп%", "%Ривер%", "%ривер%",
                 "%показ%", "%Тёрн%", "%тёрн%")
        filtered = query + " AND (" + " OR ".join("r.raw_text LIKE ?" for _ in terms) + ")"
        rows = db.execute(filtered, terms).fetchall()
        event_ids = list(event_by_rec)
        for start in range(0, len(event_ids), 500):
            chunk = event_ids[start:start + 500]
            rows.extend(db.execute(query + " AND r.recognition_id IN (" + ",".join("?" for _ in chunk) + ")", chunk).fetchall())
    selected = {}
    for rid, raw, revision, oid, relative, digest, frame, session, row_id in rows:
        events = event_by_rec.get(rid, [])
        if not events and not SIGNAL_RE.search(raw):
            continue
        prior = selected.get(oid)
        if prior is None or revision >= prior["revision"]:
            selected[oid] = {"recognition_id": rid, "ocr_text": raw,
                             "revision": revision, "observation_id": oid,
                             "relative_path": relative, "image_sha256": digest,
                             "frame_id": frame, "table_session_id": session,
                             "row_id": row_id, "event_ids": [x[0] for x in events],
                             "event_types": sorted({x[1] for x in events})}
    # Preserve event-associated rows even when another OCR revision had no signal.
    by_rec = {r[0]: r for r in rows}
    for rid, events in event_by_rec.items():
        if rid not in by_rec:
            continue
        _, raw, revision, oid, relative, digest, frame, session, row_id = by_rec[rid]
        if oid not in selected:
            selected[oid] = {"recognition_id": rid, "ocr_text": raw,
                             "revision": revision, "observation_id": oid,
                             "relative_path": relative, "image_sha256": digest,
                             "frame_id": frame, "table_session_id": session,
                             "row_id": row_id, "event_ids": [], "event_types": []}
        selected[oid]["event_ids"] = sorted(set(selected[oid]["event_ids"]) | {x[0] for x in events})
        selected[oid]["event_types"] = sorted(set(selected[oid]["event_types"]) | {x[1] for x in events})
    candidates = []
    for row in selected.values():
        path = (run / row.pop("relative_path")).resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file():
            continue
        if verify_hashes and file_hash(path) != row["image_sha256"]:
            raise ValueError(f"Crop SHA mismatch: {path}")
        phash, width, height = perceptual_hash(path)
        hint = proposed(row["ocr_text"])
        street = next((x for x in row["event_types"] if x in {"FLOP", "TURN", "RIVER"}), None)
        kind = "BOARD" if street or hint and len(hint.split()) in {3, 4, 5} else "CARD_ROW"
        row.update({"candidate_id": "cand_" + hashlib.sha256((source[1] + row["observation_id"]).encode()).hexdigest()[:24],
                    "source_project": "Новый парсер", "source_video": Path(source[2]).name,
                    "source_video_sha256": source[1], "artifact_path": str(path),
                    "artifact_type": "CHAT_ROW", "table_id": session_slots.get(row["table_session_id"]),
                    "hand_id": None, "event_type": ",".join(row["event_types"]) or None,
                    "street": street, "perceptual_hash": phash,
                    "width": width, "height": height, "proposed_gt": hint,
                    "proposed_gt_source": "OCR_UNVERIFIED" if hint else None,
                    "gt_status": "UNREVIEWED", "verified_gt": None,
                    "source_run": str(run), "kind": kind})
        candidates.append(row)
    inventory = []
    for kind, count, unique in stats:
        inventory.append({"source_video": Path(source[2]).name,
                          "source_video_sha256": source[1], "source_project": "Новый парсер",
                          "artifact_path": str(run / ("crops/rows" if kind == "CHAT_ROW" else "crops")),
                          "artifact_type": "PHYSICAL_ROW" if kind == "CHAT_ROW" else "CHAT_CROP",
                          "image_count": count, "unique_hash_count": unique,
                          "unique_hash_basis": "journal recorded PNG SHA",
                          "has_frame_id": True, "has_table_id": True,
                          "has_row_id": kind == "CHAT_ROW", "has_event_type": kind == "CHAT_ROW",
                          "has_gt": False, "has_ocr_text": kind == "CHAT_ROW",
                          "has_existing_prediction": kind == "CHAT_ROW",
                          "usable_for_board": kind == "CHAT_ROW",
                          "usable_for_card": kind == "CHAT_ROW",
                          "notes": "Filter card-related rows; OCR/events are unreviewed hints" if kind == "CHAT_ROW" else "Whole chat ROI; lower priority than rows"})
    return inventory, candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--skip-verify-hashes", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory, candidates = [], []
    for run in RUNS:
        rows, found = one_run(run, not args.skip_verify_hashes)
        inventory.extend(rows)
        candidates.extend(found)
    candidates.sort(key=lambda r: (r["source_video"], r["table_id"] or 0, r["frame_id"], r["candidate_id"]))
    with (args.output / "current_card_row_candidates.jsonl").open("w", encoding="utf-8") as stream:
        for row in candidates:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / "current_crop_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"inventory_sets": len(inventory), "card_row_candidates": len(candidates),
                      "per_video": dict(collections.Counter(r["source_video"] for r in candidates)),
                      "event_tagged": sum(bool(r["event_type"]) for r in candidates),
                      "proposed_gt": sum(bool(r["proposed_gt"]) for r in candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
