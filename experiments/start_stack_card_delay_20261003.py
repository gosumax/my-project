"""Bounded diagnostic: card-pair appearance delay after reviewed Smoke14 gates.

Uses only saved parent TABLE crops and manually reviewed selected hand windows.
No production changes, OCR, candidate changes, or image writes outside output.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "runs/start_stack_frame_ledger_wide_smoke14_20261003/card_deal_delay"
WIDE = ROOT / "runs/start_stack_frame_ledger_wide_smoke14_20261003"
PARENT = Path("C:/ParserData/runs/smoke14_full_tiny_ssd_20260929_v3")
DBPATH = PARENT / "journal.sqlite3"
PROFILE_PATH = ROOT / "configs/layout_847x404.json"
MANIFEST_PATH = ROOT / "production_manifest.json"
THRESHOLD = 0.95
CONSECUTIVE = 3
MAX_FRAMES_PER_HAND = 60


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def database():
    return sqlite3.connect(DBPATH.resolve().as_uri() + "?mode=ro", uri=True)


def fetch_table(db, session: str, frame: int):
    row = db.execute(
        """SELECT o.crop_path,o.crop_sha256,f.source_pts,f.time_base_num,f.time_base_den
           FROM observations o JOIN frames f ON f.source_id=o.source_id AND f.frame_id=o.frame_id
           WHERE o.roi_type='table' AND o.table_session_id=? AND o.frame_id=?""",
        (session, frame),
    ).fetchone()
    if not row:
        raise ValueError(f"MISSING_TABLE_FRAME {session} {frame}")
    path = PARENT / row[0]
    file_hash = sha_file(path)
    if file_hash != row[1]:
        raise ValueError(f"TABLE_FILE_HASH_MISMATCH {session} {frame}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (404, 847):
        raise ValueError(f"TABLE_IMAGE_GEOMETRY_INVALID {session} {frame}")
    return image, {
        "frame": frame,
        "table_path": str(path),
        "table_file_sha256": file_hash,
        "source_pts": row[2],
        "time_base": [row[3], row[4]],
    }


def crop_hash(crop):
    return hashlib.sha256(crop.tobytes()).hexdigest()


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    selected = read_jsonl(WIDE / "selected_hands.jsonl")
    reviews = read_jsonl(WIDE / "visual_reviews.jsonl")
    review_by_key = {(r["table"], r["hand_id"]): r for r in reviews}
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    # First selected, manually reviewed hand per slot supplies fixed seat templates.
    calibration = {}
    for c in selected:
        key = (c["table"], c["hand_id"])
        rv = review_by_key[key]
        if rv.get("deal_complete_status") != "CONFIRMED":
            raise ValueError(f"UNCONFIRMED_REVIEWED_GATE {key}")
        slot = int(c["table"][1:])
        calibration.setdefault(slot, (c, rv))
    db = database()
    templates = {}
    template_records = []
    for slot, (c, rv) in sorted(calibration.items()):
        frame = rv["deal_complete_frame"]
        image, meta = fetch_table(db, c["session"], frame)
        for seat in rv["occupied_seats"]:
            box = profile["slots"][f"slot_{slot:02d}"]["seats"][seat]["shown_cards"]
            x1, y1, x2, y2 = box
            template = image[y1:y2, x1:x2].copy()
            templates[(slot, seat)] = template
            template_records.append({
                "slot": slot, "table": c["table"], "hand_id": c["hand_id"],
                "seat": seat, "roi_xyxy": box, "gate_frame": frame,
                "table_path": meta["table_path"],
                "table_file_sha256": meta["table_file_sha256"],
                "source_pts": meta["source_pts"], "time_base": meta["time_base"],
                "template_bgr_sha256": crop_hash(template),
                "template_shape": list(template.shape),
            })

    scores = []
    missing = []
    distinct_source_paths = set()
    for c in selected:
        rv = review_by_key[(c["table"], c["hand_id"])]
        slot = int(c["table"][1:])
        gate = rv["deal_complete_frame"]
        # Bounded scan is inclusive from hand start through gate+3 (at most 60 frames).
        lo = c["start"]
        hi = min(gate + 3, lo + MAX_FRAMES_PER_HAND - 1)
        if hi < gate:
            raise ValueError(f"GATE_OUTSIDE_SCAN {c['table']} {c['hand_id']}")
        for frame in range(lo, hi + 1):
            try:
                image, meta = fetch_table(db, c["session"], frame)
            except ValueError as exc:
                if str(exc).startswith("MISSING_TABLE_FRAME"):
                    missing.append({"table": c["table"], "hand_id": c["hand_id"],
                                    "frame": frame, "reason": str(exc)})
                    continue
                raise
            distinct_source_paths.add(meta["table_path"])
            for seat in rv["occupied_seats"]:
                box = profile["slots"][f"slot_{slot:02d}"]["seats"][seat]["shown_cards"]
                x1, y1, x2, y2 = box
                candidate = image[y1:y2, x1:x2]
                templ = templates.get((slot, seat))
                if templ is None:
                    # A seat not occupied in the calibration hand has no safe
                    # per-seat template. Keep unknown rather than borrowing one.
                    scores.append({"table": c["table"], "hand_id": c["hand_id"],
                                   "slot": slot, "seat": seat, **meta,
                                   "roi_xyxy": box, "score": None,
                                   "candidate_bgr_sha256": crop_hash(candidate),
                                   "status": "NO_CALIBRATION_TEMPLATE"})
                    continue
                score = float(cv2.matchTemplate(candidate, templ, cv2.TM_CCOEFF_NORMED)[0, 0])
                scores.append({"table": c["table"], "hand_id": c["hand_id"],
                               "slot": slot, "seat": seat, **meta,
                               "roi_xyxy": box, "score": score,
                               "candidate_bgr_sha256": crop_hash(candidate),
                               "status": "MATCH" if score >= THRESHOLD else "NO_MATCH"})
    db.close()

    scoremap = {(r["table"], r["hand_id"], r["seat"], r["frame"]): r for r in scores}
    hand_results = []
    seat_results = []
    calibration_keys = {(c["table"], c["hand_id"]) for c, _ in calibration.values()}
    for c in selected:
        rv = review_by_key[(c["table"], c["hand_id"])]
        slot = int(c["table"][1:])
        gate = rv["deal_complete_frame"]
        lo, hi = c["start"], min(gate + 3, c["start"] + MAX_FRAMES_PER_HAND - 1)
        first_by_seat = {}
        per_seat_run_start = {}
        for seat in rv["occupied_seats"]:
            hit_frames = []
            for f in range(lo, hi + 1):
                item = scoremap.get((c["table"], c["hand_id"], seat, f))
                hit_frames.append(bool(item and item["score"] is not None and item["score"] >= THRESHOLD))
            run_start = None
            for i in range(len(hit_frames) - CONSECUTIVE + 1):
                if all(hit_frames[i:i + CONSECUTIVE]):
                    run_start = lo + i
                    break
            per_seat_run_start[seat] = run_start
            first_by_seat[seat] = next((lo + i for i, hit in enumerate(hit_frames) if hit), None)
            seat_results.append({
                "table": c["table"], "hand_id": c["hand_id"], "seat": seat,
                "occupied_reviewed": True,
                "first_threshold_frame": first_by_seat[seat],
                "first_three_consecutive_frame": run_start,
                "confirm_frame": run_start + CONSECUTIVE - 1 if run_start is not None else None,
                "deal_complete_frame": gate,
                "deal_chat_frame": c["deal_chat_frame"],
                "offset_to_first_threshold": (first_by_seat[seat] - gate) if first_by_seat[seat] is not None else None,
                "first_threshold_offset_from_deal_chat": (first_by_seat[seat] - c["deal_chat_frame"]) if first_by_seat[seat] is not None else None,
                "offset_to_three_consecutive": (run_start - gate) if run_start is not None else None,
                "offset_to_three_consecutive_from_hand_start": (run_start - c["start"]) if run_start is not None else None,
                "offset_to_three_consecutive_from_deal_chat": (run_start - c["deal_chat_frame"]) if run_start is not None else None,
                "template_source_hand": calibration[slot][0]["hand_id"],
                "split": "CALIBRATION" if (c["table"], c["hand_id"]) in calibration_keys else "HOLDOUT",
            })
        # Earliest frame that starts a 3-consecutive match in every reviewed occupied seat.
        all_start = None
        for f in range(lo, hi - CONSECUTIVE + 2):
            exact = True
            for seat in rv["occupied_seats"]:
                exact &= all(
                    (q := scoremap.get((c["table"], c["hand_id"], seat, f + j))) is not None
                    and q["score"] is not None and q["score"] >= THRESHOLD
                    for j in range(CONSECUTIVE)
                )
            if exact:
                all_start = f
                break
        hand_results.append({
            "table": c["table"], "hand_id": c["hand_id"], "slot": slot,
            "split": "CALIBRATION" if (c["table"], c["hand_id"]) in calibration_keys else "HOLDOUT",
            "hand_start_frame": lo, "deal_chat_frame": c["deal_chat_frame"], "deal_complete_frame": gate,
            "scan_end_frame": hi,
            "gate_evidence_table_path": rv["gate_evidence"]["table_path"],
            "gate_evidence_table_sha256": rv["gate_evidence"]["table_file_sha256"],
            "earliest_all_occupied_three_consecutive_frame": all_start,
            "all_occupied_confirm_frame": all_start + CONSECUTIVE - 1 if all_start is not None else None,
            "all_occupied_offset_from_hand_start": all_start - lo if all_start is not None else None,
            "all_occupied_confirm_offset_from_hand_start": all_start + CONSECUTIVE - 1 - lo if all_start is not None else None,
            "all_occupied_offset_from_deal_chat": all_start - c["deal_chat_frame"] if all_start is not None else None,
            "all_occupied_confirm_offset_from_deal_chat": all_start + CONSECUTIVE - 1 - c["deal_chat_frame"] if all_start is not None else None,
            "all_occupied_offset_from_gate": all_start - gate if all_start is not None else None,
            "occupied_seat_count": len(rv["occupied_seats"]),
            "seats_without_template": [s for s in rv["occupied_seats"] if (slot, s) not in templates],
            "seat_first_three_consecutive_frames": per_seat_run_start,
        })

    # Small visual contact sheet requested for T01's first hand, frames 681..694.
    from PIL import Image, ImageDraw
    t01 = next(c for c in selected if c["table"] == "T01" and c["hand_id"] == "2875498927")
    sheet = Image.new("RGB", (4 * 847, 4 * 430), "white")
    draw = ImageDraw.Draw(sheet)
    for i, frame in enumerate(range(681, 695)):
        image, meta = fetch_table(database(), t01["session"], frame)
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        x, y = (i % 4) * 847, (i // 4) * 430
        draw.text((x + 4, y + 2), f"T01 #{t01['hand_id']} frame {frame}", fill="black")
        sheet.paste(pil, (x, y + 24))
    sheet.save(OUT / "t01_frames_681_694_contact_sheet.png")

    with (OUT / "template_sources.jsonl").open("w", encoding="utf-8") as f:
        for row in template_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "frame_scores.jsonl").open("w", encoding="utf-8") as f:
        for row in scores:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "seat_results.jsonl").open("w", encoding="utf-8") as f:
        for row in seat_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "hand_results.jsonl").open("w", encoding="utf-8") as f:
        for row in hand_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_inputs = [WIDE / "selected_hands.jsonl", WIDE / "visual_reviews.jsonl",
                       PROFILE_PATH, MANIFEST_PATH, Path(__file__).resolve()]
    manifest = {
        "inputs": [{"path": str(p.resolve()), "sha256": sha_file(p)} for p in manifest_inputs],
        "journal": str(DBPATH.resolve()), "journal_access": "READ_ONLY_SQLITE_URI",
        "source_frames_scanned": len({r["table_path"] for r in scores}),
        "source_file_paths_unique": len(distinct_source_paths),
        "seat_frame_scores": len(scores), "missing_table_frames": missing,
        "production_changes": False,
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    calibration_hands = [r for r in hand_results if r["split"] == "CALIBRATION"]
    holdout_hands = [r for r in hand_results if r["split"] == "HOLDOUT"]
    detected = [r["all_occupied_offset_from_gate"] for r in hand_results
                if r["all_occupied_offset_from_gate"] is not None]
    start_offsets = [r["all_occupied_offset_from_hand_start"] for r in hand_results
                     if r["all_occupied_offset_from_hand_start"] is not None]
    seat_offsets = [r["offset_to_three_consecutive"] for r in seat_results
                    if r["offset_to_three_consecutive"] is not None]
    split_results = {}
    for split in ["CALIBRATION", "HOLDOUT"]:
        values = [r["all_occupied_offset_from_hand_start"] for r in hand_results
                  if r["split"] == split and r["all_occupied_offset_from_hand_start"] is not None]
        split_results[split] = {
            "hands_detected": len(values),
            "earliest_all_seat_start_offset_min": min(values) if values else None,
            "earliest_all_seat_start_offset_median": float(np.median(values)) if values else None,
            "earliest_all_seat_start_offset_max": max(values) if values else None,
        }
    t01res = next(r for r in hand_results if r["table"] == "T01" and r["hand_id"] == "2875498927")
    summary = {
        "schema": "experimental.card-deal-delay-template.v1",
        "verdict": "DIAGNOSTIC_ONLY",
        "method": "fixed per-slot/per-seat shown_cards ROI template from first reviewed gate; cv2.TM_CCOEFF_NORMED >=0.95; 3 consecutive source frames",
        "threshold": THRESHOLD, "consecutive_frames": CONSECUTIVE,
        "calibration_hand_count": len(calibration_hands),
        "holdout_hand_count": len(holdout_hands),
        "split_results": split_results,
        "hands_scanned": len(hand_results), "seat_rows": len(seat_results),
        "all_occupied_detected_hands": len(detected),
        "all_occupied_offset_min": min(detected) if detected else None,
        "all_occupied_offset_median": float(np.median(detected)) if detected else None,
        "all_occupied_offset_max": max(detected) if detected else None,
        "per_seat_three_consecutive_offset_min": min(seat_offsets) if seat_offsets else None,
        "per_seat_three_consecutive_offset_median": float(np.median(seat_offsets)) if seat_offsets else None,
        "per_seat_three_consecutive_offset_max": max(seat_offsets) if seat_offsets else None,
        "all_occupied_start_offset_min": min(start_offsets) if start_offsets else None,
        "all_occupied_start_offset_median": float(np.median(start_offsets)) if start_offsets else None,
        "all_occupied_start_offset_max": max(start_offsets) if start_offsets else None,
        "t01_first_hand": t01res,
        "timing_proposal": {
            "bounded_probe_offsets_after_CHAT_DEAL": [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
            "dense_probe_interval_frames": 1,
            "dense_probe_through_offset": 22,
            "sparse_followup_probe_offsets": [25, 30, 35, 40, 45, 50, 55, 60],
            "sparse_confirmation_frames_after_each_offset": 2,
            "rule": "After current-hand CHAT DEAL, sample every frame through at least +22; accept only after every occupied seat scores >=0.95 in the same 3 consecutive frames (first is candidate appearance, third is causal confirmation). If not found, open 3-frame probe blocks beginning at +25,+30,...,+60. This is only an observation schedule: hand opening still requires current-hand CHAT boundary, forced-post/seat evidence and direct current-table visual state; never infer exact deal onset or certify from card-template match alone.",
        },
        "limitations": [
            "Nine calibration hands are also in-sample; eleven holdouts use templates fitted only on the first selected hand of that slot.",
            "Fixed template match is a visual detector, not an authoritative deal boundary or player identity resolver.",
            "The scan stops at gate+3 and is bounded to at most 60 frames per hand; missing saved table observations are explicit.",
            "First three-consecutive frame marks causal confirmation at its third frame, while the candidate onset is the first frame of that triple.",
        ],
        "wall_seconds": time.perf_counter() - started,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["verdict", "calibration_hand_count", "holdout_hand_count", "split_results", "all_occupied_detected_hands", "all_occupied_start_offset_min", "all_occupied_start_offset_median", "all_occupied_start_offset_max", "all_occupied_offset_min", "all_occupied_offset_median", "all_occupied_offset_max", "per_seat_three_consecutive_offset_min", "per_seat_three_consecutive_offset_median", "per_seat_three_consecutive_offset_max", "t01_first_hand"]}, indent=2))


if __name__ == "__main__":
    main()
