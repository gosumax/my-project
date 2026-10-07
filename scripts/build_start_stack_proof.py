"""Build hash-checked real-hand START STACK proof from saved Smoke14 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.stack_temporal import hand_windows, prove_hand


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_sources(source_run: Path, ram_metadata: Path, review: dict) -> dict:
    metadata = json.loads(ram_metadata.read_text("utf-8"))
    with sqlite3.connect(source_run / "journal.sqlite3") as db:
        rows = db.execute("SELECT path,sha256 FROM sources").fetchall()
    if len(rows) != 1:
        raise ValueError("Exactly one source video is required")
    path, digest = rows[0]
    video = Path(path).resolve(strict=True)
    if video != Path(metadata["video"]).resolve(strict=True):
        raise ValueError("RAM timeline and table crops refer to different videos")
    with video.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest or review["source_video_sha256"] != digest:
        raise ValueError("Source video SHA256 mismatch")
    return {"video": str(video), "sha256": digest,
            "journal": str((source_run / "journal.sqlite3").resolve())}


def verify_reviewed_tables(source_run: Path, reviews: list[dict]) -> dict:
    paths = {}
    with sqlite3.connect(source_run / "journal.sqlite3") as db:
        for hand in reviews:
            for seat_review in hand["seats"].values():
                for prefix in ("overlay", "stack"):
                    if f"{prefix}_frame" not in seat_review:
                        continue
                    frame = seat_review[f"{prefix}_frame"]
                    expected = seat_review[f"{prefix}_table_sha256"]
                    row = db.execute("""SELECT o.crop_path,o.crop_sha256
                        FROM observations o JOIN table_sessions s
                          ON s.table_session_id=o.table_session_id
                        WHERE o.roi_type='table' AND o.frame_id=?
                          AND json_extract(s.geometry_json,'$.slot_id')=?""",
                        (frame, int(hand["table_id"][1:]))).fetchone()
                    if row is None or row[1] != expected:
                        raise ValueError("Reviewed table frame differs from journal")
                    path = (source_run / row[0]).resolve(strict=True)
                    if not path.is_relative_to(source_run.resolve()):
                        raise ValueError("Reviewed crop escapes source run")
                    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                        raise ValueError("Reviewed table crop SHA256 mismatch")
                    paths[(hand["table_id"], hand["hand_number"], frame)] = str(path)
    return paths


def build_proof(source_run: Path, ram_timeline: Path, ram_metadata: Path,
                observer_paths: list[Path], review_path: Path) -> dict:
    review = json.loads(review_path.read_text("utf-8"))
    if review.get("schema") != "pokerdom.start-stack-visual-review.v1":
        raise ValueError("Unexpected visual review schema")
    source = verify_sources(source_run, ram_metadata, review)
    table_paths = verify_reviewed_tables(source_run, review["hands"])
    observations = []
    observer_summaries = []
    for path in observer_paths:
        summary = json.loads((path.parent / "start_stack_observer_summary.json").read_text("utf-8"))
        if summary["status"] != "RAW_EVIDENCE_ONLY" or summary["metrics"].get("dropped_candidates", 0):
            raise ValueError("Observer evidence is incomplete or dropped candidates")
        observer_summaries.append(summary)
        for item in read_jsonl(path):
            crop = (path.parent / item["crop_path"]).resolve(strict=True)
            if not crop.is_relative_to(path.parent.resolve()) or (
                    hashlib.sha256(crop.read_bytes()).hexdigest() != item["crop_file_sha256"]):
                raise ValueError("Observer crop path or SHA256 mismatch")
            item["crop_path"] = str(crop)
            observations.append(item)
    rows = read_jsonl(ram_timeline)
    reviews = {(x["table_id"], x["hand_number"]): x for x in review["hands"]}
    proofs = []
    for window in hand_windows(rows):
        key = window["table_id"], window["hand_number"]
        if key in reviews:
            hand_proof = prove_hand(window, observations, reviews[key])
            hand_proof["reviewed_table_crops"] = {
                str(frame): table_paths[(key[0], key[1], frame)]
                for seat in reviews[key]["seats"].values()
                for frame in (seat["stack_frame"],
                              *([seat["overlay_frame"]] if "overlay_frame" in seat else []))}
            proofs.append(hand_proof)
    if len(proofs) != len(reviews):
        raise ValueError("Some reviewed hands are absent from the raw timeline")
    counts = Counter(player["status"] for hand in proofs for player in hand["players"])
    return {"schema": "pokerdom.start-stack-proof.v1", "source": source,
        "timeline": str(ram_timeline.resolve()), "review": str(review_path.resolve()),
        "observer_paths": [str(p.resolve()) for p in observer_paths],
        "observer_summaries": observer_summaries,
        "reviewed_hands": len(proofs), "certified": counts["CERTIFIED"],
        "unknown": counts["UNKNOWN"], "partial_start": counts["PARTIAL_START"],
        "hands": proofs}


def markdown(proof: dict) -> str:
    lines = ["# START STACK real-hand proof", "",
        f"Source SHA-256: `{proof['source']['sha256']}`", "",
        f"Reviewed hands: {proof['reviewed_hands']}; CERTIFIED: {proof['certified']}; UNKNOWN: {proof['unknown']}.", ""]
    for hand in proof["hands"]:
        lines += [f"## T{int(hand['table_id'][1:]):02d} — #{hand['hand_id']}", "",
            f"Hand first observed at frame {hand['hand_start_frame']}.", ""]
        lines += ["Reviewed table frames: " + ", ".join(
            f"[frame {frame}](<{path}>)" for frame, path in sorted(
                hand["reviewed_table_crops"].items(), key=lambda x: int(x[0]))) + ".", ""]
        for player in hand["players"]:
            lines += [f"### {player['seat']} — {player.get('nickname', 'UNKNOWN')}", "",
                f"Status: **{player['status']}**. Reason: {player.get('reason') or 'none'}.", ""]
            if player["status"] == "CERTIFIED":
                forced = ", ".join(f"{x['event_type']} {x['amount']} @ chat frame {x['frame_index']}"
                    for x in player["forced_bets_already_posted"]) or "none"
                before = ", ".join(f"{x['event_type']} {x['amount']} @ chat frame {x['frame_index']}"
                    for x in player["voluntary_actions_before_stack"]) or "none"
                after = ", ".join(f"{x['event_type']} {x['amount']} @ chat frame {x['frame_index']}"
                    for x in player["voluntary_actions_after_stack"]) or "none"
                lines += [f"Role {player['role']}; first numeric stack frame {player['stack_frame']}.",
                    f"Forced bets already posted: {forced}. Voluntary actions before stack: {before}. After stack: {after}.",
                    f"Visible {player['raw_visible_stack']} + proven debits {player['player_contribution_before_stack_frame']} = **{player['start_stack']}**.",
                    f"[Stack crop](<{player['crop_path']}>) (pixel SHA-256 `{player['crop_sha256']}`).", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--ram-timeline", type=Path, required=True)
    parser.add_argument("--ram-metadata", type=Path, required=True)
    parser.add_argument("--observer", type=Path, action="append", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proof = build_proof(args.source_run, args.ram_timeline, args.ram_metadata,
                        args.observer, args.review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, ensure_ascii=False, indent=2), "utf-8")
    args.output.with_suffix(".md").write_text(markdown(proof), "utf-8")


if __name__ == "__main__":
    main()
