"""Capture sparse, predeclared raw frame windows from one nine-table source.

Sampling is declared in a JSON plan before reading frames. Deliberately skipped
frames are not loss events, but they prevent claims of complete hand capture.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import av

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.annotation import sha256, write_json
from parser_core.journal import Journal, stable_id
from parser_core.layout import load_fixed_layout
from scripts.capture_video import save_crop


def checked_resume_frames(output: Path, source_id: str, selected: set[int],
                          sessions: dict[int, str], plan_sha256: str) -> dict[int, tuple]:
    """Accept only complete, unchanged frames from the same raw capture."""
    db_path = output / "journal.sqlite3"
    if not db_path.is_file() or (output / "summary.json").exists():
        raise ValueError("Resume requires an unfinished capture journal")
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        source = db.execute("SELECT source_id,state FROM sources").fetchall()
        if source != [(source_id, "OPEN")]:
            raise ValueError("Resume source identity or state differs")
        session_rows = db.execute(
            "SELECT table_session_id,geometry_json FROM table_sessions").fetchall()
        actual_sessions = {row[0] for row in session_rows}
        if actual_sessions != set(sessions.values()):
            raise ValueError("Resume table sessions differ")
        if any(json.loads(geometry).get("sampling_plan_sha256") != plan_sha256
               for _, geometry in session_rows):
            raise ValueError("Resume sampling plan differs")
        for table in ("recognitions", "physical_rows", "row_decisions", "messages",
                      "event_revisions", "hand_revisions", "exports"):
            if db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]:
                raise ValueError("Resume directory is no longer a raw-only capture")
        frames = {row[0]: row[1:] for row in db.execute(
            "SELECT frame_id,source_pts,time_base_num,time_base_den,width,height FROM frames")}
        if not frames.keys() <= selected:
            raise ValueError("Resume journal contains an unselected frame")
        expected_per_frame = {(sid, kind) for sid in sessions.values()
                              for kind in ("table", "chat")}
        for frame_id in frames:
            rows = db.execute("""SELECT table_session_id,roi_type,crop_path,crop_sha256
                FROM observations WHERE source_id=? AND frame_id=?""",
                (source_id, frame_id)).fetchall()
            if {(sid, kind) for sid, kind, _, _ in rows} != expected_per_frame or len(rows) != len(expected_per_frame):
                raise ValueError(f"Incomplete stored frame {frame_id}; retain for audit")
            for _, _, relative, digest in rows:
                crop = (output / relative).resolve(strict=True)
                if not crop.is_relative_to(output) or sha256(crop) != digest:
                    raise ValueError(f"Stored crop changed: {crop}")
        return frames


def selected_frames(windows: list[dict], frame_limit: int | None = None) -> set[int]:
    selected, last = set(), -1
    for window in windows:
        first, end, step = window["first_frame"], window["last_frame"], window["step"]
        if (any(type(x) is not int for x in (first, end, step)) or first < 0 or
                end < first or step < 1 or first <= last or
                (frame_limit is not None and end >= frame_limit)):
            raise ValueError("Windows must be ordered, disjoint and within the source")
        selected.update(range(first, end + 1, step))
        last = end
    if not selected:
        raise ValueError("No frames selected")
    return selected


def capture(plan: dict, plan_path: Path, output: Path, resume: bool = False) -> dict:
    if output.exists() and not resume:
        raise ValueError("Select a new raw capture directory")
    if resume and not output.is_dir():
        raise ValueError("Resume directory does not exist")
    video = Path(plan["video"]).resolve(strict=True)
    if sha256(video) != plan["source_sha256"]:
        raise ValueError("Video SHA differs from frozen sampling plan")
    layout_path = ROOT / "configs" / "nine_table_layout.json"
    profile_path = ROOT / "configs" / "layout_847x404.json"
    calibration_path = ROOT / "configs" / "chat_log_bottom_rois.json"
    layout = load_fixed_layout(layout_path, profile_path, calibration_path)
    config_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in
                     (layout_path, profile_path, calibration_path)}
    source_id = stable_id("source", plan["source_sha256"], video.stat().st_size)
    slots = {slot.slot_id: slot for slot in layout.slots}
    chosen_slots = plan.get("slots", list(slots))
    if len(set(chosen_slots)) != len(chosen_slots) or any(x not in slots for x in chosen_slots):
        raise ValueError("Invalid selected slots")
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        chosen = selected_frames(plan["windows"], stream.frames or None)
        if not chosen_slots:
            raise ValueError("No slots selected")
        sessions = {number: stable_id("table", source_id, layout.layout_id,
                    config_hashes, number, slots[number].table_box, slots[number].chat_box)
                    for number in chosen_slots}
        existing = checked_resume_frames(output, source_id, chosen, sessions,
                                         sha256(plan_path)) if resume else {}
        journal = Journal(output / "journal.sqlite3")
        captured = []
        try:
            journal.add_source(source_id, str(video), plan["source_sha256"], video.stat().st_size)
            for number in chosen_slots:
                slot = slots[number]
                journal.add_session(sessions[number], source_id, layout.layout_id,
                    {"slot_id": number, "table_roi": slot.table_box, "chat_roi": slot.chat_box,
                     "config_sha256": config_hashes, "sampling_plan_sha256": sha256(plan_path)},
                    identity_status="UNRESOLVED")
            last_frame = max(chosen)
            for frame_id, frame in enumerate(container.decode(stream)):
                if frame_id > last_frame:
                    break
                if frame_id not in chosen:
                    continue
                if frame_id in existing:
                    time_base = frame.time_base or stream.time_base
                    if existing[frame_id] != (frame.pts, time_base.numerator,
                                              time_base.denominator, frame.width, frame.height):
                        raise ValueError(f"Stored frame metadata changed: {frame_id}")
                    captured.append({"frame_id": frame_id, "source_pts": frame.pts,
                                     "time_base": [time_base.numerator, time_base.denominator]})
                    continue
                if (frame.width, frame.height) != (layout.source_width, layout.source_height):
                    raise ValueError(f"Frame {frame_id} dimensions differ from frozen layout")
                time_base = frame.time_base or stream.time_base
                # A committed frame must always include every selected ROI.
                # Small batches also bound replay work after an abrupt exit.
                if not journal._batched:
                    journal.begin_batch()
                journal.add_frame(source_id, frame_id, frame.pts,
                    time_base.numerator if frame.pts is not None else None,
                    time_base.denominator if frame.pts is not None else None,
                    frame.width, frame.height)
                image = frame.to_image()
                for number in chosen_slots:
                    slot = slots[number]
                    for roi_type, box in (("table", slot.table_box), ("chat", slot.chat_box)):
                        observation_id = stable_id("obs", source_id, frame_id,
                            sessions[number], roi_type, box)
                        relative = Path("crops") / sessions[number] / f"{frame_id:09d}_{roi_type}.png"
                        digest = save_crop(image, list(box), output / relative)
                        journal.add_observation(observation_id, source_id, frame_id,
                            sessions[number], roi_type, list(box), str(relative), digest, "RAW")
                captured.append({"frame_id": frame_id, "source_pts": frame.pts,
                                 "time_base": [time_base.numerator, time_base.denominator]})
                if len(captured) % 10 == 0:
                    journal.end_batch()
                if len(captured) % 30 == 0:
                    print(f"Captured {len(captured)}/{len(chosen)} frames", flush=True)
            if len(captured) != len(chosen):
                raise ValueError(f"Source ended after {len(captured)}/{len(chosen)} selected frames")
            if journal._batched:
                journal.end_batch()
            journal.set_source_state(source_id, "PARTIAL")
            summary = {"source_id": source_id, "source_sha256": plan["source_sha256"],
                "sampling_plan_sha256": sha256(plan_path), "config_sha256": config_hashes,
                "table_sessions": sessions, "selected_frames": len(chosen),
                "selected_slots": chosen_slots, "frame_windows": plan["windows"],
                "first_selected_pts": captured[0]["source_pts"],
                "last_selected_pts": captured[-1]["source_pts"],
                "source_state": "PARTIAL", "interpretation": "RAW_SAMPLED_ONLY_NO_COMPLETE_HH",
                "counts": journal.counts()}
            write_json(output / "summary.json", summary)
            return summary
        except Exception:
            if journal._batched:
                journal.end_batch(commit=False)
            journal.set_source_state(source_id, "FAILED")
            raise
        finally:
            journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Verify and continue an unfinished raw capture")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("Output must remain inside this project")
    result = capture(json.loads(args.plan.read_text(encoding="utf-8")), args.plan, output,
                     resume=args.resume)
    print(json.dumps({"selected_frames": result["selected_frames"],
                      "observations": result["counts"]["observations"],
                      "source_state": result["source_state"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
