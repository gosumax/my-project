"""Capture configured fixed-layout table and chat pixels from one video decoder.

The slot sessions are provisional: this capture does not prove table identity,
window movement, replacement, or hand completeness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.journal import Journal, stable_id
from parser_core.layout import load_fixed_layout
from scripts.capture_video import save_crop, sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_BATCH_FRAMES = 25


def crop_pixel_sha256(image, box: tuple[int, int, int, int]) -> str:
    crop = image.crop(tuple(box))
    return hashlib.sha256(crop.tobytes()).hexdigest()


def save_or_reuse_crop(image, box: tuple[int, int, int, int], destination: Path,
                       previous: dict[str, object] | None) -> tuple[str, str, bool]:
    pixel_sha = crop_pixel_sha256(image, box)
    if previous is not None and previous["pixel_sha256"] == pixel_sha:
        return str(previous["crop_sha256"]), pixel_sha, True
    return save_crop(image, list(box), destination), pixel_sha, False


def timed_crop(image, box, destination, previous):
    tick = time.perf_counter()
    pixel_sha = crop_pixel_sha256(image, box)
    hash_wall = time.perf_counter() - tick
    if previous is not None and previous["pixel_sha256"] == pixel_sha:
        return str(previous["crop_sha256"]), pixel_sha, True, hash_wall, 0.0
    tick = time.perf_counter()
    digest = save_crop(image, list(box), destination)
    return digest, pixel_sha, False, hash_wall, time.perf_counter() - tick


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--layout", type=Path,
                        default=ROOT / "configs" / "nine_table_layout.json")
    parser.add_argument("--profile", type=Path,
                        default=ROOT / "configs" / "layout_847x404.json")
    parser.add_argument("--calibration", type=Path,
                        default=ROOT / "configs" / "chat_log_bottom_rois.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, required=True,
                        help="Bounded fixed-layout experiment; source remains PARTIAL")
    parser.add_argument("--db-batch-frames", type=int, default=DEFAULT_DB_BATCH_FRAMES,
                        help="Commit capture journal writes every N frames")
    parser.add_argument("--capture-workers", type=int, default=1)
    args = parser.parse_args()
    if args.start_frame < 0 or args.max_frames < 1:
        parser.error("--start-frame must be nonnegative and --max-frames positive")
    if args.db_batch_frames < 1:
        parser.error("--db-batch-frames must be positive")
    if args.capture_workers < 1:
        parser.error("--capture-workers must be positive")

    video = args.video.resolve(strict=True)
    layout_path = args.layout.resolve(strict=True)
    profile_path = args.profile.resolve(strict=True)
    calibration_path = args.calibration.resolve(strict=True)
    layout = load_fixed_layout(layout_path, profile_path, calibration_path)
    layout_sha = sha256(layout_path)
    profile_sha = sha256(profile_path)
    calibration_sha = sha256(calibration_path)
    source_sha = sha256(video)
    source_id = stable_id("source", source_sha, video.stat().st_size)
    sessions = {
        slot.slot_id: stable_id("table", source_id, layout.layout_id, layout_sha,
                                profile_sha, calibration_sha, slot.slot_id,
                                slot.table_box, slot.chat_box)
        for slot in layout.slots
    }
    output = args.output.resolve()
    journal = Journal(output / "journal.sqlite3")
    count = 0
    saved_crops = 0
    reused_crops = 0
    batch_frames = 0
    batch_commits = 0
    metrics = {
        "decode_wall_seconds": 0.0,
        "frame_to_image_wall_seconds": 0.0,
        "pixel_hash_wall_seconds": 0.0,
        "png_encode_write_wall_seconds": 0.0,
        "db_write_wall_seconds": 0.0,
        "db_commit_wall_seconds": 0.0,
    }
    last_by_roi: dict[tuple[str, str], dict[str, object]] = {}
    state = "FAILED"
    batch_active = False
    buffered_frames = []
    buffered_observations = []
    executor = ThreadPoolExecutor(max_workers=args.capture_workers)
    started_tick = time.perf_counter()
    started_cpu = time.process_time()

    def commit_batch() -> None:
        nonlocal batch_frames, batch_commits, batch_active
        if not buffered_frames:
            return
        tick = time.perf_counter()
        journal.begin_batch()
        batch_active = True
        for values in buffered_frames:
            journal.add_frame(*values)
        for values, quality in buffered_observations:
            journal.add_observation(*values, quality=quality)
        metrics["db_write_wall_seconds"] += time.perf_counter() - tick
        tick = time.perf_counter()
        journal.end_batch()
        metrics["db_commit_wall_seconds"] += time.perf_counter() - tick
        batch_active = False
        batch_frames = 0
        batch_commits += 1
        buffered_frames.clear()
        buffered_observations.clear()

    def begin_batch_if_needed() -> None:
        nonlocal batch_active
        if not batch_active:
            journal.begin_batch()
            batch_active = True

    try:
        tick = time.perf_counter()
        journal.add_source(source_id, str(video), source_sha, video.stat().st_size)
        for slot in layout.slots:
            journal.add_session(
                sessions[slot.slot_id], source_id, layout.layout_id,
                {"slot_id": slot.slot_id, "table_roi": slot.table_box,
                 "chat_roi": slot.chat_box, "layout_sha256": layout_sha,
                 "profile_sha256": profile_sha,
                 "calibration_sha256": calibration_sha},
                identity_status="UNRESOLVED",
            )
        metrics["db_write_wall_seconds"] += time.perf_counter() - tick
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            frame_iter = enumerate(container.decode(stream))
            while True:
                decode_tick = time.perf_counter()
                try:
                    frame_id, frame = next(frame_iter)
                except StopIteration:
                    metrics["decode_wall_seconds"] += time.perf_counter() - decode_tick
                    break
                metrics["decode_wall_seconds"] += time.perf_counter() - decode_tick
                if frame_id < args.start_frame:
                    continue
                if count >= args.max_frames:
                    break
                if (frame.width, frame.height) != (layout.source_width,
                                                    layout.source_height):
                    raise ValueError(f"Frame {frame_id} dimensions "
                                     f"{frame.width}x{frame.height} differ from "
                                     f"layout {layout.source_width}x{layout.source_height}")
                time_base = frame.time_base or stream.time_base
                buffered_frames.append((source_id, frame_id, frame.pts,
                                  time_base.numerator if frame.pts is not None else None,
                                  time_base.denominator if frame.pts is not None else None,
                                  frame.width, frame.height))
                tick = time.perf_counter()
                image = frame.to_image()
                metrics["frame_to_image_wall_seconds"] += time.perf_counter() - tick
                crop_jobs = []
                for slot in layout.slots:
                    session_id = sessions[slot.slot_id]
                    for roi_type, box in (("table", slot.table_box),
                                          ("chat", slot.chat_box)):
                        observation_id = stable_id("obs", source_id, frame_id,
                                                   session_id, roi_type, box)
                        key = (session_id, roi_type)
                        previous = last_by_roi.get(key)
                        candidate_relative = (Path("crops") / session_id /
                                              f"{frame_id:09d}_{roi_type}.png")
                        future = executor.submit(timed_crop, image, box,
                                                 output / candidate_relative, previous)
                        crop_jobs.append((session_id, roi_type, box, observation_id,
                                          key, previous, candidate_relative, future))
                for (session_id, roi_type, box, observation_id, key, previous,
                     candidate_relative, future) in crop_jobs:
                        digest, pixel_sha, reused, hash_wall, png_wall = future.result()
                        metrics["pixel_hash_wall_seconds"] += hash_wall
                        metrics["png_encode_write_wall_seconds"] += png_wall
                        if reused:
                            relative = previous["relative"]
                            reused_crops += 1
                        else:
                            relative = candidate_relative
                            last_by_roi[key] = {
                                "pixel_sha256": pixel_sha,
                                "crop_sha256": digest,
                                "relative": relative,
                                "frame_id": frame_id,
                            }
                            saved_crops += 1
                        buffered_observations.append(((observation_id, source_id, frame_id,
                                                session_id, roi_type, list(box),
                                                str(relative), digest, "RAW"), {
                                                    "capture_disposition": (
                                                        "EXACT_PIXEL_REUSE" if reused
                                                        else "SAVED_PIXEL_CHANGE"
                                                    ),
                                                    "native_pixel_sha256": pixel_sha,
                                                    "selected_frame_id": (
                                                        previous["frame_id"] if reused
                                                        else frame_id
                                                    ),
                                                }))
                count += 1
                batch_frames += 1
                if batch_frames >= args.db_batch_frames:
                    commit_batch()
        if count == 0:
            raise ValueError("Selected source window contains no frames")
        commit_batch()
        state = "PARTIAL"
        tick = time.perf_counter()
        journal.set_source_state(source_id, state)
        metrics["db_write_wall_seconds"] += time.perf_counter() - tick
    except Exception:
        if batch_active:
            journal.end_batch(commit=False)
            batch_active = False
        try:
            journal.set_source_state(source_id, "FAILED")
        except Exception:
            pass
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        summary = {
            "source_id": source_id, "source_sha256": source_sha,
            "layout_id": layout.layout_id, "layout_sha256": layout_sha,
            "profile_sha256": profile_sha,
            "calibration_sha256": calibration_sha,
            "table_sessions": sessions,
            "start_frame": args.start_frame, "frames_this_run": count,
            "crop_storage": {
                "policy": "EXACT_PIXEL_REUSE_V1",
                "saved_crops": saved_crops,
                "reused_observations": reused_crops,
            },
            "performance": {
                "db_batch_frames": args.db_batch_frames,
                "db_batch_commits": batch_commits,
                "capture_workers": args.capture_workers,
                "wall_seconds": round(time.perf_counter() - started_tick, 6),
                "process_cpu_seconds": round(time.process_time() - started_cpu, 6),
                **{key: round(value, 6) for key, value in metrics.items()},
            },
            "state": state, "identity_status": "UNRESOLVED",
            "interpretation": "RAW_ONLY_FIXED_LAYOUT_EXPERIMENT",
            "counts": journal.counts(),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        journal.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
