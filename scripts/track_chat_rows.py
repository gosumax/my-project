"""Track physical chat rows in saved full-chat observations with fail-closed gaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.chat_rows import (ROW_TRACKER_VERSION, Alignment, align_upward,
                                   exclude_first_visible_span, match_rows,
                                   padded_row_span, physical_spans)
from parser_core.journal import Journal, stable_id


def read_gray(path: Path, expected_sha256: str) -> np.ndarray:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"Crop hash mismatch: {path}")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def save_row(gray: np.ndarray, span: tuple[int, int], path: Path) -> str:
    ok, encoded = cv2.imencode(".png", gray[span[0]:span[1]])
    if not ok:
        raise ValueError("Row crop encoding failed")
    data = encoded.tobytes()
    digest = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Existing row crop differs: {path}")
    else:
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    return digest


def save_or_reuse_row(gray, span, relative, run, cache):
    pixels = gray[span[0]:span[1]]
    key = (pixels.shape, hashlib.sha256(pixels.tobytes()).hexdigest())
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached[0], cached[1], True
    digest = save_row(gray, span, run / relative)
    cache[key] = (digest, relative)
    if len(cache) > 2048:
        cache.popitem(last=False)
    return digest, relative, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--max-observations", type=int)
    parser.add_argument("--watch-until", type=Path,
                        help="Consume committed capture rows until this completion marker exists")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    journal = Journal(run / "journal.sqlite3")
    query = """SELECT o.source_id,o.table_session_id,o.frame_id,
        f.source_pts,o.observation_id,o.bbox_json,o.crop_path,o.crop_sha256
        FROM observations o JOIN frames f ON f.source_id=o.source_id AND f.frame_id=o.frame_id
        WHERE o.roi_type='chat' AND o.frame_id>? ORDER BY o.frame_id,o.table_session_id"""
    input_count = 0
    started_tick = time.perf_counter()
    started_cpu = time.process_time()
    metrics = {"read_decode_wall_seconds": 0.0, "geometry_wall_seconds": 0.0,
               "row_png_wall_seconds": 0.0, "db_commit_wall_seconds": 0.0,
               "idle_wall_seconds": 0.0}
    prior_by_session: dict[str, dict] = {}
    resume_frame = -1
    if args.resume:
        session_count = journal.db.execute("SELECT COUNT(*) FROM table_sessions").fetchone()[0]
        row = journal.db.execute("""SELECT MAX(frame_id) FROM (
            SELECT frame_id FROM row_suppressions WHERE algorithm_version=?
            GROUP BY frame_id HAVING COUNT(DISTINCT table_session_id)=?)""",
            (ROW_TRACKER_VERSION, session_count)).fetchone()
        resume_frame = row[0] if row and row[0] is not None else -1
        if resume_frame >= 0:
            session_crops = journal.db.execute("""SELECT o.table_session_id,o.crop_path,
                o.crop_sha256,f.source_pts FROM observations o JOIN frames f
                ON f.source_id=o.source_id AND f.frame_id=o.frame_id
                WHERE o.roi_type='chat' AND o.frame_id=?""", (resume_frame,)).fetchall()
            for session_id, crop_path, digest, pts in session_crops:
                gray = read_gray(run / crop_path, digest)
                spans, suppressed = exclude_first_visible_span(physical_spans(gray))
                restored_rows = []
                stored = journal.db.execute("""SELECT p.physical_row_id,p.chat_epoch,
                    p.state,p.reason,o.quality_json FROM row_decisions d
                    JOIN observations o ON o.observation_id=d.observation_id
                    JOIN physical_rows p ON p.physical_row_id=d.physical_row_id
                    WHERE o.frame_id=? AND o.table_session_id=? AND d.algorithm_version=?""",
                    (resume_frame, session_id, ROW_TRACKER_VERSION)).fetchall()
                for row_id, epoch, origin, reason, quality_json in stored:
                    index = json.loads(quality_json)["source_span_index"] - len(suppressed)
                    restored_rows.append((index, {"id": row_id, "span": spans[index],
                                                  "origin": origin, "reason": reason}, epoch))
                restored_rows.sort(key=lambda item: item[0])
                prior_by_session[session_id] = {"image": gray,
                    "rows": [item[1] for item in restored_rows],
                    "epoch": (restored_rows[0][2] if restored_rows else
                              f"{ROW_TRACKER_VERSION}:{stable_id('epoch', session_id, resume_frame)}"),
                    "digest": digest,
                    "frame_id": resume_frame, "pts": pts, "suppressed": suppressed}
            if len(prior_by_session) != session_count:
                raise ValueError("Missing table session in tracker checkpoint")

    def stream_observations():
        last_frame = resume_frame
        while True:
            # Release the writer lock before waiting for the capture producer.
            if journal._batched:
                tick = time.perf_counter()
                journal.end_batch()
                metrics["db_commit_wall_seconds"] += time.perf_counter() - tick
            finished = args.watch_until is None or args.watch_until.exists()
            batch = journal.db.execute(query, (last_frame,)).fetchall()
            for row in batch:
                yield row
            if batch:
                last_frame = batch[-1][2]
            if finished:
                return
            tick = time.perf_counter()
            time.sleep(0.2)
            metrics["idle_wall_seconds"] += time.perf_counter() - tick
    counts: Counter[str] = Counter()
    row_crop_cache = OrderedDict()
    last_progress_tick = 0.0
    try:
        for sequence, (source_id, session_id, frame_id, pts, _, bbox_json, relative, digest) in enumerate(stream_observations(), 1):
            if args.max_observations is not None and sequence > args.max_observations:
                break
            input_count += 1
            if time.perf_counter() - last_progress_tick >= 5:
                last_progress_tick = time.perf_counter()
                (run / "tracking_progress.json").write_text(json.dumps({
                    "input_chat_observations": input_count,
                    "frame_id": frame_id, "decisions": dict(counts),
                    "performance": metrics,
                    "wall_seconds": time.perf_counter() - started_tick}), encoding="utf-8")
            if not journal._batched:
                journal.begin_batch()
            crop = (run / relative).resolve(strict=True)
            if not crop.is_relative_to(run):
                raise ValueError(f"Crop outside run: {crop}")
            previous = prior_by_session.get(session_id)
            if previous is not None and digest == previous["digest"]:
                for span in previous["suppressed"]:
                    journal.add_row_suppression(source_id, frame_id, session_id,
                                                span, ROW_TRACKER_VERSION,
                                                "FIRST_VISIBLE_ROW_EXCLUDED")
                    counts["SUPPRESSED_FIRST_VISIBLE_ROW"] += 1
                for old in previous["rows"]:
                    journal.add_physical_row(old["id"], session_id, previous["epoch"],
                                             pts, old["origin"], old["reason"])
                counts["unchanged_chat_frames"] += 1
                if sequence % 10 == 0:
                    journal.end_batch()
                    journal.begin_batch()
                continue
            tick = time.perf_counter()
            gray = read_gray(crop, digest)
            metrics["read_decode_wall_seconds"] += time.perf_counter() - tick
            tick = time.perf_counter()
            all_spans = physical_spans(gray)
            spans, suppressed = exclude_first_visible_span(all_spans)
            for span in suppressed:
                journal.add_row_suppression(source_id, frame_id, session_id,
                                            span, ROW_TRACKER_VERSION,
                                            "FIRST_VISIBLE_ROW_EXCLUDED")
                counts["SUPPRESSED_FIRST_VISIBLE_ROW"] += 1
            bbox = json.loads(bbox_json)
            if previous is None:
                epoch = f"{ROW_TRACKER_VERSION}:{stable_id('epoch', session_id, frame_id)}"
                matches = {}
                alignment = None
            else:
                old_spans = [row["span"] for row in previous["rows"]]
                zero_matches = match_rows(previous["image"], old_spans, gray, spans, 0)
                if len(zero_matches) == len(spans) == len(old_spans):
                    alignment = Alignment(0, 0.0, 0.0, True)
                else:
                    alignment = align_upward(previous["image"], gray)
                proposed_matches = (zero_matches if alignment.shift_px == 0 else
                                    match_rows(previous["image"], old_spans,
                                               gray, spans, alignment.shift_px))
                minimum_overlap = (max(1, min(len(old_spans), len(spans)) // 2)
                                   if alignment.shift_px == 0 else 3)
                if alignment.proven and len(proposed_matches) < minimum_overlap:
                    alignment = Alignment(alignment.shift_px, alignment.score,
                                          alignment.zero_score, False)
                if alignment.proven:
                    epoch = previous["epoch"]
                    matches = proposed_matches
                else:
                    epoch = f"{ROW_TRACKER_VERSION}:{stable_id('epoch', session_id, frame_id)}"
                    matches = {}
                    journal.add_gap(stable_id("gap", source_id, session_id,
                                              previous["frame_id"], frame_id),
                                    source_id, session_id, previous["pts"], pts,
                                    "CHAT_ALIGNMENT_UNRESOLVED")
                    counts["unresolved_alignments"] += 1
            metrics["geometry_wall_seconds"] += time.perf_counter() - tick
            current_rows = []
            for index, span in enumerate(spans):
                source_span_index = index + len(suppressed)
                if index in matches:
                    old = previous["rows"][matches[index]]
                    row_id, origin, reason = old["id"], old["origin"], old["reason"]
                    observation_status = "TRACKED"
                elif previous is None:
                    row_id = stable_id("row", epoch, frame_id, source_span_index)
                    origin, reason = "INITIAL_VISIBLE", "FIRST_OBSERVED_FRAME"
                    observation_status = origin
                else:
                    row_id = stable_id("row", epoch, frame_id, source_span_index)
                    newly_exposed = (alignment is not None and alignment.proven and (
                        (alignment.shift_px > 0 and span[0] >= gray.shape[0] - alignment.shift_px - 8)
                        or (alignment.shift_px == 0 and previous["rows"] and
                            span[0] > max(row["span"][1] for row in previous["rows"]) + 5)))
                    origin = "NEW_CANDIDATE" if newly_exposed else "UNRESOLVED_IDENTITY"
                    reason = None if newly_exposed else (
                        "CHAT_ALIGNMENT_UNRESOLVED" if not alignment or not alignment.proven
                        else "NO_PROVEN_ROW_MATCH")
                    observation_status = origin
                crop_span = padded_row_span(all_spans, source_span_index,
                                            gray.shape[0])
                full_bbox = [bbox[0], bbox[1] + crop_span[0],
                             bbox[2], bbox[1] + crop_span[1]]
                obs_id = stable_id("obs", source_id, frame_id, session_id,
                                   "CHAT_ROW", full_bbox)
                row_relative = Path("crops") / "rows" / session_id / f"{frame_id:09d}_{source_span_index:03d}.png"
                tick = time.perf_counter()
                row_digest, row_relative, reused = save_or_reuse_row(
                    gray, crop_span, row_relative, run, row_crop_cache)
                counts["reused_row_crops" if reused else "saved_row_crops"] += 1
                metrics["row_png_wall_seconds"] += time.perf_counter() - tick
                edge_contact = span[0] == 0 or span[1] == gray.shape[0]
                journal.add_observation(obs_id, source_id, frame_id, session_id,
                                        "CHAT_ROW", full_bbox, str(row_relative),
                                        row_digest,
                                        "RAW_PARTIAL" if edge_contact else "RAW",
                                        "CHAT_SOURCE_EDGE_CONTACT" if edge_contact else None,
                                        quality={
                                            "tight_text_span_local_y": list(span),
                                            "crop_span_local_y": list(crop_span),
                                            "source_span_index": source_span_index,
                                            "target_height_px": 16,
                                            "neighbour_guard_px": 2,
                                            "source_top_contact": span[0] == 0,
                                            "source_bottom_contact": span[1] == gray.shape[0],
                                            "segmentation_excluded_right_px": 12 if gray.shape[1] >= 50 else 0,
                                        })
                journal.add_physical_row(row_id, session_id, epoch, pts, origin, reason)
                journal.link_row_observation(row_id, obs_id, frame_id)
                journal.add_row_decision(row_id, obs_id, ROW_TRACKER_VERSION,
                                         observation_status, reason,
                                         alignment.shift_px if alignment else None,
                                         alignment.score if alignment else None,
                                         alignment.zero_score if alignment else None)
                current_rows.append({"id": row_id, "span": span, "origin": origin,
                                     "reason": reason})
                counts[observation_status] += 1
            prior_by_session[session_id] = {"image": gray, "rows": current_rows,
                                            "epoch": epoch, "digest": digest,
                                            "frame_id": frame_id, "pts": pts,
                                            "suppressed": suppressed}
            counts["changed_chat_frames"] += 1
            if sequence % 10 == 0:
                journal.end_batch()
                journal.begin_batch()
        if journal._batched:
            journal.end_batch()
    except Exception:
        if journal._batched:
            journal.end_batch(commit=False)
        raise
    finally:
        totals = journal.counts()
        journal.close()
    report = {"input_chat_observations": input_count, "decisions": dict(counts),
              "resumed_after_frame": resume_frame,
              "performance": {"wall_seconds": round(time.perf_counter() - started_tick, 6),
                              "process_cpu_seconds": round(time.process_time() - started_cpu, 6),
                              **{k: round(v, 6) for k, v in metrics.items()}},
              "journal_counts": totals, "status": "ROW_CANDIDATES_ONLY"}
    target = run / "row_tracking_summary.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
