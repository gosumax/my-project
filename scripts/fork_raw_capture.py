"""Fork frozen raw frames/ROIs into a clean journal for algorithm A/B runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.journal import Journal


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source_run.resolve(strict=True)
    output = args.output.resolve()
    if output == source or source.is_relative_to(output) or output.is_relative_to(source):
        parser.error("Source and output runs must be separate directories")
    old = Journal(source / "journal.sqlite3")
    new = Journal(output / "journal.sqlite3")
    try:
        sources = old.db.execute("SELECT * FROM sources").fetchall()
        sessions = old.db.execute("SELECT * FROM table_sessions").fetchall()
        frames = old.db.execute("SELECT * FROM frames ORDER BY source_id,frame_id").fetchall()
        observations = old.db.execute("""SELECT observation_id,source_id,frame_id,
            table_session_id,roi_type,bbox_json,crop_path,crop_sha256,status,reason
            FROM observations WHERE roi_type IN ('chat','table')
            ORDER BY source_id,frame_id,roi_type""").fetchall()
        for source_id, path, digest, size, _ in sources:
            new.add_source(source_id, path, digest, size)
        for session_id, source_id, layout_id, geometry_json, identity_status in sessions:
            new.add_session(session_id, source_id, layout_id,
                            json.loads(geometry_json), identity_status)
        new.begin_batch()
        for source_id, frame_id, pts, numerator, denominator, width, height, state in frames:
            new.add_frame(source_id, frame_id, pts, numerator, denominator,
                          width, height, state)
        for obs_id, source_id, frame_id, session_id, roi_type, bbox_json, relative, digest, status, reason in observations:
            origin = (source / relative).resolve(strict=True)
            destination = (output / relative).resolve()
            if not origin.is_relative_to(source) or not destination.is_relative_to(output):
                raise ValueError(f"Crop escapes run directory: {relative}")
            if sha256(origin) != digest:
                raise ValueError(f"Source crop hash mismatch: {origin}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha256(destination) != digest:
                    raise ValueError(f"Existing output crop differs: {destination}")
            else:
                shutil.copy2(origin, destination)
            new.add_observation(obs_id, source_id, frame_id, session_id, roi_type,
                                json.loads(bbox_json), relative, digest, status, reason)
        new.end_batch()
        for source_id, _, _, _, state in sources:
            new.set_source_state(source_id, state)
        report = {"forked_from": str(source), "raw_frames": len(frames),
                  "raw_observations": len(observations),
                  "counts": new.counts(), "purpose": "FROZEN_INPUT_REPLAY"}
        (output / "fork_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception:
        if new._batched:
            new.end_batch(commit=False)
        raise
    finally:
        old.close()
        new.close()


if __name__ == "__main__":
    main()
