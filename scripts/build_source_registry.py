"""Hash source videos and record documented exposure; never infer independence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.annotation import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if not target.is_relative_to(ROOT) or target.exists():
        parser.error("Select a new output file within this project")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    sources = []
    for item in plan["sources"]:
        path = Path(item["path"]).resolve(strict=True)
        print(f"Hashing {path.name} ({path.stat().st_size} bytes)", flush=True)
        record = {**item, "sha256": sha256(path), "size_bytes": path.stat().st_size,
                  "evidence_documents": [{"path": p, "sha256": sha256(Path(p))}
                                         for p in item["evidence_documents"]]}
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            record["stream_metadata"] = {"width": stream.width, "height": stream.height,
                "reported_frames": stream.frames, "duration_pts": stream.duration,
                "time_base": str(stream.time_base), "average_rate": str(stream.average_rate),
                "note": "Container metadata only; no frame decode or layout validation"}
        sources.append(record)
    write_json(target, {"version": "source_registry_v1", "plan_sha256": sha256(args.plan),
                       "sources": sources, "independent_test_sources": [],
                       "session_grouping": "UNKNOWN: all documented legacy sources kept in DEV",
                       "note": "Prior exposure is documented; absence of exposure cannot be proved by filenames."})
    print(f"Saved {len(sources)} source hashes to {target}")


if __name__ == "__main__":
    main()
