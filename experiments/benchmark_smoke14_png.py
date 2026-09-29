"""Compare lossless PNG encoding settings on fixed saved Smoke14 crops."""

import argparse
import io
import json
import sqlite3
import time
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    with sqlite3.connect(run / "journal.sqlite3") as db:
        samples = []
        for roi in ("table", "chat"):
            rows = db.execute("SELECT crop_path FROM observations WHERE roi_type=? AND "
                              "quality_json LIKE '%SAVED_PIXEL_CHANGE%' ORDER BY frame_id",
                              (roi,)).fetchall()
            samples.extend((roi, str(run / rows[i * len(rows) // 100][0])) for i in range(100))
    images = [(roi, Image.open(path).copy()) for roi, path in samples]
    result = []
    for level in (None, 1, 0):
        started = time.perf_counter()
        total = 0
        for roi, image in images:
            buffer = io.BytesIO()
            options = {} if level is None else {"compress_level": level}
            image.save(buffer, format="PNG", **options)
            data = buffer.getvalue()
            total += len(data)
            with Image.open(io.BytesIO(data)) as decoded:
                if decoded.mode != image.mode or decoded.size != image.size or decoded.tobytes() != image.tobytes():
                    raise ValueError(f"Pixel parity failure: {roi}, {level}")
        result.append({"compress_level": level, "images": len(images),
                       "wall_seconds": time.perf_counter() - started,
                       "bytes": total, "pixel_parity": True})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
