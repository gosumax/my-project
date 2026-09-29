"""Compare bounded OCR retries on saved row PNGs without changing the journal."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.chat_rows import ROW_TRACKER_VERSION


ROOT = Path(__file__).resolve().parents[1]
PROBE_VERSION = "upscale3_psm7_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_upscaled(crop: Path, tesseract: Path, tessdata: Path) -> tuple[int, str, str]:
    with Image.open(crop) as opened:
        gray = opened.convert("L")
        upscaled = gray.resize((gray.width * 3, gray.height * 3),
                                Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        upscaled.save(buffer, format="PNG")
    result = subprocess.run(
        [str(tesseract), "stdin", "stdout", "-l", "rus+eng", "--psm", "7",
         "--tessdata-dir", str(tessdata)],
        input=buffer.getvalue(), capture_output=True, timeout=60)
    return (result.returncode, result.stdout.decode("utf-8", "replace").strip(),
            result.stderr.decode("utf-8", "replace").strip()[:1000])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controls", type=int, default=20)
    parser.add_argument("--max-unreadable", type=int, default=100)
    parser.add_argument("--tesseract", type=Path,
                        default=Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"))
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    output = args.output.resolve()
    if not run.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        parser.error("Input run and output must remain inside the new parser project")
    if args.controls < 0 or args.max_unreadable < 1:
        parser.error("Controls must be nonnegative; max-unreadable must be positive")
    tesseract = args.tesseract.resolve(strict=True)
    tessdata = ROOT / "models" / "tessdata"
    version = subprocess.check_output([str(tesseract), "--version"],
                                      text=True).splitlines()[0]
    with sqlite3.connect(run / "journal.sqlite3") as db:
        rows = db.execute("""SELECT o.observation_id,o.crop_path,o.crop_sha256,
            r.recognition_id,r.raw_text,r.status
            FROM physical_rows p
            JOIN row_observations ro ON ro.physical_row_id=p.physical_row_id
            JOIN observations o ON o.observation_id=ro.observation_id
            JOIN recognitions r ON r.observation_id=o.observation_id
            WHERE p.chat_epoch LIKE ? AND r.engine='Tesseract'
              AND ro.ordinal=(SELECT MIN(ro2.ordinal) FROM row_observations ro2
                              WHERE ro2.physical_row_id=p.physical_row_id)
              AND r.revision=(SELECT MAX(r2.revision) FROM recognitions r2
                              WHERE r2.observation_id=o.observation_id
                                AND r2.engine='Tesseract')
            ORDER BY o.observation_id""",
            (ROW_TRACKER_VERSION + ":%",)).fetchall()
    unreadable = [row for row in rows if not (row[4] or "").strip()]
    readable = [row for row in rows if (row[4] or "").strip()]
    controls = sorted(readable, key=lambda row: hashlib.sha256(
        row[0].encode("utf-8")).hexdigest())[:args.controls]
    selected = [("UNREADABLE", row) for row in unreadable[:args.max_unreadable]]
    selected += [("CONTROL", row) for row in controls]
    results = []
    for group, (observation_id, relative, digest, recognition_id, raw_text,
                primary_status) in selected:
        crop = (run / relative).resolve(strict=True)
        if not crop.is_relative_to(run) or sha256(crop) != digest:
            raise ValueError(f"Crop path/hash mismatch: {crop}")
        code, retry_text, stderr = read_upscaled(crop, tesseract, tessdata)
        results.append({"group": group, "observation_id": observation_id,
                        "crop_path": relative, "crop_sha256": digest,
                        "primary_recognition_id": recognition_id,
                        "primary_raw_text": raw_text,
                        "primary_status": primary_status,
                        "retry_raw_text": retry_text,
                        "retry_exit_code": code, "retry_stderr": stderr})
    summary = {"unreadable_selected": len(selected) - len(controls),
               "unreadable_retry_nonempty": sum(item["group"] == "UNREADABLE" and
                                                bool(item["retry_raw_text"]) for item in results),
               "controls_selected": len(controls),
               "controls_retry_changed": sum(item["group"] == "CONTROL" and
                                             item["primary_raw_text"] != item["retry_raw_text"]
                                             for item in results),
               "controls_retry_empty": sum(item["group"] == "CONTROL" and
                                           not item["retry_raw_text"] for item in results)}
    report = {"probe_version": PROBE_VERSION, "interpretation": "RAW_ALTERNATIVES_ONLY",
              "run": str(run), "row_tracker_version": ROW_TRACKER_VERSION,
              "tesseract_version": version,
              "language_sha256": {language: sha256(tessdata / f"{language}.traineddata")
                                  for language in ("rus", "eng")},
              "retry_preprocessing": {"source": "saved_grayscale_row_png",
                                      "resize": "3x_LANCZOS", "psm": 7,
                                      "languages": ["rus", "eng"]},
              "summary": summary, "results": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
