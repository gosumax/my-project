"""Compare decoded table-video context to archived snapshots at hand starts."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "annotation" / "dev_smoke11_queue_20260927"
SNAPSHOTS = Path(r"D:\Проэкты\V2.0 PD\outputs\nine_table_eventlog_smoke11_final_20260927\snapshots")


def read_image(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def main() -> None:
    manifest = json.loads((QUEUE / "manifest.json").read_text(encoding="utf-8"))
    errors = []
    for hand in manifest["hand_windows"]:
        pattern = f"T{int(hand['table']):02d}_f{hand['start_frame']:08d}_*.png"
        reference = next(SNAPSHOTS.glob(pattern), None)
        if reference is None:
            continue
        decoded = read_image(ROOT / hand["start_image"]["local"])
        archived = read_image(reference)
        if decoded is None or archived is None or decoded.shape != archived.shape:
            raise ValueError(f"Cannot compare {hand['table']}:{hand['start_frame']}")
        errors.append(float(np.mean(cv2.absdiff(decoded, archived))))
    print({"pairs": len(errors), "mean_abs_min": min(errors),
           "mean_abs_max": max(errors), "mean_abs_mean": sum(errors) / len(errors)})


if __name__ == "__main__":
    main()
