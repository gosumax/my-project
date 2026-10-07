"""Bounded, source-linked CHAT card-row sampling from Smoke15 video.

This diagnostic is not a full tracking run and assigns no card ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parser_core.chat_rows import exclude_first_visible_span, padded_row_span, physical_spans
from parser_core.layout import load_fixed_layout
from parser_core.visual_tile_queue import card_tile_candidate
from experiments.build_verified_chat_card_tiles import spans_like_production


VIDEO = ROOT / "video/smoke_15.mp4"
OUT = ROOT / "runs/smoke15_chat_card_sample_v2_20260930"
CONFIGS = (ROOT / "configs/nine_table_layout.json", ROOT / "configs/layout_847x404.json",
           ROOT / "configs/chat_log_bottom_rois.json")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def png(path: Path, pixels: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise ValueError(f"PNG encoding failed: {path}")
    payload = encoded.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"Conflicting PNG at {path}")
    path.write_bytes(payload)
    return sha(payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-seconds", type=int, default=20)
    ap.add_argument("--max-samples", type=int, default=34)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--video", type=Path, default=VIDEO)
    args = ap.parse_args()
    if args.step_seconds < 1 or args.max_samples < 1 or args.output.exists():
        ap.error("Positive bounds and a fresh output directory are required")
    layout = load_fixed_layout(*CONFIGS)
    with args.video.open("rb") as stream:
        video_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    config_sha = {str(path): sha(path.read_bytes()) for path in CONFIGS}
    args.output.mkdir(parents=True)
    records = []
    sampled = []
    rejection_ledger = []
    with av.open(str(args.video)) as source:
        stream = source.streams.video[0]
        time_base = stream.time_base
        duration = float(stream.duration * time_base) if stream.duration else None
        for sample_index in range(args.max_samples):
            target_s = sample_index * args.step_seconds
            if duration is not None and target_s >= duration:
                break
            source.seek(max(0, int(target_s / time_base)), stream=stream, backward=True)
            frame = None
            for candidate in source.decode(stream):
                if candidate.pts is not None and float(candidate.pts * time_base) >= target_s:
                    frame = candidate
                    break
            if frame is None:
                break
            rgb = frame.to_ndarray(format="rgb24")
            if (rgb.shape[1], rgb.shape[0]) != (layout.source_width, layout.source_height):
                raise ValueError("Video resolution differs from fixed layout")
            actual_s = float(frame.pts * time_base)
            sampled.append({"sample_index": sample_index, "requested_time_s": target_s,
                            "decoded_time_s": actual_s, "source_pts": frame.pts})
            for slot in layout.slots:
                x1, y1, x2, y2 = slot.chat_box
                roi_rgb = rgb[y1:y2, x1:x2]
                ok, packed = cv2.imencode(".png", cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR),
                                          [cv2.IMWRITE_PNG_COMPRESSION, 0])
                if not ok:
                    raise ValueError("Chat ROI encoding failed")
                gray = cv2.imdecode(packed, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise ValueError("Chat ROI grayscale conversion failed")
                roi_path = args.output / "chat_rois" / f"s{sample_index:03d}_t{slot.slot_id:02d}.png"
                roi_sha = png(roi_path, gray)
                all_spans = physical_spans(gray)
                usable, suppressed = exclude_first_visible_span(all_spans)
                candidates = []
                for index, _ in enumerate(usable):
                    source_index = index + len(suppressed)
                    row_y0, row_y1 = padded_row_span(all_spans, source_index, gray.shape[0])
                    row = np.ascontiguousarray(gray[row_y0:row_y1])
                    reason = None
                    if not card_tile_candidate(row):
                        reason = "CARD_TILE_CANDIDATE_FALSE"
                    elif row.shape[0] != 16 or row.shape[1] not in (224, 226, 228, 230):
                        reason = "NONPRODUCTION_GEOMETRY"
                    spans = []
                    if reason is None:
                        trimmed = row[:, :-12]
                        spans = spans_like_production(trimmed)
                        gap_ok = all(2 <= spans[i+1][0] - spans[i][1] - 1 <= 5 for i in range(len(spans)-1))
                        if not (2 <= len(spans) <= 5 and gap_ok):
                            reason = "SPAN_COUNT_OR_GAP_MISMATCH"
                    rejection_ledger.append({"sample_index": sample_index, "source_pts": frame.pts,
                                             "table_id": slot.slot_id, "chat_roi_png": str(roi_path),
                                             "chat_roi_sha256": roi_sha, "source_span_index": source_index,
                                             "row_y0_y1": [row_y0, row_y1],
                                             "row_pixel_sha256": sha(row.tobytes()),
                                             "raw_size_wh": [row.shape[1], row.shape[0]],
                                             "detected_card_spans_inclusive": spans,
                                             "status": "CANDIDATE" if reason is None else "EXCLUDED",
                                             "reason": reason})
                    if reason is None:
                        candidates.append((source_index, row_y0, row_y1, row, spans))
                if not candidates:
                    continue
                for source_index, row_y0, row_y1, row, spans in candidates:
                    row_path = args.output / "chat_rows" / f"s{sample_index:03d}_t{slot.slot_id:02d}_r{source_index:03d}.png"
                    row_sha = png(row_path, row)
                    records.append({"source_video": str(args.video.resolve()), "source_video_sha256": video_sha,
                                    "sample_index": sample_index, "requested_time_s": target_s,
                                    "decoded_time_s": actual_s, "source_pts": frame.pts,
                                    "time_base_num": time_base.numerator, "time_base_den": time_base.denominator,
                                    "table_id": slot.slot_id, "chat_box_xyxy": slot.chat_box,
                                    "source_span_index": source_index, "row_y0_y1": [row_y0, row_y1],
                                    "chat_roi_png": str(roi_path), "chat_roi_sha256": roi_sha,
                                    "chat_row_png": str(row_path), "chat_row_sha256": row_sha,
                                    "row_pixel_sha256": sha(row.tobytes()),
                                    "raw_size_wh": [row.shape[1], row.shape[0]],
                                    "production_right_trim_px": 12,
                                    "detected_card_spans_inclusive": spans,
                                    "gt_status": "UNREVIEWED"})
    with (args.output / "candidates.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.output / "row_selection_ledger.jsonl").open("w", encoding="utf-8") as output:
        for record in rejection_ledger:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"status": "BOUNDED_CHAT_SAMPLE_UNREVIEWED", "source_video": str(args.video.resolve()),
               "source_video_sha256": video_sha, "layout_config_sha256": config_sha,
               "step_seconds": args.step_seconds, "max_samples": args.max_samples,
               "sampled_frames": sampled, "candidate_rows": len(records),
               "unique_row_pixels": len({r["row_pixel_sha256"] for r in records}),
               "saved_chat_rois": len(sampled) * len(layout.slots),
               "row_selection_decisions": len(rejection_ledger),
               "excluded_rows": sum(r["status"] == "EXCLUDED" for r in rejection_ledger),
               "note": "No card labels or independent test status assigned; unsampled intervals and excluded rows are unresolved for card presence."}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"sampled_frames", "layout_config_sha256"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
