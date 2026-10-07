"""Derived card crops from hash-verified saved table observations."""

from __future__ import annotations

import hashlib
import base64
import io
import json
import sqlite3
from pathlib import Path

from PIL import Image
import numpy as np


def exact_pixel_sha256(image: Image.Image) -> str:
    """Hash the exact decoded RGB crop pixels, independent of PNG encoding."""
    rgb = image.convert("RGB")
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def board_crop_box(slot: dict, street: str) -> list[int]:
    fields = {"FLOP": ("board_flop",), "TURN": ("board_flop", "board_turn"),
              "RIVER": ("board_flop", "board_turn", "board_river")}
    boxes = [slot["global"][field] for field in fields[street]]
    if any(box is None for box in boxes):
        raise ValueError("Visual crop geometry is not calibrated")
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def snapshot_ram_board(table_rgb: np.ndarray, slot: dict, **identity) -> dict:
    """Retain only the board union from an exact decoded frame, never a TABLE PNG."""
    box = board_crop_box(slot, "RIVER")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= table_rgb.shape[1] and 0 <= y1 < y2 <= table_rgb.shape[0]):
        raise ValueError("BOARD snapshot is outside table pixels")
    pixels = np.ascontiguousarray(table_rgb[y1:y2, x1:x2]).copy()
    return {**identity, "pixels": pixels, "union_box": box,
            "street_boxes": {s: board_crop_box(slot, s) for s in ("FLOP", "TURN", "RIVER")},
            "snapshot_pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "source_table_pixel_sha256": hashlib.sha256(table_rgb.tobytes()).hexdigest()}


def crop_ram_board(snapshot: dict, street: str) -> tuple[bytes, dict]:
    pixels = snapshot["pixels"]
    if hashlib.sha256(pixels.tobytes()).hexdigest() != snapshot["snapshot_pixel_sha256"]:
        raise ValueError("RAM BOARD snapshot pixels changed")
    box = snapshot["street_boxes"][street]
    left, top = snapshot["union_box"][:2]
    image = Image.fromarray(pixels).crop((box[0]-left, box[1]-top, box[2]-left, box[3]-top))
    packed = io.BytesIO()
    image.save(packed, format="PNG")
    data = packed.getvalue()
    metadata = {k: v for k, v in snapshot.items() if k not in {"pixels", "union_box", "street_boxes"}}
    return data, {**metadata, "crop_type": "BOARD", "street": street,
                  "crop_box_local": box, "crop_path": None, "image_transport": "ram",
                  "crop_png_base64": base64.b64encode(data).decode("ascii"),
                  "crop_sha256": hashlib.sha256(data).hexdigest(),
                  "board_sha": exact_pixel_sha256(image)}


def crop_saved_table(baseline: Path, profile: Path, output: Path, *,
                     session_id: str, table_id: int, frame_id: int,
                     kind: str, street: str | None = None,
                     seat: str | None = None, row_y1: int | None = None,
                     row_height: int | None = None) -> dict:
    """Read the exact source frame; missing evidence stays unresolved."""
    with sqlite3.connect(baseline / "journal.sqlite3") as db:
        result = db.execute("""SELECT o.crop_path, o.crop_sha256, o.bbox_json,
                                   f.source_pts, f.time_base_num, f.time_base_den,
                                   o.source_id, s.sha256
            FROM observations o JOIN frames f ON f.source_id=o.source_id
              AND f.frame_id=o.frame_id
              JOIN sources s ON s.source_id=o.source_id
            WHERE o.table_session_id=? AND o.frame_id=? AND o.roi_type='table'""",
            (session_id, frame_id)).fetchone()
    if result is None:
        raise FileNotFoundError(f"No exact table observation for {session_id}:{frame_id}")
    relative_path, parent_sha, source_box, pts, base_num, base_den, source_id, video_sha = result
    source_path = (baseline / relative_path).resolve(strict=True)
    if not source_path.is_relative_to(baseline.resolve()):
        raise ValueError("Table crop path escapes baseline")
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != parent_sha:
        raise ValueError("Saved table crop SHA256 mismatch")
    layout = json.loads(profile.read_text("utf-8"))
    slot = layout["slots"][f"slot_{table_id:02d}"]
    if kind == "BOARD":
        boxes = [board_crop_box(slot, street)]
    elif kind == "SHOWDOWN" and seat in slot["seats"]:
        boxes = [slot["seats"][seat]["shown_cards"]]
    elif kind == "CHAT_ROW_CARDS" and row_y1 is not None and row_height is not None:
        chat = slot["chat"]["chat_log"]
        boxes = [[chat[0], chat[1] + row_y1,
                  chat[2], chat[1] + row_y1 + row_height]]
    else:
        raise ValueError("Unresolved visual crop type or player seat")
    if any(box is None for box in boxes):
        raise ValueError("Visual crop geometry is not calibrated")
    box = [min(item[0] for item in boxes), min(item[1] for item in boxes),
           max(item[2] for item in boxes), max(item[3] for item in boxes)]
    with Image.open(io.BytesIO(source)) as opened:
        image = opened.convert("RGB")
        if not (0 <= box[0] < box[2] <= image.width and
                0 <= box[1] < box[3] <= image.height):
            raise ValueError("Visual crop is outside saved table image")
        crop = image.crop(tuple(box))
        pixel_sha = exact_pixel_sha256(crop)
        packed = io.BytesIO()
        crop.save(packed, format="PNG")
    data = packed.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and hashlib.sha256(output.read_bytes()).hexdigest() != digest:
        raise ValueError("Existing visual crop conflicts with source pixels")
    output.write_bytes(data)
    return {"crop_type": kind, "street": street, "seat": seat,
            "table_id": table_id, "table_session_id": session_id,
            "frame_id": frame_id, "source_pts": pts,
            "time_base_num": base_num, "time_base_den": base_den,
            "source_table_path": str(source_path),
            "source_id": source_id, "source_video_sha256": video_sha,
            "source_table_sha256": parent_sha,
            "source_table_box": json.loads(source_box),
            "crop_box_local": box, "crop_path": str(output.resolve()),
            "crop_sha256": digest, "board_sha": pixel_sha if kind == "BOARD" else None}
