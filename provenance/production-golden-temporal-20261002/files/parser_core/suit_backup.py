"""Hash-pinned, validation-gated NCC suit channel for CHAT card tiles."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Reuse the exact ROI, resize, normalization and scoring tested offline.
from experiments.suit_template_experiment import features, predict


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SuitBackup:
    def __init__(self, config_path: Path):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["mode"] not in ("OFF", "SHADOW", "BACKUP"):
            raise ValueError("Unknown suit backup mode")
        if (config["method"] != "grayscale_ncc" or config["size"] != 16 or
                config["roi_xyxy_exclusive"] != [10, 2, 19, 13]):
            raise ValueError("Suit backup algorithm geometry changed")
        implementation = Path(__file__).resolve().parents[1] / "experiments/suit_template_experiment.py"
        if sha(implementation.read_bytes()) != config["implementation_sha256"]:
            raise ValueError("Suit backup implementation changed")
        self.mode = config["mode"]
        self.best_threshold = float(config["best_score_threshold"])
        self.margin_threshold = float(config["margin_threshold"])
        self.config_sha256 = sha(config_path.read_bytes())
        self.bank = None
        if self.mode == "OFF":
            return
        manifest = Path(config["train_manifest"])
        if sha(manifest.read_bytes()) != config["train_manifest_sha256"]:
            raise ValueError("Suit TRAIN manifest changed")
        results_path = Path(config["validation_results"])
        if sha(results_path.read_bytes()) != config["validation_results_sha256"]:
            raise ValueError("Suit validation results changed")
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        if len(rows) != 431 or {Path(r["source_video"]).name.lower() for r in rows} != {"smoke_12.mp4", "smoke_13.mp4"}:
            raise ValueError("Suit TRAIN source changed")
        images = []
        for row in rows:
            payload = Path(row["tile_png"]).read_bytes()
            if sha(payload) != row["tile_sha256"]:
                raise ValueError("Suit template PNG changed")
            with Image.open(io.BytesIO(payload)) as opened:
                rgb = np.asarray(opened.convert("RGB"))
            if sha(rgb.tobytes()) != row["tile_pixel_sha256"]:
                raise ValueError("Suit template pixels changed")
            images.append(rgb)
        self.bank, _ = features(images, 16, "grayscale_ncc")
        self.suits = np.array([r["ground_truth"][1] for r in rows])
        if set(self.suits) != set("shdc"):
            raise ValueError("Suit template bank incomplete")
        with results_path.open(encoding="utf-8-sig", newline="") as stream:
            validation = [r for r in csv.DictReader(stream) if r["split"] == "validation" and
                          r["method"] == "grayscale_ncc" and r["size"] == "16"]
        if len(validation) != 318 or any(r["true_suit"] != r["predicted_suit"] for r in validation):
            raise ValueError("Suit validation result changed")
        self.validation_pass = sum(float(r["best_score"]) >= self.best_threshold and
                                   float(r["margin"]) >= self.margin_threshold for r in validation)

    def read(self, rgb: np.ndarray) -> dict:
        if self.mode == "OFF":
            return {"template_suit": "UNRESOLVED", "template_confident": False,
                    "suit_backup_mode": "OFF"}
        if rgb.shape[0] != 16 or rgb.shape[1] not in (20, 21) or rgb.shape[2] != 3:
            return {"template_suit": "UNRESOLVED", "template_confident": False,
                    "suit_backup_mode": self.mode, "template_reason": "UNSUPPORTED_TILE_GEOMETRY"}
        query, reds = features([rgb], 16, "grayscale_ncc")
        suit, best, second, other = predict(self.bank, self.suits, query[0], bool(reds[0]), "grayscale_ncc")
        margin = best-second
        return {"template_suit": suit, "template_best_score": best,
                "template_second_score": second, "template_margin": margin,
                "template_second_suit": other,
                "template_confident": best >= self.best_threshold and margin >= self.margin_threshold,
                "suit_backup_mode": self.mode, "suit_backup_config_sha256": self.config_sha256}


def decide_suit(primary: str | None, primary_confident: bool, template: dict) -> dict:
    """Keep conflicts visible; SHADOW never changes the primary outcome."""
    mode = template["suit_backup_mode"]
    candidate = template.get("template_suit")
    confident = template.get("template_confident", False)
    primary = primary if primary is not None and primary in "shdc" else None
    if mode == "OFF":
        return {"suit": primary or "UNRESOLVED", "suit_source": "primary" if primary else "UNRESOLVED"}
    if primary and candidate == primary:
        return {"suit": primary, "suit_source": "agreement"}
    if mode == "SHADOW":
        return {"suit": primary or "UNRESOLVED", "suit_source": "primary_shadow" if primary else "UNRESOLVED"}
    if primary and primary_confident and confident:
        return {"suit": "UNRESOLVED", "suit_source": "conflict", "suit_status": "SUIT_CONFLICT"}
    if confident and (not primary or not primary_confident):
        return {"suit": candidate, "suit_source": "template_backup",
                "suit_status": "PRIMARY_UNCALIBRATED_DISAGREEMENT" if primary and primary != candidate else None}
    return {"suit": primary or "UNRESOLVED", "suit_source": "primary" if primary else "UNRESOLVED",
            "suit_status": "TEMPLATE_LOW_CONFIDENCE" if not confident else None}
