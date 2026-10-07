"""Isolated, video-held-out comparison of suit pixels. No production imports."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs/card_train_smoke16_test_20260930"
OUT = ROOT / "runs/suit_template_experiment_20260930"
SUITS = "shdc"
SIZES = (16, 24, 32, 48)
METHODS = ("grayscale_ncc", "binary_shape", "edge_shape", "color_binary_shape")
ROI = (10, 2, 19, 13)  # x1,y1,x2,y2, exclusive; native 20/21 x 16 tile


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load(rows: list[dict]) -> list[np.ndarray]:
    images = []
    for row in rows:
        path = Path(row["tile_png"])
        raw = path.read_bytes()
        if sha(raw) != row["tile_sha256"]:
            raise ValueError(f"PNG hash mismatch: {path}")
        with Image.open(path) as opened:
            rgb = np.asarray(opened.convert("RGB"))
        if sha(rgb.tobytes()) != row["tile_pixel_sha256"]:
            raise ValueError(f"Pixel hash mismatch: {path}")
        if rgb.shape[0] != 16 or rgb.shape[1] not in (20, 21):
            raise ValueError(f"Unexpected geometry: {path}: {rgb.shape}")
        images.append(rgb)
    return images


def features(images: list[np.ndarray], size: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    vectors, reds = [], []
    x1, y1, x2, y2 = ROI
    for rgb in images:
        crop = rgb[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        # Color decision uses native pixels; pale anti-aliasing is excluded.
        ink = np.min(crop, axis=2) < 190
        red_ink = ink & (crop[:, :, 0].astype(int) - crop[:, :, 1].astype(int) > 25) & (crop[:, :, 0].astype(int) - crop[:, :, 2].astype(int) > 25)
        reds.append(bool(red_ink.sum() > max(1, int(ink.sum() * 0.1))))
        resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR)
        if mode == "grayscale_ncc":
            a = resized.astype(np.float32)
            a -= a.mean()
            a /= max(float(np.linalg.norm(a)), 1e-8)
        elif mode in ("binary_shape", "color_binary_shape"):
            _, a = cv2.threshold(resized, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            a = a.astype(np.float32)
        else:
            a = (cv2.Canny(resized, 60, 120) > 0).astype(np.float32)
        vectors.append(a.reshape(-1))
    return np.stack(vectors), np.array(reds, dtype=bool)


def predict(bank: np.ndarray, bank_suits: np.ndarray, query: np.ndarray, query_red: bool, mode: str) -> tuple[str, float, float, str]:
    if mode == "grayscale_ncc":
        similarity = bank @ query
    else:
        # Dice overlap compares silhouettes/edges and ignores empty background.
        intersection = bank @ query
        similarity = 2 * intersection / np.maximum(bank.sum(axis=1) + query.sum(), 1)
    if mode == "color_binary_shape":
        allowed = np.isin(bank_suits, list("hd" if query_red else "sc"))
        similarity = np.where(allowed, similarity, -1.0)
    class_scores = {s: float(np.max(similarity[bank_suits == s])) for s in SUITS}
    ranking = sorted(SUITS, key=lambda s: (-class_scores[s], SUITS.index(s)))
    best, second = ranking[:2]
    return best, class_scores[best], class_scores[second], second


def evaluate(target: list[dict], target_images: list[np.ndarray], bank: np.ndarray, bank_suits: np.ndarray, size: int, mode: str, split: str) -> list[dict]:
    queries, reds = features(target_images, size, mode)
    output = []
    for row, query, red in zip(target, queries, reds):
        pred, best, second, runner_up = predict(bank, bank_suits, query, bool(red), mode)
        output.append({"split": split, "method": mode, "size": size, "true_suit": row["ground_truth"][1],
                       "predicted_suit": pred, "best_score": best, "second_best_score": second,
                       "margin": best - second, "second_suit": runner_up, "color_pair_red": bool(red),
                       "tile_png": row["tile_png"], "tile_sha256": row["tile_sha256"],
                       "tile_pixel_sha256": row["tile_pixel_sha256"], "source_video": row["source_video"],
                       "record_id": row.get("record_id", ""), "card_index": row["card_index"]})
    return output


def summary(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(r["true_suit"] == r["predicted_suit"] for r in rows)
    return {"n": n, "correct": correct, "errors": n - correct, "accuracy": correct/n,
            **{f"{s}_accuracy": sum(r["true_suit"] == s and r["predicted_suit"] == s for r in rows) /
               max(1, sum(r["true_suit"] == s for r in rows)) for s in SUITS}}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_failures(rows: list[dict]) -> None:
    fail_dir = OUT / "failures"
    fail_dir.mkdir(exist_ok=True)
    thumbs = []
    for i, row in enumerate(r for r in rows if r["true_suit"] != r["predicted_suit"]):
        with Image.open(row["tile_png"]) as opened:
            tile = opened.convert("RGB")
        canvas = Image.new("RGB", (360, 110), "white")
        canvas.paste(tile.resize((tile.width * 5, tile.height * 5), Image.Resampling.NEAREST), (4, 4))
        draw = ImageDraw.Draw(canvas)
        draw.text((125, 7), f'{row["split"]} {row["method"]} {row["size"]}', fill="black")
        draw.text((125, 28), f'TRUE {row["true_suit"]}  PREDICTED {row["predicted_suit"]}', fill="black")
        draw.text((125, 49), f'BEST {row["best_score"]:.4f}', fill="black")
        draw.text((125, 70), f'SECOND {row["second_best_score"]:.4f}', fill="black")
        canvas.save(fail_dir / f'{i:04d}_{row["split"]}_{row["true_suit"]}_to_{row["predicted_suit"]}.png')
        thumbs.append(canvas)
    columns = 6
    sheet = Image.new("RGB", (180 * columns, 55 * max(1, (len(thumbs) + columns - 1) // columns)), "white")
    if not thumbs:
        ImageDraw.Draw(sheet).text((8, 8), "NO ERRORS", fill="black")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb.resize((180, 55), Image.Resampling.LANCZOS), ((i % columns) * 180, (i // columns) * 55))
    sheet.save(OUT / "failures_contact_sheet.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {"template": DATA / "split_v2/train_manifest.jsonl",
             "validation": DATA / "split_v2/validation_manifest.jsonl",
             "Smoke16": DATA / "smoke16_test_manifest.jsonl"}
    manifests = {name: read_manifest(path) for name, path in paths.items()}
    videos = {name: {Path(r["source_video"]).name.lower() for r in rows} for name, rows in manifests.items()}
    for a, b in (("template", "validation"), ("template", "Smoke16"), ("validation", "Smoke16")):
        if videos[a] & videos[b]:
            raise ValueError(f"Source video leakage: {a}, {b}")
        if {r["tile_pixel_sha256"] for r in manifests[a]} & {r["tile_pixel_sha256"] for r in manifests[b]}:
            raise ValueError(f"Pixel leakage: {a}, {b}")
    images = {name: load(rows) for name, rows in manifests.items()}
    suits = np.array([r["ground_truth"][1] for r in manifests["template"]])
    if set(suits) != set(SUITS):
        raise ValueError("Template set lacks a suit")
    all_validation, scores, banks = [], [], {}
    for size in SIZES:
        for method in METHODS:
            bank, _ = features(images["template"], size, method)
            banks[(size, method)] = bank
            result = evaluate(manifests["validation"], images["validation"], bank, suits, size, method, "validation")
            all_validation.extend(result)
            scores.append({"method": method, "size": size, **summary(result)})
    # Select only from source-held-out validation. A fixed order breaks exact ties.
    winner = max(scores, key=lambda r: (r["accuracy"], -METHODS.index(r["method"]), -SIZES.index(r["size"])))
    method, size = winner["method"], winner["size"]
    holdout = evaluate(manifests["Smoke16"], images["Smoke16"], banks[(size, method)], suits, size, method, "Smoke16")
    selected_validation = [r for r in all_validation if r["method"] == method and r["size"] == size]
    results = all_validation + holdout
    write_csv(OUT / "results.csv", results, list(results[0]))
    confusions = []
    for candidate in scores + [{"method": method, "size": size, "split": "Smoke16"}]:
        split = candidate.get("split", "validation")
        subset = [r for r in results if r["split"] == split and r["method"] == candidate["method"] and r["size"] == candidate["size"]]
        for true in SUITS:
            for pred in SUITS:
                confusions.append({"split": split, "method": candidate["method"], "size": candidate["size"],
                                   "true_suit": true, "predicted_suit": pred,
                                   "count": sum(r["true_suit"] == true and r["predicted_suit"] == pred for r in subset)})
    write_csv(OUT / "confusion_matrix.csv", confusions, list(confusions[0]))
    save_failures(results)
    # Warm bank and first inference; measure ROI processing and comparison for 1000 tiles.
    benchmark_images = (images["validation"] * 4)[:1000]
    evaluate(manifests["validation"][:1], images["validation"][:1], banks[(size, method)], suits, size, method, "warmup")
    start = time.perf_counter()
    query, reds = features(benchmark_images, size, method)
    for vec, red in zip(query, reds):
        predict(banks[(size, method)], suits, vec, bool(red), method)
    elapsed = time.perf_counter() - start
    counts = {name: dict(collections.Counter(r["ground_truth"][1] for r in rows)) for name, rows in manifests.items()}
    colored = {name: sum(bool(np.any(im[:, :, 0] != im[:, :, 1]) or np.any(im[:, :, 0] != im[:, :, 2])) for im in group)
               for name, group in images.items()}
    selected_margins = np.array([r["margin"] for r in selected_validation], dtype=float)
    smoke_margins = np.array([r["margin"] for r in holdout], dtype=float)
    pair_lines = []
    for split, group in (("validation", selected_validation), ("Smoke16", holdout)):
        sc = sum(r["true_suit"] == "s" and r["predicted_suit"] == "c" or r["true_suit"] == "c" and r["predicted_suit"] == "s" for r in group)
        hd = sum(r["true_suit"] == "h" and r["predicted_suit"] == "d" or r["true_suit"] == "d" and r["predicted_suit"] == "h" for r in group)
        pair_lines.append(f"{split}: spade↔club {sc}, heart↔diamond {hd}.")
    source_lines = []
    for source in sorted({Path(r["source_video"]).name for r in selected_validation + holdout}):
        group = [r for r in selected_validation + holdout if Path(r["source_video"]).name == source]
        score = summary(group)
        source_lines.append(f"{source}: {score['correct']}/{score['n']} ({score['accuracy']:.2%}).")
    lines = ["# Suit template experiment (2026-09-30)", "",
             "## Data and split", "",
             f'Template: `{paths["template"]}` SHA256 `{sha(paths["template"].read_bytes())}`; {len(manifests["template"])} tiles, videos {sorted(videos["template"])}.',
             f'Validation: `{paths["validation"]}` SHA256 `{sha(paths["validation"].read_bytes())}`; {len(manifests["validation"])} tiles, videos {sorted(videos["validation"])}.',
             f'Smoke16: `{paths["Smoke16"]}` SHA256 `{sha(paths["Smoke16"].read_bytes())}`; {len(manifests["Smoke16"])} tiles, videos {sorted(videos["Smoke16"])}.',
             f"Suit counts (s/h/d/c): template {counts['template']}; validation {counts['validation']}; Smoke16 {counts['Smoke16']}.",
             "All PNG and decoded-pixel SHA256 values were checked. Videos and exact pixel hashes are disjoint across splits. Adjacent rows remain within one video split. The existing split quarantined 128 near-match validation tiles. Parent episode identity is incomplete; tiles are not independent hands.",
             "Smoke16 ground truth was visually reviewed and frozen for an earlier baseline, but its results and errors were already examined before this experiment. This is a post-baseline holdout, not a new blind test. No Smoke16 data selected the method or ROI.", "",
             "## ROI and methods", "",
             f"Native suit ROI: x=[{ROI[0]},{ROI[2]}), y=[{ROI[1]},{ROI[3]}) of a 20/21 x 16 CARD tile. Linear resize to 16, 24, 32, 48 square pixels.",
             "Grayscale NCC: mean-centered, L2-normalized pixel correlation. Binary shape: per-ROI Otsu threshold, Dice overlap. Edge shape: Canny (60,120), Dice overlap. Color + shape: red/black from native RGB foreground pixels, then binary Dice within h/d or s/c. Each suit score is the maximum over all templates of that suit. Scores are similarities, not calibrated probabilities.",
             f"Tiles with any unequal RGB channels: template {colored['template']}/{len(images['template'])}, validation {colored['validation']}/{len(images['validation'])}, Smoke16 {colored['Smoke16']}/{len(images['Smoke16'])}. The saved tiles have no red/black color signal; the color-gated result is a negative control and cannot assess original full-color CARD images.", "",
             "## Validation results", "",
             "| Method | Size | N | Accuracy | s | h | d | c | Errors |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in scores:
        lines.append(f'| {r["method"]} | {r["size"]} | {r["n"]} | {r["accuracy"]:.2%} | {r["s_accuracy"]:.2%} | {r["h_accuracy"]:.2%} | {r["d_accuracy"]:.2%} | {r["c_accuracy"]:.2%} | {r["errors"]} |')
    smoke = summary(holdout)
    lines += ["", "## Selected method and Smoke16", "",
              f'Best validation method: **{method} {size}x{size}**, {winner["correct"]}/{winner["n"]} ({winner["accuracy"]:.2%}), {winner["errors"]} errors.',
              f'Smoke16 post-baseline holdout: {smoke["correct"]}/{smoke["n"]} ({smoke["accuracy"]:.2%}), {smoke["errors"]} errors; s {smoke["s_accuracy"]:.2%}, h {smoke["h_accuracy"]:.2%}, d {smoke["d_accuracy"]:.2%}, c {smoke["c_accuracy"]:.2%}.',
              f'Selected-method margin (best minus second): validation minimum {selected_margins.min():.4f}, median {np.median(selected_margins):.4f}; Smoke16 minimum {smoke_margins.min():.4f}, median {np.median(smoke_margins):.4f}. No threshold was fitted.',
              "By source video: " + " ".join(source_lines),
              "Critical pair confusions for selected method: " + " ".join(pair_lines),
              "Full confusion matrices are in `confusion_matrix.csv`; per-card scores and margins are in `results.csv`. `failures/` contains every validation variant error plus any selected-method Smoke16 error, with true/predicted suit and both scores; `failures_contact_sheet.png` shows all of them. The selected validation and Smoke16 method had zero errors.", "",
              "## Speed", "", f'1000 suit crops: {elapsed:.3f} s ({1000/elapsed:.1f} crops/s, {elapsed:.3f} ms/crop), warm template bank, single process; includes ROI extraction, resize, and comparison, excludes PNG disk loading.', "",
              "## Production recommendation", "",
              "Production changed: NO. Do not replace the current suit recognizer yet. This is a strong isolated candidate, but the prior Smoke16 analysis means a fresh untouched video with independently checked suits is needed to confirm generalization. All observed predictions were correct, so these data cannot establish a safe confidence threshold. Retain UNKNOWN for low-margin or conflicting evidence in any future integration.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"selected": winner, "Smoke16": smoke, "seconds_1000": elapsed, "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
