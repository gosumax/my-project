"""RAM-only native row diagnostics. Scores are measurements, never identity verdicts.

Support is the existing segmentation gray > 70; small alignment never resizes
glyphs or discards nonoverlapping foreground. dx/dy moves the AFTER image.
"""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class RowContent:
    pixels: np.ndarray
    support: np.ndarray
    bbox: tuple | None
    metadata: dict


def represent_row(gray: np.ndarray, scrollbar_px: int = 12) -> RowContent:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Expected native two-dimensional uint8 gray crop")
    if scrollbar_px < 0:
        raise ValueError("Negative scrollbar exclusion")
    width = max(0, gray.shape[1] - scrollbar_px)
    pixels = gray[:, :width].copy()
    support = pixels > 70
    ys, xs = np.nonzero(support)
    bbox = None if not len(xs) else (int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1))
    trims = None if bbox is None else (bbox[0], bbox[1], width-bbox[2], gray.shape[0]-bbox[3])
    metadata = {"original_shape": list(gray.shape), "usable_shape": list(pixels.shape),
                "source_sha256": hashlib.sha256(gray.tobytes()).hexdigest(),
                "scrollbar_excluded_px": gray.shape[1]-width, "bbox": bbox,
                "trims_left_top_right_bottom": trims,
                "content_shape": None if bbox is None else [bbox[3]-bbox[1], bbox[2]-bbox[0]],
                "empty": bbox is None, "support_pixels": int(support.sum())}
    pixels.flags.writeable = False
    support.flags.writeable = False
    return RowContent(pixels, support, bbox, metadata)


def _canvas_pair(a, b, dx, dy):
    ah, aw = a.pixels.shape
    bh, bw = b.pixels.shape
    x0, y0 = min(0, dx), min(0, dy)
    shape = (max(ah, dy+bh)-y0, max(aw, dx+bw)-x0)
    arrays = []
    for row, x, y in ((a, -x0, -y0), (b, dx-x0, dy-y0)):
        p = np.zeros(shape, np.uint8)
        s = np.zeros(shape, bool)
        covered = np.zeros(shape, bool)
        content = np.zeros(shape, bool)
        edge = np.zeros(shape, bool)
        h, w = row.pixels.shape
        p[y:y+h, x:x+w] = row.pixels
        s[y:y+h, x:x+w] = row.support
        covered[y:y+h, x:x+w] = True
        local_edge = np.zeros((h, w), bool)
        local_edge[:2] = local_edge[-2:] = True
        local_edge[:, :2] = local_edge[:, -2:] = True
        edge[y:y+h, x:x+w] = local_edge
        if row.bbox is not None:
            l, t, r, bottom = row.bbox
            content[y+t:y+bottom, x+l:x+r] = True
        arrays.append((p, s, covered, content, edge))
    return arrays


def _metric(diff, mask):
    count = int(mask.sum())
    return {"pixels": count, "mae": float(diff[mask].mean()) if count else None,
            "changed_pixels": int(np.count_nonzero(diff[mask]))}


def _score(a, b, dx, dy):
    (ap, ass, ac, ab, ae), (bp, bs, bc, bb, be) = _canvas_pair(a, b, dx, dy)
    diff = np.abs(ap.astype(np.int16)-bp.astype(np.int16))
    union = ass | bs
    coverage = ac | bc
    bbox_union = ab | bb
    missing_a = bs & ~ac
    missing_b = ass & ~bc
    return {"dx": dx, "dy": dy, "foreground_union": _metric(diff, union),
            "content_region": _metric(diff, bbox_union),
            "outside_content": _metric(diff, coverage & ~bbox_union),
            "dark_background": _metric(diff, coverage & ~union),
            "edge_diff": _metric(diff, ae | be),
            "full_crop": _metric(diff, coverage),
            "support_xor_pixels": int(np.count_nonzero(ass ^ bs)),
            "missing_foreground_before": int(missing_a.sum()),
            "missing_foreground_after": int(missing_b.sum()),
            "coverage_only_before": int((ac & ~bc).sum()),
            "coverage_only_after": int((bc & ~ac).sum()),
            "exact_support_equal_after_alignment": bool(union.any() and np.array_equal(ass, bs)
                and not np.any(diff[union]) and not missing_a.any() and not missing_b.any())}


def _edge_descriptor(row):
    p = row.pixels
    mask = np.zeros(p.shape, bool)
    mask[:2, :] = True
    mask[-2:, :] = True
    mask[:, :2] = True
    mask[:, -2:] = True
    values = p[mask]
    return {"pixels": int(values.size), "dark_le_70": int((values <= 70).sum()),
            "light_gt_70": int((values > 70).sum()),
            "mean": float(values.mean()) if values.size else None,
            "sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def compare_rows(before: np.ndarray, after: np.ndarray, scrollbar_px: int = 12) -> dict:
    a, b = represent_row(before, scrollbar_px), represent_row(after, scrollbar_px)
    scores = [_score(a, b, dx, dy) for dy in range(-2, 3) for dx in range(-2, 3)]
    def key(s):
        mae = s["foreground_union"]["mae"]
        return (float("inf") if mae is None else mae, s["support_xor_pixels"],
                abs(s["dx"])+abs(s["dy"]), s["dy"], s["dx"])
    best = min(scores, key=key)
    def native(row):
        if row.bbox is None:
            return None
        l, t, r, bottom = row.bbox
        return row.pixels[t:bottom, l:r]
    na, nb = native(a), native(b)
    return {"before": a.metadata, "after": b.metadata,
            "before_alignment": next(s for s in scores if s["dx"] == s["dy"] == 0),
            "best_alignment": best, "alignment_candidates": scores,
            "original_full_crop": _score(represent_row(before, 0), represent_row(after, 0), 0, 0)["full_crop"],
            "exact_content_native": bool(na is not None and nb is not None and np.array_equal(na, nb)),
            "before_outer_2px": _edge_descriptor(a), "after_outer_2px": _edge_descriptor(b),
            "source_arrays_unchanged": a.metadata["source_sha256"] == hashlib.sha256(before.tobytes()).hexdigest()
                and b.metadata["source_sha256"] == hashlib.sha256(after.tobytes()).hexdigest()}
