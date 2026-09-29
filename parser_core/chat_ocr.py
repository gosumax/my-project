"""Versioned OCR input and secondary-reading policy for fixed chat rows."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import numpy as np
from PIL import Image


CHAT_OCR_RIGHT_TRIM_PX = 12
CHAT_OCR_PRIMARY_SCALE = 2
CHAT_OCR_PROFILE = "chat_row_trim12_upscale2_v1"
NICKNAME_ROUTE_VERSION = "vl15_nickname_route_v3"
NICKNAME_SUBSTITUTION_VERSION = "vl15_uncertain_glyph_guard_v4"
BOARD_ROUTE_VERSION = "vl15_board_tiles_v1"

_DEALER_NAME = re.compile(r"^\s*Дилер\s*:\s*(\S+)", re.IGNORECASE)
_ACTION = re.compile(
    r"\b(?:сбрасывает|ставит|повышает|выигрывает|получает|уравнивает|проверяет|"
    r"делает\s+чек|не\s+показывает|показывает)\b",
    re.IGNORECASE,
)
_SYSTEM_START = {"раздаются", "раздаётся", "раздается", "начинается"}
_LATIN_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
_LATIN_CONFUSABLES = str.maketrans({
    "а": "a", "А": "A", "е": "e", "Е": "E", "о": "o", "О": "O",
    "с": "c", "С": "C", "р": "p", "Р": "P", "х": "x", "Х": "X",
    "і": "i", "І": "I",
})


def prepare_row_png(path: Path, expected_sha256: str, *, scale: int = 2,
                    right_trim_px: int = CHAT_OCR_RIGHT_TRIM_PX) -> tuple[bytes, str]:
    """Derive OCR pixels while preserving the original saved observation."""
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError(f"Crop hash mismatch: {path}")
    with Image.open(io.BytesIO(source)) as opened:
        image = opened.convert("RGB")
    if not (0 <= right_trim_px < image.width) or scale < 1:
        raise ValueError("Invalid chat OCR crop transform")
    image = image.crop((0, 0, image.width - right_trim_px, image.height))
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale),
                             Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="PNG")
    result = output.getvalue()
    return result, hashlib.sha256(result).hexdigest()


def dealer_nickname(text: str | None) -> tuple[str, tuple[int, int]] | None:
    match = _DEALER_NAME.match(text or "")
    if not match or match.group(1).casefold() in _SYSTEM_START:
        return None
    action = _ACTION.search(text, match.end(1))
    if action is None:
        return match.group(1), match.span(1)
    start = match.start(1)
    name = text[start:action.start()].rstrip()
    return (name, (start, start + len(name))) if name else None


def nickname_route_reasons(text: str | None) -> tuple[str, ...]:
    parsed = dealer_nickname(text)
    if parsed is None:
        return ()
    nickname = parsed[0]
    reasons = []
    if re.search(r"[А-Яа-яЁё]", nickname):
        reasons.append("CYRILLIC_IN_NICKNAME")
    if re.search(r"[iIlL]", nickname):
        reasons.append("LATIN_I_OR_L_IN_NICKNAME")
    if "?" in nickname or "\ufffd" in nickname:
        reasons.append("UNCERTAIN_GLYPH_IN_NICKNAME")
    return tuple(reasons)


def latinize_mixed_nickname(nickname: str) -> str | None:
    """Fix only visual Cyrillic/Latin confusables in an otherwise Latin name."""
    if _LATIN_NAME.fullmatch(nickname):
        return nickname
    if len(re.findall(r"[A-Za-z]", nickname)) < 2:
        return None
    converted = nickname.translate(_LATIN_CONFUSABLES)
    return converted if _LATIN_NAME.fullmatch(converted) else None


def uncertain_glyph_conflict(tesseract_text: str, vl_text: str) -> bool:
    """Flag disagreement outside unknown marks and the routed i/l/1 ambiguity."""
    primary = dealer_nickname(tesseract_text)
    secondary = dealer_nickname(vl_text)
    if primary is None or secondary is None:
        return False
    old = primary[0]
    if "?" not in old and "\ufffd" not in old:
        return False
    new = latinize_mixed_nickname(secondary[0])
    if new is None or len(old) != len(new):
        return True
    for left, right in zip(old, new):
        if left in "?\ufffd" or left == right:
            continue
        if left in "iIlL1" and right in "iIlL1":
            continue
        return True
    return False


def substitute_nickname(tesseract_text: str, vl_text: str) -> str | None:
    """Replace only the name span; preserve Tesseract action and amount."""
    primary = dealer_nickname(tesseract_text)
    secondary = dealer_nickname(vl_text)
    if primary is None or secondary is None:
        return None
    old, span = primary
    new = latinize_mixed_nickname(secondary[0])
    if new is None or old == new:
        return None
    primary_action = _ACTION.search(tesseract_text[span[1]:])
    secondary_action = _ACTION.search(vl_text[secondary[1][1]:])
    if primary_action is None or secondary_action is None or (
            primary_action.group().casefold() != secondary_action.group().casefold()):
        return None
    return tesseract_text[:span[0]] + new + tesseract_text[span[1]:]


def board_tiles_visible(path: Path, expected_sha256: str) -> bool:
    """Recognize the bright three-card strip in this fixed chat layout."""
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError(f"Crop hash mismatch: {path}")
    with Image.open(io.BytesIO(source)) as opened:
        image = np.asarray(opened.convert("L"))
    if image.shape[1] < 200 or image.shape[0] > 40:
        return False
    return int(np.count_nonzero(image[:, :100] > 180)) >= 300
