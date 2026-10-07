"""Fail-closed validator for ERNIEKit poker BOARD/CARD SFT JSONL files."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image


BOARD_PROMPT = "Read the poker board. Return only cards left to right using rank+suit notation."
CARD_PROMPT = "Read the poker card. Return only rank+suit."
CARD = re.compile(r"^[2-9TJQKA][cdhs]$")


def validate_line(line, line_no, split, manifest):
    value = json.loads(line)
    if set(value) != {"image_info", "text_info"}:
        raise ValueError(f"{split}:{line_no}: unexpected top-level fields")
    images, texts = value["image_info"], value["text_info"]
    if not isinstance(images, list) or len(images) != 1 or set(images[0]) != {"image_url", "matched_text_index"}:
        raise ValueError(f"{split}:{line_no}: invalid image_info")
    if images[0]["matched_text_index"] != 0:
        raise ValueError(f"{split}:{line_no}: image must precede prompt")
    if not isinstance(texts, list) or len(texts) != 2 or texts[0].get("tag") != "mask" or texts[1].get("tag") != "no_mask":
        raise ValueError(f"{split}:{line_no}: prompt/target mask sequence invalid")
    if any(set(item) != {"text", "tag"} for item in texts):
        raise ValueError(f"{split}:{line_no}: invalid text_info fields")
    path = Path(images[0]["image_url"])
    meta = manifest.get(str(path.resolve()))
    if meta is None:
        raise ValueError(f"{split}:{line_no}: image absent from verified manifest")
    kind = meta.get("kind")
    if kind not in {"BOARD", "CARD"} or texts[0]["text"] != (BOARD_PROMPT if kind == "BOARD" else CARD_PROMPT):
        raise ValueError(f"{split}:{line_no}: unexpected kind or prompt")
    target = texts[1]["text"]
    cards = target.split(" ")
    if target != " ".join(cards) or any(not CARD.fullmatch(c) for c in cards) or len(set(cards)) != len(cards):
        raise ValueError(f"{split}:{line_no}: noncanonical target")
    if kind == "CARD" and len(cards) != 1:
        raise ValueError(f"{split}:{line_no}: CARD must have one card")
    if kind == "BOARD" and len(cards) != {"FLOP": 3, "TURN": 4, "RIVER": 5}.get(meta.get("street")):
        raise ValueError(f"{split}:{line_no}: BOARD card count disagrees with street")
    if meta.get("gt_status") != "VISUALLY_REVIEWED" or not meta.get("board_identity") or not meta.get("hand_id"):
        raise ValueError(f"{split}:{line_no}: GT or group identity absent")
    if not path.is_file():
        raise ValueError(f"{split}:{line_no}: image absent: {path}")
    with Image.open(path) as im:
        im.verify()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != meta.get("sha256"):
        raise ValueError(f"{split}:{line_no}: SHA mismatch")
    if kind == "CARD" and (not meta.get("parent_board") or not isinstance(meta.get("card_index"), int)
                           or not isinstance(meta.get("crop_box"), list) or len(meta["crop_box"]) != 4):
        raise ValueError(f"{split}:{line_no}: CARD parent mapping incomplete")
    return kind, digest, meta["board_identity"], meta["hand_id"], meta.get("source_video_sha256")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="Reviewed split metadata keyed by absolute image path")
    args = parser.parse_args()
    manifest_rows = json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
    manifest = {str(Path(row["image"]).resolve()): row for row in manifest_rows}
    observed = {}
    counts = {}
    for split in ("train", "val", "test"):
        path = args.directory / f"{split}.jsonl"
        if not path.is_file():
            raise ValueError(f"Missing {path}")
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        counts[split] = len(rows)
        for line_no, line in enumerate(rows, 1):
            kind, digest, board, hand, video = validate_line(line, line_no, split, manifest)
            # A board and every card derived from it share the same split.
            for group in (("sha256", digest), ("board", board), ("hand", video, hand)):
                previous = observed.setdefault(group, split)
                if previous != split:
                    raise ValueError(f"Leakage: {group} occurs in {previous} and {split}")
    print(json.dumps({"status": "VALID", "counts": counts}))


if __name__ == "__main__":
    main()
