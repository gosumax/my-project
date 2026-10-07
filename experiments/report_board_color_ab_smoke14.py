"""Summarize fixed-manifest BOARD color A/B without altering pipeline outputs."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/smoke14_board_color_ab_20260930"
RANK = re.compile(r"10|[2-9TJQKA]", re.IGNORECASE)
SUIT = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}


def load_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def score(raw: str, truth: list[str], parsed: list[str] | None, stop: str) -> dict:
    rank_matches = list(RANK.finditer(raw))
    ranks = ["T" if m.group().upper() == "10" else m.group().upper()
             for m in rank_matches]
    count_ok = len(ranks) == len(truth)
    rank_correct = sum(a == b[0] for a, b in zip(ranks, truth)) if count_ok else 0
    suit_correct = 0
    if count_ok:
        for index, match in enumerate(rank_matches):
            end = rank_matches[index + 1].start() if index + 1 < len(rank_matches) else len(raw)
            between = raw[match.end():end].strip(" ,;|/[](){}:.-\n\t")
            suit = SUIT.get(between, between.lower() if between.lower() in "shdc" and len(between) == 1 else None)
            suit_correct += suit == truth[index][1]
    failures = []
    if not rank_matches:
        failures.append("NO_CARDS")
    if 0 < len(rank_matches) < len(truth):
        failures.append("PARTIAL_CARDS")
    if count_ok and rank_correct < len(truth):
        failures.append("WRONG_RANKS")
    if count_ok and suit_correct < len(truth):
        failures.append("WRONG_SUITS")
    if len(rank_matches) > len(truth):
        failures.append("REPETITION_OR_EXTRA_CARDS")
    if stop == "TOKEN_LIMIT":
        failures.append("TOKEN_LIMIT")
    if parsed is None and raw:
        failures.append("MALFORMED_OUTPUT")
    return {"rank_tokens": len(ranks), "card_count_correct": count_ok,
            "rank_correct": rank_correct, "suit_correct": suit_correct,
            "valid_parse": parsed is not None, "full_board_exact": parsed == truth,
            "failures": failures}


def main() -> None:
    dataset = json.loads((RUN / "dataset.json").read_text("utf-8"))
    summary = json.loads((RUN / "summary.json").read_text("utf-8"))
    rows = dataset["rows"]
    color = load_lines(RUN / "results_CURRENT_RGB.jsonl")
    gray = load_lines(RUN / "results_GRAY_EXPERIMENT.jsonl")
    assert len(rows) == len(color) == len(gray) == 32
    pairs = []
    metrics = {variant: Counter() for variant in ("CURRENT_RGB", "GRAY_EXPERIMENT")}
    for source, c, g in zip(rows, color, gray):
        evidence_id = source["board_evidence_id"]
        assert c["board_evidence_id"] == g["board_evidence_id"] == evidence_id
        truth = source["ground_truth"]
        cs = score(c["raw_text"], truth, c["parsed_cards"], c["stop_reason"])
        gs = score(g["raw_text"], truth, g["parsed_cards"], g["stop_reason"])
        for variant, scored in (("CURRENT_RGB", cs), ("GRAY_EXPERIMENT", gs)):
            m = metrics[variant]
            m["boards"] += 1
            m["card_slots"] += len(truth)
            for key in ("rank_correct", "suit_correct", "valid_parse", "full_board_exact", "card_count_correct"):
                m[key] += scored[key]
            m.update(scored["failures"])
        if cs["full_board_exact"] and not gs["full_board_exact"]:
            result = "COLOR_FIXED_FULL_BOARD"
        elif gs["full_board_exact"] and not cs["full_board_exact"]:
            result = "COLOR_WORSE"
        elif cs["full_board_exact"] and gs["full_board_exact"]:
            result = "NO_CHANGE"
        elif cs["suit_correct"] > gs["suit_correct"] and cs["rank_correct"] == gs["rank_correct"]:
            result = "COLOR_FIXED_SUIT"
        elif cs["rank_correct"] > gs["rank_correct"] and cs["suit_correct"] == gs["suit_correct"]:
            result = "COLOR_FIXED_RANK"
        elif cs["rank_correct"] > gs["rank_correct"] and cs["suit_correct"] > gs["suit_correct"]:
            result = "COLOR_IMPROVED_RANK_AND_SUIT"
        elif cs["rank_correct"] < gs["rank_correct"] or cs["suit_correct"] < gs["suit_correct"]:
            result = "COLOR_WORSE"
        elif c["raw_text"] == g["raw_text"]:
            result = "NO_CHANGE"
        else:
            result = "BOTH_WRONG"
        pairs.append({"manifest_index": source["manifest_index"], "board_evidence_id": evidence_id,
                      "source_sha256": source["source_sha256"],
                      "color_pixel_sha256": source["current_rgb_pixel_sha256"],
                      "grayscale_pixel_sha256": source["gray_experiment_pixel_sha256"],
                      "frame_id": source["frame_id"], "table_id": source["table_id"],
                      "street": source["street"], "crop_size": source["crop_size"],
                      "ground_truth": truth, "ground_truth_status": source["ground_truth_status"],
                      "grayscale_response": g["raw_text"], "color_response": c["raw_text"],
                      "grayscale_parse": g["parsed_cards"], "color_parse": c["parsed_cards"],
                      "grayscale_rank_correct": gs["rank_correct"],
                      "color_rank_correct": cs["rank_correct"],
                      "grayscale_suit_correct": gs["suit_correct"],
                      "color_suit_correct": cs["suit_correct"], "result": result})
    (RUN / "pairwise.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in pairs), "utf-8")
    with (RUN / "pairwise.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    report = {"metrics": {key: dict(value) for key, value in metrics.items()},
              "timing": summary, "pairwise": dict(Counter(x["result"] for x in pairs))}
    (RUN / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
