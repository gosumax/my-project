"""Summarize fixed-corpus VL experiments without modifying production parsing."""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/board_vl_quality_audit_20260930"
RANK = re.compile(r"10|[2-9TJQKA]", re.I)
CARD = re.compile(r"(10|[2-9TJQKA])\s*([shdc♠♥♦♣])", re.I)
SUITS = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}


def read(path):
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x]


def rank_stream(raw):
    raw = re.sub(r'"cards"\s*:', "", raw, flags=re.I)
    return [("T" if x.upper() == "10" else x.upper()) for x in RANK.findall(raw)]


def repeated_sequence(ranks):
    # A repeated adjacent span of at least two ranks is a diagnostic loop.
    for size in range(2, len(ranks) // 2 + 1):
        for start in range(len(ranks) - 2 * size + 1):
            if ranks[start:start + size] == ranks[start + size:start + 2 * size]:
                return True
    return False


def summarize(rows):
    total_cards = sum(len(r["expected"] if "expected" in r else r["ground_truth"]) for r in rows)
    rank_ok = suit_ok = exact = valid = strict = count_ok = malformed = repeat = limited = 0
    for r in rows:
        expected = r.get("expected", r.get("ground_truth"))
        raw = r["raw_text"]
        normalized = r.get("normalized_cards", r.get("strict_normalized"))
        strict_cards = r.get("strict_cards", r.get("strict_normalized"))
        valid += normalized is not None
        strict += strict_cards is not None
        exact += normalized == expected
        malformed += normalized is None
        ranks = [c[0] for c in normalized] if normalized else rank_stream(raw)
        count_ok += len(ranks) == len(expected)
        rank_ok += sum(a == b[0] for a, b in zip(ranks, expected))
        if normalized:
            suit_ok += sum(a[1] == b[1] for a, b in zip(normalized, expected))
        else:
            candidates = CARD.findall(raw)
            suit_ok += sum(SUITS.get(a[1], a[1].lower()) == b[1]
                           for a, b in zip(candidates, expected))
        repeat += r.get("primary_failure_reason") == "REPETITION_LOOP" if "primary_failure_reason" in r else repeated_sequence(ranks)
        limited += r.get("stop_reason") == "TOKEN_LIMIT" or r.get("original_worker_stop_status") == "TOKEN_LIMIT"
    return {"n": len(rows), "cards": total_cards,
            "full_board_exact": exact, "valid_parse": valid, "strict_parse": strict,
            "rank_correct_slots": rank_ok, "rank_slots": total_cards,
            "suit_correct_slots": suit_ok, "suit_slots": total_cards,
            "correct_rank_token_count": count_ok, "malformed_response": malformed,
            "repetition_primary": repeat, "token_limit": limited}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="*", type=Path)
    args = p.parse_args()
    files = args.paths or [OUT / "baseline32.jsonl"] + sorted(OUT.glob("*/results.jsonl"))
    for path in files:
        print(path.parent.name, json.dumps(summarize(read(path)), ensure_ascii=False))


if __name__ == "__main__":
    main()
