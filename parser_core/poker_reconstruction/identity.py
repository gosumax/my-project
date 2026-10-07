"""Aliases require independent identity evidence, before any chip equations."""
from difflib import SequenceMatcher


def identity_hypotheses(candidates, max_hypotheses=128):
    valid, rejected = [], []
    for candidate in candidates or []:
        a, b = candidate.get("a"), candidate.get("b")
        ok = (isinstance(a, str) and isinstance(b, str) and a != b and
              candidate.get("seat_a") is not None and candidate.get("seat_a") == candidate.get("seat_b") and
              candidate.get("continuous_identity") is True and candidate.get("same_table_hand_context") is True and
              bool(candidate.get("identity_evidence_refs")) and SequenceMatcher(None, a, b).ratio() >= 0.8)
        (valid if ok else rejected).append(candidate)
    hypotheses = [{"aliases": {}, "decisions": []}]
    exhausted = False
    for candidate in valid:
        expanded = []
        for hypothesis in hypotheses:
            a, b = candidate["a"], candidate["b"]
            expanded.append({"aliases": dict(hypothesis["aliases"]),
                             "decisions": hypothesis["decisions"] + [dict(candidate, hypothesis="A!=B")]})
            mapping = dict(hypothesis["aliases"])
            def canonical(name):
                while name in mapping and mapping[name] != name:
                    name = mapping[name]
                return name
            ca, cb = canonical(a), canonical(b)
            if ca != cb:
                mapping[cb] = ca
            expanded.append({"aliases": mapping,
                             "decisions": hypothesis["decisions"] + [dict(candidate, hypothesis="A==B")]})
        if len(expanded) > max_hypotheses:
            exhausted = True
            break
        hypotheses = expanded
    return hypotheses, exhausted, rejected
