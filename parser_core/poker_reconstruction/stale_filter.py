"""Candidates only. A duplicate signature alone never authorizes deletion."""


def drop_candidate(event, prior, previous_hand, future=()):
    if event.kind in {"START_HAND", "INFO", "UNKNOWN", "UNPARSED_ACTION"}:
        return None
    if event.signature() in {e.signature() for e in (*prior, *future)}:
        return "REPEATED_CANDIDATE_REQUIRES_UNIQUE_STATE_AND_EXACT_POT"
    if event.kind in {"ANTE", "SB", "BB"} and any(e.kind in {"DEAL", "FOLD", "CHECK", "CALL", "BET", "RAISE", "STREET"} for e in prior):
        return "FORCED_BET_AFTER_ACTION_PHASE"
    if event.kind in {"WIN", "RETURN"} and event.signature() in {e.signature() for e in previous_hand}:
        return "PREVIOUS_HAND_CANDIDATE_REQUIRES_UNIQUE_STATE_AND_EXACT_POT"
    return None
