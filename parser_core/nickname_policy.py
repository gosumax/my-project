"""User-selected NICK admission; CHAT, EMPTY and numeric policies are separate."""

NICK_CONFIDENCE_MIN = .90
NICK_REVIEW_MIN = .85
FIRST_NICK_POLICY_V1 = 'FIRST_QUALIFIED_NICK_FRAME_PER_HAND_V1'
FIRST_NICK_POLICY = 'FIRST_QUALIFIED_NICK_FRAME_PER_HAND_V2'
FIRST_NICK_POLICIES = frozenset((FIRST_NICK_POLICY_V1, FIRST_NICK_POLICY))


def nickname_witness_floor(policy):
    if policy == FIRST_NICK_POLICY:
        return NICK_CONFIDENCE_MIN
    if policy in (None, FIRST_NICK_POLICY_V1):
        return .95
    raise ValueError('NICK_WITNESS_POLICY_UNKNOWN')
