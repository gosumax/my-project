"""Experimental deterministic Hold'em card/class and payout eligibility checks."""
from collections import Counter
from itertools import combinations

VERSION = 'experimental-cards-1'
RANKS = '23456789TJQKA'
CLASSES = ['HIGH_CARD', 'PAIR', 'TWO_PAIR', 'THREE_OF_A_KIND', 'STRAIGHT', 'FLUSH', 'FULL_HOUSE', 'FOUR_OF_A_KIND', 'STRAIGHT_FLUSH']


def five_rank(cards):
    values = sorted((RANKS.index(c[0])+2 for c in cards), reverse=True)
    counts = Counter(values)
    groups = sorted(((count, value) for value, count in counts.items()), reverse=True)
    flush = len({c[1] for c in cards}) == 1
    unique = sorted(set(values), reverse=True)
    straight = (unique[0] if len(unique) == 5 and unique[0]-unique[-1] == 4 else
                5 if unique == [14, 5, 4, 3, 2] else 0)
    if flush and straight:
        return (8, straight)
    if groups[0][0] == 4:
        return (7, groups[0][1], groups[1][1])
    if [g[0] for g in groups] == [3, 2]:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5, *values)
    if straight:
        return (4, straight)
    if groups[0][0] == 3:
        return (3, groups[0][1], *sorted((g[1] for g in groups[1:]), reverse=True))
    pairs = sorted((v for count, v in groups if count == 2), reverse=True)
    if len(pairs) == 2:
        return (2, *pairs, next(v for count, v in groups if count == 1))
    if pairs:
        return (1, pairs[0], *sorted((v for count, v in groups if count == 1), reverse=True))
    return (0, *values)


def best_rank(cards):
    if not 5 <= len(cards) <= 7 or len(set(cards)) != len(cards):
        raise ValueError('Invalid physical cards')
    if any(len(c) != 2 or c[0] not in RANKS or c[1] not in 'shdc' for c in cards):
        raise ValueError('Invalid card token')
    return max(five_rank(combo) for combo in combinations(cards, 5))


def described_class(text):
    low = text.lower()
    # More specific terms first: 'Two pairs' must never collapse to Pair.
    for words, value in [(['straight flush', 'стрит-флеш'], 'STRAIGHT_FLUSH'),
                         (['four of a kind', 'каре'], 'FOUR_OF_A_KIND'),
                         (['full house', 'фулл'], 'FULL_HOUSE'),
                         (['flush', 'флеш'], 'FLUSH'), (['straight', 'стрит'], 'STRAIGHT'),
                         (['three of a kind', 'тройка', 'сет'], 'THREE_OF_A_KIND'),
                         (['two pairs', 'two pair', 'две пары'], 'TWO_PAIR'),
                         (['pair', 'пара'], 'PAIR'), (['high card', 'старшая карта'], 'HIGH_CARD')]:
        if any(word in low for word in words):
            return value
    return None
