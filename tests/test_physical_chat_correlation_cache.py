"""Full native proof parity and cache lifetime across reused frame buffers."""
from pathlib import Path

import numpy as np
import pytest

from parser_core import physical_chat_admission as admission
from parser_core.chat_rows import physical_spans

FIXTURE = Path(__file__).parent / 'fixtures/physical_chat_admission_smoke19.npz'


class Uncached:
    """Run the same proof gates with every original correlation evaluated."""
    def __call__(self, a, b):
        return admission.correlation(a, b)

    def finish(self, result):
        return result


@pytest.mark.parametrize('key', [
    'T01_2085', 'T03_6455', 'T03_9001', 'T03_9028',
    'T09_8025', 'T01_9701', 'T04_1009', 'T08_1672',
])
def test_all_native_proof_scores_candidates_and_gates_equal_uncached(monkeypatch, key):
    with np.load(FIXTURE) as saved:
        old, new = saved[key + '_before'], saved[key + '_after']
    old_copy, new_copy = old.copy(), new.copy()
    spans = physical_spans(old), physical_spans(new)
    original = admission.correlation
    counts = []

    def counted(a, b):
        counts.append(1)
        return original(a, b)

    monkeypatch.setattr(admission, 'correlation', counted)
    with monkeypatch.context() as context:
        context.setattr(admission, '_CorrelationMemo', Uncached)
        expected = admission.physical_tail_proof(old, new, *spans)
    uncached_calls = len(counts)
    counts.clear()
    actual = admission.physical_tail_proof(old, new, *spans)
    assert actual == expected
    assert len(counts) <= uncached_calls
    if key == 'T01_2085':
        assert len(counts) < uncached_calls
    assert np.array_equal(old, old_copy) and np.array_equal(new, new_copy)


def test_no_cache_survives_new_proof_on_same_mutated_frame_buffers(monkeypatch):
    with np.load(FIXTURE) as saved:
        old, new = saved['T01_2085_before'], saved['T01_2085_after']
    spans = physical_spans(old), physical_spans(new)
    before = admission.physical_tail_proof(old, new, *spans)
    # Reuse identical addresses with new pixels: cross-call pointer caching
    # would incorrectly reuse the earlier proof's native scores.
    np.bitwise_xor(new, 127, out=new)
    with monkeypatch.context() as context:
        context.setattr(admission, '_CorrelationMemo', Uncached)
        expected = admission.physical_tail_proof(old, new, *spans)
    actual = admission.physical_tail_proof(old, new, *spans)
    assert actual == expected and actual != before
