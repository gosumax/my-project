"""Current-hand numeric fallback on retained native pixels, without OCR/video."""
import copy
import json
from pathlib import Path

import pytest

from parser_core.adjacent_stack_carry import SavedCarrySources
from parser_core.constant_stack_balance import frozen_runtime_files
from parser_core.direct_start_stack import DirectStartStackProvider, FreshDirectWindows, native_key
from parser_core.first_visible_single_numeric import SAVED_SINGLE_CONTRACT, prove
from parser_core.native_nick_who import NativeNickWhoSource

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs/hh_all_hands_20261009/current_full_replay_v5_identity_single_numeric'


@pytest.fixture(scope='module')
def late_native():
    records = [json.loads(line) for line in
        (OUTPUT / 'logical_hand_analysis.jsonl').read_text('utf8').splitlines() if line.strip()]
    source = Path(json.loads((OUTPUT / 'replay_receipt.json').read_text('utf8'))['source'])
    record = next(r for r in records if native_key(r['observed_hand'])[::2] == (2, '2878367229'))
    key = native_key(record['observed_hand'])
    native = NativeNickWhoSource(source, [r['observed_hand'] for r in records])
    registry = SavedCarrySources(source, records, [])
    # COMPLETE hand, retained full current-hand sources only.
    provider = object.__new__(DirectStartStackProvider)
    provider.inventory = FreshDirectWindows(source, [key], actor_policy=True, native_who=native)
    provider.registry = registry
    provider.runtime_files = frozen_runtime_files()
    provider.single_first_numeric = True
    case = provider.inventory.case_for(key, 'tolbin3', 'seat5')
    result = provider._prove(case, record)
    provider.cases = {(key, 'tolbin3'): (case, record)}
    provider.proofs = {(key, 'tolbin3'): result}
    return provider, key, record


def test_complete_retained_numeric_and_consumer(late_native):
    provider, key, record = late_native
    proof = provider.proofs[(key, 'tolbin3')]
    assert proof['status'] == 'PROVEN'
    assert proof['displayed_stack'] == 398697 and proof['numeric_frames'] == [8049]
    assert proof['first_visible_proof']['contract'] == SAVED_SINGLE_CONTRACT
    assert proof['calculated_start_stack'] == 401822
    assert provider.for_hand(record['observed_hand']) == [proof]


def test_expanded_inventory_ends_before_next_hand(late_native):
    provider, key, _ = late_native
    expanded = provider.inventory.current_hand_inventory(key)
    key = next(k for k in expanded.contexts if k[2] == '2878367229')
    low, high = expanded.native_who.bounds[key]
    assert expanded.contexts[key]['capture_end_inclusive'] == high
    foreign = [r for r in expanded.native_who.checkpoints[key[:2]]
               if r['frame'] > low and r['hand_id'] != key[2]]
    assert foreign and high == foreign[0]['frame'] - 1
    captures = expanded.case_for(key, 'tolbin3', 'seat5')['raw_stack_captures']
    assert all(low <= r['raw_record']['source_frame'] <= high for r in captures)


def test_changed_native_hand_bounds_do_not_grant_numeric_proof(late_native):
    provider, key, _ = late_native
    expanded = copy.copy(provider.inventory.current_hand_inventory(key))
    expanded.native_who = copy.copy(expanded.native_who)
    expanded.native_who.bounds = dict(expanded.native_who.bounds)
    low, high = expanded.native_who.bounds[key]
    expanded.native_who.bounds[key] = (low, high - 1)
    result = prove(expanded.case_for(key, 'tolbin3', 'seat5'), expanded, lambda: None)
    assert result['status'] == 'UNKNOWN'


def test_missing_native_hand_bounds_stay_unknown():
    inventory = object.__new__(FreshDirectWindows)
    inventory.current_hand_numeric = False
    class MissingBounds:
        bounds = {}
    inventory.native_who = MissingBounds()
    with pytest.raises(ValueError, match='FIRST_SAVED_CURRENT_HAND_BOUNDS_UNKNOWN'):
        inventory.current_hand_inventory((2, 'session', 'hand'))
