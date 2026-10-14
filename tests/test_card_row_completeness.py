"""A resolved subset of visible cards must never become a complete card row."""
import hashlib
import base64
import copy
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from parser_core.board_tiles import (card_spans, secondary_card_spans,
    prove_split_card_partition, verify_card_geometry_resolution, SPLIT_PARTITION_RULE)
from parser_core.card_hybrid import CardHybrid

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/card_primary_subset_native.png'
META = json.loads(FIXTURE.with_suffix('.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def reader():
    return CardHybrid(ROOT / 'configs/card_hybrid_local.json',
                      ROOT / 'configs/suit_backup_local.json',
                      rank_ncc_config=ROOT / 'configs/rank_ncc_production.json')


def read_pixels(reader, pixels):
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format='PNG')
    raw = stream.getvalue()
    return reader.read_bytes(raw, hashlib.sha256(raw).hexdigest(), '')


def test_native_primary_subset_requires_proved_full_partition(reader):
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == META['native_png_sha256']
    pixels = np.asarray(Image.open(io.BytesIO(raw)).convert('L'))
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == META['native_gray_sha256']
    assert card_spans(pixels, recover_split=True)[0] == [(35, 54), (58, 77)]
    assert secondary_card_spans(pixels)[0] == [(12, 31), (35, 54), (58, 77)]
    result = reader.read_bytes(raw, META['native_png_sha256'], '')
    assert result['reason'] is None
    assert result['cards'] == META['visually_observed_cards']
    assert result['detected_spans_inclusive'] == [[12, 31], [35, 54], [58, 77]]
    assert result['row_geometry_source'] == SPLIT_PARTITION_RULE
    assert result['primary_incomplete_spans_inclusive'] == [[35, 54], [58, 77]]
    assert result['secondary_partition_proof']['bridged_spans'][0]['ink_gap_inclusive'] == [16, 18]
    assert verify_card_geometry_resolution(pixels, result)


def test_agreeing_scanlines_retain_all_native_boxes(reader):
    pixels = np.asarray(Image.open(FIXTURE).convert('L')).copy()
    # Synthetic control: restore one background pixel so the already supported
    # first card also passes the existing two-pixel split recovery.
    pixels[3, 18] = pixels[2, 18]
    result = read_pixels(reader, pixels)
    assert result['detected_spans_inclusive'] == [[12, 31], [35, 54], [58, 77]]
    assert result.get('reason') != 'CARD_SPAN_COVERAGE_UNRESOLVED'
    assert len(result['tiles']) == 3
    assert not result.get('cards') or result['cards'] == META['visually_observed_cards']


def test_rank_or_suit_scores_cannot_override_source_coverage_conflict(reader, monkeypatch):
    import parser_core.card_hybrid as module
    monkeypatch.setattr(module, 'secondary_card_spans',
                        lambda _: ([(13, 32), (36, 55), (59, 78)], []))
    def cannot_supply_authority(*args, **kwargs):
        pytest.fail('A per-tile score cannot resolve missing source coverage')
    monkeypatch.setattr(reader.rank_ncc, 'read', cannot_supply_authority)
    monkeypatch.setattr(reader.suit_backup, 'read', cannot_supply_authority)
    raw = FIXTURE.read_bytes()
    result = reader.read_bytes(raw, META['native_png_sha256'], '')
    assert result['reason'] == 'CARD_SPAN_COVERAGE_UNRESOLVED'
    assert not result.get('cards')


def test_real_two_card_row_and_unqualified_noise_do_not_gain_a_card(reader):
    pixels = np.asarray(Image.open(FIXTURE).convert('L')).copy()
    pixels[:, :35] = 0
    result = read_pixels(reader, pixels)
    assert result['cards'] == ['Kh', '4d']
    assert result['detected_spans_inclusive'] == [[35, 54], [58, 77]]
    # A bright card-width strip with a dark y3 gap has no independent glyphs.
    pixels[2, 12:32] = 240
    pixels[3, 12:16] = 240
    pixels[3, 19:32] = 240
    assert prove_split_card_partition(pixels[:, :-12]) is None
    noisy = read_pixels(reader, pixels)
    assert noisy['cards'] == result['cards']
    assert noisy['detected_spans_inclusive'] == result['detected_spans_inclusive']


@pytest.mark.parametrize('mutation', ['support', 'rank_ink', 'width', 'spacing'])
def test_partition_rejects_unproved_geometry(mutation):
    pixels = np.asarray(Image.open(FIXTURE).convert('L')).copy()
    if mutation == 'support':
        pixels[2, 17] = 0
    elif mutation == 'rank_ink':
        pixels[4:12, 13:21] = 240  # Disconnect the dark gap from a valid rank.
    elif mutation == 'width':
        pixels[2, 31:34] = 240
    else:
        pixels[2, 32:35] = 240  # Removes the independently dark card separation.
    assert prove_split_card_partition(pixels[:, :-12]) is None


@pytest.mark.parametrize('mutation', ['proof', 'spans', 'recovery', 'source', 'strip'])
def test_strict_consumer_replays_partition_receipt(reader, mutation):
    import copy
    raw = FIXTURE.read_bytes()
    pixels = np.asarray(Image.open(io.BytesIO(raw)).convert('L'))
    result = copy.deepcopy(reader.read_bytes(raw, META['native_png_sha256'], ''))
    if mutation == 'proof':
        result['secondary_partition_proof']['bridged_spans'][0]['ink_gap_inclusive'][0] += 1
    elif mutation == 'spans':
        result['detected_spans_inclusive'][0][0] += 1
    elif mutation == 'recovery':
        result['recovered_split_spans'] = []
    elif mutation == 'source':
        pixels = pixels.copy(); pixels[2, 17] = 0
    else:
        for field in ('secondary_partition_proof', 'row_geometry_source', 'recovered_split_spans'):
            result.pop(field, None)
    assert not verify_card_geometry_resolution(pixels, result)


def test_proved_geometry_still_requires_existing_card_classifier(reader, monkeypatch):
    import parser_core.card_hybrid as module
    monkeypatch.setattr(module, 'decide_suit', lambda *a, **k: dict(
        suit='UNRESOLVED', suit_source='UNRESOLVED', suit_status='TEMPLATE_LOW_CONFIDENCE'))
    result = reader.read_bytes(FIXTURE.read_bytes(), META['native_png_sha256'], '')
    assert result['row_geometry_source'] == SPLIT_PARTITION_RULE
    assert result['reason'] == 'SUIT_UNRESOLVED' and not result.get('cards')


def recovered_win_source(reader):
    """Synthetic geometry control made solely from retained native card pixels."""
    case = copy.deepcopy(json.loads((ROOT/'tests/fixtures/win_tail_t09_cross_epoch_source.json').read_text('utf8')))
    reading = case['reading']
    native = np.asarray(Image.open(FIXTURE).convert('L'))
    pixels = np.zeros_like(native)
    pixels[:, 12:32] = native[:, 12:32]
    for index, tile in enumerate(reading['hybrid']['tiles'][1:], 1):
        source = np.asarray(Image.open(io.BytesIO(base64.b64decode(tile['tile_png_base64']))).convert('L'))
        start = 12+23*index
        pixels[:, start:start+source.shape[1]] = source
        # Synthetic complete background support; glyph/classifier pixels retain
        # the native tile. Some original tails have a narrower y2 border.
        pixels[2, start:start+source.shape[1]] = 240
    stream = io.BytesIO(); Image.fromarray(pixels).save(stream, format='PNG')
    raw = stream.getvalue(); digest = hashlib.sha256(raw).hexdigest()
    pixel_digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    hybrid = reader.read_bytes(raw, digest, '')
    assert hybrid['cards'] == reading['cards'] and hybrid['reason'] is None
    assert hybrid['row_geometry_source'] == SPLIT_PARTITION_RULE
    reading.update(crop_png_base64=base64.b64encode(raw).decode('ascii'), crop_sha256=digest,
        row_pixel_sha256=pixel_digest, hybrid=hybrid, status='RAW_UNREVIEWED', reason=None)
    reading.pop('hybrid_authoritative_rejection', None)
    reading['raw_pixel_response']['source_sha256'] = digest
    # This control has just regenerated its complete hybrid receipt with the
    # current reader. Its response envelope must identify that same config,
    # rather than the historical fixture reader used to supply source tiles.
    reading['raw_pixel_response']['rank_config_sha256'] = reader.rank_ncc_config_sha256
    identity = f"{reading['table_session_id']}:{reading['frame_id']}:{reading['source_span_index']}:{digest}"
    reading['card_evidence_id'] = 'chatcard_'+hashlib.sha256(identity.encode()).hexdigest()[:24]
    for row in case['rows']+case['context']:
        if row['stable_row_id'] == reading['row_id']:
            row['row_pixel_sha256'] = pixel_digest
    return case


@pytest.mark.parametrize('mutation', [None, 'proof', 'recovery', 'strip'])
def test_recovered_partition_passes_real_strict_consumers_only_with_replayed_proof(reader, mutation):
    from parser_core.card_set_evidence import components
    from parser_core.win_tail_evidence import prove_win_tail
    case = recovered_win_source(reader)
    reading = case['reading']; hybrid = reading['hybrid']
    if mutation == 'proof':
        hybrid['secondary_partition_proof']['model_gray_sha256'] = '0'*64
    elif mutation == 'recovery':
        hybrid['recovered_split_spans'] = []
    elif mutation == 'strip':
        for field in ('secondary_partition_proof', 'row_geometry_source', 'recovered_split_spans'):
            hybrid.pop(field, None)
    # WIN tails have a distinct association envelope; their pixel proof is identical.
    import copy
    win_reading = copy.deepcopy(reading)
    win_reading.update(status='UNRESOLVED', reason='NO_NEARBY_CARD_EVENT')
    if mutation:
        with pytest.raises(ValueError, match='CARD_GEOMETRY_RESOLUTION_NOT_REPRODUCIBLE'):
            components(reading, case['rows'][-1], 5, complete=True)
        assert prove_win_tail(case['rows'], win_reading, case['context'], hand_context=True) is None
    else:
        domains, unknown = components(reading, case['rows'][-1], 5, complete=True)
        assert not unknown and domains == [[c] for c in reading['cards']]
        proof = prove_win_tail(case['rows'], win_reading, case['context'], hand_context=True)
        assert proof and proof['cards'] == reading['cards']


@pytest.mark.parametrize('width', [18, 19, 20, 22])
def test_valid_primary_widths_are_not_replaced_by_secondary_width_policy(reader, width):
    pixels = np.zeros((16, 224), dtype=np.uint8)
    pixels[:, 12:12 + width] = 240
    pixels[:, 12 + width + 3:12 + width * 2 + 3] = 240
    before = pixels.copy()
    expected = [(12, 11 + width), (15 + width, 14 + width * 2)]
    result = read_pixels(reader, pixels)
    assert result['detected_spans_inclusive'] == [list(span) for span in expected]
    assert result.get('reason') != 'CARD_SPAN_COVERAGE_UNRESOLVED'
    assert not result.get('cards')  # Blank backgrounds have no rank authority.
    assert np.array_equal(pixels, before)


def test_incompatible_secondary_boxes_do_not_replace_primary(reader, monkeypatch):
    import parser_core.card_hybrid as module
    monkeypatch.setattr(module, 'secondary_card_spans',
                        lambda _: ([(13, 32), (36, 55), (59, 78)], [{'source': 'CONFLICT_CONTROL'}]))
    raw = FIXTURE.read_bytes()
    result = reader.read_bytes(raw, META['native_png_sha256'], '')
    assert result['reason'] == 'CARD_SPAN_COVERAGE_UNRESOLVED'
    assert result['detected_spans_inclusive'] == [[35, 54], [58, 77]]
    assert not result.get('cards')
