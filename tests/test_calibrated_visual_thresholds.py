"""Native controls for the registered pixel-only calibration candidates."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from parser_core.card_hybrid import CardHybrid
from parser_core.sitout_text_pattern import MODEL, SitOutTextPattern

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests/fixtures/threshold_calibration'
SITOUT = json.loads((FIX / 'sitout/fixture_manifest.json').read_text('utf8'))
CARDS = json.loads((FIX / 'card_rows.json').read_text('utf8'))


@pytest.fixture(scope='module')
def reader():
    return CardHybrid(ROOT / 'configs/card_hybrid_local.json',
                      ROOT / 'configs/suit_backup_local.json',
                      rank_ncc_config=ROOT / 'configs/rank_ncc_production.json')


def test_actual_rank_policy_and_provenance_are_calibrated(reader):
    assert (reader.rank_ncc.threshold, reader.rank_ncc.margin) == (.71, .015)
    assert reader.rank_ncc_config_sha256 == hashlib.sha256(
        (ROOT / 'configs/rank_ncc_production.json').read_bytes()).hexdigest()
    assert reader.rank_mode == 'PIXEL_ONLY' and not reader.development_only


@pytest.mark.parametrize('change', [
    {'best_score_threshold': .80}, {'margin_threshold': .01},
    {'best_score_threshold': .70}, {'margin_threshold': .014},
])
def test_unregistered_production_rank_policy_is_rejected(tmp_path, change):
    data = json.loads((ROOT / 'configs/rank_ncc_production.json').read_text('utf8'))
    data.update(change)
    path = tmp_path / 'rank.json'
    path.write_text(json.dumps(data), 'utf8')
    with pytest.raises(ValueError, match='Rank policy'):
        CardHybrid(ROOT / 'configs/card_hybrid_local.json',
                   ROOT / 'configs/suit_backup_local.json', rank_ncc_config=path)


@pytest.mark.parametrize('case', CARDS, ids=lambda c: c['review_id'])
def test_new_native_card_admissions_match_blind_pixel_labels(reader, case):
    raw = (FIX / case['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == case['png_sha256']
    gray = np.asarray(Image.open(io.BytesIO(raw)).convert('L'))
    assert hashlib.sha256(gray.tobytes()).hexdigest() == case['gray_sha256']
    out = reader.read_bytes(raw, case['png_sha256'], '')
    assert out.get('cards') == case['cards'] and not out.get('reason')
    assert out['rank_ncc_config_sha256'] == reader.rank_ncc_config_sha256


@pytest.mark.parametrize('case', SITOUT['fixtures'], ids=lambda c: c['review_id'])
def test_native_sitout_text_and_ui_controls_keep_only_pixel_authority(case):
    raw = (FIX / 'sitout' / case['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == case['file_sha256']
    gray = np.asarray(Image.open(io.BytesIO(raw)).convert('L'))
    assert hashlib.sha256(gray.tobytes()).hexdigest() == case['gray_sha256']
    out = SitOutTextPattern().evaluate(gray)
    assert out['matched'] is case['expected_at_086']
    assert out['score'] == case['current_score']
    assert out['threshold'] == .86 and out['numeric_authority'] is False
    assert not any(k in out for k in ('amount', 'actor', 'hand_id', 'seat'))


@pytest.mark.parametrize('threshold', [.85, .87, .95, True, None])
def test_stale_or_unregistered_sitout_model_threshold_is_rejected(tmp_path, threshold):
    data = json.loads(MODEL.read_text('utf8'))
    data['threshold'] = threshold
    path = tmp_path / 'sitout.json'
    path.write_text(json.dumps(data), 'utf8')
    with pytest.raises(ValueError, match='MODEL_CONTRACT_CONFLICT'):
        SitOutTextPattern(path)
