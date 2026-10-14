"""Current clipped native UI must be proved from pixels, never OCR aliases."""
import copy
import json
from pathlib import Path

import pytest
import numpy as np

from parser_core import chat_nick_actor as policy
from parser_core.simple_seat_who import nick_observation_constraint


def source(raw):
    fixture = Path(__file__).parent / 'fixtures/smoke18_nick_ui_sources.json'
    return json.loads(fixture.read_text('utf-8'))['observations'][raw]['native_source']


def observation(raw):
    native = source(raw)
    response = native['ocr_response']
    return dict(table_id=native['table_id'], session_id=native['session_id'],
        hand_id=native['hand_id'], seat=native['seat'], frame=native['source_frame'],
        raw_text=response['raw_text'], confidence=response['score'],
        ocr_status=response['status'], item_id=response['item_id'], roi=native['roi'],
        pixel_sha256=native['pixel_sha256'], source_pts=native['source_pts'],
        time_base=native['time_base'], native_nick_source=native)


@pytest.mark.parametrize('raw,canonical', [
    ('ANE', 'ANTE'), ('ANe_F202', 'ANTE'), ('5OD', 'FOLD'),
    ('5OLD', 'FOLD'), ('5On', 'FOLD'), ('5On_F512', 'FOLD'), ('5on_F513', 'FOLD')])
def test_actual_current_native_ui_retains_raw_and_fixed_geometry(raw, canonical):
    row = observation(raw)
    original = copy.deepcopy(row)
    proof = policy.prove_nick_ui_overlay(row)
    assert proof is not None
    assert proof['canonical_ui'] == canonical
    assert proof['roi'] == row['roi'] == [139, 277, 221, 295]
    assert proof['pixel_sha256'] == row['pixel_sha256']
    assert nick_observation_constraint(row) is None
    assert row == original


def test_actual_current_r0bot_remains_nickname():
    row = observation('r0bot')
    original = copy.deepcopy(row)
    assert policy.prove_nick_ui_overlay(row) is None
    assert nick_observation_constraint(row) == dict(kind='EXACT', text='r0bot')
    assert row == original


@pytest.mark.parametrize('raw', ['ANE', 'ANCTE', '5OD', '5On', 'FOLn'])
def test_new_spelling_without_native_pixels_has_no_ui_authority(raw):
    assert policy.prove_nick_ui_overlay(dict(raw_text=raw)) is None
    assert nick_observation_constraint(dict(raw_text=raw)) == dict(kind='EXACT', text=raw)


@pytest.mark.parametrize('raw', ['ANE', 'ANCTE', '5OD', '5On', 'FOLn'])
def test_ui_like_spelling_on_actual_nickname_pixels_is_retained(raw):
    native = source('r0bot')
    native['ocr_response']['raw_text'] = raw
    assert policy.prove_nick_ui_overlay(native) is None


@pytest.mark.parametrize('field,value', [
    ('hand_id', 'foreign'), ('frame', -1), ('source_pts', None),
    ('roi', [0, 0, 82, 18]), ('pixel_sha256', 'bad'), ('raw_text', 'different')])
def test_unlisted_ui_wrapper_tamper_rejects(field, value):
    row = observation('5OD')
    row[field] = value
    with pytest.raises(ValueError):
        policy.prove_nick_ui_overlay(row)


def test_unlisted_ui_source_hash_tamper_rejects():
    native = source('5OD')
    native['source_ref'] = 'bad'
    with pytest.raises(ValueError):
        policy.prove_nick_ui_overlay(native)


@pytest.mark.parametrize('raw', ['r0bot', 'ANE'])
def test_unlisted_native_unknown_pts_retains_literal_without_ui_authority(raw):
    row = observation(raw)
    row['source_pts'] = row['native_nick_source']['source_pts'] = None
    original = copy.deepcopy(row)
    assert policy.prove_nick_ui_overlay(row) is None
    assert nick_observation_constraint(row) == dict(kind='EXACT', text=raw)
    assert row == original


def test_legacy_alias_unknown_pts_keeps_original_clock_rejection():
    row = observation('5OLD')
    row['source_pts'] = row['native_nick_source']['source_pts'] = None
    with pytest.raises(ValueError, match='NICK_UI_SOURCE_CLOCK_UNKNOWN'):
        policy.prove_nick_ui_overlay(row)


def test_pixel_proof_does_not_depend_on_a_new_text_whitelist():
    native = source('5OD')
    native['ocr_response']['raw_text'] = 'unlisted-spelling'
    assert policy.prove_nick_ui_overlay(native)['canonical_ui'] == 'FOLD'


def test_ambiguous_physical_families_retain_literal_candidate(monkeypatch):
    fixture = Path(__file__).parent / 'fixtures/nick_ui_overlay_native_sources.json'
    native = json.loads(fixture.read_text('utf-8'))['1185']
    native['ocr_response']['raw_text'] = 'unlisted-spelling'
    cores = copy.deepcopy(policy.NICK_UI_CORES)
    cores['ANTE'] = copy.deepcopy(cores['FOLD'])
    monkeypatch.setattr(policy, 'NICK_UI_CORES', cores)
    monkeypatch.setattr(policy, 'NICK_UI_PIXEL_TWINS', {})
    assert policy.prove_nick_ui_overlay(native) is None


def bottom_sources():
    return json.loads((Path(__file__).parent/'fixtures/smoke19_bottom_nick_sources.json').read_text('utf-8'))['native_sources']


@pytest.mark.parametrize('frame',['2089','2092'])
def test_real_bottom_only_ante_is_skipped_without_reading_its_spelling(frame):
    row=bottom_sources()[frame];before=copy.deepcopy(row)
    proof=policy.prove_nick_ui_overlay(row)
    assert proof and proof['canonical_ui']=='BOTTOM_CLIPPED_SERVICE_TEXT'
    assert proof['upper_ink_count']==0 and proof['glyph_band'][1]==11 and proof['glyph_band'][3]==15
    assert row==before
    row['ocr_response']['raw_text']='arbitrary-unlisted-spelling'
    assert policy.prove_nick_ui_overlay(row)['canonical_ui']=='BOTTOM_CLIPPED_SERVICE_TEXT'


def test_real_full_height_gosteeer_is_first_one_frame_nick_after_clipped_ante():
    from parser_core.nickname_roster_binding import _nickname_triples
    sources=bottom_sources();rows=list(sources.values());native=sources['2132']
    assert policy.prove_nick_ui_overlay(native) is None
    key=(native['table_id'],native['session_id'],native['hand_id'])
    result=_nickname_triples(rows,key,2085,2132,.95,
        dict(source_frame=2107,source_pts=2107*512,time_base=[1,15360]),(2108,2108*512))
    assert result['seat7'][0]['frames']==[2132]
    assert result['seat7'][0]['constraint']==dict(kind='EXACT',text='gosteeer')


def test_bottom_rule_rejects_empty_frame_border_and_retains_upper_glyph_continuation():
    pixels=np.full((15,73),32,np.uint8)
    assert policy._prove_bottom_clipped_service_text(pixels) is None
    pixels[-1,:]=255
    assert policy._prove_bottom_clipped_service_text(pixels) is None
    pixels[:]=32;pixels[-3:,25:35]=255
    assert policy._prove_bottom_clipped_service_text(pixels)
    pixels[4,27]=255
    assert policy._prove_bottom_clipped_service_text(pixels) is None
