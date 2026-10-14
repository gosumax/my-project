import copy
import json
from pathlib import Path

import pytest

from parser_core.poker_hand_assembly import make_message, observed_wrapped_continuation
from parser_core.poker_reconstruction.assembly import candidate_assemblies


SOURCES = json.loads((Path(__file__).parent / 'fixtures/inline_show_native_sources.json').read_text('utf-8'))


@pytest.mark.parametrize('source', SOURCES, ids=lambda s: s['hand'] + ':' + s['rows'][0]['stable_row_id'])
def test_native_inline_show_retains_the_bound_card_row_as_one_source(source):
    rows = copy.deepcopy(source['rows'])
    for i, row in enumerate(rows):
        row['source_index'] = i
    before = copy.deepcopy(rows)
    assert observed_wrapped_continuation(rows[:1], rows[1])
    message = make_message(rows, require_pixel_cards=True)
    assert message['action'] == 'SHOW'
    assert message['cards'] == rows[0]['cards']
    assert message['field_sources']['cards'] == 'OBSERVED_PIXEL'
    assemblies, exhausted, _ = candidate_assemblies(rows)
    assert not exhausted and len(assemblies) == 1
    assert len(assemblies[0]) == 1 and assemblies[0][0].kind == 'SHOW'
    assert assemblies[0][0].source['rows'] == rows
    assert rows == before


@pytest.mark.parametrize('mutation', [
    'frame', 'row_gap', 'table', 'session', 'epoch', 'row_id', 'row_hash',
    'crop_hash', 'identity', 'recognizer', 'mode', 'financial_suffix', 'opaque_head',
])
def test_inline_show_cannot_gain_group_authority_from_a_broken_source(mutation):
    rows = copy.deepcopy(SOURCES[1]['rows'])
    head, tail = rows
    if mutation == 'frame': tail['frame_id'] += 1
    elif mutation == 'row_gap': tail['row_index'] += 1
    elif mutation == 'table': tail['table_id'] = 'T99'
    elif mutation == 'session': tail['session_id'] = 'foreign'
    elif mutation == 'epoch': tail['epoch'] += 1
    elif mutation == 'row_id': tail['stable_row_id'] = 'foreign'
    elif mutation == 'row_hash': tail['row_pixel_sha256'] = '0' * 64
    elif mutation == 'crop_hash': head['card_crop_pixel_sha256'] = '0' * 64
    elif mutation == 'identity': head['identity_status'] = 'UNKNOWN'
    elif mutation == 'recognizer': head['raw_pixel_response']['recognizer'] = 'OTHER'
    elif mutation == 'mode': head['recognition_mode'] = 'OCR'
    elif mutation == 'financial_suffix': head['raw_ocr_text'] += ' calls 100'
    elif mutation == 'opaque_head': head['raw_ocr_text'] = 'Dealer: gekcon555 shows Pair, 7?'
    if mutation in {'financial_suffix', 'opaque_head'}:
        head['raw_text'] = head['raw_ocr_text']
    assert not observed_wrapped_continuation(rows[:1], rows[1])
