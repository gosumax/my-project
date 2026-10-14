import base64
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from parser_core.numeric_text_evidence import exact_text_temporal_binding
from parser_core.start_stack_visibility import VisibilitySource


@pytest.fixture
def source():
    data = json.loads((Path(__file__).parent / 'fixtures/numeric_text_temporal_sources.json').read_text('utf-8'))
    cap = data['captures'][1]
    vs = VisibilitySource()
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(cap['png_base64']), np.uint8), 1), cv2.COLOR_BGR2RGB)
    x, y, X, Y = cap['roi']
    a, b, A, B = vs.rois['slot_03']['seats']['seat1']['stack']
    gray = cv2.cvtColor(rgb[b-y:B-y, a-x:A-x], cv2.COLOR_RGB2GRAY)
    return data, gray, vs.gate


def test_native_codec_background_change_keeps_exact_text_and_one_original_ocr(source):
    data, gray, gate = source
    before = copy.deepcopy(data)
    proof = exact_text_temporal_binding(gray, data['state'], data['observation'], 2696, gate)
    assert proof and proof['text_rectangle'] == [22, 2, 61, 14]
    assert proof['original_pixel_sha256'] != proof['target_pixel_sha256']
    assert proof['evidence_kind'] == 'SOURCE_TEXT_PIXELS_IDENTICAL'
    assert proof['ocr_inference_count'] == 1 and not proof['new_ocr_performed']
    assert data == before


def test_real_first_stack_uses_the_same_text_proof_through_all_visibility_gates():
    from parser_core.joint_start_stack import prove
    data=json.loads((Path(__file__).parent/'fixtures/joint_text_positive_native_source.json').read_text('utf-8'))
    before=copy.deepcopy(data)
    proof=prove(data['case'],data['database_metadata_raw'])
    assert proof['status']=='PROVEN',proof.get('reason')
    assert proof['calculated_start_stack']==175474
    assert proof['first_visible_proof']['frames']==[5382,5383,5384]
    assert proof['sql_numeric_bindings'][-1]['numeric_source_evidence']['evidence_kind']=='SOURCE_TEXT_PIXELS_IDENTICAL'
    assert data==before
    numeric=next(s for s in data['case']['raw_sql_visual_states'] if s['frame_index']==5382)
    evidence=json.loads(numeric['digit_gate_evidence'])
    evidence['temporal_frames'][-1]['hash']='0'*64
    numeric['digit_gate_evidence']=json.dumps(evidence)
    payload=json.loads(numeric['provenance']);payload['state_payload']['digit_gate_evidence']=evidence
    numeric['provenance']=json.dumps(payload)
    assert prove(data['case'],data['database_metadata_raw'])['status']=='UNKNOWN'


@pytest.mark.parametrize('mutation', ['digit', 'subthreshold_text', 'position', 'extra_digit', 'edge',
    'temporal_hash', 'temporal_frame', 'unconfirmed', 'late_frame', 'low_confidence', 'numeric', 'saved_boxes'])
def test_different_text_or_unbound_source_has_no_numeric_equivalence(source, mutation):
    data, gray, gate = source
    state, obs, frame = data['state'], data['observation'], 2696
    if mutation == 'digit': gray[6, 25] = 255 - gray[6, 25]
    elif mutation == 'subthreshold_text': gray[2, 22] ^= 1
    elif mutation == 'position': gray[:] = np.roll(gray, 1, axis=1)
    elif mutation == 'extra_digit': gray[3:13, 65:67] = 230
    elif mutation == 'edge': gray[3:13, 1:3] = 230
    elif mutation == 'late_frame': frame += 2
    elif mutation == 'low_confidence': state['ocr_confidence'] = .949
    elif mutation == 'numeric': state['ocr_numeric'] = '89214'
    evidence = json.loads(state['digit_gate_evidence'])
    if mutation == 'temporal_hash': evidence['temporal_frames'][0]['hash'] = '0' * 64
    elif mutation == 'temporal_frame': evidence['temporal_frames'][0]['frame'] += 1
    elif mutation == 'unconfirmed': evidence['temporal_status'] = 'STATIC_ONLY'
    elif mutation == 'saved_boxes': evidence['glyphs'][0]['box'][0] += 1
    # Mutated pixels get a matching receipt: shape/text checks must still reject.
    if mutation in {'digit', 'subthreshold_text', 'position', 'extra_digit', 'edge'}:
        evidence['temporal_frames'][0]['hash'] = hashlib.sha256(gray.tobytes()).hexdigest()
    state['digit_gate_evidence'] = json.dumps(evidence)
    payload = json.loads(state['provenance'])
    payload['state_payload']['digit_gate_evidence'] = evidence
    for k in ('ocr_confidence', 'ocr_numeric'):
        payload['state_payload'][k] = state[k]
    state['provenance'] = json.dumps(payload)
    assert exact_text_temporal_binding(gray, state, obs, frame, gate) is None
