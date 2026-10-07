"""Receipt safety tests only; no parser/edge/validator execution."""
import copy
import json

import pytest

from experiments.audit_remaining_fixed_window_candidates import BASE,OVERLAY,ROOT,read,verify_receipts


@pytest.fixture(scope='module')
def receipt():
    return (read(ROOT/'runs/production_smoke17_20261001_013909/ordered_timeline.jsonl'),
        read(OVERLAY/'CORE_280/classified_rows.jsonl'),
        json.loads((BASE/'CORE_FIXED_WINDOWS.assembled_hands.json').read_text('utf-8')),
        read(BASE/'CORE_FIXED_WINDOWS.hands.jsonl'),
        read(OVERLAY/'CORE_280/row_reason_attribution.jsonl'))


def test_current_receipt_parity(receipt):verify_receipts(*receipt)


@pytest.mark.parametrize('field,value',[('raw_ocr_text','changed'),('epoch',999),
    ('stable_row_id','replacement'),('table_id','T09'),('status','ERROR'),
    ('row_pixel_sha256','replacement')])
def test_original_field_mutation_rejected(receipt,field,value):
    data=copy.deepcopy(receipt);data[1][0][field]=value
    with pytest.raises(ValueError,match='Source field/order mutation'):verify_receipts(*data)


def test_source_order_mutation_rejected(receipt):
    data=copy.deepcopy(receipt);data[1][0],data[1][1]=data[1][1],data[1][0]
    with pytest.raises(ValueError,match='Source field/order mutation'):verify_receipts(*data)


def test_blocker_deletion_rejected(receipt):
    data=copy.deepcopy(receipt);data[3][0]['fully_valid_blocking_checks'].pop()
    with pytest.raises(ValueError,match='Blocker loss'):verify_receipts(*data)


def test_false_session_rejected(receipt):
    data=copy.deepcopy(receipt);data[2][0]['session_id']='other'
    with pytest.raises(ValueError,match='Hand context mismatch'):verify_receipts(*data)


def test_missing_start_source_rejected(receipt):
    data=copy.deepcopy(receipt)
    hand,record=data[2][0],data[3][0]
    hand['messages']=hand['messages'][1:]
    record['source_row_ids']=[rid for m in hand['messages'] for rid in m['source_row_ids']]
    with pytest.raises(ValueError,match='Missing fixed start'):verify_receipts(*data)
