import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from parser_core.table_card_back_evidence import CardBackSupplement

ROOT=Path(__file__).resolve().parents[1]
ROWS=json.loads((ROOT/'tests/fixtures/table_card_back_smoke19.json').read_text('utf-8'))


@pytest.fixture(scope='module')
def model():return CardBackSupplement(ROOT)


@pytest.mark.parametrize('table,frame,positive',[(7,151,False),(7,161,False),(7,171,False),
    (7,181,True),(7,191,True),(7,201,True),(7,211,True),(8,131,False),(8,141,False),
    (8,151,True),(8,161,True),(8,171,True),(8,181,True),(8,191,True)])
def test_reviewed_blank_partial_and_complete_native_card_regions(model,table,frame,positive):
    row=next(r for r in ROWS if r['table_id']==table and r['source_frame']==frame)
    bgr=np.frombuffer(base64.b64decode(row['pixels_bgr_base64']),np.uint8).reshape(row['shape'])
    assert hashlib.sha256(bgr.tobytes()).hexdigest()==row['raw_bgr_sha256']
    matched=model.match(table,row['seat'],row['roi'],bgr)
    assert (matched is not None)==positive
    assert model.match(table,'seat1',row['roi'],bgr) is None
    assert model.match(table,row['seat'],[0,0,58,35],bgr) is None
    assert model.match(table,row['seat'],row['roi'],np.full_like(bgr,40)) is None
