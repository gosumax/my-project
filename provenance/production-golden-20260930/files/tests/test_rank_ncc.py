"""Rank-only contracts: no CardHybrid/suit decisions are altered."""
import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from parser_core.rank_ncc import RankNCC,rank_feature,RANKS
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'runs/card_train_smoke16_test_20260930/split_v2/train_manifest.jsonl'

@pytest.fixture(scope='module')
def reader():return RankNCC(MANIFEST,threshold=.98,margin=.03)

def image(row):return np.asarray(Image.open(row['tile_png']).convert('RGB'))

def test_train_provenance_pin_and_tamper(tmp_path):
    assert RankNCC(MANIFEST,expected_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest()).manifest_sha256
    with pytest.raises(ValueError,match='manifest changed'):
        RankNCC(MANIFEST,expected_manifest_sha256='0'*64)
    row=json.loads(MANIFEST.read_text('utf8').splitlines()[0]);row['tile_sha256']='0'*64
    path=tmp_path/'manifest.jsonl';path.write_text(json.dumps(row)+'\n','utf8')
    with pytest.raises(ValueError,match='PNG changed'):RankNCC(path)

def test_source_exclusion_full_path_equals_basename(reader):
    rgb=image(reader.rows[0]);source=Path(reader.rows[0]['source_video']).name
    bare=reader.read(rgb,excluded_source=source)
    full=reader.read(rgb,excluded_source=str(ROOT/'video'/source))
    assert bare==full
    assert not bare['rank_accepted']
    assert bare['rank_reason']=='SOURCE_HELD_OUT_CLASS_MISSING'

@pytest.mark.parametrize('shape',[(16,17,3),(16,23,3),(32,40,3),(16,20),(16,20,4)])
def test_unverified_geometry_fails_closed(reader,shape):
    out=reader.read(np.zeros(shape,dtype=np.uint8))
    assert out['rank_result']=='UNRESOLVED' and not out['rank_accepted']
    assert out['rank_reason']=='UNSUPPORTED_CHAT_GEOMETRY'

def test_empty_card_no_forced_rank(reader):
    result=reader.read(np.full((16,20,3),240,np.uint8))
    assert result['rank_roi'] is None
    assert result['rank_candidate']=='UNRESOLVED' and not result['rank_accepted']

def test_similar_rank_tie_is_rejected_without_gt_runtime(reader):
    rgb=image(reader.rows[0]);vector,_,reason=rank_feature(rgb)
    assert reason is None
    original_bank,original_labels=reader.bank,reader.labels
    try:
        reader.bank=np.repeat(vector[None,:],len(reader.bank),axis=0)
        # Every rank now has identical visual evidence; margin must prevent acceptance.
        result=reader.read(rgb)
        assert result['rank_margin']==0
        assert result['rank_result']=='UNRESOLVED'
    finally:reader.bank,reader.labels=original_bank,original_labels

def test_component_origin_preserves_feature_and_brightness_stays_guarded(reader):
    rgb=image(reader.rows[0]);feature,box,reason=rank_feature(rgb)
    assert reason is None
    shifted=rgb.copy();x1,y1,x2,y2=box
    shifted[y1:y2,x1:x2]=255
    shifted[y1:y2,x1-1:x2-1]=rgb[y1:y2,x1:x2]
    other,otherbox,reason=rank_feature(shifted)
    assert reason is None and otherbox[0]==box[0]-1
    assert np.array_equal(feature,other)
    brighter=(rgb.astype(float)*.75+30).astype(np.uint8)
    other,_,reason=rank_feature(brighter)
    # The existing absolute ink locator may change bounds under contrast;
    # normalized NCC itself is affine invariant, the locator is not.
    assert reason is None
    assert reader.read(brighter)['rank_result'] in ('UNRESOLVED',reader.rows[0]['ground_truth'][:-1])
    fixed,_,reason=rank_feature(rgb,'fixed_ncc')
    adjusted,_,reason2=rank_feature(brighter,'fixed_ncc')
    assert reason is None and reason2 is None and float(fixed@adjusted)>.999

def test_default_policy_is_diagnostic_only():
    model=RankNCC(MANIFEST)
    assert all(not model.read(image(next(r for r in model.rows if r['ground_truth'][:-1]==rank)))['rank_accepted'] for rank in RANKS)
