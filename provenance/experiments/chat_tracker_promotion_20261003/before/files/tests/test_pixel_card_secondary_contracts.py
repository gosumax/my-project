"""Behavior contracts for secondary geometry and experimental pixel-only cards."""
import base64,hashlib,io,json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from parser_core.board_tiles import card_spans,secondary_card_spans
from parser_core.card_hybrid import CardHybrid
from parser_core.visual_tile_queue import VisualTileQueue
from tests.test_visual_tile_queue import Row

ROOT=Path(__file__).resolve().parents[1]
FIX=ROOT/'tests/fixtures/smoke15_card_contracts'

def glyph_row(gap=3,count=2):
    gray=np.zeros((16,226),dtype=np.uint8)
    for x in [12+i*(20+gap) for i in range(count)]:
        gray[:,x:x+20]=240
        gray[4:11,x+3:x+6]=30
        gray[4:11,x+11:x+16]=30
    return gray

def test_secondary_requires_complete_native_background_and_two_separated_glyphs():
    gray=glyph_row();gray[3,15:18]=30;before=gray.copy()
    primary,_=card_spans(gray,recover_split=True)
    assert primary==[(35,54)]
    spans,proof=secondary_card_spans(gray)
    assert spans==[(12,31),(35,54)]
    assert all(x['source']=='SECONDARY_NATIVE_BACKGROUND_Y2' for x in proof)
    assert np.array_equal(gray,before)
    missing=gray.copy();missing[4:11,23:28]=240
    assert secondary_card_spans(missing)==([],[])
    clipped=gray.copy();clipped[2,12:14]=0
    assert secondary_card_spans(clipped)==([],[])

@pytest.mark.parametrize('gap,count',[(1,2),(6,2),(3,1),(3,6)])
def test_secondary_rejects_wrong_spacing_or_card_count(gap,count):
    assert secondary_card_spans(glyph_row(gap,count))==([],[])

@pytest.fixture(scope='module')
def ordinary_reader():
    return CardHybrid(ROOT/'configs/card_hybrid_local.json',ROOT/'configs/suit_backup_local.json')

def test_admissible_primary_boxes_are_preserved_without_secondary_rebinding(ordinary_reader):
    gray=glyph_row();spans,_=card_spans(gray,recover_split=True)
    buf=io.BytesIO();Image.fromarray(gray).save(buf,format='PNG');raw=buf.getvalue()
    out=ordinary_reader.read_bytes(raw,hashlib.sha256(raw).hexdigest(),'J J')
    assert 'row_geometry_source' not in out
    assert [t['box_xyxy_exclusive'] for t in out['tiles']]==[[a,0,b+1,16] for a,b in spans]

def config(tmp_path,**updates):
    obj=json.loads((ROOT/'runs/rank_ncc_accuracy/rank_candidate_config.json').read_text('utf8'))
    obj.update(mode='PIXEL_ONLY',development_only=True);obj.update(updates)
    path=tmp_path/'rank.json';path.write_text(json.dumps(obj),'utf8');return path

def test_pixel_only_queue_never_creates_vl_client_and_preserves_sha_provenance(tmp_path):
    raw=(FIX/'006.png').read_bytes();gray=np.asarray(Image.open(io.BytesIO(raw)).convert('L'))
    calls=[]
    def forbidden_factory(worker):
        calls.append(worker);raise AssertionError('PIXEL_ONLY must never instantiate VL')
    pool=VisualTileQueue(forbidden_factory,tmp_path,workers=1,queue_size=1,
        card_hybrid_config=ROOT/'configs/card_hybrid_local.json',
        suit_backup_config=ROOT/'configs/suit_backup_local.json',rank_ncc_config=config(tmp_path),image_transport='ram')
    assert pool.submit(Row(pixels=gray,pixel_sha256=hashlib.sha256(gray.tobytes()).hexdigest()))
    pool.finish();assert calls==[] and pool.errors==[]
    out=pool.results['1:5:2'];payload=base64.b64decode(out['crop_png_base64'])
    assert hashlib.sha256(payload).hexdigest()==out['crop_sha256']==out['response_source_sha256']
    assert out['recognition_mode']=='PIXEL_ONLY' and out['vl_status']=='NOT_USED'
    assert out['raw_vl_response'] is None and out['raw_pixel_response']['source_sha256']==out['crop_sha256']
    assert out['identity_status']=='CHAT_ROW_SHA_ROUNDTRIP_VERIFIED' and out['card_evidence_id']
    for t in out['hybrid']['tiles']:
        tile_raw=base64.b64decode(t['tile_png_base64']);assert hashlib.sha256(tile_raw).hexdigest()==t['tile_sha256']

def test_pixel_only_cards_do_not_depend_on_vl_rank_text_or_legacy_cards(tmp_path):
    reader=CardHybrid(ROOT/'configs/card_hybrid_local.json',ROOT/'configs/suit_backup_local.json',rank_ncc_config=config(tmp_path))
    raw=(FIX/'006.png').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    empty=reader.read_bytes(raw,digest,'');garbage=reader.read_bytes(raw,digest,'2s 3s 4s',primary_cards=['2s','3s','4s'])
    assert [t['card'] for t in empty['tiles']]==[t['card'] for t in garbage['tiles']]
    assert all(t['rank_source']!='VL_POSITIONAL' for t in garbage['tiles'])
    with pytest.raises(ValueError,match='SHA mismatch'):reader.read_bytes(raw,'0'*64,'')

@pytest.mark.parametrize('updates',[
    {'best_score_threshold':.79},{'margin_threshold':.02},{'development_only':False},
    {'implementation_sha256':'0'*64},{'train_manifest_sha256':'0'*64}])
def test_tampered_frozen_rank_policy_fails_closed(tmp_path,updates):
    with pytest.raises(ValueError):CardHybrid(ROOT/'configs/card_hybrid_local.json',ROOT/'configs/suit_backup_local.json',rank_ncc_config=config(tmp_path,**updates))

def test_component_confident_disagreement_with_native_suit_is_rejected(ordinary_reader,monkeypatch):
    raw=(FIX/'006.png').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(ordinary_reader.suit_backup,'read',lambda _:dict(template_suit='h',template_confident=False,suit_backup_mode='BACKUP'))
    monkeypatch.setattr(ordinary_reader.component_suit,'read',lambda _:dict(template_suit='s',template_confident=True,suit_backup_mode='BACKUP'))
    out=ordinary_reader.read_bytes(raw,digest,'9 T 3')
    assert out['tiles'][0]['raw_suit2_suit']=='d'
    assert out['tiles'][0]['suit_status']=='SUIT_CONFLICT' and out['tiles'][0]['card']=='UNRESOLVED'
    assert out['reason']=='SUIT_CONFLICT' and 'cards' not in out
