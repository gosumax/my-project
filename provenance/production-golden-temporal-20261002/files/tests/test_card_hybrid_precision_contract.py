import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from parser_core.board_tiles import card_spans, read_board_tiles_bytes
from parser_core.card_hybrid import CardHybrid, parse_rank_tokens
from parser_core.visual_tile_queue import VisualTileQueue
from parser_core.visual_cards import associate_visual_evidence, project_chat_card_events

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT/'tests/fixtures/smoke15_card_contracts'
CASES = json.loads((FIX/'manifest.json').read_text('utf8'))

@pytest.fixture(scope='module')
def reader():
    return CardHybrid(ROOT/'configs/card_hybrid_local.json', ROOT/'configs/suit_backup_local.json')

def primary(raw, digest, text):
    reading = read_board_tiles_bytes(raw, digest, text)
    suits = {'♠':'s','♥':'h','♦':'d','♣':'c'}
    return [('T' if c[:-1]=='10' else c[:-1])+suits[c[-1]] for c in reading.cards] if reading else None

@pytest.mark.parametrize('case', CASES, ids=lambda c:c['png'])
def test_frozen_pixel_contracts_have_no_wrong_published_tile(reader, case):
    raw=(FIX/case['png']).read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert digest==case['source_sha256']
    gray=np.asarray(Image.open(io.BytesIO(raw)).convert('L'))
    result=reader.read_bytes(raw,digest,case['vl_raw'],primary_cards=primary(raw,digest,case['vl_raw']))
    assert len(result['tiles'])==len(case['gt_cards'])
    for tile,gt in zip(result['tiles'],case['gt_cards']):
        x1,y1,x2,y2=tile['box_xyxy_exclusive']
        rgb=np.asarray(Image.open(io.BytesIO(base64.b64decode(tile['tile_png_base64']))))
        assert np.array_equal(rgb,np.repeat(gray[y1:y2,x1:x2,None],3,axis=2))
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==tile['tile_pixel_sha256']
        assert tile['card'] in ('UNRESOLVED',gt)

def test_literal_parser_never_guesses_unicode_or_reads_prose():
    assert parse_rank_tokens('Cards: Qh 10♦ Ts 5h')==['Q','T','T','5']
    assert parse_rank_tokens('♀ 🔍 ♡ B 0 ✈ ✦ ↓')==[]
    assert parse_rank_tokens('7 ♥ J ♥ 7 ♣ 4 ♦ ♀ ♦')==['7','J','7','4']
    assert parse_rank_tokens('Q1 a♠ a♦')==['A','A']

def test_split_background_needs_independent_support_and_preserves_native_boxes():
    gray=np.zeros((16,150),dtype=np.uint8)
    gray[:,12:32]=240;gray[:,35:55]=240;gray[3,17:19]=30
    before=gray.copy()
    old,_=card_spans(gray)
    new,evidence=card_spans(gray,recover_split=True)
    assert old==[(35,54)] and new==[(12,31),(35,54)]
    assert evidence[0]['support_scanline_y']==2
    assert np.array_equal(gray,before)
    gray[2,17:19]=30
    assert card_spans(gray,recover_split=True)[0]==old

def test_native_primary_keeps_legacy_disagreement_visible(reader):
    case=next(c for c in CASES if c['png']=='026.png');raw=(FIX/case['png']).read_bytes()
    result=reader.read_bytes(raw,case['source_sha256'],case['vl_raw'],primary_cards=['9h','9c'])
    tile=result['tiles'][1]
    assert tile['legacy_board_suit']=='c'
    assert tile['primary_suit']==tile['template_suit']=='d'
    assert tile['card']=='9d'

def test_genuine_native_ncc_conflict_and_low_ncc_stay_unresolved(reader, monkeypatch):
    case=next(c for c in CASES if c['png']=='006.png');raw=(FIX/case['png']).read_bytes()
    monkeypatch.setattr(reader.suit_backup,'read',lambda _:dict(template_suit='s',template_confident=True,suit_backup_mode='BACKUP'))
    result=reader.read_bytes(raw,case['source_sha256'],case['vl_raw'])
    assert result['tiles'][0]['suit_status']=='SUIT_CONFLICT'
    assert result['tiles'][0]['card']=='UNRESOLVED'
    monkeypatch.setattr(reader.suit_backup,'read',lambda _:dict(template_suit='h',template_confident=False,suit_backup_mode='BACKUP'))
    monkeypatch.setattr(reader.component_suit,'read',lambda _:dict(template_suit='h',template_confident=False,suit_backup_mode='BACKUP'))
    result=reader.read_bytes(raw,case['source_sha256'],case['vl_raw'])
    assert all(t['card']=='UNRESOLVED' for t in result['tiles'])

def test_equal_count_wrong_vl_preserves_raw_and_explicit_pixel_rank_source(reader):
    case=next(c for c in CASES if c['png']=='019.png');raw=(FIX/case['png']).read_bytes()
    result=reader.read_bytes(raw,case['source_sha256'],case['vl_raw'])
    tile=result['tiles'][3]
    assert tile['vl_rank']=='4' and tile['final_rank']=='A'
    assert tile['rank_status']=='VL_RANK_CORRECTED_FROM_PIXELS'
    assert tile['rank_source']=='EXACT_VERIFIED_TRAIN_RANK_GLYPH'
    assert result['vl_rank_source']=='VL_WHOLE_CHAT_ROW'
    assert result['rank_source']=='MIXED_VL_AND_EXACT_PIXEL_TEMPLATES'

def test_pixel_veto_keeps_other_tiles_and_withholds_row_cards(reader):
    case=next(c for c in CASES if c['png']=='116.png');raw=(FIX/case['png']).read_bytes()
    result=reader.read_bytes(raw,case['source_sha256'],case['vl_raw'])
    assert result['tiles'][0]['rank_status']=='RANK_VISUAL_CONFLICT'
    assert result['tiles'][0]['vl_rank']=='A'
    assert result['tiles'][0]['final_rank']=='UNRESOLVED'
    assert result['tiles'][1]['card']=='Jc'
    assert result['reason']=='RANK_VISUAL_CONFLICT' and 'cards' not in result

@pytest.mark.parametrize('failure',['NEW_REASON','EXCEPTION'])
def test_backup_never_leaks_legacy_cards_after_hybrid_rejection(tmp_path,monkeypatch,failure):
    import parser_core.visual_tile_queue as queue_module
    monkeypatch.setattr(queue_module,'read_board_tiles_bytes',lambda *a:SimpleNamespace(cards=('Q♦','7♣'),template_sha256='legacy'))
    class Client:
        def read(self,path,sha,trim):return {'status':'OK','raw_text':'Qd 7c'}
        def close(self):pass
    class Hybrid:
        suit_backup=SimpleNamespace(mode='BACKUP')
        def read(self,*a,**kw):
            if failure=='EXCEPTION':raise ValueError('frozen failure')
            return {'reason':'NEW_REASON','tiles':[]}
    pixels=np.zeros((16,226),dtype=np.uint8);pixels[:,:50]=255
    row=SimpleNamespace(table_id=1,frame_id=5,source_span_index=2,session_id='session',video_time=.2,source_pts=3072,time_base_num=1,time_base_den=15360,pixel_sha256='raw',pixels=pixels)
    pool=VisualTileQueue(lambda _:Client(),tmp_path,workers=1,queue_size=1);pool.hybrid=Hybrid()
    pool.submit(row);pool.finish();result=pool.results['1:5:2']
    assert result['cards'] is None and result['status']=='UNRESOLVED'
    assert result['legacy_board_cards']==['Qd','7c']
    assert result['hybrid_authoritative_rejection']

def test_unknown_rejection_survives_association_and_projection():
    from tests.test_chat_card_projection import fixture
    rows,r=fixture();r.update(reason='NEW_REASON',hybrid_authoritative_rejection=True)
    associate_visual_evidence(rows,{'1:8:2':r})
    assert r['cards'] is None and r['reason']=='NEW_REASON'
    assert r['status']=='UNRESOLVED'
    assert project_chat_card_events(rows,{'1:8:2':r})==0
