"""Bounded diagnostic regressions for short tiles with separate native NCC ROI."""
import base64,hashlib,io,json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from parser_core.card_hybrid import CardHybrid
from experiments.compare_card_templates import predict
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope='module')
def reader():return CardHybrid(ROOT/'configs/card_hybrid_local.json',ROOT/'configs/suit_backup_local.json')
def read(reader,row,ranks=2):
 b=io.BytesIO();Image.fromarray(row).save(b,format='PNG');raw=b.getvalue();return reader.read_bytes(raw,hashlib.sha256(raw).hexdigest(),' '.join(['J']*ranks))
def manifest_tiles():
 for r in [json.loads(x) for x in (ROOT/'runs/card_train_smoke16_test_20260930/split_v2/validation_manifest.jsonl').read_text('utf8').splitlines()]:
  raw=Path(r['tile_png']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['tile_sha256']
  with Image.open(io.BytesIO(raw)) as im:
   assert hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==r['tile_pixel_sha256'];yield r,np.asarray(im.convert('L')).copy()
def native_evidence(tile,row):
 raw=base64.b64decode(tile['tile_png_base64']);assert hashlib.sha256(raw).hexdigest()==tile['tile_sha256'];im=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'));assert hashlib.sha256(im.tobytes()).hexdigest()==tile['tile_pixel_sha256'];x1,y1,x2,y2=tile['box_xyxy_exclusive'];assert np.array_equal(im,np.repeat(row[y1:y2,x1:x2,None],3,axis=2))
def test_short_roi_validation_predictions_and_untouched_tiles(reader):
 counts={'cases':0,'accepted':0,'wrong_accepted':0,'rejected':0,'query_evidence':0};wrong=[]
 for r,full in manifest_tiles():
  tile=full[:,:20]
  for left,right in ((1,0),(2,0),(0,1),(0,2),(1,1)):
   for whole_edge in (False,True):
    row=np.zeros((16,140),dtype=np.uint8)
    for x in (12,35):row[:,x:x+20]=tile
    ys=slice(None) if whole_edge else 3
    if left:row[ys,12:12+left]=150
    if right:row[ys,32-right:32]=150
    preserved=row.copy();result=read(reader,row);assert np.array_equal(row,preserved);actual=result['tiles'][0];counts['cases']+=1
    assert actual['box_xyxy_exclusive']==[12+left,0,32-right,16];assert actual['size_wh']==[20-left-right,16];native_evidence(actual,row)
    assert actual['raw_suit2_suit']=='UNRESOLVED'
    if 'ncc_query_box_xyxy_exclusive' in actual:
     counts['query_evidence']+=1;qx,_,end,_=actual['ncc_query_box_xyxy_exclusive'];sx,sy,sw,sh=actual['suit_glyph_box_xywh'];assert [sx,2,sx+9,13]==actual['ncc_suit_roi_box_xyxy_exclusive'];assert 12<=sx and sx+9<=32-right
     raw=base64.b64decode(actual['ncc_query_png_base64']);assert hashlib.sha256(raw).hexdigest()==actual['ncc_query_sha256'];q=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'));assert hashlib.sha256(q.tobytes()).hexdigest()==actual['ncc_query_pixel_sha256'];assert np.array_equal(q,np.repeat(row[:,qx:end,None],3,axis=2))
     expected=reader.suit_backup.read(q)
     fixed=actual.get('component_suit_original_result',actual)
     for k,v in expected.items():assert fixed[k]==v
    if 'component_suit_original_result' in actual:
     native=np.asarray(Image.open(io.BytesIO(base64.b64decode(actual['tile_png_base64']))).convert('RGB'))
     for k,v in reader.component_suit.read(native).items():assert actual[k]==v
    if actual['template_confident']:
     counts['accepted']+=1;assert actual['suit_source']=='template_backup'
     if actual['suit']!=r['ground_truth'][1]:counts['wrong_accepted']+=1;wrong.append({'gt':r['ground_truth'],'pred':actual['suit'],'left':left,'right':right,'whole_edge':whole_edge,'score':actual['template_best_score']})
    else:counts['rejected']+=1;assert actual['card']=='UNRESOLVED'
 out=ROOT/'runs/smoke15_ram_port_20260930/short_roi_validation_regression.json';out.write_text(json.dumps({'counts':counts,'wrong':wrong},indent=2),'utf8')
 assert counts['cases']==3180
 assert counts['wrong_accepted']==0,counts

def test_normal_20_21_matches_historical_raw_and_direct_NCC(reader):
 for r,tile in manifest_tiles():
  w=tile.shape[1];row=np.zeros((16,140),dtype=np.uint8)
  for x in (12,12+w+3):row[:,x:x+w]=tile
  result=read(reader,row);first=result['tiles'][0];native_evidence(first,row)
  pred,best,second,_=predict(reader.bank,reader.labels,tile[:,:20],'raw_suit2');assert first['raw_suit2_suit']==(pred[1] if pred!='UNRESOLVED' else 'UNRESOLVED');assert first['best_distance']==best and first['runner_up_other_card_distance']==second
  direct=reader.suit_backup.read(np.repeat(tile[:,:,None],3,axis=2))
  preserved=first.get('ncc_original_query_result',first)
  for k,v in direct.items():assert preserved[k]==v
  if 'ncc_original_query_result' in first:
   assert not direct['template_confident'] and first['template_confident']
   assert first['template_suit']==r['ground_truth'][1]
  if 'ncc_query_png_base64' in first:
   query_raw=base64.b64decode(first['ncc_query_png_base64'])
   assert hashlib.sha256(query_raw).hexdigest()==first['ncc_query_sha256']
   query=np.asarray(Image.open(io.BytesIO(query_raw)).convert('RGB'))
   x1,y1,x2,y2=first['ncc_query_box_xyxy_exclusive']
   assert np.array_equal(query,np.repeat(row[y1:y2,x1:x2,None],3,axis=2))
   fixed=first.get('component_suit_original_result',first)
   for k,v in reader.suit_backup.read(query).items():assert fixed[k]==v
  else:assert 'ncc_query_box_xyxy_exclusive' not in first
  if 'component_suit_original_result' in first:
   native=np.asarray(Image.open(io.BytesIO(base64.b64decode(first['tile_png_base64']))).convert('RGB'))
   for k,v in reader.component_suit.read(native).items():assert first[k]==v

@pytest.mark.parametrize('components',[0,3])
def test_no_or_multiple_glyph_components_does_not_force_acceptance(reader,components):
 row=np.zeros((16,140),dtype=np.uint8)
 for x in (12,35):row[:,x:x+20]=240
 row[3,12:14]=150
 for x in (16,21,26)[:components]:row[5:9,x:x+2]=30
 first=read(reader,row)['tiles'][0];native_evidence(first,row)
 assert 'ncc_query_box_xyxy_exclusive' not in first
 assert first['card']=='UNRESOLVED'

def test_suit_query_out_of_bounds_does_not_force_acceptance(reader):
 row=np.zeros((16,140),dtype=np.uint8);row[:,0:20]=240;row[:,23:43]=240;row[3,0:2]=150
 row[5:9,3:5]=30;row[5:9,7:9]=30
 first=read(reader,row)['tiles'][0];assert 'ncc_query_box_xyxy_exclusive' not in first;assert first['card']=='UNRESOLVED'
