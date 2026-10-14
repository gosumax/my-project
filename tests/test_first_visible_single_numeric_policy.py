import copy,json,math,sys
from pathlib import Path
import pytest
from parser_core.first_stack_ocr import single_first_numeric,single_first_numeric_policy
from parser_core.start_stack_visibility import first_visible_proof,verify_first_visible_fact,SINGLE_CONTRACT
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'production_manifest.json').exists())
sys.path.insert(0,str(ROOT/'tests'))
from test_direct_numeric_policy import native_context,attach_bound_scale

@pytest.mark.parametrize('score',[.93,.929999,True,False,None,'0.99',float('nan'),float('inf'),-.1,1.01])
def test_invalid_or_boundary_score(score):
 assert single_first_numeric(dict(raw_text='12350',confidence=score,ocr_status='OK')) is None
@pytest.mark.parametrize('score',[.930001,.934602,.999996066,1.0])
def test_strictly_above_threshold(score):
 assert single_first_numeric(dict(raw_text='12350',confidence=score,ocr_status='OK'))=='12350'
@pytest.mark.parametrize('literal',['-100','+100','12 chips','Sit out','1 00','PENSEAT',''])
def test_literal_rejected(literal):
 assert single_first_numeric(dict(raw_text=literal,confidence=.999,ocr_status='OK')) is None
def test_bound_scale_cannot_uplift_primary():
 r=native_context()['readings'][-1];r['confidence']=.93;r['ocr_response']['score']=.93;attach_bound_scale(r,r['raw_text'],.999)
 assert single_first_numeric(r) is None
def test_bound_scale_disagreement_withheld():
 r=native_context()['readings'][-1];attach_bound_scale(r,'999999',.999)
 assert single_first_numeric(r) is None
def test_named_single_cannot_enter_old_generic_fact():
 c=native_context();readings={r['source_frame']:r for r in c['readings']};p=first_visible_proof(readings,[284],start_frame=258,confidence=.93,single_first_policy=single_first_numeric_policy())
 assert p['status']=='PROVEN' and p['contract']==SINGLE_CONTRACT
 f=copy.deepcopy(c['fact']);f['first_visible_proof']=p
 assert not verify_first_visible_fact(f)

import copy,json
from pathlib import Path
import pytest
from parser_core.first_stack_ocr import single_first_numeric_policy
from parser_core.start_stack_visibility import first_visible_proof
from parser_core import direct_start_stack as direct,first_visible_single_numeric as single

def checked(readings,frame=284):
 return first_visible_proof(readings,[frame],start_frame=258,confidence=.93,single_first_policy=single_first_numeric_policy())
@pytest.mark.parametrize('mutation,reason',[('notfirst','EARLIER_NUMERIC_OR_AMBIGUOUS_SOURCE'),('earlier_unexplained','UNEXPLAINED_PRE_NUMERIC_PIXELS'),('seat','FIRST_VISIBLE_SOURCE_CONTEXT_CONFLICT'),('hand','FIRST_VISIBLE_SOURCE_CONTEXT_CONFLICT'),('session','FIRST_VISIBLE_SOURCE_CONTEXT_CONFLICT'),('frame','FIRST_VISIBLE_SOURCE_CONTEXT_CONFLICT'),('crop','FIRST_VISIBLE_SOURCE_PIXELS_UNPROVEN')])
def test_single_intrinsic_native_pixel_firstness_and_context(mutation,reason):
 c=native_context();readings={r['source_frame']:r for r in c['readings']}
 assert checked(readings)['status']=='PROVEN'
 frame=284
 if mutation in ('notfirst','earlier_unexplained'):
  frame=285
  if mutation=='earlier_unexplained':readings[284]['confidence']=.93;readings[284]['ocr_response']['score']=.93
 elif mutation in ('seat','hand','session'):
  key={'seat':'seat','hand':'hand_id','session':'session_id'}[mutation];readings[280][key]='FOREIGN'
 elif mutation=='frame':readings[280]['source_frame']=279
 else:readings[280]['pixel_sha256']='0'*64
 p=checked(readings,frame);assert p['status']=='UNKNOWN' and p['reason']==reason

def test_native_sameframe_raw_sql_direct_conflict_intrinsic():
 # Function-level contradiction control. This cannot grant a proof or replace FreshInventory.
 fixture=Path(__file__).parent/'fixtures/first_single_native_proof.jsonl';assert __import__('hashlib').sha256(fixture.read_bytes()).hexdigest()=='ed58cb6154ea95bb6dc677de40571d01615cc7c560712a0e4672767fbd81cc29';p=json.loads(fixture.read_text('utf8').splitlines()[0]);c=p['source_case'];readings=direct.full_readings(c);_,_,obs,states=direct.numeric_sources(c,c['database_metadata'])
 actual,sources=single.original_numeric_readings(c,readings,obs,states);assert sources[4704][0]['raw_text']=='12350'
 changed=copy.deepcopy(readings);row=changed[4704];row.update(raw_text='12351',confidence=.999,ocr_status='OK',ocr_response={'item_id':'3:4704:6'})
 with pytest.raises(ValueError,match='SINGLE_NATIVE_OCR_VALUE_CONFLICT'):single.original_numeric_readings(c,changed,obs,states)
