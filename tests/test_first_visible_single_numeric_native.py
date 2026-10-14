import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from parser_core import direct_start_stack as direct,first_visible_single_numeric as single
from parser_core.adjacent_stack_carry import SavedCarrySources
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'production_manifest.json').exists());BASE=ROOT/'runs/hh_all_hands_20261009/practical_actor_policy/single_first_numeric_production_validation'
SRC=Path('C:/Users/Admin/.codex/tmp/hh95_source_cache_20261008/full_v1')
@pytest.fixture(scope='module')
def native():
 records=[json.loads(x) for x in (ROOT/'runs/hh_all_hands_20261009/current_full_replay_v2/logical_hand_analysis.jsonl').read_text('utf8').splitlines() if x.strip()]
 rows=json.loads((ROOT/'runs/hh_all_hands_20261009/current_full_replay_v4_cross_source/joint_start_stack_audit.json').read_text('utf8'))['rows']
 wanted={(3,'2878366511','vasyajin'),(6,'2878365993','bashershadow')}
 rows=[r for r in rows if (r['table_id'],r['hand_id'],r['actor']) in wanted];assert len(rows)==2
 keys={(r['table_id'],r['session_id'],r['hand_id']) for r in rows}
 selected=[r for r in records if direct.native_key(r['observed_hand']) in keys]
 registry=SavedCarrySources(SRC,records,[])
 provider=direct.DirectStartStackProvider(SRC,selected,SimpleNamespace(rows=rows),registry,actor_policy=True,single_first_numeric=True)
 out=BASE/'measurement/focused_native';out.mkdir(parents=True,exist_ok=True);provider.write_audit(out)
 print('NATIVE_FOCUSED_AUDIT '+json.dumps(provider.audit),flush=True)
 return provider,selected
def native_case(native,hand='2878366511'):
 provider,records=native;k=next(k for k in provider.inventory.contexts if k[2]==hand);actor,seat=('vasyajin','seat7') if hand=='2878366511' else ('bashershadow','seat6');return provider,provider.inventory.case_for(k,actor,seat)
def test_actual_first_12350_is_one_source_proven(native):
 provider,c=native_case(native);p=provider.proofs[(direct.native_key(c),c['actor'])]
 assert p['status']=='PROVEN' and single.is_single_proof(p)
 assert p['numeric_frames']==[4704] and p['displayed_stack']==12350
 assert p['sql_numeric_authority'] is False
 assert any(s['raw_text']=='12350' and abs(s['confidence']-.999996066)<1e-7 for s in p['native_numeric_sources'])
def test_raw_934602_sql_remains_unaccepted(native):
 provider,c=native_case(native,'2878365993');s=[s for s in c['raw_sql_visual_states'] if s['frame_index']==2256 and s['ocr_raw'].replace(' ','')=='168531'];assert len(s)==1
 assert abs(s[0]['ocr_confidence']-.934602)<1e-6 and s[0]['ocr_numeric'] is None
 sql,*_=direct.numeric_sources(c,c['database_metadata']);assert not any(str(r.get('raw_text')).replace(' ','')=='168531' and r.get('ocr_status')=='OK' for r in sql.values())
 readings=direct.full_readings(c);_,_,obs,states=direct.numeric_sources(c,c['database_metadata']);fresh,sources=single.original_numeric_readings(c,readings,obs,states)
 frames=sorted(f for f,r in fresh.items() if single.single_first_numeric(r) is not None);assert frames[0]==2256 and single.single_first_numeric(fresh[2256])=='168531'
 rawsql=[r for r in sources[2256] if r['kind']=='NATIVE_CROP_BOUND_RAW_OCR_IN_SQL_CONTAINER'];assert len(rawsql)==1
 assert rawsql[0]['raw_text'].replace(' ','')=='168531' and abs(rawsql[0]['confidence']-.9346024394)<1e-9
 assert rawsql[0]['original_sql_status']=='UNKNOWN' and rawsql[0]['original_sql_numeric'] is None and rawsql[0]['sql_numeric_authority'] is False
 row=next(r for r in provider.audit if r['hand_key'][2]=='2878365993');assert row['status']=='PROVEN' and row['reason'] is None
 proof=provider.proofs[(direct.native_key(c),c['actor'])]
 assert proof['first_visible_proof']['contract']==single.SAVED_SINGLE_CONTRACT
 assert proof['displayed_stack']==168531 and proof['numeric_frames']==[2256]
def test_actual_positive_consumer_accepts_json_roundtrip(native):
 original,records=native;provider=copy.copy(original);provider.proofs=json.loads(json.dumps(list(original.proofs.values())))
 provider.proofs={(direct.native_key(p),p['actor']):p for p in provider.proofs}
 hand=next(r['observed_hand'] for r in records if r['observed_hand']['hand_id']=='2878366511');actual=provider.for_hand(hand)
 expected=[p for (k,a),p in original.proofs.items() if k==direct.native_key(hand)]
 assert actual and actual==expected and json.loads(json.dumps(actual))==actual
 assert actual[0]['calculated_start_stack']==13020 and actual[0]['numeric_frames']==[4704]
@pytest.mark.parametrize('mutation',['wrong_amount','seat','hand','crop','not_first','earlier_unexplained','source_conflict'])
def test_real_inventory_rejects_altered_native_case(native,mutation):
 provider,c=native_case(native);c=copy.deepcopy(c)
 if mutation=='wrong_amount':
  row=next(r for r in c['raw_sql_visual_states'] if r['frame_index']==4704);row['ocr_raw']='99999'
 elif mutation=='seat':c['seat']='seat6'
 elif mutation=='hand':c['hand_id']='wronghand'
 elif mutation=='crop':c['raw_stack_captures'][0]['raw_record']['pixel_sha256']='0'*64
 elif mutation=='not_first':c['raw_stack_captures']=[r for r in c['raw_stack_captures'] if r['raw_record']['source_frame']!=4704]
 elif mutation=='earlier_unexplained':c['raw_stack_captures'][0]['raw_record']['source_frame']-=1
 else:c['raw_sql_visual_states'][0]['raw_gray_sha256']='0'*64
 p=single.prove(c,provider.inventory,lambda:None);assert p['status']=='UNKNOWN' and p['reason']=='FRESH_COMPLETE_WINDOW_INVENTORY_REQUIRED'
@pytest.mark.parametrize('mutation',['wrong_amount','seat','hand','crop'])
def test_current_consumer_rejects_altered_saved_proof(native,mutation):
 original,records=native;provider=copy.copy(original);provider.proofs=copy.deepcopy(original.proofs)
 key=next(k for k in provider.proofs if k[0][2]=='2878366511');p=provider.proofs[key]
 if mutation=='wrong_amount':p['calculated_start_stack']+=1
 elif mutation=='seat':p['seat']='seat6'
 elif mutation=='hand':p['hand_id']='wronghand'
 else:p['native_numeric_sources'][0]['source_rgb_sha256']='0'*64
 hand=next(r['observed_hand'] for r in records if r['observed_hand']['hand_id']=='2878366511');assert provider.for_hand(hand)==[]
