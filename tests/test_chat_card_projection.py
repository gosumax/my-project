import copy,hashlib
import pytest
from parser_core.visual_cards import project_chat_card_events,associate_visual_evidence

def fixture(kind='BOARD',hand=None):
 trigger={'table_id':'T01','session_id':'session','epoch':4,'frame_id':8,'row_index':1,'raw_ocr_text':'Dealer: Dealing Flop' if kind=='BOARD' else 'Dealer: alice shows','event_type':'UNKNOWN_EVENT'}
 physical={**trigger,'row_index':2,'raw_ocr_text':'cards','stable_row_id':'session:R2','row_pixel_sha256':'b'*64};sha='a'*64
 reading={'type':kind,'street':'FLOP' if kind=='BOARD' else None,'player':None if kind=='BOARD' else 'alice','crop_type':'CHAT_ROW_CARDS','identity_status':'CHAT_ROW_SHA_ROUNDTRIP_VERIFIED','crop_sha256':sha,'row_pixel_sha256':'b'*64,'row_id':'session:R2','table_id':1,'table_session_id':'session','frame_id':8,'source_span_index':2,'epoch':4,'row_key':'1:8:2','trigger_row_key':'1:8:1','status':'RAW_UNREVIEWED','raw_vl_response':{'request_id':'1','source_sha256':sha,'status':'OK'},'cards':['Ah','7c','2s'] if kind=='BOARD' else ['Ah','7c'],'hand_id':hand,'vl_raw_text':'Ah 7c 2s' if kind=='BOARD' else 'Ah 7c'}
 reading['card_evidence_id']='chatcard_'+hashlib.sha256(f'session:8:2:{sha}'.encode()).hexdigest()[:24];return [trigger,physical],reading
@pytest.mark.parametrize('kind',['BOARD','SHOWDOWN'])
@pytest.mark.parametrize('hand',[None,'123'])
def test_verified_chat_projection_preserves_partial_context(kind,hand):
 rows,r=fixture(kind,hand);assert project_chat_card_events(rows,{'1:8:2':r})==1;assert rows[0]['cards']==r['cards'];assert rows[0]['evidence_context_status']==('HAND_CONTEXT_AVAILABLE' if hand else 'PARTIAL_HAND_CONTEXT')
@pytest.mark.parametrize('field,value',[('table_id',2),('table_session_id','foreign'),('frame_id',9),('epoch',5),('row_id','foreign'),('row_pixel_sha256','c'*64),('crop_sha256','d'*64),('trigger_row_key','1:9:1'),('row_key','1:8:99')])
def test_foreign_chat_evidence_cannot_project(field,value):
 rows,r=fixture();r[field]=value;assert project_chat_card_events(rows,{'1:8:2':r})==0;assert 'cards' not in rows[0]
@pytest.mark.parametrize('changed_context',['table','session','frame','epoch'])
def test_association_does_not_borrow_trigger_context(changed_context):
 rows,r=fixture();field={'table':'table_id','session':'session_id','frame':'frame_id','epoch':'epoch'}[changed_context];rows[0][field]={'table':'T02','session':'other','frame':7,'epoch':3}[changed_context];associate_visual_evidence(rows,{'1:8:2':r});assert r['status']=='UNRESOLVED';assert project_chat_card_events(rows,{'1:8:2':r})==0

def test_projection_rejects_foreign_trigger_semantics():
 rows,r=fixture();rows[0]['raw_ocr_text']='Dealer: alice folds';assert project_chat_card_events(rows,{'1:8:2':r})==0


def test_verified_chat_suits_do_not_come_from_vl_full_cards():
 rows,r=fixture();r['vl_status']='OK';r['vl_raw_text']='As 7d 2h';r['hybrid']={'cards':r['cards']};associate_visual_evidence(rows,{'1:8:2':r});assert r['cards']==['Ah','7c','2s'];assert r['status']=='RAW_UNREVIEWED';assert project_chat_card_events(rows,{'1:8:2':r})==1

def test_verified_chat_missing_hybrid_cannot_fill_suits_from_vl():
 rows,r=fixture();r['cards']=None;r['vl_status']='OK';r['vl_raw_text']='Ah 7c 2s';r['status']='UNRESOLVED';associate_visual_evidence(rows,{'1:8:2':r});assert r.get('cards') is None;assert r['status']=='UNRESOLVED';assert project_chat_card_events(rows,{'1:8:2':r})==0

def test_new_hand_marker_without_number_clears_previous_context():
 rows,r=fixture();old={**rows[0],'frame_id':6,'row_index':1,'raw_ocr_text':'#123','event_type':'HAND_ID'};new={**rows[0],'frame_id':7,'row_index':1,'raw_ocr_text':'Dealer: new hand','event_type':'NEW_HAND'};timeline=[old,new]+rows
 associate_visual_evidence(timeline,{'1:8:2':r});assert r['hand_id'] is None;assert r['continuity_status']=='PARTIAL_HAND_CONTEXT';assert project_chat_card_events(timeline,{'1:8:2':r})==1;assert rows[0]['evidence_context_status']=='PARTIAL_HAND_CONTEXT'

def test_repeated_hand_number_in_new_epoch_does_not_borrow_flop():
 oldrows,old=fixture();newrows,new=fixture();
 for item in oldrows:item.update(frame_id=2,epoch=3)
 old.update(frame_id=2,epoch=3,row_key='1:2:2',trigger_row_key='1:2:1')
 old['card_evidence_id']='chatcard_'+hashlib.sha256(f"session:2:2:{old['crop_sha256']}".encode()).hexdigest()[:24]
 newrows[0]['raw_ocr_text']='Dealer: Dealing Turn';new.update(street='TURN',cards=['Ah','7c','2s','Kd'])
 hand1={**oldrows[0],'frame_id':1,'raw_ocr_text':'#123','event_type':'HAND_ID'};hand2={**newrows[0],'frame_id':7,'raw_ocr_text':'#123','event_type':'HAND_ID'}
 timeline=[hand1]+oldrows+[hand2]+newrows;associate_visual_evidence(timeline,{'1:2:2':old,'1:8:2':new});assert old['status']=='RAW_UNREVIEWED';assert new['hand_id']=='123';assert new['continuity_status']=='PREVIOUS_STREET_NOT_AVAILABLE'

@pytest.mark.parametrize('which',['trigger','physical'])
@pytest.mark.parametrize('field,value',[('origin','UNRESOLVED_IDENTITY'),('status','UNRESOLVED')])
def test_unresolved_identity_is_never_overwritten_by_projection(which,field,value):
 rows,r=fixture();rows[0 if which=='trigger' else 1][field]=value;assert project_chat_card_events(rows,{'1:8:2':r})==0;assert 'cards' not in rows[0]

def test_unresolved_physical_row_cannot_associate_reading():
 rows,r=fixture();rows[1]['origin']='UNRESOLVED_IDENTITY';associate_visual_evidence(rows,{'1:8:2':r});assert r['status']=='UNRESOLVED';assert r['reason']=='ROW_IDENTITY_UNRESOLVED'


def test_supplemental_show_row_lineage_stays_unknown_after_pixel_projection():
 rows,r=fixture('SHOWDOWN');rows[0].update(row_identity_status='UNKNOWN',admission_ref='session:8:physical_new:1')
 assert project_chat_card_events(rows,{'1:8:2':r})==1
 assert rows[0]['row_identity_status']=='UNKNOWN'
 assert rows[0]['identity_status']=='CHAT_ROW_SHA_ROUNDTRIP_VERIFIED'
