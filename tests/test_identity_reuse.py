import copy,json,pytest,hashlib
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/"production_manifest.json").is_file())
from parser_core import chat_nick_actor as p

def inputs(anchor=100):
 c=dict(table_id=1,session_id='s',hand_id='100',header_frame=0,capture_end_inclusive=200)
 c[p.SOURCE_KEY]=dict(policy=p.RULE,cross_source_policy=p.CROSS_RULE,source_sha256={'native':'sealed'},context_checkpoints=[dict(frame=0,table_id=1,session_id='s',hand_id='100')],conflict_inventory=dict(state_receipts=[],unresolved_state_events=[]))
 q=dict(status='PROVEN',rule=p.CROSS_RULE,table_id=1,session_id='s',hand_id='100',actor='actor',seat='seat2',anchor_frame=anchor,source_files_sha256={'native':'sealed'})
 return c,q
def bind(c,q,frames=None,low=0,high=150):return p.cross_identity_frame_binding(c,q,frames or [70,71,72],start_frame=low,end_frame=high)

def test_selector_without_nick_text_or_score():
 c,q=inputs();before=copy.deepcopy((c,q));r=bind(c,q)
 assert r['seat']=='seat2' and r['selected_frames']==[70,71,72]
 assert r['nickname_ocr_rechecked'] is False
 assert not r['numeric_authority'] and not r['status_authority']
 assert (c,q)==before
 assert bind(*json.loads(json.dumps([c,q])))==r

@pytest.mark.parametrize('checkpoint',[None,'foreign'])
def test_h_none_h_and_foreign_reject(checkpoint):
 c,q=inputs();c[p.SOURCE_KEY]['context_checkpoints'] += [dict(frame=f,table_id=1,session_id='s',hand_id=h) for f,h in [(80,checkpoint),(90,'100')]]
 with pytest.raises(ValueError,match='HAND_INTERVAL_BARRIER'):bind(c,q)

@pytest.mark.parametrize('frame',[80,110,149])
@pytest.mark.parametrize('kind',['EMPTY','REPLACE','SIT_IN'])
def test_target_transition_before_after_later_anchor_beyond64(frame,kind):
 c,q=inputs();c[p.SOURCE_KEY]['conflict_inventory']['state_receipts']=[dict(frame=frame,seat='seat2',hand_id='100',kind=kind,status='APPLIED',evidence_status='PROVEN',actor='actor')]
 with pytest.raises(ValueError,match='SEAT_INTERVAL_CONFLICT'):bind(c,q)

@pytest.mark.parametrize('mutation',['actor','otherseat','foreignhand','unresolved'])
def test_observed_conflicts(mutation):
 c,q=inputs();r=dict(frame=85,seat='seat2',hand_id='100',kind='OCCUPIED',status='APPLIED',evidence_status='PROVEN',actor='actor')
 if mutation=='actor':r['actor']='different'
 elif mutation=='otherseat':r['seat']='seat3'
 elif mutation=='foreignhand':r['hand_id']='foreign'
 else:r.update(kind='REPLACE',status='PENDING',evidence_status='UNKNOWN')
 key='unresolved_state_events' if mutation=='unresolved' else 'state_receipts';c[p.SOURCE_KEY]['conflict_inventory'][key].append(r)
 with pytest.raises(ValueError,match='CONFLICT|UNRESOLVED'):bind(c,q)

@pytest.mark.parametrize('mutation',['late','shortend','foreignseatcontext','clock','sources','notcross','nonconsecutive'])
def test_interval_and_policy_guards(mutation):
 c,q=inputs();frames=[70,71,72];high=150
 if mutation=='late':q['anchor_frame']=151
 elif mutation=='shortend':c['capture_end_inclusive']=64
 elif mutation=='foreignseatcontext':q['session_id']='foreign'
 elif mutation=='clock':c[p.SOURCE_KEY]['context_checkpoints'][0]['table_id']=2
 elif mutation=='sources':q['source_files_sha256']={}
 elif mutation=='notcross':q['rule']=p.RULE
 else:frames=[70,72,73]
 with pytest.raises(ValueError):bind(c,q,frames,high=high)

def test_transition_after_interval_does_not_invent_conflict():
 c,q=inputs();c[p.SOURCE_KEY]['conflict_inventory']['state_receipts']=[dict(frame=151,seat='seat2',hand_id='foreign',kind='REPLACE',status='APPLIED',evidence_status='PROVEN',actor='different')]
 assert bind(c,q)['valid_interval']==[0,150]

def test_first_triple_is_frame_only():
 assert p.first_consecutive_frames([100,70,71,72,101,102],'missing')==[70,71,72]
 with pytest.raises(ValueError,match='missing'):p.first_consecutive_frames([70,72,73],'missing')

def test_legacy_joint_positive_proofs_exact_to_v4():
 import importlib.util
 from pathlib import Path
 from parser_core import joint_start_stack as new
 root=ROOT;oldpath=Path(__file__).resolve().parent/'fixtures/identity_reuse_v4_joint_start_stack.py'
 assert hashlib.sha256(oldpath.read_bytes()).hexdigest()=='8744be2091cf4a9204875f2c2639148a04a2cc1b9beb70c977b05f88b5abdd7d'
 spec=importlib.util.spec_from_file_location('parser_core._old_joint_control',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.ROOT=root
 src=json.loads((root/'tests/fixtures/joint_stack_native_sources.json').read_text('utf8'))
 for actor in ('kalmar_204','vasyajin'):
  case=next(c for c in src['cases'] if c['actor']==actor);a=old.prove(case,src['database_metadata_raw']);b=new.prove(case,src['database_metadata_raw'])
  assert a['status']=='PROVEN' and a==b
