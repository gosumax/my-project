import copy,json,pytest
from pathlib import Path
from parser_core import joint_start_stack as joint,sitout_hand_evidence as sitout,chat_nick_actor as actor
ROOT=next(p for p in Path(__file__).resolve().parents if (p/"production_manifest.json").is_file())
BASE=Path(__file__).resolve().parent
PROOFS=json.loads((BASE/'fixtures/identity_reuse_native_cross_joint_proofs.json').read_text('utf8'))
CROSS=[p for p in PROOFS if p['identity_proof']['rule']==actor.CROSS_RULE]
assert len(CROSS)==4

@pytest.mark.parametrize('proof',CROSS,ids=lambda p:f"T{p['table_id']}_{p['hand_id']}_{p['actor']}")
def test_genuine_cross_positive_reproduces_exact(proof):
 assert joint.prove(proof['source_case'],proof['database_metadata'])==proof

@pytest.mark.parametrize('proof',CROSS,ids=lambda p:f"T{p['table_id']}_{p['hand_id']}_{p['actor']}")
@pytest.mark.parametrize('mutation',['missing_capture','foreign_seat','missing_sql','foreign_numeric'])
def test_genuine_cross_who_cannot_prove_missing_or_foreign_money(proof,mutation):
 c=copy.deepcopy(proof['source_case']);identity=actor.prove_actor_point(c['source_context'],c['actor'])
 assert identity==proof['identity_proof'] and identity['status']=='PROVEN'
 if mutation=='missing_capture':c['raw_stack_captures']=[]
 elif mutation=='foreign_seat':
  for row in c['raw_stack_captures']:row['raw_record']['seat']='seat9'
 elif mutation=='missing_sql':c['raw_sql_observations']=[];c['raw_sql_visual_states']=[]
 else:
  for row in c['raw_sql_visual_states']:
   if row['status']=='CONFIRMED_NUMERIC':row['ocr_numeric']='999999999'
 assert actor.prove_actor_point(c['source_context'],c['actor'])==identity
 result=joint.prove(c,proof['database_metadata']);assert result['status']=='UNKNOWN',result
 assert result.get('calculated_start_stack') is None

@pytest.mark.parametrize('proof',CROSS,ids=lambda p:f"T{p['table_id']}_{p['hand_id']}_{p['actor']}")
def test_genuine_cross_who_cannot_prove_missing_native_status(proof):
 c=copy.deepcopy(proof['source_case']);point=actor.prove_actor_point(c['source_context'],c['actor']);assert point['rule']==actor.CROSS_RULE
 c['raw_stack_captures']=[]
 with pytest.raises(ValueError,match='FIRST_SOURCE_SIT_OUT_TRIPLE_NOT_PROVEN'):
  sitout.initial_status(c,point,None,interval=[c['source_context']['header_frame'],c['source_context']['capture_end_inclusive']])
