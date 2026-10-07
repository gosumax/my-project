from experiments.text_first_identity_audit import evaluate, content_pass, digest, readable_text

def fixture():
    pair={'pair_id':'a->b','decision_frame':2,'old':{'track_id':'a','normalized_text':'Dealer: person calls 10','table_id':'T','session_id':'S','last_frame':1},'new':{'track_id':'b','normalized_text':'Dealer: person calls 10','table_id':'T','session_id':'S','first_frame':2},'features':{'delta_frame':1,'causal_visibility_overlap':False,'hand_boundary_frames':[]},'motion':{'aligned':True,'local_loss_bridge':False,'target_new_index':None}}
    texts={'a':{'frame_id':0,'normalized_text':'Dealer: person calls 10','raw_ocr_text':'Dealer: person calls 10'},'b':{'frame_id':2,'normalized_text':'Dealer: person calls 10','raw_ocr_text':'Dealer: person calls 10'}}
    n={'neighbors':[{'old_id':'x','new_id':'x','center_residual':0},{'old_id':'y','new_id':'y','center_residual':1}]}
    return pair,texts,n

def test_unique_geometry_with_neighbors_support():
    p,t,n=fixture();r=evaluate(p,['a'],['b'],['a'],t,n,{'b':1},True)
    assert r['simulation_verdict']=='TEXT_FIRST_SAME_CANDIDATE'

def test_duplicate_viewport_blocks_even_with_support():
    p,t,n=fixture();t['c']=dict(t['a']);r=evaluate(p,['a'],['b','c'],['a'],t,n,{'b':1,'c':30},True)
    assert r['text_class']=='TEXT_DUPLICATED_IN_VIEWPORT' and r['simulation_verdict']=='AMBIGUOUS'

def test_future_ocr_is_unknown_not_unique():
    p,t,n=fixture();t['c']={'frame_id':3,'normalized_text':'different'};r=evaluate(p,['a'],['b','c'],[],t,n,{'b':1},True)
    assert r['unknown_viewport_text_ids']==['c'] and r['simulation_verdict']=='AMBIGUOUS'

def test_previous_duplicate_and_missing_neighbor_block():
    p,t,n=fixture();t['c']=dict(t['a']);n['neighbors']=n['neighbors'][:1]
    r=evaluate(p,['a'],['b'],['c'],t,n,{'b':1},True)
    assert r['text_class']=='TEXT_DUPLICATED_IN_TIME_WINDOW' and not r['conditions']['two_consistent_neighbors']

def test_no_exact_pixel_requirement_but_content_required():
    p,t,n=fixture();r=evaluate(p,['a'],['b'],[],t,n,{'b':1},False)
    assert r['mapping']=='UNIQUE_TEXT_GEOMETRY_MAPPING' and r['blocking_reasons']==['SUPPORTING_CONTENT']

def test_gap_and_boundary_are_not_overridden_by_content():
    p,t,n=fixture();p['features']['delta_frame']=3;p['features']['hand_boundary_frames']=[2]
    r=evaluate(p,['a'],['b'],[],t,n,{'b':1},True)
    assert r['simulation_verdict']=='AMBIGUOUS'
    assert {'IMMEDIATE_CONTINUITY','NO_BOUNDARY'}<=set(r['blocking_reasons'])

def test_unproven_alignment_cannot_claim_unique_mapping():
    p,t,n=fixture();p['motion']['aligned']=False
    r=evaluate(p,['a'],['b'],[],t,n,{'b':0},True)
    assert r['mapping']=='NO_TEXT_GEOMETRY_MAPPING' and not r['mapping_context_proven']

def test_content_enum_requires_explicit_pass():
    for value in ('FAIL','UNKNOWN',False,None):
        assert not content_pass({'within_causal_neighbor_control_envelope':value})
    assert content_pass({'within_causal_neighbor_control_envelope':'PASS'})

def test_content_hash_changes_with_source_bytes(tmp_path):
    p=tmp_path/'content.jsonl';p.write_text('one');first=digest(p);p.write_text('two')
    assert digest(p)!=first

def test_empty_competing_ocr_prevents_uniqueness():
    p,t,n=fixture();t['c']={'frame_id':1,'normalized_text':'','raw_ocr_text':''}
    r=evaluate(p,['a'],['b','c'],[],t,n,{'b':1,'c':0},True)
    assert r['unknown_viewport_text_ids']==['c'] and not r['conditions']['text_unique']

def test_readable_requires_raw_and_normalized_source_fields():
    assert not readable_text({'frame_id':1,'normalized_text':'known'},2)
    assert not readable_text({'frame_id':1,'raw_ocr_text':'known'},2)
    assert readable_text({'frame_id':1,'raw_ocr_text':'known','normalized_text':'known','confidence':0.01},2)
