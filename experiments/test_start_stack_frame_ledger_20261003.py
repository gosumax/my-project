from start_stack_frame_ledger_20261003 import observed_ledger, unresolved_prefix, validate_supplement
import pytest

def event(frame,kind,amount,player='p'):
    return {'frame_id':frame,'action':kind,'amount':amount,'player':player,'source_row_ids':[str(frame)]}

def test_raise_delta_street_reset_return_and_strict_causal_cutoff():
    rows=[event(1,'ANTE','10'),event(2,'BIG_BLIND','20'),event(3,'RAISE','100'),
          event(4,'FLOP',None,None),event(5,'BET','50'),event(6,'UNCALLED_RETURN','30')]
    before=observed_ledger(rows,'p',6)
    assert (before['gross'],before['returns'],before['net'])==('160','0','160')
    after=observed_ledger(rows,'p',7)
    assert (after['gross'],after['returns'],after['net'])==('160','30','130')
    assert after['trace'][2]['delta']=='80'

def test_unbound_betting_actor_and_unknown_row_are_blockers():
    rows=[event(2,'CALL','20','p_ocr_variant'),event(3,'UNKNOWN_EVENT',None,None)]
    assert unresolved_prefix(rows,{'p'},2)==[]
    assert unresolved_prefix(rows,{'p'},3)==['UNBOUND_MONETARY_ACTOR']
    assert unresolved_prefix(rows,{'p'},4)==['UNBOUND_MONETARY_ACTOR','UNPARSED_CHAT_ROW_LEDGER_COMPLETENESS_UNKNOWN']

def test_supplement_must_match_visible_text_and_reviewed_frame_hash():
    rv={'seats':{'seat1':{'stack_frame':2,'stack_table_sha256':'abc'}}}
    w={'start_frame':1,'end_frame':5}
    s={'raw_visible_text':'Dealer: p posts Ante 10','player':'p','event_type':'ANTE','amount':'10','frame_index':2,'table_sha256':'abc'}
    validate_supplement(s,rv,w)
    with pytest.raises(ValueError): validate_supplement(dict(s,amount='11'),rv,w)
    with pytest.raises(ValueError): validate_supplement(dict(s,table_sha256='other'),rv,w)

def test_repeated_ante_and_post_payout_scope_remain_explicit():
    l=observed_ledger([event(1,'ANTE','10'),event(2,'ANTE','10'),event(3,'WIN','100')],'p',4)
    assert l['net']=='20'
    assert l['payouts']=='100'
    assert l['issues']==['PAYOUT_APPLIED_FORMULA_OUT_OF_SCOPE','REPEATED_FORCED_POST_UNRESOLVED']
