from start_stack_wide_20261003 import classify,numeric,uncertain_row_identity,first_search_complete
from start_stack_frame_ledger_20261003 import observed_ledger

def record(value,gross,blockers=()):
    return {'candidate_start_stack':value,'gross_committed_before_frame':gross,
            'returns_before_frame':'0','payouts_before_frame':'0','blockers':list(blockers)}

def test_different_commitments_required_for_multi_state():
    assert classify([record('100','10'),record('100','10')])=='SAME_STATE_REPEAT'
    assert classify([record('100','10'),record('100','30')])=='MULTI_STATE_EXACT'
    assert classify([record('100','10'),record('101','30')])=='CONFLICT'

def test_unknown_later_ledger_does_not_create_multi_state():
    assert classify([record('100','10'),record('100','30',['LEDGER_UNKNOWN'])])=='LEDGER_UNKNOWN'
    assert classify([record('100','10',['TIMING_UNKNOWN']),record('100','30',['TIMING_UNKNOWN'])])=='TIMING_UNKNOWN'
    assert classify([record('100','10',['IDENTITY_UNKNOWN'])])=='IDENTITY_UNKNOWN'

def test_overlay_low_confidence_and_numeric_grammar():
    assert numeric({'ocr':{'raw_text':'CALL','score':.99}}) is None
    assert numeric({'ocr':{'raw_text':'12 345','score':.899}}) is None
    assert numeric({'ocr':{'raw_text':'12 345','score':.90}})=='12345'

def test_future_stability_frames_never_supply_future_return_or_payout():
    def e(f,k,n):return {'frame_id':f,'action':k,'amount':n,'player':'p','source_row_ids':[str(f)]}
    ms=[e(1,'ANTE','10'),e(2,'BIG_BLIND','20'),e(3,'RAISE','100'),e(10,'UNCALLED_RETURN','30'),e(11,'WIN','200')]
    own=observed_ledger(ms,'p',10)
    assert (own['gross'],own['returns'],own['payouts'])==('110','0','0')
    later=observed_ledger(ms,'p',12)
    assert 270+int(later['gross'])-int(later['returns'])-int(later['payouts'])==150

def test_opponent_money_and_street_source_uncertainty_cannot_be_silent():
    for kind,player in [('CALL','opponent'),('FLOP',None)]:
        m={'action':kind,'player':player,'source_rows':[{'message_reason':'ROW_IDENTITY_UNRESOLVED'}]}
        assert uncertain_row_identity(m)

def test_first_numeric_triple_beyond_exhaustive_interval_is_unusable():
    assert first_search_complete(728,729)
    assert first_search_complete(728,741)
    assert not first_search_complete(728,742)
