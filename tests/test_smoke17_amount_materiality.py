from experiments import report_smoke17_amount_materiality as audit


def message(action,player=None,amount=None,raw=None,all_in=False,epoch=0):
    return dict(action=action,player=player,amount=amount,raw_text=raw or action,
                all_in=all_in,cards=['4c','2c','4h'] if action=='FLOP' else [],
                source_row_ids=[action],source_rows=[dict(epoch=epoch,origin='NEW_CANDIDATE',status='OK')])


def hand(messages,**kw):
    return dict(table_id='T06',session_id='test',hand_id='2876718617',
                start_observed=True,end_observed=True,boundary_conflicts=[],messages=messages,**kw)


def test_t06_zero_prior_both_actions_identical_with_return():
    rows=[message('NEW_HAND')]+[message('ANTE',str(p),600) for p in range(7)]
    rows += [message('SMALL_BLIND','sb',2500),message('BIG_BLIND','bb',5000),
             message('RAISE','kohb_blr',10750,'kohb_blr raises to 10750'),
             message('CALL','billmurray',10750),message('DEAL'),message('FLOP'),
             message('CHECK','kohb_blr'),message('BET','billmurray',10956),
             message('FOLD','kohb_blr'),message('UNCALLED_RETURN','billmurray',10956),
             message('WIN','billmurray',33200)]
    result=audit.analyze(hand(rows),'test','test')
    assert result['class_']=='A'
    assert result['equivalent_global_full_histories']
    snap=result['global_ADDED']['snapshot']
    assert snap['totals']['nominal_net']==33200
    assert snap['totals']['nominal_remaining_pot']==0
    assert snap['refunds']['billmurray']==10956


def test_deal_does_not_reset_prior_contribution():
    h=hand([message('NEW_HAND'),message('SMALL_BLIND','p',100),
            message('BIG_BLIND','q',200),message('DEAL'),message('CALL','p',200)])
    added=audit.run_arm(h,'ADDED');total=audit.run_arm(h,'TOTAL')
    assert added['own']['p']==300
    assert total['own']['p']==200
    assert added['hash']!=total['hash']


def test_final_cancellation_does_not_prove_equal_histories():
    h=hand([message('NEW_HAND'),message('SMALL_BLIND','p',100),
            message('BIG_BLIND','q',200),message('CALL','p',200),
            message('RAISE','p',500,'p raises to 500')])
    r=audit.analyze(h,'test','test')
    assert r['equivalent_global_final_totals']
    assert not r['equivalent_global_full_histories']
    assert r['class_']!='A'


def test_known_target_can_resolve_material_call_branch():
    h=hand([message('NEW_HAND'),message('SMALL_BLIND','p',100),
            message('BIG_BLIND','q',200),message('RAISE','q',300,'q raises to 300'),
            message('CALL','p',300),message('FOLD','q'),message('WIN','p',600)])
    r=audit.analyze(h,'test','test')
    assert r['class_']=='B'
    assert r['viable_branches']==1


def test_unknown_prior_not_zero():
    h=hand([message('NEW_HAND'),message('BET','p',100),message('WIN','p',100)])
    h['start_observed']=False
    r=audit.analyze(h,'test','test')
    assert r['class_']=='C'
    assert r['global_TOTAL']['trace'][1]['prior_contribution']=='UNKNOWN'
    assert r['global_TOTAL']['trace'][1]['derived_increment']=='UNKNOWN'


def test_unknown_identity_and_epoch_prevent_proof():
    for variant in ('identity','epoch','boundary'):
        rows=[message('NEW_HAND'),message('BET','p',100),message('WIN','p',100)]
        h=hand(rows)
        if variant=='identity':rows[1]['source_rows'][0]['origin']='UNRESOLVED_IDENTITY'
        if variant=='epoch':rows[1]['source_rows'][0]['epoch']=1
        if variant=='boundary':h['boundary_conflicts']=['conflict']
        assert audit.analyze(h,'test','test')['class_']=='C'


def test_repeated_board_does_not_silently_reset():
    h=hand([message('NEW_HAND'),message('FLOP'),message('BET','p',100),
            message('FLOP'),message('BET','p',100)])
    s=audit.run_arm(h,'ADDED')
    assert s['own']['p']==200
    assert not s['precise']


def test_missing_amount_never_invented():
    h=hand([message('NEW_HAND'),message('BET','p'),message('CALL','q',100)])
    r=audit.analyze(h,'test','test')
    assert r['class_']=='C'
    assert 'p' not in r['global_ADDED']['snapshot']['per_player_ledger']
    assert r['global_ADDED']['trace'][1]['derived_increment']=='UNKNOWN'


def test_capped_search_not_proof(monkeypatch):
    monkeypatch.setattr(audit,'CAP',1)
    h=hand([message('NEW_HAND'),message('SMALL_BLIND','p',100),
            message('BIG_BLIND','q',200),message('CALL','p',200)])
    r=audit.analyze(h,'test','test')
    assert r['branch_search']['incomplete']
    assert r['class_']=='C'


def test_unobserved_short_call_stays_unknown():
    h=hand([message('NEW_HAND'),message('BET','q',300),message('CALL','p',200)])
    s=audit.run_arm(h,'ADDED')
    assert 'SHORT_CALL_REQUIRES_UNOBSERVED_ALLIN_OR_STACK' in s['trace'][-1]['legality_unknown']
    assert not s['trace'][-1]['errors']


def test_explicit_raise_to_is_not_ambiguous():
    assert not audit.disputed(message('RAISE','p',300,'p raises to 300'))
    assert audit.disputed(message('RAISE','p',300,'p raises 300'))


def test_raise_to_unknown_prior_increment_unknown():
    h=hand([message('RAISE','p',300,'p raises to 300')])
    h['start_observed']=False
    assert audit.run_arm(h,'ADDED')['trace'][0]['derived_increment']=='UNKNOWN'


def test_report_serialization_keeps_decimal_exact(tmp_path):
    import json
    p=tmp_path/'report.json'
    audit.write(p,{'amount':audit.Decimal('10956'),'unknown':'UNKNOWN'})
    assert json.loads(p.read_text(encoding='utf-8'))=={'amount':'10956','unknown':'UNKNOWN'}
