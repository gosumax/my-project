"""Independent evidence labels; never an alternate HH certification gate."""


def evidence_axes(record, reconstruction, *, proven_stacks=None, proven_seats=None, certified=False,
                  hh_contract=None):
    hand = record['observed_hand']
    derived = reconstruction.get('derived_final_validation', {})
    validation = derived.get('validation') if derived.get('accepted') else record['validation']
    diagnostics = derived.get('diagnostics') if derived.get('accepted') else record['diagnostics']
    money = reconstruction.get('money_only_assessment', reconstruction)
    diagnostic_exact = ('money_only_assessment' in reconstruction and
                        money.get('strict_source_uniqueness') is True and
                        money.get('full_hh_authority') is False and money.get('required_blinds') == ['BB'])
    source_exact = (diagnostic_exact or
                    reconstruction.get('strict_source_uniqueness') and
                    not reconstruction.get('dropped_events') and not reconstruction.get('inferred_events') and
                    not reconstruction.get('chosen_identity_hypothesis', {}).get('aliases'))
    exact = (source_exact and money.get('solution_count') == 1 and not money.get('exhausted') and
             money.get('pot_delta') == 0 and money.get('payout_eligible') is True)
    observed = diagnostics.get('math', {})
    math_status = ('PASS' if exact else 'UNKNOWN' if not observed or observed.get('missing_fields') or
                   observed.get('suspicious_rows') else 'PASS' if observed.get('pass') else 'FAIL')
    actors = {m['player'] for m in hand['messages'] if m.get('player')}
    stacks = proven_stacks or {}
    complete_stacks = (bool(actors) and actors == set(stacks) and
                       len({s['seat_id'] for s in stacks.values()}) == len(stacks))
    seats=stacks if proven_seats is None else proven_seats
    complete_seats=(bool(actors) and actors==set(seats) and
                    len({s['seat_id'] for s in seats.values()})==len(seats))
    action = validation.get('action_logic_status', 'UNKNOWN')
    if exact:
        action = 'PASS'
    if action == 'NOT_VERIFIABLE':
        action = 'UNKNOWN'
    return dict(
        action_logic=dict(status=action, scope='OBSERVED_SEQUENCE_RULES; SEAT_ORDER_NOT_PROVEN'),
        pot_accounting=dict(status=math_status, scope='OBSERVED_CONTRIBUTIONS_MINUS_RETURNS_EQUALS_PAYOUTS'),
        payout_eligibility=dict(status='PASS' if exact else 'UNKNOWN', scope='NET_CONTRIBUTION_LAYERS; CARD_WINNERS_SEPARATE'),
        cards=dict(status=validation.get('card_status', 'UNKNOWN')),
        actor_identity=dict(status=validation.get('identity_status', 'UNKNOWN'),
                            scope='EXISTING_ROW_PROVENANCE_CHECK; CROSS_SPELLING_ACTOR_IDENTITY_NOT_INFERRED'),
        seats=dict(status='PASS' if complete_seats else 'UNKNOWN'),
        start_stacks=dict(status='PASS' if complete_stacks else 'UNKNOWN'),
        boundaries=dict(status='PASS' if hand.get('start_observed') and hand.get('end_observed')
                        and not hand.get('boundary_conflicts') else 'UNKNOWN'),
        forced_blind_state=dict(status='UNKNOWN' if reconstruction.get('money_only_assessment')
                                else 'PASS' if {'SMALL_BLIND','BIG_BLIND'}.issubset(
                                    {m.get('action') for m in hand['messages']}) else 'UNKNOWN'),
        full_hh=dict(status='PASS' if certified else 'UNKNOWN', certified=bool(certified),
                     **({'contract':hh_contract} if hh_contract else {})),
        diagnostic_only=True)
