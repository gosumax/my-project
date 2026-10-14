"""Source preserving experimental diagnostics; never a production gate."""
import copy
from parser_core.poker_logic_engine import validate_hand
from parser_core.poker_math_engine import audit_observed_ledger


EXPECTED = {
    'MISSING_BLINDS_BEFORE_ACTION': 'Observed Small Blind and Big Blind before voluntary action',
    'MISSING_AMOUNT': 'Positive observed amount on this money action',
    'MISSING_ACTION': 'Recognizable source-supported event',
    'ACTION_AFTER_FOLD': 'Folded player cannot act again',
    'ACTION_AFTER_ALLIN': 'All-in player cannot act again',
    'CHECK_FACING_BET': 'Player contribution equals current bet before checking',
    'INVALID_CALL': 'Added call equals amount to call, or smaller positive all-in',
    'CALL_WITH_NOTHING_TO_CALL': 'Positive amount to call',
    'INVALID_RAISE': 'Raise-to target greater than current bet',
    'OLD_WIN': 'Completed betting before payout',
    'BETTING_ROUND_INCOMPLETE': 'All active players have matched the bet and acted',
}


def diagnose_hand(hand, *, hidden_card_source=None,no_sb_source=None):
    """One first action break, independent complete observed ledger/card results."""
    prefix = validate_hand(hand, diagnostic_stop=True,no_sb_source=no_sb_source)
    ledger = audit_observed_ledger(hand)
    # Existing card evaluator remains separate. Its action cascade is neither
    # exposed as new diagnoses nor used as root-cause evidence.
    layers = validate_hand(hand, hidden_card_source=hidden_card_source,no_sb_source=no_sb_source)
    issue = prefix['first_issue']
    first = None
    if issue:
        index = issue['message_index']
        msg = hand['messages'][index] if index >= 0 else None
        before = prefix['trace'][index]['state_before'] if msg else prefix['prefix_state']
        first = {**issue, 'severity': 'DIAGNOSTIC', 'RULE_TRIGGER': issue['code'], 'PROVEN_CAUSE': 'UNKNOWN',
                 'expected': EXPECTED.get(issue['code'], issue.get('detail') or issue['code']),
                 'observed': msg['raw_text'] if msg else 'END_OF_SEGMENT',
                 'state_before': before, 'state_after': before,
                 'virtual_boundary_check': msg is None,
                 'math_delta_at_break': before['prefix_math']['delta'],
                 'prefix_delta_is_conservation_failure': False}
    card_issues = [i for i in layers['issues'] if i['domain'] == 'CARDS']
    if first:
        # These conclusions depend on betting state beyond the trusted prefix.
        card_issues = [i for i in card_issues if i['detail'] not in
                       {'Folded player showdown', 'Payouts contradict deterministic side-pot winners'}]
    card_status = ('CONFLICT' if any(i['evidence_type'] == 'CONTRADICTION' for i in card_issues)
                   else 'NOT_VERIFIABLE' if card_issues or first else 'PASS')
    missing = ledger['missing_fields']
    critical_identity = layers['identity_status'] == 'IDENTITY_CRITICAL'
    math_status = 'FAIL' if not ledger['pass'] else 'NOT_VERIFIABLE' if missing or ledger['suspicious_rows'] else 'PASS'
    logic_status = 'FAIL' if first and not first['virtual_boundary_check'] else 'NOT_VERIFIABLE' if first else 'PASS'
    fully = (first is None and layers['complete_candidate'] and math_status == 'PASS'
             and card_status == 'PASS' and layers['side_pot_status'] == 'PASS'
             and not critical_identity)
    status = ('FULLY_VALID' if fully else 'MISSING_AMOUNT' if first and first['code'] == 'MISSING_AMOUNT'
              else 'LOGIC_FAIL' if logic_status == 'FAIL' else 'MATH_FAIL' if math_status == 'FAIL' else 'UNRESOLVED')
    suspects = list(ledger['suspicious_rows'])
    if first and first['message_index'] >= 0:
        suspects.append({'message_index': first['message_index'], 'RULE_TRIGGER': first['code'], 'PROVEN_CAUSE': 'UNKNOWN'})
    suspects.sort(key=lambda v: v['message_index'])
    ledger['first_suspicious_action'] = suspects[0] if suspects else None
    ledger['localization_note'] = ('Suspicious observation, not proof that delta began here' if suspects else
                                 'No source-supported localization of the aggregate delta')
    ledger['ledger_certainty'] = 'UNPROVEN' if missing or ledger['suspicious_rows'] or first or critical_identity else 'OBSERVED_COMPLETE'
    for step, math in zip(prefix['trace'], ledger['steps']):
        for issue in step['issues']:
            if issue['severity'] == 'BLOCKER':
                issue['severity'] = 'DIAGNOSTIC'
        step['observed_math'] = math
        step['rule_result'] = ('NOT_EVALUATED_AFTER_FIRST_BREAK' if step['status'] == 'AFTER_BREAK' else
                               first['code'] if step['status'] == 'FIRST_BREAK' else 'OK')
        step['PROVEN_CAUSE'] = 'UNKNOWN'
    for issue in card_issues:
        if issue['severity'] == 'BLOCKER':
            issue['severity'] = 'DIAGNOSTIC'
    return {'first_break': first, 'trace': prefix['trace'], 'logic_status': logic_status,
            'math': ledger, 'math_status': math_status, 'card_status': card_status,
            'card_issues': card_issues,
            'provenance_status': layers['identity_status'], 'complete_start': bool(hand.get('start_observed')),
            'complete_end': bool(hand.get('end_observed')), 'final_status': status,
            'logic_ms': prefix['logic_ms'], 'PROVEN_CAUSE': 'UNKNOWN'}


def revalidate_candidate(observed, candidate):
    """No new observations; candidate must retain immutable physical evidence."""
    if candidate != observed:
        from parser_core.jev_poker_resolver import candidate_view
        inferences = candidate.get('inferences', [])
        if len(inferences) != 1 or not inferences[0].get('decision'):
            return {'accepted': False, 'reject_reason': 'UNAUTHORIZED_CANDIDATE_TRANSFORMATION', 'revalidation': None}
        authorized, rejection = candidate_view(observed, validate_hand(observed), inferences[0]['decision'])
        if rejection or authorized != candidate:
            return {'accepted': False, 'reject_reason': rejection or 'CANDIDATE_DIFF_OUTSIDE_PROVEN_TRANSFORMATION', 'revalidation': None}
    def rows(hand):
        return {rid: row for msg in hand['messages'] for rid, row in zip(msg['source_row_ids'], msg['source_rows'])}
    original_rows, candidate_rows = rows(observed), rows(candidate)
    if original_rows != candidate_rows:
        return {'accepted': False, 'reject_reason': 'OBSERVED_SOURCE_ROWS_CHANGED_OR_INVENTED', 'revalidation': None}
    # Exact authorized==candidate comparison above proves the bounded field
    # transformation, including evidence from a separate blind-level row.
    result = diagnose_hand(copy.deepcopy(candidate))
    return {'accepted': result['final_status'] == 'FULLY_VALID',
            'reject_reason': None if result['final_status'] == 'FULLY_VALID' else 'REVALIDATION_NOT_FULLY_VALID',
            'revalidation': result}
