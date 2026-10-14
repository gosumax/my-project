"""Retained native CHAT service clauses; financial or incomplete tails stay unknown."""
import copy
import json
from pathlib import Path

import pytest

from parser_core.poker_hand_assembly import assemble_messages, parse_text as observed_parse
from parser_core.poker_reconstruction.assembly import candidate_assemblies
from parser_core.poker_reconstruction.timeline import parse_rows, parse_text
from parser_core.table_service_events import describe_service
from parser_core.service_text_normalizer import normalize_service, literal_win_pot_continuation


FIXTURE = json.loads((Path(__file__).parent / 'fixtures/smoke18_terminal_notices.json').read_text('utf-8'))


@pytest.mark.parametrize('name', ['T03_BREAK', 'T05_BREAK'])
def test_native_break_is_one_actorless_service_message(name):
    rows = copy.deepcopy(FIXTURE['groups'][name])
    original = copy.deepcopy(rows)
    messages = assemble_messages(rows, require_adjacency=True, require_pixel_cards=True)
    assert len(messages) == 1
    assert messages[0]['action'] == 'TABLE_INFO'
    assert messages[0]['player'] is None and messages[0]['amount'] is None
    assert messages[0]['source_rows'] == rows
    events = parse_rows(rows)
    assert len(events) == 1 and events[0].kind == 'INFO'
    assert events[0].player is None and events[0].amount is None
    assert events[0].source['service_event']['stack_transfer'] is False
    paths, exhausted, _ = candidate_assemblies(rows)
    assert not exhausted and len(paths) == 1
    assert len(paths[0]) == 1 and paths[0][0].kind == 'INFO'
    assert paths[0][0].source['rows'] == rows
    assert rows == original


@pytest.mark.parametrize('minutes', [1, 5, 15, 60])
def test_break_duration_is_not_a_chip_amount(minutes):
    text = f'Dealer: Break has started and will last {minutes} min after the end of active hands.'
    assert describe_service(text) == dict(kind='BREAK', actor=None, amount=None, stack_transfer=False)
    assert observed_parse(text)['action'] == 'TABLE_INFO'
    event = parse_text(text, {})
    assert event.kind == 'INFO' and event.player is None and event.amount is None


@pytest.mark.parametrize('text', [
    'Dealer: Break has started and will last',
    '5 min after the end of active hands.',
    'Dealer: Break has started and will last 5 min',
    'Dealer: Break has started and will last 5 min after the end of active hands. alice bets 100',
    'Dealer: Break has started and will last 5 min after the end of active hands. 100',
    'Dealer: Break has started and will last 0 min after the end of active hands.',
    'Dealer: Break has started and will last -5 min after the end of active hands.',
])
def test_incomplete_or_financial_break_text_is_not_harmless(text):
    assert describe_service(text) is None
    assert observed_parse(text)['action'] != 'TABLE_INFO'
    assert parse_text(text, {}).kind != 'INFO'


@pytest.mark.parametrize('field,value', [
    ('frame_id', 1889), ('row_index', 19), ('table_id', 'other'),
    ('session_id', 'other'), ('epoch', 2),
    ('hand_id', 'other'), ('source_file', 'other'),
])
def test_break_fragments_cannot_cross_physical_context(field, value):
    rows = copy.deepcopy(FIXTURE['groups']['T03_BREAK'])
    if field in {'hand_id', 'source_file'}:
        rows[0][field] = 'original'
    rows[1][field] = value
    assert len(assemble_messages(rows, require_adjacency=True)) == 2
    assert all(e.kind == 'UNKNOWN' for e in parse_rows(rows))
    paths, _, _ = candidate_assemblies(rows)
    assert all(all(e.kind != 'INFO' for e in path) for path in paths)


def test_native_win_terminal_spelling_keeps_amount_actor_and_all_raw_sources():
    rows = copy.deepcopy(FIXTURE['groups']['T03_WIN'])
    original = copy.deepcopy(rows)
    assert literal_win_pot_continuation(rows)
    messages = assemble_messages(rows, require_adjacency=True, require_pixel_cards=True)
    assert len(messages) == 1
    message = messages[0]
    assert (message['action'], message['player'], int(message['amount'])) == ('WIN', 'bankir', 233500)
    assert message['raw_text'].endswith('main potO') and message['normalized_text'].endswith('main pot')
    assert message['source_rows'] == original
    assert not message['cards'] and 'win_best_five_cards' not in message
    events = parse_rows(rows)
    assert len(events) == 1 and (events[0].kind, events[0].player, events[0].amount) == ('WIN', 'bankir', 233500)
    change = events[0].source['service_normalizations'][0]
    assert change['rule_id'] == 'SERVICE_WIN_MAIN_POT_TERMINAL_SPELLING_V1'
    assert change['observed_variant'] == 'potO' and change['canonical_token'] == 'pot'
    paths, exhausted, _ = candidate_assemblies(rows)
    assert not exhausted and len(paths) == 1 and len(paths[0]) == 1
    assert (paths[0][0].kind, paths[0][0].player, paths[0][0].amount) == ('WIN', 'bankir', 233500)
    assert paths[0][0].source['rows'] == original and rows == original


@pytest.mark.parametrize('actor,amount', [('potO_nick', '100'), ('literal_99', '1 234 567')])
def test_complete_win_spelling_is_general_and_only_terminal_token_changes(actor, amount):
    raw = f'Dealer: {actor} wins {amount} from the main potO'
    result = normalize_service(raw)
    assert result.raw_ocr == raw and result.normalized_text == raw[:-1]
    change = result.changes[0]
    assert change['source_span'] == [raw.rfind('potO'), len(raw)]
    event = parse_text(raw, {})
    assert event.kind == 'WIN' and event.player == actor
    assert event.amount == int(amount.replace(' ', ''))


@pytest.mark.parametrize('raw', [
    'main potO', 'Dealer: a wins 100 from the main pot0',
    'Dealer: a wins 100 from the main potOO', 'Dealer: a wins 100 from the main potO9',
    'Dealer: a wins 100 from the main potOther', 'Dealer: a wins 100 from the main potO calls 100',
    'Dealer: a wins 100 from the main potO 200', 'Dealer: a wins from the main potO',
    'Dealer: a wins 1O0 from the main potO', 'Dealer: a calls 100 from the main potO',
    'Dealer: potO_nick wins 100 from the main pot',
])
def test_other_win_or_financial_spelling_cannot_be_repaired(raw):
    result = normalize_service(raw)
    assert result.raw_ocr == result.normalized_text == raw and not result.changes


@pytest.mark.parametrize('tail,kind', [('main pot0', 'UNKNOWN'), ('main potOO', 'UNKNOWN'),
    ('main potO9', 'UNKNOWN'), ('main potOther', 'UNKNOWN'), ('main potO calls 100', 'CALL'),
    ('main potO 100', 'UNKNOWN')])
def test_unknown_or_financial_payout_tail_is_never_absorbed(tail, kind):
    rows = copy.deepcopy(FIXTURE['groups']['T03_WIN'])
    rows[1]['raw_ocr_text'] = tail
    assert not literal_win_pot_continuation(rows)
    assert len(assemble_messages(rows, require_adjacency=True)) == 2
    assert len(parse_rows(rows)) == 2 and parse_rows(rows)[1].kind == kind
    assert all(len(path) == 2 and path[-1].kind == kind for path in candidate_assemblies(rows)[0])


@pytest.mark.parametrize('field,value', [
    ('frame_id', 1884), ('row_index', 19), ('table_id', 'other'),
    ('session_id', 'other'), ('epoch', 2), ('hand_id', 'other'), ('source_file', 'other'),
])
def test_win_spelling_cannot_bridge_source_context(field, value):
    rows = copy.deepcopy(FIXTURE['groups']['T03_WIN'])
    if field in {'hand_id', 'source_file'}: rows[0][field] = 'original'
    rows[1][field] = value
    assert not literal_win_pot_continuation(rows)
    assert len(assemble_messages(rows, require_adjacency=True)) == 2
    assert len(parse_rows(rows)) == 2 and parse_rows(rows)[1].kind == 'UNKNOWN'
    assert all(len(path) == 2 and path[-1].kind == 'UNKNOWN' for path in candidate_assemblies(rows)[0])


@pytest.mark.parametrize('head', ['Dealer: bankir calls 233 500',
    'Dealer: bankir wins from the', 'Dealer: bankir wins 233 5OO from the'])
def test_win_tail_requires_complete_literal_amountful_win_head(head):
    rows = copy.deepcopy(FIXTURE['groups']['T03_WIN'])
    rows[0]['raw_ocr_text'] = head
    assert not literal_win_pot_continuation(rows)
    assert len(assemble_messages(rows, require_adjacency=True)) == 2
    assert len(parse_rows(rows)) == 2 and parse_rows(rows)[1].kind == 'UNKNOWN'
