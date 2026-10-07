"""Offline official transport and local MCP contracts; never read credentials."""
import json
from types import SimpleNamespace
import pytest
import parser_core.jev_poker_resolver as resolver
from parser_core.poker_hand_assembly import make_message
from scripts.jev_typesafe_mcp import dispatch, evaluate


@pytest.fixture
def client(monkeypatch):
    # Replace module sys only: bypass Windows registry/environment credential path.
    monkeypatch.setattr(resolver, 'sys', SimpleNamespace(platform='offline-test'))
    return resolver.TypeSafeJevClient()


def test_official_endpoint_and_direct_response(client):
    assert client.config['endpoint'] == 'https://api.typesafe.ai/v1/systemone'
    response = {'model': 'jev-latest', 'answers': {'hypothesis': {'choice': 'UNRESOLVED', 'confidence': .5}}}
    assert client.unwrap_response(response) == {'code': 0, 'data': response}
    with pytest.raises(ValueError, match='TYPESAFE_ENDPOINT_MUST_BE_OFFICIAL'):
        resolver.TypeSafeJevClient({'endpoint': 'https://example.invalid'})


@pytest.mark.parametrize('response', [None, {}, {'model': 'jev-latest', 'answers': []}])
def test_direct_response_schema_rejected(client, response):
    with pytest.raises(ValueError, match='TYPESAFE_RESPONSE_SCHEMA_INVALID'):
        client.unwrap_response(response)


def test_whole_hand_larger_than_32k_preserves_all_actions(client):
    texts = ['A folds ' + ('x' * 500) + str(i) for i in range(100)]
    messages = [make_message([{'row_id': f'r{i}', 'raw_text': t}]) for i, t in enumerate(texts)]
    hand = {'table_id': 'T01', 'session_id': 's', 'hand_id': '1', 'messages': messages}
    encoded, _ = client.prepare_request({'hand': hand, 'conflicts': []})
    assert len(encoded) > 32768
    payload = json.loads(encoded)
    assert [a[5] for a in payload['state']['hand']['ordered_actions']] == texts
    assert len(payload['state']['source_rows']) == 100
    assert payload['model'] == 'jev-latest'


def test_mcp_initialize_notifications_list_and_call():
    init = dispatch({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-03-26'}})
    assert init['result']['protocolVersion'] == '2025-03-26'
    assert dispatch({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) is None
    listed = dispatch({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'})
    assert listed['result']['tools'][0]['name'] == 'jev_decide'
    data = {'model': 'jev-latest', 'answers': {'q': {'choice': 'yes'}}}
    seen = []
    result = dispatch({'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'jev_decide', 'arguments': {'state': 'whole-hand'}}}, lambda args: seen.append(args) or data)
    assert seen == [{'state': 'whole-hand'}]
    assert result['result']['structuredContent'] == data
    assert json.loads(result['result']['content'][0]['text']) == data


def test_mcp_evaluate_large_state_fake_transport():
    class Fake:
        def request_json(self, encoded):
            self.payload = json.loads(encoded)
            return {'model': 'jev-latest', 'answers': {'q': {'choice': 'yes'}}}
        def unwrap_response(self, response):
            return {'data': response}
    fake = Fake()
    state = 'complete source ' * 4000
    data = evaluate({'state': state, 'questions': {'q': {'type': 'choice', 'instructions': 'Select', 'criteria': {'yes': 'Yes', 'no': 'No'}}}}, fake)
    assert fake.payload['state'] == state
    assert data['answers']['q']['choice'] == 'yes'


@pytest.mark.parametrize('arguments', [None, {'state': 1, 'questions': {}}, {'state': 1, 'questions': {'q': {'type': 'choice', 'instructions': 'x', 'criteria': {'one': 'x'}}}}])
def test_malformed_arguments_do_not_reach_transport(arguments):
    class Fake:
        def request_json(self, encoded):
            pytest.fail('Malformed arguments reached transport')
    with pytest.raises(ValueError):
        evaluate(arguments, Fake())


def test_invalid_params_and_exception_redaction():
    bad = dispatch({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call', 'params': ['invalid']})
    assert bad['error']['code'] == -32602
    def fail(args):
        raise RuntimeError('Authorization: Bearer synthetic-secret')
    result = dispatch({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {'name': 'jev_decide', 'arguments': {}}}, fail)
    assert result['result']['isError'] is True
    assert result['result']['content'][0]['text'] == 'JEV_TRANSPORT_OR_SCHEMA_ERROR'
    assert 'synthetic-secret' not in json.dumps(result)


def test_cross_origin_redirect_never_followed_without_network(client, monkeypatch):
    seen = []
    class OfflineOpener:
        def open(self, request, timeout):
            seen.append(request.full_url)
            assert redirect.redirect_request(request, None, 302, 'Moved', {}, 'https://other.invalid') is None
            raise RuntimeError('OFFLINE_REDIRECT_BLOCKED')
    def opener_factory(handler):
        nonlocal redirect
        redirect = handler
        return OfflineOpener()
    redirect = None
    monkeypatch.setattr(resolver.urllib.request, 'build_opener', opener_factory)
    # Disable credentials entirely; the test exercises only redirect construction.
    client.config['api_key_env'] = None
    with pytest.raises(RuntimeError, match='OFFLINE_REDIRECT_BLOCKED'):
        client.request_json(b'{}')
    assert seen == ['https://api.typesafe.ai/v1/systemone']


@pytest.mark.parametrize('kind,reason', [('UNRESOLVED', 'JEV_UNRESOLVED'), ('CARD_EVIDENCE_CONFLICT', 'CARD_EVIDENCE_CONFLICT')])
def test_explicit_nonrepair_decision_retains_no_candidate(kind, reason):
    hand = {'messages': []}
    candidate, status = resolver.candidate_view(hand, {'issues': []}, {'decision': kind, 'target_rows': [], 'evidence_rows': []})
    assert candidate is None
    assert status == reason
    assert hand == {'messages': []}


def test_choice_with_eleven_options_reaches_fake_transport():
    class Fake:
        def request_json(self, encoded):
            self.payload = json.loads(encoded)
            return {'model': 'jev-latest', 'answers': {}}
        def unwrap_response(self, response):
            return {'data': response}
    fake = Fake()
    choices = {str(i): f'Option {i}' for i in range(11)}
    evaluate({'state': {}, 'questions': {'q': {'type': 'choice', 'instructions': 'Choose', 'criteria': choices}}}, fake)
    assert fake.payload['questions']['q']['criteria'] == choices
