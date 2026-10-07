"""Local stdio MCP bridge to the official TypeSafe Jev API (experimental).

No credential in arguments/configuration/output. Windows User environment is
used when an already-running desktop has not inherited TYPESAFE_API_KEY yet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.jev_poker_resolver import TypeSafeJevClient, canonical

PROTOCOLS = ('2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05')
TOOLS = [{
    'name': 'jev_decide',
    'description': 'Evaluate explicit state using official Jev typed Choice, Noul, or Score questions. Returns probabilities and token usage. Confidence alone does not prove source facts.',
    'inputSchema': {
        'type': 'object', 'required': ['state', 'questions'], 'additionalProperties': False,
        'properties': {
            'model': {'type': 'string', 'default': 'jev-latest'},
            'state': {'description': 'Complete observed state; never truncate a poker hand.'},
            'questions': {'type': 'object', 'minProperties': 1,
                          'additionalProperties': {
                              'type': 'object', 'required': ['type', 'instructions'],
                              'properties': {'type': {'enum': ['choice', 'noul', 'score']},
                                             'instructions': {}, 'criteria': {}},
                              'additionalProperties': False}},
        },
    },
    'annotations': {'readOnlyHint': True, 'destructiveHint': False,
                    'idempotentHint': False, 'openWorldHint': True},
}]


def evaluate(arguments, client=None):
    if not isinstance(arguments, dict) or set(arguments) - {'model', 'state', 'questions'}:
        raise ValueError('INVALID_JEV_ARGUMENTS')
    if 'state' not in arguments or not isinstance(arguments.get('questions'), dict) or not arguments['questions']:
        raise ValueError('JEV_STATE_AND_QUESTIONS_REQUIRED')
    for name, question in arguments['questions'].items():
        if not isinstance(name, str) or not isinstance(question, dict):
            raise ValueError('INVALID_JEV_QUESTION')
        if set(question) - {'type', 'instructions', 'criteria'} or question.get('type') not in {'choice', 'noul', 'score'} or 'instructions' not in question:
            raise ValueError('INVALID_JEV_QUESTION')
        criteria = question.get('criteria')
        if question['type'] == 'choice' and (not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255):
            raise ValueError('JEV_CHOICE_REQUIRES_2_TO_255_OPTIONS')
        if question['type'] == 'score' and (not isinstance(criteria, list) or not 2 <= len(criteria) <= 10):
            raise ValueError('JEV_SCORE_REQUIRES_2_TO_10_LEVELS')
    payload = dict(arguments)
    payload.setdefault('model', 'jev-latest')
    encoded = canonical(payload).encode('utf-8')
    if len(encoded) > 1048576:
        raise ValueError('WHOLE_HAND_EXCEEDS_CLIENT_PAYLOAD_LIMIT')
    client = client or TypeSafeJevClient({'timeout_seconds': 45, 'prefer_windows_user_key': True})
    return client.unwrap_response(client.request_json(encoded))['data']


def dispatch(message, evaluate_fn=evaluate):
    """Notification requests never emit responses, including initialized/cancelled."""
    if not isinstance(message, dict) or message.get('jsonrpc') != '2.0':
        return {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32600, 'message': 'Invalid request'}}
    if 'id' not in message:
        return None
    result = None
    error = None
    method = message.get('method')
    params = message.get('params') or {}
    if not isinstance(params, dict):
        error = {'code': -32602, 'message': 'Invalid params'}
    elif method == 'initialize':
        protocol = params.get('protocolVersion')
        result = {'protocolVersion': protocol if protocol in PROTOCOLS else PROTOCOLS[0],
                  'capabilities': {'tools': {'listChanged': False}},
                  'serverInfo': {'name': 'jev-typesafe-local', 'version': '1.0.0'},
                  'instructions': 'Evaluate evidence with Jev via official TypeSafe API. Never treat probability as proof. Preserve observed source rows; repairs require a separate candidate view and full deterministic revalidation.'}
    elif method == 'ping':
        result = {}
    elif method == 'tools/list':
        result = {'tools': TOOLS}
    elif method == 'tools/call':
        if params.get('name') != 'jev_decide':
            error = {'code': -32602, 'message': 'Unknown tool'}
        else:
            try:
                data = evaluate_fn(params.get('arguments'))
                result = {'content': [{'type': 'text', 'text': canonical(data)}],
                          'structuredContent': data, 'isError': False}
            except Exception as exc:
                # Only internal error identifiers, never raw exceptions/headers/key.
                code = str(exc)
                if not code or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789' for c in code):
                    code = 'JEV_TRANSPORT_OR_SCHEMA_ERROR'
                result = {'content': [{'type': 'text', 'text': code}], 'isError': True}
    else:
        error = {'code': -32601, 'message': 'Method not found'}
    return {'jsonrpc': '2.0', 'id': message['id'], **({'error': error} if error else {'result': result})}


def main():
    if hasattr(sys.stdin, 'reconfigure'):
        sys.stdin.reconfigure(encoding='utf-8')
        sys.stdout.reconfigure(encoding='utf-8')
    for line in sys.stdin:
        try:
            response = dispatch(json.loads(line))
        except json.JSONDecodeError:
            response = {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'Parse error'}}
        except Exception:
            response = {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32603, 'message': 'Internal error'}}
        if response is not None:
            print(canonical(response), flush=True)


if __name__ == '__main__':
    main()
