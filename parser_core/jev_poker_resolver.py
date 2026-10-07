"""Experimental Jev hypotheses. Evidence gates precede deterministic revalidation."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
import urllib.request
import subprocess
import sys
from pathlib import Path
from parser_core.poker_hand_assembly import make_message

VERSION = 'jev-evidence-gated-2-native-decisions'
PROMPT = '''You are Jev, a poker hypothesis resolver, never a source of facts.
Return one JSON object with decision, target_rows, reason, confidence, evidence_rows,
candidate_amount (optional). Allowed decisions: LIKELY_STALE_DUPLICATE,
LIKELY_MISSING_AMOUNT, LIKELY_MULTILINE_CONTINUATION, LIKELY_SAME_PLAYER,
STALE_PREVIOUS_HAND_ROW, CARD_EVIDENCE_CONFLICT, UNRESOLVED.
Do not invent actions, winners, cards, or final HH. Cite physical row IDs.
Use the whole provided hand. Confidence is not evidence.'''


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


class JevHTTPClient:
    """Jev native Decisions API; selects evidence-linked structured hypotheses."""
    def __init__(self, config):
        self.config = config

    def prepare_request(self, request):
        options = hypothesis_options(request['hand'], request['conflicts'])
        criteria = {'UNRESOLVED': 'No supplied hypothesis is justified by observed evidence.',
                    'CARD_EVIDENCE_CONFLICT': 'Card evidence conflicts; keep unresolved, do not change cards.'}
        criteria.update({f'H{i}': canonical(option) for i, option in enumerate(options)})
        # Send all messages/row provenance needed by this hand, but exclude pixel blobs
        # and unrelated diagnostic fields. Never truncate a hand to fit API limits.
        compact = {k: copy.deepcopy(v) for k, v in request.items() if k not in {'hand', 'conflicts', 'prompt'}}
        compact['hand'] = {k: copy.deepcopy(v) for k, v in request['hand'].items() if k in
                           {'table_id', 'session_id', 'hand_id', 'hand_epoch', 'blind_level',
                            'boundary_conflicts', 'boundary_context', 'start_observed', 'end_observed', 'semantics'}}
        # One lossless row dictionary; references avoid repeating long stable IDs in
        # actions and every conflict. Raw text is retained in assembled message order.
        physical = {rid: row for m in request['hand']['messages'] for rid, row in zip(m['source_row_ids'], m['source_rows'])}
        ids = list(physical)
        row_index = {rid: index for index, rid in enumerate(ids)}
        compact['source_rows'] = [[rid, physical[rid].get('row_pixel_sha256'), physical[rid].get('origin'),
                                   physical[rid].get('epoch')] for rid in ids]
        actions = []
        compact['action_columns'] = ['timestamp_ms', 'frame_id', 'action', 'player', 'amount',
                                     'raw_text', 'rows', 'all_in', 'cards', 'showdown_class', 'field_sources']
        for m in request['hand']['messages']:
            action = [m.get('timestamp_ms'), m.get('frame_id'), m['action'], m.get('player'), m.get('amount'),
                      m['raw_text'], [row_index[rid] for rid in m['source_row_ids']], m.get('all_in', False),
                      m.get('cards', []), m.get('showdown_class'),
                      {k: v for k, v in m.get('field_sources', {}).items() if v != 'OBSERVED'}]
            actions.append(action)
        compact['default_field_source_type'] = 'OBSERVED'
        compact['hand']['ordered_actions'] = actions
        compact['conflicts'] = [[i['message_index'], i['code'], i.get('detail', '')] for i in request['conflicts']]
        if 'diagnostic_packet' in compact:
            # A lossless source-ID dictionary avoids repeating 150-character
            # IDs in every context, missing-field and ledger record.
            packet_ids = list(ids)
            for row in compact['diagnostic_packet'].get('physical_source_context', []):
                rid = row['source_row_id']
                if rid not in packet_ids:
                    packet_ids.append(rid)
            refs = {rid: i for i, rid in enumerate(packet_ids)}
            def references(value):
                if isinstance(value, dict):
                    return {k: references(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [references(v) for v in value]
                return {'row_ref': refs[value]} if isinstance(value, str) and value in refs else value
            compact['diagnostic_packet'] = references(compact['diagnostic_packet'])
            compact['diagnostic_source_id_dictionary'] = packet_ids
        payload = {'model': self.config.get('model', 'typesafe-ai/jev'), 'state': compact,
                   'questions': {'hypothesis': {'type': 'choice', 'instructions': PROMPT, 'criteria': criteria}}}
        if self.config.get('use_service_default_model'):
            payload.pop('model')
        encoded = canonical(payload).encode('utf-8')
        if len(encoded) > self.config.get('max_request_bytes', 32768):
            raise ValueError(self.config.get('payload_limit_error', 'WHOLE_HAND_EXCEEDS_JEV_32K_LIMIT'))
        return encoded, options

    def __call__(self, request):
        encoded, options = self.prepare_request(request)
        data = self.unwrap_response(self.request_json(encoded))
        if data.get('code') != 0:
            raise RuntimeError('JEV_API_NONZERO_CODE')
        answer = data['data']['answers']['hypothesis']
        choice = answer['choice']
        if choice.startswith('H') and choice[1:].isdigit() and int(choice[1:]) < len(options):
            decision = copy.deepcopy(options[int(choice[1:])])
        elif choice in {'UNRESOLVED', 'CARD_EVIDENCE_CONFLICT'}:
            decision = {'decision': choice, 'target_rows': [], 'reason': 'Native typed Jev decision', 'evidence_rows': []}
        else:
            raise ValueError('JEV_CHOICE_OUTSIDE_CANDIDATES')
        decision.update(confidence=answer['confidence'], native_response=data,
                        reason_source='DETERMINISTIC_CANDIDATE_TEMPLATE',
                        transport=self.config.get('transport_label', 'JEV_NATIVE_DECISIONS'))
        return decision

    def request_json(self, encoded):
        """Verified HTTP transport shared by typed poker requests and the MCP bridge."""
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'}
        if self.config.get('api_key_env'):
            key = os.environ.get(self.config['api_key_env'])
            if not key:
                raise RuntimeError('Configured API key environment variable is unset')
            headers['Authorization'] = 'Bearer '+key
        endpoint = self.config.get('endpoint', 'https://www.jevai.org/api/v1/decisions')
        if not endpoint.startswith('https://'):
            raise ValueError('Jev requires verified HTTPS')
        if sys.platform == 'win32':
            # Windows' verified TLS succeeds where this workstation's Python trust
            # chain fails. No -SkipCertificateCheck or certificate callbacks.
            environment = dict(os.environ, JEV_RESOLVER_ENDPOINT=endpoint,
                               JEV_RESOLVER_KEY_VARIABLE=self.config.get('api_key_env', ''),
                               JEV_RESOLVER_TIMEOUT=str(self.config.get('timeout_seconds', 60)))
            command = '''$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$body = [Console]::In.ReadToEnd()
$headers = @{}
$key = [Environment]::GetEnvironmentVariable($env:JEV_RESOLVER_KEY_VARIABLE)
if ($key) { $headers['Authorization'] = 'Bearer ' + $key }
try {
  $response = Invoke-WebRequest -Uri $env:JEV_RESOLVER_ENDPOINT -Method Post -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec ([int]$env:JEV_RESOLVER_TIMEOUT) -MaximumRedirection 0 -UseBasicParsing
  [Console]::Write($response.Content)
} catch {
  $status = 0
  if ($_.Exception.Response) { $status = [int]$_.Exception.Response.StatusCode }
  $detail = [string]$_.ErrorDetails.Message
  if (-not $detail -and $_.Exception.Response) {
    try { $reader = [IO.StreamReader]::new($_.Exception.Response.GetResponseStream()); $detail = $reader.ReadToEnd(); $reader.Close() } catch { }
  }
  if ($key) { $detail = $detail.Replace($key, '[REDACTED]') }
  if ($detail.Length -gt 600) { $detail = $detail.Substring(0, 600) }
  [Console]::Error.Write((@{ status = $status; detail = $detail } | ConvertTo-Json -Compress)); exit 2
}
'''
            process = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                                     input=encoded, capture_output=True, env=environment,
                                     timeout=self.config.get('timeout_seconds', 60)+5,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            if process.returncode:
                try:
                    diagnostic = json.loads(process.stderr.decode('utf-8', errors='replace'))
                    exception = RuntimeError('JEV_HTTP_STATUS_'+str(int(diagnostic['status'])))
                    exception.api_error_detail = diagnostic.get('detail', '')
                    raise exception
                except (ValueError, KeyError):
                    raise RuntimeError('JEV_VERIFIED_WINDOWS_HTTP_FAILED')
            data = json.loads(process.stdout.decode('utf-8-sig'))
        else:
            req = urllib.request.Request(endpoint, data=encoded, headers=headers)
            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    return None
            opener = urllib.request.build_opener(NoRedirect())
            with opener.open(req, timeout=self.config.get('timeout_seconds', 60)) as response:
                data = json.load(response)
        return data

    def unwrap_response(self, data):
        return data


class TypeSafeJevClient(JevHTTPClient):
    """Official TypeSafe API. The legacy third-party Jev transport stays separate."""
    def __init__(self, config=None):
        config = dict(config or {})
        if config.get('endpoint', 'https://api.typesafe.ai/v1/systemone') != 'https://api.typesafe.ai/v1/systemone':
            raise ValueError('TYPESAFE_ENDPOINT_MUST_BE_OFFICIAL')
        super().__init__({'model': 'jev-latest', 'max_request_bytes': 1048576,
                          'payload_limit_error': 'WHOLE_HAND_EXCEEDS_CLIENT_PAYLOAD_LIMIT',
                          'transport_label': 'TYPESAFE_SYSTEMONE', **config,
                          'endpoint': 'https://api.typesafe.ai/v1/systemone',
                          'api_key_env': 'TYPESAFE_API_KEY'})
        # Desktop sessions may predate the user's environment-variable update.
        if sys.platform == 'win32' and (self.config.get('prefer_windows_user_key', False) or not os.environ.get('TYPESAFE_API_KEY')):
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as registry:
                    key, _ = winreg.QueryValueEx(registry, 'TYPESAFE_API_KEY')
                if key:
                    os.environ['TYPESAFE_API_KEY'] = key
            except FileNotFoundError:
                pass

    def unwrap_response(self, data):
        if not isinstance(data, dict) or not isinstance(data.get('answers'), dict) or not data.get('model'):
            raise ValueError('TYPESAFE_RESPONSE_SCHEMA_INVALID')
        return {'code': 0, 'data': data}


class JevMCPClient(JevHTTPClient):
    """Stateless Streamable HTTP MCP, jev_decide with native typed questions."""
    def __init__(self, config):
        super().__init__(dict(config, endpoint='https://www.jevai.org/api/mcp'))

    def prepare_request(self, request):
        encoded, options = super().prepare_request(request)
        payload = {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                   'params': {'name': 'jev_decide', 'arguments': json.loads(encoded)}}
        encoded = canonical(payload).encode('utf-8')
        if len(encoded) > 32768:
            raise ValueError('WHOLE_HAND_EXCEEDS_JEV_32K_LIMIT')
        return encoded, options

    def unwrap_response(self, data):
        result = data.get('result') or {}
        if data.get('error') or result.get('isError'):
            error = RuntimeError('JEV_MCP_TOOL_ERROR')
            detail = data.get('error') or [c.get('text') for c in result.get('content', []) if c.get('type') == 'text']
            error.api_error_detail = canonical(detail)[:600]
            raise error
        structured = result.get('structuredContent')
        if structured is None:
            blocks = [c['text'] for c in result.get('content', []) if c.get('type') == 'text']
            structured = json.loads(blocks[0]) if len(blocks) == 1 else None
        if not isinstance(structured, dict):
            raise ValueError('JEV_MCP_STRUCTURED_CONTENT_MISSING')
        return structured if 'code' in structured else {'code': 0, 'data': structured}


def hypothesis_options(hand, conflicts):
    options = []
    for index in sorted({i['message_index'] for i in conflicts if i['message_index'] >= 0}):
        msg = hand['messages'][index]
        for previous in hand['messages'][:index]:
            if previous['raw_text'] == msg['raw_text']:
                options.append({'decision': 'LIKELY_STALE_DUPLICATE', 'target_rows': msg['source_row_ids'],
                                'evidence_rows': previous['source_row_ids'], 'reason': 'Repeated observed text; candidate requires independent rule/pixel proof'})
                break
        if msg['amount'] is None and msg['action'] in {'SMALL_BLIND', 'BIG_BLIND'}:
            level = hand.get('blind_level', {})
            amount = level.get(msg['action'].lower())
            if amount is not None:
                options.append({'decision': 'LIKELY_MISSING_AMOUNT', 'target_rows': msg['source_row_ids'],
                                'candidate_amount': str(amount), 'evidence_rows': level.get('source_row_ids', []),
                                'reason': 'Explicit observed blind-level candidate'})
    return options


class JevResolver:
    def __init__(self, client=None, cache_dir=None, client_identity='UNAVAILABLE'):
        self.client, self.client_identity = client, client_identity
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.live_calls, self.cache_hits, self.latencies, self.errors = 0, 0, [], 0
        self.consecutive_errors = 0
        self.circuit_open = False
        self.success_latencies = []
        self.request_locks = {}
        self.failed_requests = {}

    def resolve(self, hand, initial):
        from parser_core.poker_logic_engine import VERSION as logic_version
        from parser_core.poker_math_engine import VERSION as math_version
        request = {'hand': hand, 'player_states': initial['players'], 'street': initial['street'],
                   'board': initial['board'], 'showdown': initial['showdown'], 'math': initial['math'],
                   'conflicts': initial['issues'], 'logic_version': logic_version,
                   'math_version': math_version, 'resolver_version': VERSION, 'prompt': PROMPT,
                   'client_identity': self.client_identity}
        if hand.get('diagnostic_packet') is not None:
            request['diagnostic_packet'] = copy.deepcopy(hand['diagnostic_packet'])
        # Timings do not affect the semantic request identity.
        request['math'] = {k: v for k, v in request['math'].items() if k != 'elapsed_ms'}
        digest = hashlib.sha256(canonical(request).encode('utf-8')).hexdigest()
        with self.lock:
            request_lock = self.request_locks.setdefault(digest, threading.Lock())
        with request_lock:
            if digest in self.failed_requests:
                return dict(copy.deepcopy(self.failed_requests[digest]), live_call=False, replayed_failure=True)
            result = self._resolve_request(hand, initial, request, digest)
            if result.get('rejection') == 'JEV_TRANSPORT_OR_SCHEMA_ERROR':
                self.failed_requests[digest] = copy.deepcopy(result)
            return result

    def _resolve_request(self, hand, initial, request, digest):
        result = {'request_hash': digest, 'accepted': False, 'affected_rows': []}
        cache_path = self.cache_dir / (digest+'.json') if self.cache_dir else None
        if cache_path and cache_path.exists():
            envelope = json.loads(cache_path.read_text(encoding='utf-8'))
            if envelope['request_hash'] != digest or envelope['request'] != request:
                return dict(result, rejection='CACHE_IDENTITY_MISMATCH')
            response = envelope['response']
            with self.lock:
                self.cache_hits += 1
            result['cache_hit'] = True
        elif self.client is None:
            return dict(result, rejection='JEV_UNAVAILABLE', live_call=False)
        else:
            with self.lock:
                if self.circuit_open:
                    return dict(result, rejection='JEV_SERVICE_CIRCUIT_OPEN_AFTER_REPEATED_FAILURES', live_call=False)
            if isinstance(self.client, JevHTTPClient):
                try:
                    self.client.prepare_request(request)
                except ValueError as exc:
                    return dict(result, rejection=str(exc), live_call=False)
            began = time.perf_counter()
            with self.lock:
                self.live_calls += 1
            try:
                response = self.client(copy.deepcopy(request))
                canonical(response)
            except Exception as exc:
                with self.lock:
                    self.errors += 1
                    self.consecutive_errors += 1
                    if self.consecutive_errors >= 3:
                        self.circuit_open = True
                    self.latencies.append((time.perf_counter()-began)*1000)
                safe_code = str(exc) if re.fullmatch(r'JEV_[A-Z_]+(?:\d+)?', str(exc)) else None
                return dict(result, rejection='JEV_TRANSPORT_OR_SCHEMA_ERROR', error_type=type(exc).__name__,
                            error_code=safe_code, api_error_detail=getattr(exc, 'api_error_detail', None),
                            latency_ms=(time.perf_counter()-began)*1000, live_call=True)
            elapsed = (time.perf_counter()-began)*1000
            with self.lock:
                self.latencies.append(elapsed)
                self.success_latencies.append(elapsed)
                self.consecutive_errors = 0
            result.update(live_call=True, latency_ms=elapsed, cache_hit=False)
            if cache_path:
                temp = cache_path.with_suffix('.'+str(threading.get_ident())+'.tmp')
                temp.write_text(canonical({'request_hash': digest, 'request': request, 'response': response}), encoding='utf-8')
                temp.replace(cache_path)
        result['jev_decision'] = response
        if not isinstance(response, dict) or isinstance(response.get('confidence'), bool) or not isinstance(response.get('confidence'), (int, float)) or not 0 <= response['confidence'] <= 1:
            return dict(result, rejection='INVALID_STRUCTURED_DECISION')
        result.update(jev_confidence=response['confidence'], jev_evidence=response.get('evidence_rows', []))
        candidate, rejection = candidate_view(hand, initial, response)
        if rejection:
            return dict(result, rejection=rejection)
        result.update(candidate_hand=candidate, affected_rows=response['target_rows'])
        return result


def candidate_view(hand, initial, decision):
    """Only bounded, source-supported transformations. Original hand is immutable."""
    kind, targets = decision.get('decision'), decision.get('target_rows')
    evidence_rows = decision.get('evidence_rows', [])
    if kind in {'UNRESOLVED', 'CARD_EVIDENCE_CONFLICT'} and targets == [] and evidence_rows == []:
        return None, 'JEV_UNRESOLVED' if kind == 'UNRESOLVED' else 'CARD_EVIDENCE_CONFLICT'
    if (not isinstance(targets, list) or not targets or not all(isinstance(r, str) for r in targets)
            or len(set(targets)) != len(targets) or not isinstance(evidence_rows, list)
            or not all(isinstance(r, str) for r in evidence_rows)):
        return None, 'INVALID_TARGET_ROWS'
    indices = [i for i, m in enumerate(hand['messages']) if set(m['source_row_ids']) & set(targets)]
    if not indices or set(targets) != {r for i in indices for r in hand['messages'][i]['source_row_ids']}:
        return None, 'TARGET_MUST_REFERENCE_COMPLETE_MESSAGES'
    candidate = copy.deepcopy(hand)
    if kind == 'LIKELY_STALE_DUPLICATE':
        if len(indices) != 1:
            return None, 'ONE_DUPLICATE_AT_A_TIME'
        index = indices[0]
        msg = hand['messages'][index]
        # Equal repeated actions can be legitimate. Require an independently detected
        # rule conflict AND identical pixel evidence to an earlier observed message.
        conflicts = {i['code'] for i in initial['issues'] if i['message_index'] == index}
        if not conflicts & {'CHAT_OVERLAP_DUPLICATE', 'INVALID_RAISE', 'ACTION_AFTER_FOLD', 'ACTION_AFTER_ALLIN', 'ACTION_AFTER_HAND_END', 'OLD_WIN', 'STALE_BLIND_OR_ANTE'}:
            return None, 'NO_INDEPENDENT_DUPLICATE_CONFLICT'
        fingerprint = [r.get('row_pixel_sha256') for r in msg['source_rows']]
        match = any(old['raw_text'] == msg['raw_text'] and old['action'] == msg['action'] and
                    old.get('player') == msg.get('player') and fingerprint and all(fingerprint) and
                    [r.get('row_pixel_sha256') for r in old['source_rows']] == fingerprint
                    and set(old['source_row_ids']) <= set(decision.get('evidence_rows', []))
                    for old in hand['messages'][:index])
        if not match:
            return None, 'DUPLICATE_PIXEL_AND_ROW_EVIDENCE_MISSING'
        candidate['excluded_message_indices'] = [index]
    elif kind == 'LIKELY_MISSING_AMOUNT':
        if len(indices) != 1:
            return None, 'ONE_AMOUNT_AT_A_TIME'
        msg = candidate['messages'][indices[0]]
        if msg['action'] not in {'SMALL_BLIND', 'BIG_BLIND'} or msg['amount'] is not None:
            return None, 'ONLY_OBSERVED_MISSING_BLIND_AMOUNT_SUPPORTED'
        level = hand.get('blind_level', {})
        expected = level.get(msg['action'].lower())
        evidence = level.get('source_row_ids', [])
        if level.get('source_type') != 'OBSERVED' or not evidence or not set(evidence) <= set(decision.get('evidence_rows', [])) or expected is None or str(expected) != str(decision.get('candidate_amount')):
            return None, 'OBSERVED_BLIND_LEVEL_EVIDENCE_MISSING'
        all_rows = {r for m in hand['messages'] for r in m['source_row_ids']}
        if not set(evidence) <= all_rows:
            return None, 'BLIND_EVIDENCE_OUTSIDE_HAND'
        observed_levels = []
        for m in hand['messages']:
            if set(m['source_row_ids']) & set(evidence):
                for match in re.finditer(r'(?:blind(?: level)?|блайнды)\s*[:=]?\s*(\d+(?:[ \u00a0]\d{3})*)\s*/\s*(\d+(?:[ \u00a0]\d{3})*)', m['raw_text'], re.I):
                    observed_levels.append(tuple(v.replace(' ', '').replace('\u00a0', '') for v in match.groups()))
        pair = (str(level.get('small_blind')), str(level.get('big_blind')))
        if observed_levels != [pair]:
            return None, 'RAW_BLIND_LEVEL_NOT_PROVEN'
        msg['amount'] = str(expected)
        msg['field_sources']['amount'] = 'JEV_INFERRED'
    elif kind == 'LIKELY_MULTILINE_CONTINUATION':
        if len(indices) != 2 or indices[1] != indices[0]+1:
            return None, 'MULTILINE_MUST_BE_ADJACENT'
        a, b = (hand['messages'][i] for i in indices)
        physical = a['source_rows']+b['source_rows']
        binding = [(r.get('table_id'), r.get('session_id', r.get('table_session_id')), r.get('epoch')) for r in physical]
        # Synthetic clients can omit all bindings; real observed rows carry
        # them. Mixed/missing real bindings and cross-epoch joins are forbidden.
        if len(set(binding)) != 1 or (any(any(v is not None for v in item) for item in binding) and
                                     any(v is None for v in binding[0])):
            return None, 'MULTILINE_SOURCE_BOUNDARY_MISMATCH'
        if binding[0][0] is not None and (str(binding[0][0]) != hand.get('table_id') or str(binding[0][1]) != hand.get('session_id')):
            return None, 'MULTILINE_SOURCE_HAND_BINDING_MISMATCH'
        if not set(targets) <= set(evidence_rows):
            return None, 'MULTILINE_SOURCE_EVIDENCE_MISSING'
        combined = make_message(a['source_rows']+b['source_rows'])
        if a['action'] not in {'SMALL_BLIND', 'BIG_BLIND', 'UNCALLED_RETURN', 'RAISE', 'CALL'} or b['action'] != 'UNKNOWN_EVENT' or combined['action'] != a['action'] or (combined['amount'] is None and not combined['all_in']):
            return None, 'MULTILINE_GRAMMAR_NOT_PROVEN'
        combined['source_type'] = 'JEV_INFERRED'
        candidate['messages'][indices[0]:indices[1]+1] = [combined]
    else:
        # Identity merges and previous-hand bleed need external binding/boundary proof.
        return None, 'UNSUPPORTED_OR_UNPROVEN_HYPOTHESIS'
    candidate['inferences'] = [{'source_type': 'JEV_INFERRED', 'decision': copy.deepcopy(decision), 'affected_rows': targets}]
    return candidate, None
