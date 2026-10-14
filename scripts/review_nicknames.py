"""Local human nickname review. Enter through run_production.py."""
import argparse
import hashlib
import hmac
import json
import secrets
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from parser_core.nickname_review import PAGE, write_cases, prepare_cases, unreviewed_cases, group_spelling_cases
from parser_core.player_registry import DEFAULT_DATABASE, PlayerRegistry, digest

ROOT = Path(__file__).resolve().parents[1]


def launch_cached_review(source):
    """Open pending human review after the normal writer has finalized its files."""
    source = Path(source).resolve()
    cases = source / 'nickname_review_cases.json'
    if not cases.is_file():
        return None
    manifest = json.loads(cases.read_text('utf-8'))
    prepared, _ = prepare_cases(manifest['cases'])
    database = Path(manifest.get('database') or DEFAULT_DATABASE).resolve()
    pending = unreviewed_cases(prepared, PlayerRegistry(database))
    if not pending:
        return None
    output = source.parent / (source.name + '_nickname_review_' +
                              time.strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(3))
    command = [sys.executable, str(ROOT / 'run_production.py'), '--review-nicknames',
               str(source), '--output', str(output), '--cases-file', str(cases),
               '--player-database', str(database), '--open-browser']
    log = output.with_suffix('.log')
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=(getattr(subprocess, 'CREATE_NO_WINDOW', 0) |
                           getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)))
    receipt = dict(status='STARTING', pid=process.pid, pending=len(pending),
                   output=str(output), log=str(log), cases=str(cases))
    (source / 'nickname_review_launch.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps(dict(nickname_review=receipt), ensure_ascii=False), flush=True)
    return receipt


def verify_case(case, source):
    if case.get('error'):
        raise ValueError('Review source has an error')
    if digest({k: v for k, v in case.items() if k != 'case_id'}) != case['case_id']:
        raise ValueError('Review case was modified')
    source = Path(source).resolve()
    for name, expected in case['source_pins'].items():
        path = Path(name).resolve()
        if path.parent != source or path.name not in ('seat_forced_inputs.jsonl', 'physical_rows.jsonl'):
            raise ValueError('Review source path outside declared input')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Исходные данные изменились; создайте очередь проверки заново')


def load_cached_cases(path, source):
    manifest=json.loads(Path(path).read_text('utf-8'))
    if manifest.get('schema')!='poker.nickname-review.v1' or Path(manifest['source']).resolve()!=Path(source).resolve():
        raise ValueError('Cached review source/schema mismatch')
    for case in manifest['cases']:
        verify_case(case,source)
    return manifest


def confirm_review_cases(cases, case_id, canonical, selected, source, database):
    """Confirm displayed repetitions, retaining each hand's original receipt."""
    registry=PlayerRegistry(database)
    display=next((c for c in group_spelling_cases(unreviewed_cases(cases,registry))
                  if c['case_id']==case_id),None)
    if display is None:
        # Preserve editing an existing individual confirmation. Do not add
        # hidden siblings to an update of an already completed review.
        display=next((c for c in cases if c['case_id']==case_id),None)
    if display is None:
        raise ValueError('Unknown review case')
    choices={o['id']:(o['kind'],o['text']) for o in display['observations']}
    if (not isinstance(selected,list) or len(selected)!=len(set(selected)) or
            any(key not in choices for key in selected)):
        raise ValueError('Invalid review observation selection')
    wanted={choices[key] for key in selected}
    ids=display.get('review_group',{}).get('case_ids',[case_id])
    members=[next(c for c in cases if c['case_id']==key) for key in ids]
    # Verify every child before the first DB write. A group cannot extend or
    # substitute any child's source bounds, native crop or CHAT evidence.
    for case in members:
        verify_case(case,source)
    results=[]
    for case in members:
        result=PlayerRegistry(database).confirm(case,canonical,
            [o['id'] for o in case['observations'] if (o['kind'],o['text']) in wanted])
        results.append(result)
    return dict(results[-1],confirmed_cases=len(results),review_ids=[r['review_id'] for r in results],
                conflicts=[dict(kind=kind,raw=raw) for kind,raw in sorted(
                    {(c['kind'],c['raw']) for r in results for c in r['conflicts']})])


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, manifest, database, output):
        super().__init__(address, ReviewHandler)
        self.source_cases=list(manifest['cases'])
        prepared,audit=prepare_cases(self.source_cases)
        self.manifest = dict(manifest,cases=prepared)
        self.database = Path(database).resolve()
        self.output = Path(output).resolve()
        (self.output/'nickname_review_queue_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.job = None


class ReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _authorized(self):
        host = '127.0.0.1:' + str(self.server.server_port)
        origin = self.headers.get('Origin')
        return (self.headers.get('Host') == host and
                (origin is None or origin == 'http://' + host) and
                hmac.compare_digest(self.headers.get('X-Review-Token', ''), self.server.token))

    def _send(self, data, status=200, html=False):
        body = data.encode('utf-8') if html else json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8' if html else 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/':
            if self.headers.get('Host') != '127.0.0.1:' + str(self.server.server_port):
                return self._send(dict(error='Forbidden'), 403)
            return self._send(PAGE, html=True)
        if not self._authorized():
            return self._send(dict(error='Forbidden'), 403)
        if self.path == '/cases':
            registry = PlayerRegistry(self.server.database)
            latest = {}
            for review in registry.reviews.values():
                prior = latest.get(review['case_id'])
                if prior is None or prior['confirmed_at'] < review['confirmed_at']:
                    latest[review['case_id']] = review
            confirmations = {key:dict(canonical=r['canonical'],selected=[o['id'] for o in r['observations']])
                             for key,r in latest.items()}
            return self._send(dict(self.server.manifest,
                cases=group_spelling_cases(unreviewed_cases(self.server.manifest['cases'],registry)),confirmations=confirmations))
        if self.path == '/job':
            return self._send(self.server.job or dict(status='IDLE'))
        return self._send(dict(error='Not found'), 404)

    def do_POST(self):
        if not self._authorized():
            return self._send(dict(error='Forbidden'), 403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 65536 or self.headers.get('Content-Type') != 'application/json':
                raise ValueError('Invalid request body')
            request = json.loads(self.rfile.read(length))
            if self.path == '/confirm':
                with self.server.lock:
                    result = confirm_review_cases(self.server.manifest['cases'],request.get('case_id'),
                        request.get('canonical'),request.get('selected'),self.server.manifest['source'],self.server.database)
                return self._send(result)
            if self.path == '/reprocess':
                with self.server.lock:
                    if self.server.job and self.server.job['status'] == 'RUNNING':
                        raise ValueError('Пересчёт уже выполняется')
                    registry = PlayerRegistry(self.server.database)
                    if not any(r['context']==c['context'] and r['source_pins']==c['source_pins']
                               for r in registry.active_reviews.values() for c in self.server.source_cases):
                        raise ValueError('Сначала подтвердите ник')
                    for case in self.server.source_cases:
                        if not case.get('error'):
                            verify_case(case, self.server.manifest['source'])
                    replay = self.server.output / ('review_replay_' + time.strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(3))
                    self.server.job = dict(status='RUNNING', output=str(replay), scope='SAVED_NO_VIDEO_NO_OCR')
                    threading.Thread(target=self._replay, args=(replay,), daemon=True).start()
                return self._send(self.server.job, 202)
            return self._send(dict(error='Not found'), 404)
        except (ValueError, KeyError, TypeError, OSError) as error:
            return self._send(dict(error=str(error)), 400)

    def _replay(self, output):
        log = output.with_suffix('.log')
        try:
            command = [sys.executable, str(ROOT / 'run_production.py'), '--replay-saved-quality',
                       self.server.manifest['source'], '--output', str(output),
                       '--player-database', str(self.server.database)]
            with log.open('w', encoding='utf-8') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.server.job = dict(status='COMPLETE' if result.returncode == 0 else 'FAILED',
                exit_code=result.returncode, output=str(output), log=str(log), scope='SAVED_NO_VIDEO_NO_OCR')
        except Exception as error:
            self.server.job = dict(status='FAILED', error=str(error), output=str(output), log=str(log))


def main():
    parser = argparse.ArgumentParser(description='Проверка NICK/CHAT кропов и база игроков')
    parser.add_argument('--review-nicknames', required=True, metavar='SOURCE_PIPELINE')
    parser.add_argument('--output', required=True)
    parser.add_argument('--player-database', default=str(DEFAULT_DATABASE))
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--table', type=int)
    parser.add_argument('--hand')
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--cases-file', help='Reuse source-verified review cases to reopen the page')
    parser.add_argument('--open-browser', action='store_true', help='Open the verified local review page')
    args = parser.parse_args()
    source, output = Path(args.review_nicknames).resolve(), Path(args.output).resolve()
    if output == source or source in output.parents:
        raise ValueError('Review output must be separate from immutable input')
    output.mkdir(parents=True, exist_ok=False)
    if args.cases_file:
        if args.table is not None or args.hand is not None:
            raise ValueError('Cached review cases already define their hand scope')
        manifest=load_cached_cases(args.cases_file,source)
        manifest['database']=str(Path(args.player_database).resolve())
        path=output/'nickname_review_cases.json'
        path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    else:
        from parser_core.simple_saved_policy import SimpleSavedPolicyProvider,saved_hand_bounds,hand_key
        records = [json.loads(t) for t in (source / 'logical_hand_analysis.jsonl').read_text('utf-8').splitlines() if t.strip()]
        full_bounds=saved_hand_bounds(records)
        if args.table is not None or args.hand is not None:
            records = [r for r in records if (args.table is None or
                int(str(r['observed_hand']['table_id']).removeprefix('T')) == args.table) and
                (args.hand is None or str(r['observed_hand']['hand_id']) == args.hand)]
        if not records:
            raise ValueError('No matching hand contexts')
        provider = SimpleSavedPolicyProvider.from_saved(source, records, player_database=args.player_database,
            bounds={hand_key(r['observed_hand']):full_bounds[hand_key(r['observed_hand'])] for r in records})
        path = write_cases(provider, source, output)
        manifest = json.loads(path.read_text('utf-8'))
    print(json.dumps(dict(cases=str(path), count=len(manifest['cases']), database=args.player_database), ensure_ascii=False), flush=True)
    if args.generate_only:
        return
    with ReviewServer(('127.0.0.1', args.port), manifest, args.player_database, output) as server:
        url = f'http://127.0.0.1:{server.server_port}/#{server.token}'
        (output / 'review_server.json').write_text(json.dumps(dict(url=url, pid=__import__('os').getpid()), indent=2), 'utf-8')
        print(url, flush=True)
        if args.open_browser:
            import webbrowser
            try:
                webbrowser.open(url)
            except Exception as error:
                print('Open review page manually:',url,repr(error),flush=True)
        server.serve_forever(poll_interval=.5)


if __name__ == '__main__':
    main()
