"""Offline recovery provisioning. Never edits the original production checkout."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or ':' in value or '\\' in value:
        raise ValueError('Unsafe relative archive/Git path: ' + value)
    return Path(*path.parts)


def valid_file(path, row):
    return path.is_file() and path.stat().st_size == row['bytes'] and digest(path) == row['sha256']


def contained_plain_tree(root, target):
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Recovery target resolves outside owned destination')
    for path in (target, *target.parents):
        if path == root.parent:
            break
        if path.is_symlink() or path.is_junction():
            raise ValueError('Reparse point in recovery target: ' + str(path))
    if target.exists():
        for current, directories, files in os.walk(target, followlinks=False):
            for name in directories + files:
                path = Path(current) / name
                if path.is_symlink() or path.is_junction():
                    raise ValueError('Unexpected recovery reparse point: ' + str(path))


class Recovery:
    def __init__(self, repo, destination):
        self.repo = Path(repo).resolve(strict=True)
        self.dest = Path(destination).resolve(strict=True)
        marker = self.dest / '.poker-recovery-owned'
        if marker.read_text(encoding='utf-8').strip() != 'poker-recovery-v1':
            raise ValueError('Unknown recovery destination')
        if self.dest == self.repo or self.repo.is_relative_to(self.dest):
            raise ValueError('Recovery destination must not contain original checkout')
        self.cache = self.dest / '.recovery-cache'
        self.cache.mkdir(exist_ok=True)
        self.manifest = read_json(self.repo / 'recovery/recovery_manifest.json')
        self.lock = read_json(self.blob(self.manifest['source_lock']))

    def blob(self, row):
        rel = relative_path(row['path'])
        local = self.repo / rel
        if valid_file(local, row):
            return local
        cached = self.cache / ('blob-' + row['sha256'])
        if valid_file(cached, row):
            return cached
        for attempt in range(1, 4):
            temporary = cached.with_suffix('.partial')
            with temporary.open('wb') as output, (self.cache / 'git-fetch.stderr').open('wb') as error:
                process = subprocess.Popen(
                    ['git', '-c', 'http.version=HTTP/1.1', 'cat-file', 'blob', 'HEAD:' + rel.as_posix()],
                    cwd=self.repo, stdout=output, stderr=error,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                try:
                    code = process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    process.wait(timeout=15)
                    code = -1
            if code == 0 and valid_file(temporary, row):
                temporary.replace(cached)
                return cached
            print(f"Retry {attempt}/3: {rel.as_posix()}", flush=True)
        raise RuntimeError('Cannot obtain verified Git blob: ' + rel.as_posix())

    def catalog(self, bundle):
        row = {'path': bundle['files_catalog'], 'sha256': bundle['files_catalog_sha256'],
               'bytes': bundle['files_catalog_bytes']}
        content = json.loads(gzip.decompress(self.blob(row).read_bytes()))
        return content['files'] if isinstance(content, dict) else content

    def unpack(self, bundle, target):
        rows = self.catalog(bundle)
        target.mkdir(parents=True, exist_ok=True)
        contained_plain_tree(self.dest, target)
        if all(valid_file(target / relative_path(row['path']), row) for row in rows):
            print('Verified existing bundle: ' + bundle['name'], flush=True)
            return rows
        archive = self.cache / (bundle['name'] + '.zip')
        if not archive.is_file() or digest(archive) != bundle['archive_sha256']:
            with archive.with_suffix('.partial').open('wb') as output:
                for part in bundle['parts']:
                    with self.blob(part).open('rb') as source:
                        shutil.copyfileobj(source, output, 1024 * 1024)
            temporary = archive.with_suffix('.partial')
            if digest(temporary) != bundle['archive_sha256']:
                raise ValueError('Rejoined archive SHA mismatch: ' + bundle['name'])
            temporary.replace(archive)
        expected = {row['path']: row for row in rows}
        with zipfile.ZipFile(archive) as zipped:
            if set(zipped.namelist()) != set(expected):
                raise ValueError('Archive file list differs from catalog')
            for entry in zipped.infolist():
                path = target / relative_path(entry.filename)
                if not path.resolve().is_relative_to(target.resolve()):
                    raise ValueError('Archive escaped destination')
                if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Archive symlink rejected')
                row = expected[entry.filename]
                if valid_file(path, row):
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(entry) as source, path.open('wb') as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                if not valid_file(path, row):
                    raise ValueError('Extracted file SHA mismatch: ' + entry.filename)
        print(f"Restored {bundle['name']}: {len(rows)} exact files", flush=True)
        return rows


def check_original_lock(recovery, source):
    lock = recovery.lock
    if digest(source / 'production_manifest.json') != lock['production_manifest_sha256']:
        raise ValueError('Original production manifest differs from source lock')
    manifest = read_json(source / 'production_manifest.json')
    if manifest['hashes'] != lock['registered_hashes'] or manifest['production_version'] != lock['production_version']:
        raise ValueError('Original production registration differs from source lock')
    for rel, expected in lock['registered_hashes'].items():
        if digest(source / relative_path(rel)) != expected:
            raise ValueError('Original algorithm/config/model pin differs from lock: ' + rel)
    for row in lock['frozen_runtime_inputs']:
        if not valid_file(source / relative_path(row['path']), row):
            raise ValueError('Original frozen data differs from source lock: ' + row['path'])
    db = source / 'data/players.sqlite'
    if not lock['database_policy']['included'] and db.exists():
        raise ValueError('Nickname database must be excluded from published source bundle')


def rebind_production(recovery, source, production, runtime):
    """Whitelist-only path migration; parser/model/CSV/selection bytes stay exact."""
    original = read_json(source / 'production_manifest.json')
    changed = []
    train_rel = 'runs/card_train_smoke16_test_20260930/split_v2/train_manifest.jsonl'
    old_train = source / train_rel
    rows = [json.loads(line) for line in old_train.read_text(encoding='utf-8').splitlines()]
    for row in rows:
        old = str(row['tile_png']).replace('\\', '/')
        marker = '/runs/'
        if marker not in old:
            raise ValueError('Unexpected original tile path')
        relative = 'runs/' + old.split(marker, 1)[1]
        actual = production / relative
        pristine = source / relative
        if not pristine.is_file() or digest(actual) != digest(pristine):
            raise ValueError('Rebound tile is not byte-identical: ' + relative)
        before = copy.deepcopy(row)
        row['tile_png'] = str(actual)
        comparison = copy.deepcopy(row)
        comparison['tile_png'] = before['tile_png']
        if comparison != before:
            raise ValueError('Non-path train row mutation')
    train = production / train_rel
    train.write_text(''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
                             for row in rows), encoding='utf-8')
    changed.append(train_rel)
    train_hash = digest(train)
    edits = {
        'configs/card_hybrid_local.json': {
            'train_manifest': str(train), 'train_manifest_sha256': train_hash,
            'selection_file': str(production / 'runs/card_train_smoke16_test_20260930/stage2_template/hybrid_validation_selection.json')},
        'configs/rank_ncc_production.json': {'train_manifest': str(train), 'train_manifest_sha256': train_hash},
        'configs/suit_backup_local.json': {'train_manifest': str(train), 'train_manifest_sha256': train_hash,
                                          'validation_results': str(production / 'runs/suit_template_experiment_20260930/results.csv')},
        'configs/ocr_tiny_local.json': {'python': str(runtime / 'ocr/Scripts/python.exe'),
                                      'model_dir': str(runtime / 'paddlex-cache/official_models/PP-OCRv6_tiny_rec')},
    }
    field_changes = {}
    for rel, fields in edits.items():
        old = read_json(source / rel)
        new = copy.deepcopy(old)
        if not set(fields) <= set(old):
            raise ValueError('Unexpected config schema: ' + rel)
        new.update(fields)
        if {k: v for k, v in old.items() if k not in fields} != {k: v for k, v in new.items() if k not in fields}:
            raise ValueError('Non-path configuration mutation')
        write_json(production / rel, new)
        field_changes[rel] = {k: {'before': old[k], 'after': new[k]} for k in fields}
        changed.append(rel)
    manifest = copy.deepcopy(original)
    for rel in changed:
        if rel in manifest['hashes']:
            manifest['hashes'][rel] = digest(production / rel)
    ocr = read_json(production / 'configs/ocr_tiny_local.json')
    manifest['external_runtime']['OCR'] = ocr
    manifest['external_runtime']['node'] = str(runtime / 'node/bin/node.exe')
    for item in manifest['data_dependencies']:
        if item['config'] == 'configs/confirmed_nickname_associations.json':
            item['path'] = str(production / 'configs/confirmed_nickname_associations_source.jsonl')
        elif item['key'] == 'train_manifest':
            item['path'] = str(train)
            item['sha256'] = train_hash
        else:
            cfg = read_json(production / item['config'])
            item['path'] = cfg[item['key']]
    write_json(production / 'production_manifest.json', manifest)
    unchanged = [rel for rel, expected in original['hashes'].items()
                 if rel not in changed and digest(production / rel) == expected]
    if len(unchanged) != len(original['hashes']) - len(set(changed) & set(original['hashes'])):
        raise ValueError('Production source/model bytes changed')
    for rel in original['hashes']:
        if Path(rel).suffix in ('.py', '.mjs', '.dll', '.png') and digest(production / rel) != original['hashes'][rel]:
            raise ValueError('Algorithm/native/model byte change: ' + rel)
    return {'classification': 'RECOVERY_PATH_BINDINGS_ONLY_NO_NEW_GOLDEN',
            'original_version': original['production_version'], 'original_manifest_sha256': digest(source / 'production_manifest.json'),
            'generated_manifest_sha256': digest(production / 'production_manifest.json'),
            'config_fields': field_changes, 'train_rows': len(rows), 'unchanged_registered_files': len(unchanged),
            'changed_relative_files': changed, 'all_python_javascript_native_pixel_bytes_unchanged': True,
            'all_nonpath_config_values_unchanged': True}


def create_junction(link, target):
    if link.exists():
        if link.resolve() != target.resolve():
            raise ValueError('Unexpected existing node_modules link')
        return
    env = os.environ.copy()
    env['RECOVERY_LINK'] = str(link)
    env['RECOVERY_TARGET'] = str(target)
    subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
                    'New-Item -ItemType Junction -Path $env:RECOVERY_LINK -Target $env:RECOVERY_TARGET | Out-Null'],
                   env=env, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


def verify(destination):
    receipt = read_json(destination / 'recovery_receipt.json')
    failures = []
    expected_paths = {item['path'] for item in receipt['immutable_files']}
    importable = {'.py', '.pyc', '.pyo', '.pth', '.pyd', '.dll', '.exe', '.mjs', '.js', '.node', '.so', '.zip'}
    allowed_link = destination / 'production/scripts/node_modules'
    if allowed_link.resolve() != (destination / 'runtime/node/node_modules').resolve():
        raise ValueError('Restored Excel package binding mismatch')
    for root_name in ('runtime', 'source-original', 'production'):
        for current, directories, files in os.walk(destination / root_name, followlinks=False):
            for name in list(directories):
                path = Path(current) / name
                if path.is_symlink() or path.is_junction():
                    if path != allowed_link:
                        raise ValueError('Unexpected reparse point in recovered executable tree')
                    directories.remove(name)
            for name in files:
                path = Path(current) / name
                rel = path.relative_to(destination).as_posix()
                if path.is_symlink() or path.is_junction():
                    raise ValueError('Unexpected recovery file reparse point')
                if path.suffix.lower() in importable and rel not in expected_paths:
                    raise ValueError('Unexpected importable recovery file: ' + rel)
    for item in receipt['immutable_files']:
        if not valid_file(destination / relative_path(item['path']), item):
            failures.append(item['path'])
    if failures:
        raise ValueError('Recovered immutable file mismatch: ' + repr(failures[:10]))
    production = destination / 'production'
    manifest = read_json(production / 'production_manifest.json')
    for rel, expected in manifest['hashes'].items():
        if digest(production / rel) != expected:
            raise ValueError('Generated production manifest mismatch: ' + rel)
    database = production / 'data/players.sqlite'
    if database.exists():
        with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
            if connection.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                raise ValueError('Registry SQLite integrity failure')
    cache_model = destination / 'runtime/paddlex-cache/official_models/PP-OCRv6_tiny_rec/inference.pdiparams'
    if digest(cache_model) != manifest['external_runtime']['OCR']['weights_sha256']:
        raise ValueError('Restored OCR weight SHA mismatch')
    if (production / 'scripts/node_modules').resolve() != (destination / 'runtime/node/node_modules').resolve():
        raise ValueError('Restored Excel package binding mismatch')
    registry = 'local mutable registry integrity OK' if database.exists() else 'registry absent as requested'
    print(f"RECOVERY VERIFY PASS: {len(receipt['immutable_files'])} immutable files; {registry}", flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    recovery = Recovery(args.repo, args.destination)
    marker = recovery.dest / 'recovery_receipt.json'
    if args.verify_only or marker.exists():
        verify(recovery.dest)
        return
    runtime = recovery.dest / 'runtime'
    immutable = []
    for bundle in recovery.manifest['bundles']:
        for row in recovery.unpack(bundle, runtime):
            immutable.append(dict(row, path='runtime/' + row['path']))
    for name in ('main', 'ocr'):
        config = runtime / name / 'pyvenv.cfg'
        config.write_text('home = ' + str(runtime / 'python') + '\ninclude-system-site-packages = false\nversion = 3.13.12\n', encoding='utf-8')
        immutable.append({'path': config.relative_to(recovery.dest).as_posix(), 'bytes': config.stat().st_size, 'sha256': digest(config)})
    source = recovery.dest / 'source-original'
    source_rows = recovery.unpack(recovery.manifest['production_bundle'], source)
    check_original_lock(recovery, source)
    for row in source_rows:
        immutable.append(dict(row, path='source-original/' + row['path']))
    if 'review_sources_bundle' in recovery.manifest:
        evidence = recovery.dest / 'review-source-evidence'
        evidence_rows = recovery.unpack(recovery.manifest['review_sources_bundle'], evidence)
        locked = {row['path']: row for row in recovery.lock['review_source_inputs']}
        if {row['path'] for row in evidence_rows} != set(locked):
            raise ValueError('Review source archive differs from source lock')
        for row in evidence_rows:
            if not valid_file(evidence / relative_path(row['path']), locked[row['path']]):
                raise ValueError('Review source pin mismatch')
            immutable.append(dict(row, path='review-source-evidence/' + row['path']))
    production = recovery.dest / 'production'
    if production.exists():
        raise ValueError('Incomplete production destination exists; choose a fresh owned destination')
    shutil.copytree(source, production)
    migration = rebind_production(recovery, source, production, runtime)
    create_junction(production / 'scripts/node_modules', runtime / 'node/node_modules')
    for path in production.rglob('*'):
        if path.is_file() and path.relative_to(production).as_posix() != 'data/players.sqlite':
            # Do not traverse the junction: runtime catalog already covers it.
            if 'node_modules' not in path.relative_to(production).parts:
                immutable.append({'path': path.relative_to(recovery.dest).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest(path)})
    database = production / 'data/players.sqlite'
    receipt = {'schema': 'production-recovery-install-v1', 'destination': str(recovery.dest),
               'runtime_python': str(runtime / 'main/Scripts/python.exe'), 'production_root': str(production),
               'source_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=recovery.repo).decode().strip(),
               'source_lock': recovery.manifest['source_lock'], 'migration': migration,
               'initial_registry_sha256': digest(database) if database.exists() else None,
               'registry_initial_state': 'ABSENT_EMPTY_REGISTRY_BY_USER_REQUEST', 'registry_mutable_after_restore': True,
               'immutable_files': immutable, 'elapsed_seconds': time.monotonic() - started}
    write_json(marker, receipt)
    verify(recovery.dest)
    print('RESTORE COMPLETE: ' + str(recovery.dest), flush=True)


if __name__ == '__main__':
    main()
