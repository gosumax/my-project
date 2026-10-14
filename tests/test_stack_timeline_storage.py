"""SSD allocation, fail-closed routing and connection-lifetime contracts."""
import json
import hashlib
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from parser_core import stack_timeline_live as live_module
from parser_core import stack_timeline_storage as storage


def output_dir(tmp_path, name):
    output = tmp_path / name
    (output / 'stack_pipeline').mkdir(parents=True)
    return output


def test_unique_owned_c_paths_and_real_source_inventory(tmp_path):
    first = output_dir(tmp_path, 'first')
    second = output_dir(tmp_path, 'second')
    a, receipt = storage.create_stack_timeline_storage(first)
    b, _ = storage.create_stack_timeline_storage(second)
    assert a != b and a.drive.upper() == b.drive.upper() == 'C:'
    assert a.parent.parent == storage.RUNTIME_ROOT.resolve()
    assert a.name == b.name == 'timeline.sqlite'
    assert not (first / storage.LOGICAL_DATABASE).exists()
    assert storage.resolve_stack_timeline_path(first) == a
    key = storage.stack_timeline_source_name(first)
    assert key == str(a) and first / key == a
    assert receipt['STACK_TIMELINE_DB_RESOLVED_PATH'] == str(a.resolve())
    assert receipt['STACK_TIMELINE_STORAGE_DRIVE'] == 'C:'
    assert receipt['STACK_TIMELINE_STORAGE_CLASS'] == 'SSD'


def test_legacy_source_locator_and_inventory_key_unchanged(tmp_path):
    source = output_dir(tmp_path, 'legacy')
    path = source / storage.LOGICAL_DATABASE
    path.write_bytes(b'legacy sealed bytes')
    assert storage.resolve_stack_timeline_path(source) == path.resolve()
    assert storage.stack_timeline_source_name(source) == storage.LOGICAL_DATABASE


@pytest.mark.parametrize('on_ssd', [False, True])
def test_source_seal_rechecks_actual_database_after_mutation(tmp_path, on_ssd):
    from parser_core.constant_stack_balance import SourceSealBatch

    source = output_dir(tmp_path, 'source')
    database = storage.create_stack_timeline_storage(source)[0] if on_ssd else source / storage.LOGICAL_DATABASE
    with sqlite3.connect(database) as connection:
        connection.execute('create table seal_fixture(value integer)')
        connection.execute('insert into seal_fixture values(1)')
    config = source / 'seal_fixture_config.json'
    config.write_bytes(b'{}')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    key = storage.stack_timeline_source_name(source)
    registry = SimpleNamespace(source=source, hashes={}, config_path=config,
        config_sha256=digest(config))
    inventory = SimpleNamespace(source=source, hashes={key: digest(database)}, native_who=None)
    batch = SourceSealBatch(registry, inventory, ())
    assert (str(database.resolve()), digest(database)) in batch.expected
    assert batch.begin()
    with sqlite3.connect(database) as connection:
        connection.execute('update seal_fixture set value=2')
    assert not batch.finish()


def test_declared_c_missing_owner_never_falls_back(tmp_path):
    source = output_dir(tmp_path, 'source')
    database, _ = storage.create_stack_timeline_storage(source)
    (source / storage.LOGICAL_DATABASE).write_bytes(b'wrong D database')
    owner = database.parent / 'storage_owner.json'
    held = owner.with_name('held_test_owner.json')
    owner.rename(held)
    try:
        with pytest.raises(FileNotFoundError):
            storage.resolve_stack_timeline_path(source)
    finally:
        held.rename(owner)


def test_runtime_directory_collision_does_not_reuse_or_touch_other_run(tmp_path, monkeypatch):
    first = output_dir(tmp_path, 'first')
    second = output_dir(tmp_path, 'second')
    a, receipt = storage.create_stack_timeline_storage(first)
    a.write_bytes(b'owned first-run evidence')
    monkeypatch.setattr(storage.uuid, 'uuid4', lambda: SimpleNamespace(hex=receipt['STACK_TIMELINE_RUNTIME_RUN_ID']))
    with pytest.raises(FileExistsError):
        storage.create_stack_timeline_storage(second)
    assert a.read_bytes() == b'owned first-run evidence'
    assert not (second / storage.LOGICAL_DATABASE).exists()


def test_unavailable_c_root_fails_before_sqlite_and_never_falls_back(tmp_path, monkeypatch):
    unavailable = tmp_path / 'not_a_directory'
    unavailable.write_bytes(b'leave this file untouched')
    monkeypatch.setattr(storage, 'RUNTIME_ROOT', unavailable)
    monkeypatch.setattr(live_module.sqlite3, 'connect', lambda *a, **k: pytest.fail('SQLite must not be opened'))
    with pytest.raises(FileExistsError):
        live_module.StackTimelineLive(tmp_path, video_sha256='a'*64, sessions={}, frames=1)
    assert unavailable.read_bytes() == b'leave this file untouched'
    assert not (tmp_path / storage.LOGICAL_DATABASE).exists()


def test_non_c_resolved_root_rejected(monkeypatch):
    monkeypatch.setattr(storage, 'RUNTIME_ROOT', Path(__file__).resolve().parents[1])
    with pytest.raises(ValueError, match='MUST_RESOLVE_TO_C'):
        storage._runtime_root()


def test_declared_c_missing_db_never_uses_legacy_decoy(tmp_path):
    source = output_dir(tmp_path, 'source')
    database, _ = storage.create_stack_timeline_storage(source)
    decoy = source / storage.LOGICAL_DATABASE
    decoy.write_bytes(b'wrong D database must never be selected')
    held = database.with_name('held_test_evidence.sqlite')
    database.rename(held)
    try:
        with pytest.raises(FileNotFoundError):
            storage.resolve_stack_timeline_path(source)
        assert decoy.read_bytes() == b'wrong D database must never be selected'
    finally:
        held.rename(database)


def test_declared_c_missing_receipt_never_falls_back(tmp_path):
    source = output_dir(tmp_path, 'source')
    database, receipt = storage.create_stack_timeline_storage(source)
    (source / 'run_metadata.json').write_text(json.dumps(receipt), encoding='utf-8')
    locator = source / storage.STORAGE_RECEIPT
    held = locator.with_name('held_test_receipt.json')
    locator.rename(held)
    try:
        with pytest.raises(ValueError, match='RECEIPT_MISSING'):
            storage.resolve_stack_timeline_path(source)
        assert database.exists()
    finally:
        held.rename(locator)


def test_foreign_run_receipt_rejected(tmp_path):
    first = output_dir(tmp_path, 'first')
    second = output_dir(tmp_path, 'second')
    _, receipt = storage.create_stack_timeline_storage(first)
    (second / storage.STORAGE_RECEIPT).write_text(json.dumps(receipt), encoding='utf-8')
    with pytest.raises(ValueError, match='BINDING_INVALID'):
        storage.resolve_stack_timeline_path(second)


def test_owner_conflict_and_metadata_path_conflict_rejected(tmp_path):
    source = output_dir(tmp_path, 'source')
    database, receipt = storage.create_stack_timeline_storage(source)
    meta = {**receipt, 'STACK_TIMELINE_DB_PATH': str(source / storage.LOGICAL_DATABASE)}
    (source / 'run_metadata.json').write_text(json.dumps(meta), encoding='utf-8')
    with pytest.raises(ValueError, match='METADATA_STORAGE_CONFLICT'):
        storage.resolve_stack_timeline_path(source)
    (source / 'run_metadata.json').write_text(json.dumps(receipt), encoding='utf-8')
    owner = database.parent / 'storage_owner.json'
    owner.write_text(json.dumps({**receipt, 'STACK_TIMELINE_RUNTIME_RUN_ID': '0'*32}), encoding='utf-8')
    with pytest.raises(ValueError, match='OWNER_CONFLICT'):
        storage.resolve_stack_timeline_path(source)


def test_constructor_sql_failure_closes_partial_connection(tmp_path, monkeypatch):
    connections = []
    original = sqlite3.connect
    class Failing(sqlite3.Connection):
        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError('injected initialization failure')
    def connect(*args, **kwargs):
        connection = original(*args, **kwargs, factory=Failing)
        connections.append(connection)
        return connection
    monkeypatch.setattr(live_module.sqlite3, 'connect', connect)
    with pytest.raises(sqlite3.OperationalError, match='initialization failure'):
        live_module.StackTimelineLive(tmp_path, video_sha256='a'*64, sessions={}, frames=1)
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
        connections[0].commit()
    assert not (tmp_path / storage.LOGICAL_DATABASE).exists()


@pytest.mark.parametrize('failure', ['flush', 'client_close'])
def test_finish_closes_db_and_files_even_when_another_resource_fails(tmp_path, failure):
    live = live_module.StackTimelineLive(tmp_path, video_sha256='a'*64, sessions={}, frames=1)
    def fail():
        raise RuntimeError('injected ' + failure)
    if failure == 'flush':
        live.flush = fail
    else:
        live.client = SimpleNamespace(close=fail)
    with pytest.raises(RuntimeError, match='injected ' + failure):
        live.finish()
    assert live.closed and live.receipts.closed
    with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
        live.db.commit()
    live.finish()
    assert live.path.exists()
