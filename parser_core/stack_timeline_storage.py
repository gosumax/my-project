"""Canonical STACK SQLite location; no SQL, recognition or cleanup policy."""
import json
import re
import uuid
from pathlib import Path

RUNTIME_ROOT = Path('C:/PARSER_PD_RUNTIME/stack_timeline')
LOGICAL_DATABASE = 'stack_pipeline/timeline.sqlite'
STORAGE_RECEIPT = 'stack_pipeline/storage_receipt.json'
_SCHEMA = 'stack-timeline-storage-v1'


def _runtime_root():
    root = RUNTIME_ROOT.resolve(strict=True)
    if root.drive.upper() != 'C:':
        raise ValueError('STACK_TIMELINE_RUNTIME_MUST_RESOLVE_TO_C')
    return root


def create_stack_timeline_storage(output):
    """Reserve a fresh owned DB on C; never reuse, delete or fall back to D."""
    output = Path(output).resolve(strict=True)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    root = _runtime_root()
    run_id = uuid.uuid4().hex
    directory = root / run_id
    directory.mkdir(exist_ok=False)
    directory = directory.resolve(strict=True)
    if directory.parent != root:
        raise ValueError('STACK_TIMELINE_RUNTIME_DIRECTORY_ESCAPED')
    database = directory / 'timeline.sqlite'
    # Exclusive empty-file creation is also the real writeability probe.
    # SQLite subsequently initializes this file with its unchanged schema/SQL.
    with database.open('xb'):
        pass
    database = database.resolve(strict=True)
    receipt = dict(schema=_SCHEMA,
        STACK_TIMELINE_DB_PATH=str(database),
        STACK_TIMELINE_DB_RESOLVED_PATH=str(database),
        STACK_TIMELINE_STORAGE_DRIVE='C:', STACK_TIMELINE_STORAGE_CLASS='SSD',
        STACK_TIMELINE_RUNTIME_RUN_ID=run_id,
        STACK_TIMELINE_RUN_OUTPUT=str(output),
        STACK_TIMELINE_STORAGE_RECEIPT=str(output / STORAGE_RECEIPT))
    payload = json.dumps(receipt, ensure_ascii=False, indent=2)
    # Both locations bind the DB to its one run. Artifacts remain on errors.
    with (directory / 'storage_owner.json').open('x', encoding='utf-8') as stream:
        stream.write(payload)
    with (output / STORAGE_RECEIPT).open('x', encoding='utf-8') as stream:
        stream.write(payload)
    return database, receipt


def _declared_storage(source):
    source = Path(source).resolve()
    receipt_path = source / STORAGE_RECEIPT
    metadata_path = source / 'run_metadata.json'
    metadata = json.loads(metadata_path.read_text('utf-8')) if metadata_path.exists() else {}
    if not receipt_path.exists():
        if any(k.startswith('STACK_TIMELINE_') for k in metadata):
            raise ValueError('STACK_TIMELINE_DECLARED_STORAGE_RECEIPT_MISSING')
        return None
    receipt = json.loads(receipt_path.read_text('utf-8'))
    if receipt.get('schema') != _SCHEMA:
        raise ValueError('STACK_TIMELINE_STORAGE_SCHEMA_INVALID')
    run_id = receipt.get('STACK_TIMELINE_RUNTIME_RUN_ID')
    if not isinstance(run_id, str) or re.fullmatch('[0-9a-f]{32}', run_id) is None:
        raise ValueError('STACK_TIMELINE_RUNTIME_RUN_ID_INVALID')
    expected = _runtime_root() / run_id / 'timeline.sqlite'
    database = Path(receipt['STACK_TIMELINE_DB_PATH'])
    if (not database.is_absolute() or database != expected or
            database.resolve(strict=True) != expected or
            receipt.get('STACK_TIMELINE_DB_RESOLVED_PATH') != str(expected) or
            receipt.get('STACK_TIMELINE_STORAGE_DRIVE') != 'C:' or
            receipt.get('STACK_TIMELINE_STORAGE_CLASS') != 'SSD' or
            receipt.get('STACK_TIMELINE_RUN_OUTPUT') != str(source) or
            receipt.get('STACK_TIMELINE_STORAGE_RECEIPT') != str(receipt_path)):
        raise ValueError('STACK_TIMELINE_STORAGE_BINDING_INVALID')
    if not database.is_file():
        raise ValueError('STACK_TIMELINE_DECLARED_DATABASE_MISSING')
    owner = json.loads((database.parent / 'storage_owner.json').read_text('utf-8'))
    if owner != receipt:
        raise ValueError('STACK_TIMELINE_STORAGE_OWNER_CONFLICT')
    for name, value in receipt.items():
        if name.startswith('STACK_TIMELINE_') and name in metadata and metadata[name] != value:
            raise ValueError('STACK_TIMELINE_METADATA_STORAGE_CONFLICT:' + name)
    return database


def resolve_stack_timeline_path(source):
    """New sources require their declared C DB; old sealed sources retain D."""
    source = Path(source)
    declared = _declared_storage(source)
    return declared if declared is not None else (source / LOGICAL_DATABASE).resolve()


def stack_timeline_source_name(source):
    """Keep historical hash keys; new inventories expose the actual C path."""
    declared = _declared_storage(source)
    return str(declared) if declared is not None else LOGICAL_DATABASE
