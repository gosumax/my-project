"""Official sealed physical-producer replay; no decode, OCR, solver or Smoke."""
import ast
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from parser_core import physical_chat_admission as physical
from parser_core.physical_tail_route import install_physical_tail
from parser_core.physical_tail_validation_contract import compare


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text('utf-8'))


def validate_saved_physical_tail(root, manifest, bundle_path, output):
    root = Path(root).resolve()
    bundle_path = Path(bundle_path).resolve()
    bundle_sha = sha(bundle_path)
    if manifest.get('physical_tail_saved_validation_bundle_sha256') != bundle_sha:
        raise ValueError('PHYSICAL_VALIDATION_BUNDLE_NOT_REGISTERED')
    bundle = load(bundle_path)
    if bundle.get('schema') != 'poker.saved-physical-tail-validation.v1':
        raise ValueError('PHYSICAL_VALIDATION_BUNDLE_UNKNOWN')
    if set(bundle['files']) != {'inventory','pixels','old','native','comparator','native_receipts'}:
        raise ValueError('PHYSICAL_VALIDATION_PARENT_SET_UNKNOWN')
    files = {}
    for role, value in bundle['files'].items():
        path = Path(value['path']).resolve()
        if sha(path) != value['sha256']:
            raise ValueError('PHYSICAL_VALIDATION_SOURCE_CHANGED:' + role)
        files[role] = path
    output = Path(output).resolve()
    if output.exists() or any(output == p or output in p.parents or p in output.parents
                              for p in [bundle_path,*files.values()]):
        raise ValueError('PHYSICAL_VALIDATION_OUTPUT_NOT_FRESH_OR_OVERLAPS_INPUT')
    inventory = load(files['inventory'])
    contract_path = root / 'parser_core/physical_tail_validation_contract.py'
    def contract_nodes(path):
        tree = ast.parse(path.read_text('utf-8'))
        return [ast.dump(n, include_attributes=False) for n in tree.body
            if isinstance(n,ast.FunctionDef) and n.name in ('normalized','differences','compare')
            or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('SCORE','MARGIN') for t in n.targets)]
    if (sha(contract_path) != bundle['comparator_copy_sha256']
            or contract_nodes(contract_path) != contract_nodes(files['comparator'])):
        raise ValueError('PHYSICAL_VALIDATION_COMPARATOR_CHANGED')
    oracle, expected_native = load(files['old']), load(files['native'])
    native_receipts = {'native19:' + row['case']:row['receipt'] for row in load(files['native_receipts'])['receipts']}
    if len(native_receipts) != 8:
        raise ValueError('PHYSICAL_VALIDATION_NATIVE_RECEIPT_SET_UNKNOWN')
    if len(inventory) != 28 or set(oracle) != set(expected_native) or set(oracle) != {c['id'] for c in inventory}:
        raise ValueError('PHYSICAL_VALIDATION_EXACT_28_CLOSURE_REQUIRED')
    binding = install_physical_tail(root, manifest)
    native = manifest['speed_state']['PHYSICAL_TAIL_PROOF4_NATIVE_ACTIVE'] == 'YES'
    if native != (binding is not None):
        raise ValueError('PHYSICAL_VALIDATION_ROUTE_CONFLICT')
    before_calls = binding.calls if native else 0
    before_fallbacks = binding.fallbacks if native else 0
    output.mkdir(parents=True)
    started = time.perf_counter()
    results = {}
    with np.load(files['pixels'], allow_pickle=False) as arrays:
        for index, case in enumerate(inventory):
            before, after = arrays[f'{index}_old'], arrays[f'{index}_new']
            for name, value in (('old',before),('new',after)):
                if (hashlib.sha256(value.tobytes()).hexdigest() != case[name + '_sha256']
                        or list(value.shape) != list(case[name + '_shape'])):
                    raise ValueError('PHYSICAL_VALIDATION_PIXEL_CONFLICT:' + case['id'])
            reference = oracle[case['id']]
            lane = physical.PhysicalChatAdmission(case['table'], case['session_id'])
            if case['id'].startswith('native19:'):
                saved = native_receipts[case['id']]
                # process() returns the admission schema. FirstStage appends
                # these three sealed transport fields after process returns.
                transport = {'source_pts','time_base','source_chat_box'}
                if (set(saved) - set(reference) != transport or not set(reference) <= set(saved)
                        or {key:saved[key] for key in reference} != reference):
                    raise ValueError('PHYSICAL_VALIDATION_ORIGINAL_RECEIPT_JOIN_CONFLICT')
                lane.counter = int(saved['admissions'][0]['row_id'].rsplit(':F',1)[1])-1 if saved['admissions'] else 0
            elif not case['id'].startswith('native18:'):
                raise ValueError('PHYSICAL_VALIDATION_CASE_NAMESPACE_UNKNOWN')
            lane.previous = dict(gray=before.copy(), spans=case['old_spans'], frame=case['frame_before'])
            transition = dict(new_observed_spans=case['new_spans'], frame_to=case['frame_after'],
                classification=reference['old_classification'], reason=reference['old_reason'],
                new_edge_spans=reference['geometry']['top_edge_spans'])
            result = lane.process(after, transition)
            result = json.loads(json.dumps(result, allow_nan=False))
            results[case['id']] = result
            if (hashlib.sha256(before.tobytes()).hexdigest() != case['old_sha256']
                    or hashlib.sha256(after.tobytes()).hexdigest() != case['new_sha256']):
                raise ValueError('PHYSICAL_VALIDATION_INPUT_MUTATED')
    (output / 'actual_raw.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    semantic = compare(oracle, results)
    expected = expected_native if native else oracle
    canonical = lambda value:json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False)
    raw_exact = canonical(results) == canonical(expected)
    calls = binding.calls - before_calls if native else 0
    fallbacks = binding.fallbacks - before_fallbacks if native else 0
    if not semantic['semantic_exact'] or not raw_exact or (native and (calls != 28 or fallbacks != 0)):
        raise ValueError('PHYSICAL_VALIDATION_PARITY_OR_DISPATCH_FAIL:' + json.dumps(dict(
            semantic=semantic, raw_exact=raw_exact, native_calls=calls, fallbacks=fallbacks)))
    for role, value in bundle['files'].items():
        if sha(files[role]) != value['sha256']:
            raise ValueError('PHYSICAL_VALIDATION_SOURCE_CHANGED_DURING_RUN:' + role)
    if sha(bundle_path) != bundle_sha:
        raise ValueError('PHYSICAL_VALIDATION_BUNDLE_CHANGED_DURING_RUN')
    receipt = dict(verdict='PASS', route='run_production.py --validate-saved-physical-tail',
        mode='DEFAULT_ON_NATIVE' if native else 'DEFAULT_OFF_OLD', pairs=28,
        native_calls=calls, native_requests=None, fallbacks=fallbacks,
        native_requests_status='NOT_MEASURED_WITH_PRODUCTION_TRACE_OFF',
        raw_exact_to_sealed_reference=raw_exact, semantic_parity=semantic,
        actual_raw_sha256=sha(output/'actual_raw.json'), bundle_sha256=bundle_sha,
        speed_state=manifest['speed_state'], new_comparator_exclusions=0, score_rounding=False,
        new_video_decode_calls=0, new_ocr_calls=0, new_solver_calls=0,
        wall_seconds=time.perf_counter()-started)
    (output / 'physical_tail_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', 'utf-8')
    print(json.dumps(receipt), flush=True)
    return receipt
