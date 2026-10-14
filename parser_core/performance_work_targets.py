"""Explicit project-owned diagnostic targets; never global library patches."""
import importlib
import sys


def arg(args, kwargs, index, name, default=None):
    return args[index] if len(args) > index else kwargs.get(name, default)


def selected(value, fields):
    if not isinstance(value, dict):
        return None
    result = {field: value[field] for field in fields if value.get(field) is not None}
    return result or None


def hand(value):
    if not isinstance(value, dict):
        return None
    record = value
    value = record.get('observed_hand', record)
    key = selected(value, ('table', 'table_id', 'session', 'session_id', 'hand_id', 'hand_number'))
    if key is not None and record.get('segment_id') is not None:
        key['segment_id'] = record['segment_id']
    return key


def row(value):
    return selected(value, ('table_id', 'session_id', 'epoch', 'frame_id', 'row_index',
        'stable_row_id', 'row_id', 'row_pixel_sha256', 'crop_sha256', 'pixel_sha256',
        'source_frame', 'source_pts', 'time_base', 'source_file', 'source_line',
        'observation_id', 'state_id', 'seat', 'hand_id'))


def rows_window(values):
    # Window identity, deliberately not a digest of mutable row contents.
    if not isinstance(values, (list, tuple)) or not values:
        return None
    return dict(length=len(values), first=row(values[0]), last=row(values[-1]))


def path_arg(index=0, name='source'):
    return lambda a, k: str(arg(a, k, index, name)) if arg(a, k, index, name) is not None else None


def hand_arg(index=0, name='hand'):
    return lambda a, k: hand(arg(a, k, index, name))


def result_status(value, *unused):
    if isinstance(value, bool):
        return {'accepted' if value else 'rejected': 1}
    if isinstance(value, dict):
        return {'status/' + str(value['status']): 1} if value.get('status') is not None else {}
    return {}


def solver_result(value, *unused):
    result = result_status(value)
    if isinstance(value, dict):
        for key in ('nodes', 'solution_count'):
            if isinstance(value.get(key), (int, float)):
                result[key] = value[key]
        if 'solution_count' not in value and isinstance(value.get('solutions'), (list, tuple)):
            result['solution_count'] = len(value['solutions'])
    return result


def install(counters, runner=None):
    """Import only registered modules already used by the FULL_HH path.

    Optional saved-only seals are instrumented only if already loaded. Imports
    and installation cost belong to diagnostic startup, never to model time.
    """
    def add(module, symbol, category, **options):
        module_name = 'parser_core.' + module
        target = importlib.import_module(module_name)
        parts = symbol.split('.')
        owner = target
        for name in parts[:-1]:
            owner = getattr(owner, name)
        name = module + '.' + symbol
        counters.wrap(owner, parts[-1], name, category=category,
                      aliases=len(parts) == 1, **options)

    # Source parsing and metadata-keyed caches. No key construction calls stat.
    add('simple_saved_policy', 'read_rows', 'source/jsonl', key=path_arg(name='path'), generator=True)
    for name in ('saved_stack_observations', 'saved_nick_observations', 'saved_sitout_observations'):
        add('simple_saved_policy', name, 'source/ingest', key=path_arg())
    add('simple_saved_policy', '_source_version', 'source/stat', key=path_arg(name='path'),
        result_key=lambda r, a, k: r)
    for name in ('_forced_allin_native_inventory', '_capture_offsets', '_capture_record', '_sql_status_records'):
        add('simple_saved_policy', name, 'source/cache', key=lambda a, k: [a, k])
    add('simple_saved_policy', '_sitout_native_source', 'proof/native_status',
        key=lambda a, k: [arg(a, k, 0, 'kind'), str(arg(a, k, 1, 'path')),
                         k.get('line'), k.get('observation_id')])
    add('simple_saved_policy', 'valid_sitout_status', 'proof/sitout',
        key=lambda a, k: [row(arg(a, k, 0, 'row')), selected(k,
            ('table_id', 'session_id', 'seat', 'hand_id', 'hand_start', 'hand_end'))],
        result_counters=result_status)
    add('simple_saved_policy', 'prove_forced_allin_start_stack', 'proof/forced_allin',
        key=lambda a, k: [hand(arg(a, k, 0, 'hand')), arg(a, k, 1, 'actor'),
                         arg(a, k, 2, 'seat'), str(arg(a, k, 3, 'source'))], result_counters=result_status)
    add('simple_saved_policy', 'SimpleSavedPolicyProvider.from_saved', 'source/provider',
        key=path_arg(1), result_count=lambda r: len(r.hands) if hasattr(r, 'hands') else None)
    for cls, names in (
        ('LineageActorProvider', ('_proofs', 'for_hand', 'for_actor', 'for_seat', 'canonical_hand')),
        ('SimpleSavedPolicyProvider', ('seat_rows', 'for_hand', 'player_rows')),
    ):
        for name in names:
            add('simple_saved_policy', cls + '.' + name, 'proof/who_stack',
                key=lambda a, k: [hand(arg(a, k, 1, 'hand')),
                    arg(a, k, 2, 'actor', k.get('seat'))])
    for name in ('first_displayed_stack', 'corrected_start_stack'):
        add('simple_stack_policy', name, 'proof/numeric',
            key=(lambda a, k: [selected(k, ('table_id', 'session_id', 'seat', 'hand_start', 'hand_end')),
                rows_window(arg(a, k, 0, 'observations'))]) if name == 'first_displayed_stack' else None,
            result_counters=result_status)
    add('hand_lifecycle', 'source_bound_headers', 'source/headers', key=path_arg(name='output'))
    add('hand_lifecycle', 'post_excel_contexts', 'lifecycle', key=path_arg(name='output'))
    add('canonical_events', 'canonical_join', 'canonical', key=path_arg(name='output'))
    add('canonical_events', 'admit_saved_candidate_forced_prefixes', 'proof/forced_prefix')

    # Existing native hashes: use the digest produced by the real operation.
    for module, symbol in (('win_tail_evidence', '_sha'), ('card_hybrid', '_sha'), ('current_chat_sources', 'sha')):
        add(module, symbol, 'proof/hash', result_key=lambda r, a, k: r,
            result_counters=lambda r, a, k: {'hash_calls': 1, 'input_bytes': len(a[0])})
    add('physical_chat_admission', 'pixel_hash', 'capture/hash',
        result_key=lambda r, a, k: r,
        result_counters=lambda r, a, k: {'hash_calls': 1, 'input_bytes': a[0].nbytes})
    add('physical_chat_admission', 'correlation', 'capture/correlation',
        result_counters=lambda r, a, k: {'comparisons': 1, 'input_elements': a[0].size + a[1].size})
    add('physical_chat_admission', 'physical_tail_proof', 'capture/admission_proof',
        result_counters=lambda r, a, k: dict(result_status(r), candidates=len(r.get('candidates', ()))))
    add('physical_chat_admission', 'PhysicalChatAdmission.process', 'capture/admission',
        key=lambda a, k: [a[0].table_id, a[0].session_id,
            arg(a, k, 2, 'transition').get('frame_to')])
    add('current_chat_sources', 'load_current_chat', 'source/native_chat',
        key=lambda a, k: [str(arg(a, k, 0, 'source')), arg(a, k, 1, 'expected_source_sha256')])
    add('win_tail_evidence', '_png', 'proof/pixels',
        key=lambda a, k: [arg(a, k, 1, 'byte_sha'), arg(a, k, 2, 'pixel_sha')])
    add('win_tail_evidence', '_frozen', 'proof/config', key=lambda a, k: row(arg(a, k, 0, 'reading')))
    add('win_tail_evidence', 'prove_win_tail', 'proof/win',
        key=lambda a, k: [rows_window(arg(a, k, 0, 'rows')), row(arg(a, k, 1, 'reading'))],
        result_counters=result_status)
    add('card_set_evidence', 'prove', 'proof/cards',
        key=lambda a, k: rows_window(arg(a, k, 0, 'case', {}).get('context_rows')),
        result_counters=result_status)
    add('card_set_evidence', 'verify_current', 'proof/cards_current',
        key=lambda a, k: arg(a, k, 0, 'proof', {}).get('target_refs'), result_counters=result_status)
    add('card_hybrid', 'CardHybrid.read_bytes', 'model/cards',
        key=lambda a, k: arg(a, k, 2, 'row_sha256'))
    add('card_hybrid', 'CardHybrid.__init__', 'model/load',
        key=lambda a, k: [str(arg(a, k, 1, 'config_path')),
                         str(arg(a, k, 2, 'suit_config_path')), str(k.get('rank_ncc_config'))])
    add('tiny_chat_client', 'TinyChatClient.read_batch', 'model/ocr_request',
        key=lambda a, k: dict(weights=a[0].weights_sha256, model=a[0].model_name,
            device=a[0].device, batch_size=arg(a, k, 2, 'batch_size'),
            ordered_source_hashes=[item.get('source_sha256') for item in arg(a, k, 1, 'items', [])]))
    add('start_stack_visibility', 'VisibilitySource.verify', 'proof/native_stack',
        key=lambda a, k: row(arg(a, k, 1, 'reading')), result_counters=result_status)

    # SQLite and lookup boundaries; inlined row/hash branches also emit notes.
    add('stack_evidence', 'StackEvidenceProvider.__init__', 'source/sqlite',
        key=lambda a, k: [str(arg(a, k, 1, 'path')), k.get('video_sha256')])
    add('stack_evidence', 'StackEvidenceProvider.observations', 'lookup/stack',
        key=lambda a, k: [str(a[0].path), list(a[1:]), k], result_count=len)
    add('stack_evidence', 'StackEvidenceProvider.match_snapshot', 'proof/stack_snapshot',
        key=lambda a, k: [str(a[0].path), row(arg(a, k, 1, 'fact'))], result_counters=result_status)
    add('stack_evidence', 'StackEvidenceProvider.write_audit', 'audit/stack', key=path_arg(1, 'output'))
    add('stack_timeline_live', 'StackTimelineLive.load_state', 'source/sqlite_state',
        key=lambda a, k: [str(a[0].path), arg(a, k, 1, 'state_id')])
    for module, method in (
        ('start_stack_live', 'StartStackLiveConnection._observe_frame'),
        ('seat_forced_binding', 'SeatForcedBinding.observe_frame'),
        ('stack_timeline_live', 'StackTimelineLive.observe_frame'),
    ):
        add(module, method, 'capture/source_request',
            key=lambda a, k: [arg(a, k, i, field) for i, field in
                enumerate(('table', 'session', 'frame', 'pts', 'time_base'), 1)])
    for method in ('flush', 'write_metrics'):
        add('stack_timeline_live', 'StackTimelineLive.' + method, 'io/sqlite',
            key=lambda a, k: str(a[0].path))

    # Per-hand solver and repeated validators, with source request identity only.
    add('poker_logic_engine', 'validate_hand', 'validator/observed', key=hand_arg())
    for method in ('validate_reconstruction', 'validate_call_fallback'):
        add('reconstruction_validation', method, 'validator/reconstructed',
            key=hand_arg(0, 'observed'), result_counters=result_status)
    add('poker_reconstruction.reconstruct', 'reconstruct_hand', 'reconstruction/hand',
        key=lambda a, k: selected(k, ('table', 'hand_id')), result_counters=solver_result)
    add('poker_reconstruction.solver', 'solve', 'reconstruction/solver',
        result_counters=solver_result)
    add('poker_reconstruction_layer', 'write_reconstruction', 'reconstruction/export', key=path_arg(name='output'))
    if runner:
        # Existing Diagnostics has independently wrapped this from-import.
        counters.wrap(runner, 'write_reconstruction', 'poker_reconstruction_layer.write_reconstruction',
                      category='reconstruction/export', key=path_arg(name='output'))
    add('hh_export', 'read_jsonl', 'source/jsonl_materialize', key=path_arg(name='path'), result_count=len)
    add('hh_export', 'export_hand_histories', 'export/hh', key=path_arg(name='output'))

    # Saved-only providers are never imported just for diagnostics in live runs.
    for module, symbols in (
        ('adjacent_stack_carry', ('file_sha256',)),
        ('constant_stack_balance', ('SourceSealBatch.begin', 'SourceSealBatch.check_phase', 'SourceSealBatch.finish')),
    ):
        if 'parser_core.' + module in sys.modules:
            for symbol in symbols:
                add(module, symbol, 'proof/source_seal',
                    key=path_arg(name='path') if symbol == 'file_sha256' else None)
        else:
            counters.unavailable.append(dict(module=module, reason='SAVED_ONLY_MODULE_NOT_LOADED'))
