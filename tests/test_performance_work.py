import copy
import functools
import inspect
import io
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest

from parser_core import performance_diagnostics as diagnostics
from parser_core import performance_work


@pytest.fixture
def measured(tmp_path):
    diagnostic = diagnostics.Diagnostics(tmp_path)
    try:
        yield diagnostic.work
    finally:
        for owner, name, original in reversed(diagnostic.patches):
            if isinstance(owner, dict):
                owner[name] = original
            else:
                setattr(owner, name, original)
        diagnostic.work.events.close()
        diagnostic.intervals.close()
        for records in (diagnostic.jobs, diagnostic.samples, diagnostic.ocr_compute):
            records.close()


def operation(report, name):
    return next(row for row in report['operations'] if row['operation'] == name)


def test_repeated_requests_do_not_reuse_or_modify_results(measured):
    calls = []
    def change(value):
        calls.append(value['id'])
        value['visited'] = value.get('visited', 0) + 1
        return value
    owner = SimpleNamespace(call=change)
    measured.wrap(owner, 'call', 'change', category='test', key=lambda a, k: a[0]['id'])
    a, b = {'id': 'A'}, {'id': 'B'}
    assert owner.call(a) is a
    assert owner.call(b) is b
    assert owner.call(a) is a
    assert calls == ['A', 'B', 'A'] and a['visited'] == 2
    result = operation(measured.finish(), 'change')
    assert (result['calls'], result['unique_request_keys'], result['repeated_request_keys']) == (3, 2, 1)


def test_observer_work_span_preserves_existing_critical_path_attribution(measured):
    intervals = measured.diagnostic.intervals
    intervals.append((2., 3., 'actual_compute', 'compute'))
    intervals.append((4., 5., 'actual_wait', 'wait'))
    before = (intervals.union(0, 10, kinds=('wait',)),
              intervals.union(0, 10, kinds=('wait', 'compute')))
    measured._record('observer_with_nested_waits', 'test', None, (8., 0., 0.), 'RETURNED', span=(1., 9.))
    after = (intervals.union(0, 10, kinds=('wait',)),
             intervals.union(0, 10, kinds=('wait', 'compute')))
    assert before == after == (1., 2.)
    assert 'work/observer_with_nested_waits' not in intervals.names()


def test_high_frequency_keyless_correlations_keep_exact_counts_without_event_growth(measured):
    def record(_):
        measured._record('physical_chat_admission.correlation', 'capture/correlation',
            None, (.25, .125, .5), 'RETURNED', result_counters={'comparisons': 1, 'input_elements': 80})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(record, range(400)))
    assert measured.events.count == 0 and measured.aggregated_calls == 400
    assert measured.diagnostic.intervals.count == 0
    report = measured.finish()
    row = operation(report, 'physical_chat_admission.correlation')
    assert (row['calls'], row['unknown_identity_calls'], row['unique_request_keys'], row['repeated_request_keys']) == (400, 400, 0, 0)
    assert row['wall_seconds'] == 100 and row['thread_cpu_seconds'] == 50
    assert row['result_counters'] == {'comparisons': 400, 'input_elements': 32000}
    assert row['outcomes'] == {'RETURNED': 400} and report['events'] == 1


def test_descriptors_and_restoration(measured):
    class Example:
        @classmethod
        def name(cls, value):
            return cls, value
        @staticmethod
        def identity(value):
            return value
    before = {name: inspect.getattr_static(Example, name) for name in ('name', 'identity')}
    for name in before:
        measured.wrap(Example, name, name, category='test')
    value = object()
    assert Example.name(value) == (Example, value)
    assert Example().name(value) == (Example, value)
    assert Example.identity(value) is value and Example().identity(value) is value
    for owner, name, original in reversed(measured.diagnostic.patches):
        setattr(owner, name, original)
    assert all(inspect.getattr_static(Example, name) is descriptor for name, descriptor in before.items())


def test_key_error_and_target_exception_are_preserved(measured):
    failure = ValueError('source failure')
    def bad_key(a, k):
        raise RuntimeError('metadata failure')
    def target():
        raise failure
    owner = SimpleNamespace(call=target)
    measured.wrap(owner, 'call', 'failure', category='test', key=bad_key)
    with pytest.raises(ValueError) as caught:
        owner.call()
    assert caught.value is failure
    report = measured.finish()
    result = operation(report, 'failure')
    assert result['unknown_identity_calls'] == 1
    assert result['unique_request_keys'] == result['repeated_request_keys'] == 0
    assert result['outcomes'] == {'RAISED:ValueError': 1}
    assert report['measurement_errors'] == {'key/RuntimeError': 1}


def test_import_aliases_are_counted_once_and_restored(measured, monkeypatch):
    original = lambda value: value
    owner = ModuleType('parser_core._work_owner')
    alias = ModuleType('parser_core._work_alias')
    owner.call = alias.renamed = original
    monkeypatch.setitem(sys.modules, owner.__name__, owner)
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    measured.wrap(owner, 'call', 'alias', category='test', aliases=True)
    assert owner.call(2) == 2 and alias.renamed(3) == 3
    assert operation(measured.finish(), 'alias')['calls'] == 2


def test_threaded_counts_are_lossless(measured):
    owner = SimpleNamespace(call=lambda value: value * 2)
    measured.wrap(owner, 'call', 'parallel', category='test', key=lambda a, k: a[0])
    inputs = list(range(80)) * 5
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(owner.call, inputs)) == [v * 2 for v in inputs]
    result = operation(measured.finish(), 'parallel')
    assert (result['calls'], result['unique_request_keys'], result['repeated_request_keys']) == (400, 80, 320)
    assert len(measured.events.tail) <= 2


def test_generator_time_excludes_consumer_suspension(measured, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(performance_work.time, 'perf_counter', lambda: clock[0])
    def rows(path):
        clock[0] += 2
        yield path
        clock[0] += 3
        return 'done'
    owner = SimpleNamespace(rows=rows)
    measured.wrap(owner, 'rows', 'rows', category='test', generator=True, key=lambda a, k: a[0])
    iterator = owner.rows('source')
    assert next(iterator) == 'source'
    clock[0] += 100
    with pytest.raises(StopIteration) as ended:
        next(iterator)
    assert ended.value.value == 'done'
    result = operation(measured.finish(), 'rows')
    assert result['wall_seconds'] == 5
    assert result['returned_objects'] == 1
    assert result['outcomes'] == {'EXHAUSTED': 1}


def test_generator_factory_argument_error_is_immediate(measured):
    def rows(required):
        yield required
    owner = SimpleNamespace(rows=rows)
    measured.wrap(owner, 'rows', 'rows', category='test', generator=True)
    with pytest.raises(TypeError):
        owner.rows()
    assert operation(measured.finish(), 'rows/create')['outcomes'] == {'RAISED:TypeError': 1}


def test_generator_source_failure_and_close_are_preserved(measured):
    closed = []
    failure = RuntimeError('read failed')
    def rows():
        try:
            yield 1
            raise failure
        finally:
            closed.append(True)
    owner = SimpleNamespace(rows=rows)
    measured.wrap(owner, 'rows', 'rows', category='test', generator=True)
    iterator = owner.rows()
    assert next(iterator) == 1
    with pytest.raises(RuntimeError) as caught:
        next(iterator)
    assert caught.value is failure and closed == [True]


def test_non_generator_factory_is_not_silently_adapted(measured):
    owner = SimpleNamespace(rows=lambda: iter([1]))
    original = owner.rows
    with pytest.raises(TypeError):
        measured.wrap(owner, 'rows', 'rows', category='test', generator=True)
    assert owner.rows is original


def test_lru_methods_survive_and_reset_is_not_a_false_delta(measured):
    @functools.lru_cache(maxsize=4)
    def target(value):
        return value + 1
    owner = SimpleNamespace(call=target)
    measured.wrap(owner, 'call', 'cache', category='test', key=lambda a, k: a[0])
    assert owner.call.cache_parameters() == target.cache_parameters()
    assert owner.call(1) == owner.call(1) == 2
    assert owner.call.cache_info().hits == 1
    owner.call.cache_clear()
    for value in range(5):
        owner.call(value)
        owner.call(value)
    row = measured.finish()['actual_cache_counters']['cache']
    assert row['observed_resets'] == 1
    assert row['hits'] is None and row['misses'] is None


def test_lru_delta_without_reset(measured):
    owner = SimpleNamespace(call=functools.lru_cache(maxsize=2)(lambda value: value))
    measured.wrap(owner, 'call', 'cache', category='test')
    for value in ('A', 'B', 'A'):
        owner.call(value)
    row = measured.finish()['actual_cache_counters']['cache']
    assert (row['hits'], row['misses']) == (1, 2)


@pytest.mark.parametrize('enabled', [False, True])
def test_jsonl_content_and_invalid_number_parity(measured, monkeypatch, enabled):
    monkeypatch.setattr(diagnostics, 'ACTIVE', measured.diagnostic if enabled else None)
    source = dict(segment_id='T01:S1', raw_text='Игрок ☺', amount=125, unknown=None,
                  proof=dict(cards=['Qh', 'Tc'], valid=False))
    original = copy.deepcopy(source)
    target = io.StringIO()
    expected = json.dumps(source, ensure_ascii=False, allow_nan=False) + '\n'
    assert performance_work.write_json_line(target, source, allow_nan=False) == len(expected)
    assert target.getvalue() == expected and source == original
    with pytest.raises(ValueError):
        performance_work.write_json_line(target, {'amount': float('nan')}, allow_nan=False)
    assert target.getvalue() == expected


def test_recording_failure_does_not_change_return_or_source_exception(measured, monkeypatch):
    def fail(_):
        raise OSError('diagnostic disk')
    monkeypatch.setattr(measured.events, 'append', fail)
    owner = SimpleNamespace(call=lambda value: value)
    measured.wrap(owner, 'call', 'return', category='test')
    marker = object()
    assert owner.call(marker) is marker
    assert measured.errors['record/OSError'] == 1


def test_installed_targets_preserve_real_numeric_and_validation_calls(measured):
    from parser_core.performance_work_targets import install
    from parser_core.simple_stack_policy import first_displayed_stack
    from parser_core.card_set_evidence import verify_current
    import parser_core.simple_stack_policy as policy
    import parser_core.card_set_evidence as cards
    args = dict(table_id=1, session_id='s', seat='S1', hand_start=10, hand_end=20)
    expected = first_displayed_stack([], **args)
    rejected = verify_current({}, [])
    install(measured)
    assert policy.first_displayed_stack([], **args) == expected
    assert cards.verify_current({}, []) == rejected
    report = measured.finish()
    assert operation(report, 'simple_stack_policy.first_displayed_stack')['calls'] == 1
    assert operation(report, 'card_set_evidence.verify_current')['result_counters']['rejected'] == 1
    assert report['measurement_errors'] == {}


def test_counter_aggregation_failure_restores_parser_bindings(measured, monkeypatch):
    profile = measured.diagnostic
    profile.sampler = SimpleNamespace(join=lambda timeout: None)
    owner = SimpleNamespace(call=lambda: 'unchanged')
    original = owner.call
    profile.wrap(owner, 'call', 'call')
    monkeypatch.setattr(diagnostics, 'ACTIVE', profile)
    def failed():
        raise OSError('counter aggregation unavailable')
    monkeypatch.setattr(measured, 'finish', failed)
    profile.finish({})
    assert owner.call is original and diagnostics.ACTIVE is None
    report = json.loads((profile.output / 'performance_diagnostics.json').read_text('utf-8'))
    assert report['work_counts']['status'] == 'COUNTER_AGGREGATION_FAILED'


def test_full_install_counts_previously_wrapped_runner_once_and_restores(measured):
    marker = object()
    runner = dict(analyze_timeline=lambda *a, **k: None,
                  write_reconstruction=lambda *a, **k: marker)
    original = dict(runner)
    measured.diagnostic.install(runner)
    assert runner['write_reconstruction']('source', []) is marker
    report = measured.finish()
    result = operation(report, 'poker_reconstruction_layer.write_reconstruction')
    assert result['calls'] == 1
    assert measured.diagnostic.metrics['write_reconstruction']['wall'].calls == 1
    for owner, name, value in reversed(measured.diagnostic.patches):
        if isinstance(owner, dict):
            owner[name] = value
        else:
            setattr(owner, name, value)
    assert all(runner[name] is value for name, value in original.items())


@pytest.mark.parametrize('variant', ['positive', 'tampered', 'missing_support'])
def test_real_native_proof_on_off_exact_parity(measured, variant):
    from parser_core.performance_work_targets import install
    import parser_core.card_set_evidence as cards
    source = Path(__file__).resolve().parents[1] / 'tests/fixtures/card_set_unique_native_sources.json'
    case = json.loads(source.read_text('utf-8'))[0]
    if variant == 'tampered':
        case['target']['reading']['row_pixel_sha256'] = '0' * 64
    elif variant == 'missing_support':
        case.pop('hole')
    untouched = copy.deepcopy(case)
    expected = cards.prove(case)
    assert expected['status'] == ('PROVEN' if variant == 'positive' else 'UNKNOWN')
    install(measured)
    actual = cards.prove(case)
    assert actual == expected and case == untouched
    report = measured.finish()
    assert operation(report, 'card_set_evidence.prove')['calls'] == 1
    assert report['measurement_errors'] == {}
