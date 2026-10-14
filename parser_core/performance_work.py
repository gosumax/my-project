"""Opt-in work accounting. Request identity is not cache eligibility or proof.

No source reads, hashing, SQL or recognition are performed to construct keys.
Events are buffered to disk; exact distinct-key aggregation happens at finish.
"""
from __future__ import annotations

import functools
import inspect
import json
import sqlite3
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager, nullcontext

from .bounded_storage import DiskRecords


def current():
    module = sys.modules.get('parser_core.performance_diagnostics')
    diagnostic = getattr(module, 'ACTIVE', None)
    return getattr(diagnostic, 'work', None)


def note(name, *, category, key=None, **counts):
    work = current()
    if work is not None:
        work.note(name, category=category, key=key, **counts)


def scope(name, *, category, key=None, **counts):
    work = current()
    if work is None:
        return nullcontext()

    @contextmanager
    def measured():
        identity = work._metadata(lambda a, k: key, (), {})
        tick = work.diagnostic.tick()
        outcome = 'RETURNED'
        try:
            yield
        except BaseException as exc:
            outcome = 'RAISED:' + type(exc).__name__
            raise
        finally:
            end = time.perf_counter()
            work._record(name, category, identity,
                (end - tick[0], time.thread_time() - tick[1], time.process_time() - tick[2]),
                outcome, result_counters=counts, span=(tick[0], end))
    return measured()


def write_json_line(stream, value, *, allow_nan=True):
    """Same JSON/text operation; counters use emitted characters, not guessed bytes."""
    if current() is None:
        return stream.write(json.dumps(value, ensure_ascii=False, allow_nan=allow_nan) + '\n')
    key = dict(output=str(getattr(stream, 'name', 'UNNAMED_STREAM')),
               segment=value.get('segment_id') if isinstance(value, dict) else None)
    with scope('serialization/jsonl', category='io/serialization', key=key, rows=1):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=allow_nan) + '\n'
    with scope('serialization/text_write', category='io/write', key=key,
               rows=1, characters=len(payload)):
        return stream.write(payload)


class WorkCounters:
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        self.events = DiskRecords(diagnostic.output / 'performance_work_events.jsonl')
        self.lock = threading.RLock()
        self.errors = Counter()
        self.cache_readers = {}
        self.installed = []
        self.unavailable = []
        self.correlation_totals = None
        self.aggregated_calls = 0
        self.overhead_wall = 0.0
        self.overhead_thread_cpu = 0.0

    def note(self, name, *, category, key=None, **counts):
        identity = self._metadata(lambda a, k: key, (), {})
        self._record(name, category, identity, (0.0, 0.0, 0.0), 'OBSERVATION',
                     result_counters=counts)

    def _metadata(self, key, args, kwargs):
        tick = time.perf_counter(), time.thread_time()
        try:
            value = key(args, kwargs) if key else None
            # Keys contain only explicitly selected existing source metadata.
            return json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(',', ':')) if value is not None else None
        except Exception as exc:
            with self.lock:
                self.errors['key/' + type(exc).__name__] += 1
            return None
        finally:
            self._overhead(tick)

    def _overhead(self, tick):
        wall, cpu = time.perf_counter() - tick[0], time.thread_time() - tick[1]
        with self.lock:
            self.overhead_wall += wall
            self.overhead_thread_cpu += cpu

    def _record(self, name, category, key, elapsed, outcome, objects=None,
                result_counters=None, span=None):
        tick = time.perf_counter(), time.thread_time()
        try:
            event = dict(operation=name, category=category, request_key=key,
                wall_seconds=elapsed[0], thread_cpu_seconds=elapsed[1],
                process_cpu_interval_seconds=elapsed[2], outcome=outcome,
                returned_objects=objects, result_counters=result_counters or {})
            # Millions of successful keyless correlations need exact sums, not
            # one JSON/SQLite record per call. Failures and keyed work retain
            # individual events; this fixed one-operation aggregate is bounded.
            if (name == 'physical_chat_admission.correlation' and key is None
                    and outcome == 'RETURNED' and objects is None):
                with self.lock:
                    if self.correlation_totals is None:
                        self.correlation_totals = dict(event, calls=0,
                            wall_seconds=0., thread_cpu_seconds=0.,
                            process_cpu_interval_seconds=0., result_counters=Counter())
                    total = self.correlation_totals
                    total['calls'] += 1
                    self.aggregated_calls += 1
                    for field in ('wall_seconds', 'thread_cpu_seconds', 'process_cpu_interval_seconds'):
                        total[field] += event[field]
                    total['result_counters'].update(event['result_counters'])
            else:
                self.events.append(event)
            # Inclusive work costs are separate from the original critical
            # path. Do not store unused container intervals for every call.
        except Exception as exc:
            with self.lock:
                self.errors['record/' + type(exc).__name__] += 1
        finally:
            self._overhead(tick)

    def wrap(self, owner, attr, name, *, category, key=None, aliases=False,
             generator=False, result_count=None, result_key=None, result_counters=None):
        """Wrap explicit owned APIs; preserve descriptors, return and exceptions.

        generator=True is reserved for read-only next()-consumed row streams.
        Their timing excludes time for which the consumer holds a yielded row.
        """
        original_descriptor = owner[attr] if isinstance(owner, dict) else inspect.getattr_static(owner, attr)
        descriptor = type(original_descriptor)
        original = (original_descriptor.__func__ if descriptor in (classmethod, staticmethod)
                    else owner[attr] if isinstance(owner, dict) else getattr(owner, attr))
        if getattr(original, '_performance_work_operation', None) == name:
            return
        if generator and not inspect.isgeneratorfunction(original):
            raise TypeError('generator accounting requires an explicit Python generator function')
        if hasattr(original, 'cache_info'):
            self.cache_readers[name] = (original.cache_info, original.cache_info()._asdict(), {'resets': 0})

        @functools.wraps(original)
        def measured(*args, **kwargs):
            request_key = self._metadata(key, args, kwargs)
            tick = self.diagnostic.tick()
            outcome = 'RETURNED'
            objects = None
            counters = None
            try:
                try:
                    result = original(*args, **kwargs)
                finally:
                    end = time.perf_counter()
                    elapsed = (end - tick[0], time.thread_time() - tick[1],
                               time.process_time() - tick[2])
                if result_count or result_key or result_counters:
                    # Metadata only; never consume an iterator or traverse payloads.
                    metadata_tick = time.perf_counter(), time.thread_time()
                    try:
                        if result_count:
                            objects = result_count(result)
                        if result_key:
                            value = result_key(result, args, kwargs)
                            request_key = json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(',', ':')) if value is not None else None
                        if result_counters:
                            counters = result_counters(result, args, kwargs)
                    except Exception as exc:
                        with self.lock:
                            self.errors['result_count/' + type(exc).__name__] += 1
                    finally:
                        self._overhead(metadata_tick)
                return result
            except BaseException as exc:
                outcome = 'RAISED:' + type(exc).__name__
                raise
            finally:
                self._record(name, category, request_key, elapsed, outcome, objects,
                             counters, (tick[0], end))

        @functools.wraps(original)
        def measured_rows(*args, **kwargs):
            request_key = self._metadata(key, args, kwargs)
            tick = self.diagnostic.tick()
            outcome = 'ITERATOR_CREATED'
            try:
                iterator = original(*args, **kwargs)
            except BaseException as exc:
                outcome = 'RAISED:' + type(exc).__name__
                raise
            finally:
                elapsed = (time.perf_counter() - tick[0], time.thread_time() - tick[1],
                           time.process_time() - tick[2])
                self._record(name + '/create', category, request_key, elapsed, outcome)

            def consume():
                elapsed = [0.0, 0.0, 0.0]
                objects = 0
                outcome = 'CLOSED_BEFORE_EOF'
                failure = None
                try:
                    while True:
                        tick = self.diagnostic.tick()
                        try:
                            row = next(iterator)
                        except StopIteration as exc:
                            outcome = 'EXHAUSTED'
                            return exc.value
                        finally:
                            end = time.perf_counter()
                            elapsed[0] += end - tick[0]
                            elapsed[1] += time.thread_time() - tick[1]
                            elapsed[2] += time.process_time() - tick[2]
                        objects += 1
                        yield row
                except GeneratorExit:
                    raise
                except BaseException as exc:
                    outcome = 'RAISED:' + type(exc).__name__
                    failure = exc
                    raise
                finally:
                    close_tick = self.diagnostic.tick()
                    try:
                        if outcome != 'EXHAUSTED':
                            iterator.close()
                    except BaseException:
                        if failure is None:
                            raise
                    finally:
                        if outcome != 'EXHAUSTED':
                            self._record(name + '/close', category, request_key,
                                (time.perf_counter() - close_tick[0], time.thread_time() - close_tick[1],
                                 time.process_time() - close_tick[2]), 'TEARDOWN')
                        self._record(name, category, request_key, elapsed, outcome, objects)
            return consume()

        wrapped = measured_rows if generator else measured
        wrapped._performance_work_operation = name
        for method in ('cache_info', 'cache_parameters'):
            if hasattr(original, method):
                setattr(wrapped, method, getattr(original, method))
        if hasattr(original, 'cache_clear'):
            @functools.wraps(original.cache_clear)
            def cache_clear():
                result = original.cache_clear()
                with self.lock:
                    self.cache_readers[name][2]['resets'] += 1
                return result
            wrapped.cache_clear = cache_clear
        replacement = descriptor(wrapped) if descriptor in (classmethod, staticmethod) else wrapped
        if isinstance(owner, dict):
            owner[attr] = replacement
        else:
            setattr(owner, attr, replacement)
        self.diagnostic.patches.append((owner, attr, original_descriptor))
        self.installed.append(dict(operation=name, category=category, generator=generator,
            request_identity='EXPLICIT_SOURCE_METADATA' if key else 'UNAVAILABLE'))
        if aliases and inspect.ismodule(owner):
            # Preserve pre-existing from-import bindings in current project only.
            for module_name, module in list(sys.modules.items()):
                if not (module_name.startswith('parser_core.') or
                        module_name in ('scripts.run_smoke14_ram', 'scripts.replay_saved_quality', '__main__')):
                    continue
                if module is None or module is owner:
                    continue
                for binding, value in list(vars(module).items()):
                    if value is original:
                        setattr(module, binding, wrapped)
                        self.diagnostic.patches.append((module, binding, value))

    def finish(self):
        started = time.perf_counter()
        with self.lock:
            if self.correlation_totals is not None:
                self.events.append(self.correlation_totals)
                self.correlation_totals = None
        path = self.diagnostic.output / 'performance_work_index.sqlite'
        totals = {}
        with sqlite3.connect(path) as db:
            db.execute('pragma temp_store=FILE')
            db.execute('create table requests(operation text, request_key text)')
            pending = []
            for event in self.events:
                name = event['operation']
                total = totals.setdefault(name, dict(operation=name, category=event['category'],
                    calls=0, wall_seconds=0.0, thread_cpu_seconds=0.0,
                    process_cpu_interval_seconds=0.0, keyed_calls=0,
                    unknown_identity_calls=0, returned_objects=0, objects_counted_calls=0,
                    outcomes=Counter(), result_counters=Counter()))
                calls = event.get('calls', 1)
                total['calls'] += calls
                for field in ('wall_seconds', 'thread_cpu_seconds', 'process_cpu_interval_seconds'):
                    total[field] += event[field]
                total['outcomes'][event['outcome']] += calls
                total['result_counters'].update(event['result_counters'])
                if event['returned_objects'] is not None:
                    total['returned_objects'] += event['returned_objects']
                    total['objects_counted_calls'] += 1
                if event['request_key'] is None:
                    total['unknown_identity_calls'] += calls
                else:
                    total['keyed_calls'] += 1
                    pending.append((name, event['request_key']))
                if len(pending) >= 2048:
                    db.executemany('insert into requests values(?,?)', pending)
                    pending.clear()
            db.executemany('insert into requests values(?,?)', pending)
            db.execute('create index request_keys on requests(operation, request_key)')
            for name, count in db.execute('select operation,count(distinct request_key) from requests group by operation'):
                totals[name]['unique_request_keys'] = count
        for total in totals.values():
            total.setdefault('unique_request_keys', 0)
            total['repeated_request_keys'] = total['keyed_calls'] - total['unique_request_keys']
            total['mean_ms_per_call'] = 1000 * total['wall_seconds'] / total['calls']
        caches = {}
        for name, (reader, before, state) in self.cache_readers.items():
            after = reader()._asdict()
            stable = not state['resets'] and after['hits'] >= before['hits'] and after['misses'] >= before['misses']
            caches[name] = dict(before=before, after=after,
                hits=after['hits'] - before['hits'] if stable else None,
                misses=after['misses'] - before['misses'] if stable else None,
                observed_resets=state['resets'],
                status='COUNTER_DELTA_NO_OBSERVED_RESET' if stable else 'RESET_OR_UNAVAILABLE',
                scope='Calls through retained original cache_clear references cannot be observed')
        self.events.close()
        report = dict(schema='PERFORMANCE_WORK_COUNTS_V1', operations=list(totals.values()),
            actual_cache_counters=caches, installed=self.installed, unavailable=self.unavailable,
            measurement_errors=dict(self.errors), events=self.events.count,
            aggregated_keyless_correlation_calls=self.aggregated_calls,
            event_layout='INDIVIDUAL_EVENTS_AND_WEIGHTED_KEYLESS_CORRELATION_TOTALS',
            instrumentation=dict(metadata_and_record_wall_seconds_inclusive=self.overhead_wall,
                metadata_and_record_thread_cpu_seconds=self.overhead_thread_cpu,
                aggregation_wall_seconds=time.perf_counter() - started),
            limitations=[
                'Repeated request keys are not proof of identical mutable inputs or safely removable work.',
                'Keys use existing metadata only; no source reads, stat calls or hashes are added for identity.',
                'Wall and CPU spans are inclusive and may overlap; do not sum them as total wall.',
                'Work spans do not change existing compute/wait/unaccounted attribution.',
                'Successful keyless correlations are exact weighted totals; event rows are not call counts.',
                'Process CPU intervals include concurrent native threads; caller thread CPU is separately reported.',
                'Generator timing covers next() work only, excluding consumer yield suspension.',
                'Unknown identities are counted explicitly and excluded from unique/repeated-key counts.',
                'Only explicit listed APIs are covered; no global open/hashlib/sqlite/json monkeypatch.',
                'Counter instrumentation adds overhead; external launcher wall includes final aggregation.'
            ])
        (self.diagnostic.output / 'performance_work_counts.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return report
