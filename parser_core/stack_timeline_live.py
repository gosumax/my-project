"""Opt-in production frame adapter and continuous STACK evidence writer.

Recognition/state algorithms live in the source-preserving extracted module.
Physical seats remain unbound to players; no action or START_STACK is invented.
"""
import hashlib
import json
import sqlite3
import time
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path

import cv2
import numpy as np

from . import performance_diagnostics as diagnostics
from .stack_timeline_algorithms import Timeline, RawTiny, numeric, excluding_gate_class
from .bounded_storage import BoundedStats, HotHistory, StatePool, DiskRecords, RoutedRecords

ROOT = Path(__file__).resolve().parents[1]


class StackTimelineLive:
    def __init__(self, output, *, video_sha256, sessions, frames):
        self.output = Path(output) / 'stack_pipeline'
        self.output.mkdir()
        self.path = self.output / 'timeline.sqlite'
        self.sessions = sessions
        self.requested_frames = frames
        self.client = None
        self.closed = False
        self.pending = []
        self.frame_tables = []
        self.sources = {}
        self.counts = Counter()
        self.routes_counts = defaultdict(Counter)
        self.timings = defaultdict(BoundedStats)
        self.route_timings = defaultdict(lambda: defaultdict(BoundedStats))
        self.confidence = BoundedStats()
        self.active_route = None
        self.known = {}
        self.previous = {}
        self.unknown = Counter()
        self.written_states = self.written_observations = 0
        self.last_frame = -1
        self.last_flush_time = time.monotonic()
        self.last_flush_frame = -1
        self.dirty_observations = {}
        self.state_cache = OrderedDict()
        tables = json.loads((ROOT / 'configs/nine_table_layout.json').read_text('utf-8'))['tables']
        slots = json.loads((ROOT / 'configs/stack_roi_refined.json').read_text('utf-8'))['slots']
        self.routes = [dict(table_id=t['table'], seat_id=seat, xyxy=b['stack'])
            for t in tables for seat, b in slots[f"slot_{t['table']:02d}"]['seats'].items()]
        self.table_order = [t['table'] for t in tables]
        self.by_table = {table: [(i, r) for i, r in enumerate(self.routes) if r['table_id'] == table]
            for table in self.table_order}
        gate = excluding_gate_class()(ROOT / 'configs/stack_digit_gate.json')
        evaluate = gate.evaluate

        def timed_gate(image):
            tick = self.tick()
            try:
                result = evaluate(image)
                self.counts['gate_evaluations'] += 1
                self.counts['gate_evaluation_pass' if result['status'] == 'DIGITS_VISIBLE' else 'gate_evaluation_reject'] += 1
                return result
            finally:
                self.record('digit_gate', tick)
        gate.evaluate = timed_gate
        self.timeline = Timeline(self.routes, gate, 30.)
        self.timeline.states = HotHistory()
        self.timeline.observations = HotHistory()
        self.timeline.pools = [StatePool(self.load_state) for _ in self.routes]
        self.rejected_receipts = DiskRecords(self.output / 'visual_rejected_candidates.jsonl')
        for track,route in zip(self.timeline.tracks,self.routes):
            track.rejected = RoutedRecords(self.rejected_receipts,route)
        self.db = sqlite3.connect(self.path)
        self.db.execute('pragma foreign_keys=on')
        self.db.executescript('''
            create table metadata(key text primary key,value text);
            create table visual_states(state_id text primary key,source_state_id text,
                table_id integer,seat_id text,frame_index integer,timestamp real,origin text,
                width integer,height integer,raw_gray blob not null,exact_hash text,
                ocr_raw text,ocr_confidence real,ocr_numeric text,status text,
                digit_gate_status text,exclusion_label text,provenance text,
                ocr_called integer,digit_gate_score real,digit_gate_evidence text);
            create table observations(observation_id integer primary key,
                state_id text references visual_states(state_id),source_observation_id integer,
                previous_observation_id integer,previous_known_observation_id integer,
                table_id integer,seat_id text,frame_index integer,confirmed_frame integer,
                last_seen_frame integer,timestamp real,origin text,previous_known_value text,
                observed_delta integer,unknown_between integer,coverage_segment integer,
                delta_kind text,session_id text,source_pts integer,source_time_base text);
            create view numeric_timeline as select o.*,s.ocr_numeric as stack_value,
                s.ocr_confidence as confidence,s.status,s.exact_hash,s.exclusion_label
                from observations o join visual_states s using(state_id);
        ''')
        metadata = dict(schema='stack-evidence-timeline-v1', video_sha256=video_sha256,
            fps=30., coverage_ranges=[], requested_frames=frames,
            seat_player_bindings='NOT_PRESENT', action_onset_intervals='NOT_PRESENT',
            start_stack_authority=False, ocr_acceptance_threshold=.95,
            known_quality_limitation='Smoke19 frame96 T06 seat2 Sit out gate false admission; algorithm unchanged',
            algorithm_sha256=hashlib.sha256((ROOT / 'parser_core/stack_timeline_algorithms.py').read_bytes()).hexdigest(),
            config_hashes={name: hashlib.sha256((ROOT / 'configs' / name).read_bytes()).hexdigest()
                for name in ('stack_digit_gate.json','stack_roi_refined.json','stack_sitout_mask.json','ocr_tiny_local.json')})
        self.db.executemany('insert into metadata values(?,?)', [(k,json.dumps(v)) for k,v in metadata.items()])
        self.db.commit()
        self.receipts = (self.output / 'raw_ocr.jsonl').open('w', encoding='utf-8')
        (self.output / 'routes.json').write_text(json.dumps(self.routes, indent=2), 'utf-8')

    def tick(self):
        return time.perf_counter(), time.thread_time(), time.process_time()

    def record(self, name, tick, objects=1, kind='compute'):
        self.timings[name].append(time.perf_counter() - tick[0])
        if diagnostics.ACTIVE:
            diagnostics.ACTIVE.record('STACK_' + name, tick, objects, kind=kind,
                table=self.active_route[0] if self.active_route else 'ALL')
        if self.active_route:
            self.route_timings[self.active_route][name].append(time.perf_counter()-tick[0])

    def observe_frame(self, table, session, frame, pts, time_base, rgb, chat_gray):
        if table != self.table_order[len(self.frame_tables)]:
            raise ValueError('STACK table/frame order mismatch')
        if self.frame_tables and frame != self.last_frame:
            raise ValueError('STACK incomplete preceding frame')
        self.last_frame = frame
        self.frame_tables.append(table)
        self.sources[(table,frame)] = (session,pts,time_base)
        tick = self.tick()
        for i, route in self.by_table[table]:
            self.active_route = (table,route['seat_id'])
            x,y,X,Y = route['xyxy']
            gray = cv2.cvtColor(rgb[y:Y,x:X], cv2.COLOR_RGB2GRAY)
            if gray.shape != (Y-y,X-x) or not gray.size:
                raise ValueError('STACK ROI geometry mismatch')
            self.counts['roi_checks'] += 1
            self.routes_counts[(table,route['seat_id'])]['roi_checks'] += 1
            old = self.timeline.current[i]
            before = len(self.timeline.observations)
            state = self.timeline.feed(i, gray, frame)
            current = self.timeline.current[i]
            if old is not None:
                self.dirty_observations[old['observation_id']] = old
            if current is not None:
                self.dirty_observations[current['observation_id']] = current
            if len(self.timeline.observations) != before:
                source_frame = current['frame_index']
                current['_source'] = self.sources[(table,source_frame)]
            if state is not None:
                self.pending.append(state)
        self.active_route = None
        self.record('ROI_visual_timeline', tick, len(self.by_table[table]))
        if len(self.frame_tables) == len(self.table_order):
            # Identical per-source-frame admission order and batch<=8 contract.
            for offset in range(0,len(self.pending),8):
                batch = self.pending[offset:offset+8]
                if any(s['digit_gate_status'] != 'DIGITS_VISIBLE' for s in batch):
                    raise ValueError('STACK OCR before gate admission')
                if self.client is None:
                    tick = self.tick()
                    self.client = RawTiny(self.output)
                    self.record('OCR_startup', tick, kind='wait')
                tick = self.tick()
                for state in batch:
                    state['ocr_called'] = True
                    self.counts['ocr_calls'] += 1
                reply = self.client.read(batch)
                self.record('OCR_roundtrip', tick, len(batch), kind='wait')
                self.timings['OCR_inference'].append(reply['inference_seconds'])
                self.receipts.write(json.dumps(reply,ensure_ascii=False)+'\n')
                self.receipts.flush()
                byid = {v['item_id']:v for v in reply['results']}
                if len(byid) != len(batch):
                    raise ValueError('STACK missing/duplicate OCR result')
                for state in batch:
                    value = byid[state['state_id']]
                    state.update(ocr_raw=value['raw_text'],ocr_confidence=value['score'],
                        ocr_numeric=numeric(value['raw_text'],value['score']))
                # A batch spans seats; timing is shared, never divided into guessed per-seat compute.
                for state in batch:
                    self.route_timings[(state['table_id'],state['seat_id'])]['OCR_shared_batch_inference'].append(reply['inference_seconds'])
            self.pending.clear()
            if (frame == 0 or len(self.timeline.observations.items)>=256 or
                    frame-self.last_flush_frame>=120 or time.monotonic()-self.last_flush_time>=2.):
                self.flush()
            self.counts['frames'] += 1
            self.frame_tables.clear()
            # Only bounded recent source timestamps are needed by confirmed changes.
            self.sources = {k:v for k,v in self.sources.items() if k[1] >= frame-4}
            if frame == 0 or (frame+1)%30 == 0:
                self.write_metrics()

    def flush(self):
        tick = self.tick()
        for state in self.timeline.states[self.written_states:]:
            status = 'CONFIRMED_NUMERIC' if state['ocr_numeric'] is not None else 'UNKNOWN'
            evidence = state['digit_gate_evidence']
            self.db.execute('insert into visual_states values('+','.join('?' for _ in range(21))+')',
                (state['state_id'],state['state_id'],state['table_id'],state['seat_id'],state['frame_index'],
                 state['timestamp'],state['origin'],state['width'],state['height'],state['_raw'],state['exact_hash'],
                 state['ocr_raw'],state['ocr_confidence'],state['ocr_numeric'],status,state['digit_gate_status'],
                 evidence.get('excluded_label'),json.dumps(dict(numeric_authority='OCR_ONLY; ACTOR_UNBOUND',
                     state_payload={k:v for k,v in state.items() if k!='_raw'})),
                 state['ocr_called'],state['digit_gate_score'],json.dumps(evidence)))
            route = (state['table_id'],state['seat_id'])
            for counts in (self.counts,self.routes_counts[route]):
                counts['visual_states'] += 1
                counts['digit_gate_pass' if state['digit_gate_status']=='DIGITS_VISIBLE' else 'digit_gate_reject'] += 1
                counts['numeric_accepted' if state['ocr_numeric'] is not None else 'unknown_states'] += 1
            self.routes_counts[route]['ocr_calls'] += int(state['ocr_called'])
            if state['ocr_confidence'] is not None:self.confidence.append(state['ocr_confidence'])
        byid = {s['state_id']:s for s in self.timeline.states}
        for obs in self.timeline.observations[self.written_observations:]:
            route = (obs['table_id'],obs['seat_id'])
            state = byid.get(obs['state_id'])
            if state is None:state = self.load_state(obs['state_id'])
            value = state['ocr_numeric']
            prior = self.known.get(route)
            gap = self.unknown[route]
            delta = int(value)-prior[1] if value is not None and prior else None
            kind = 'NO_PREVIOUS_KNOWN' if delta is None else 'AGGREGATE_CONSTRAINT' if gap else 'ADJACENT_DISPLAYED_DELTA'
            source = obs.get('_source') or self.sources.get((obs['table_id'],obs['frame_index']))
            if source is None:
                raise ValueError('STACK original source timestamp unavailable')
            session,pts,base = source
            self.db.execute('insert into observations values('+','.join('?' for _ in range(20))+')',
                (obs['observation_id'],obs['state_id'],obs['observation_id'],self.previous.get(route),
                 prior[0] if prior else None,obs['table_id'],obs['seat_id'],obs['frame_index'],
                 obs['confirmed_frame'],obs['last_seen_frame'],obs['timestamp'],obs['origin'],
                 str(prior[1]) if prior else None,delta,gap,1,kind,session,pts,json.dumps(base)))
            self.previous[route] = obs['observation_id']
            if value is not None:
                self.known[route] = (obs['observation_id'],int(value))
                self.unknown[route] = 0
            else:
                self.unknown[route] += 1
            for counts in (self.counts,self.routes_counts[route]):
                counts['timeline_observations'] += 1
                counts['numeric_observations' if value is not None else 'unknown_observations'] += 1
                counts['deltas'] += int(delta is not None and delta != 0)
        self.db.executemany('update observations set last_seen_frame=? where observation_id=?',
            [(o['last_seen_frame'],o['observation_id']) for o in self.dirty_observations.values()])
        self.db.execute('update metadata set value=? where key=?', (json.dumps([[0,self.last_frame]]),'coverage_ranges'))
        commit_tick = self.tick()
        self.db.commit()
        self.record('SQLite_commit',commit_tick)
        self.counts['commits'] += 1
        self.written_states = len(self.timeline.states)
        self.written_observations = len(self.timeline.observations)
        # Release only after durable commit. Exact hash indexes retain IDs.
        for state in self.timeline.states:
            i=next(i for i,r in enumerate(self.routes) if r['table_id']==state['table_id'] and r['seat_id']==state['seat_id'])
            self.timeline.pools[i].release(state)
        self.timeline.states.release()
        self.timeline.observations.release()
        self.dirty_observations.clear()
        self.state_cache.clear()
        self.last_flush_time = time.monotonic()
        self.last_flush_frame = self.last_frame
        self.record('SQLite_evidence_write', tick, self.counts['timeline_observations'])

    def load_state(self,state_id):
        cached=self.state_cache.get(state_id)
        if cached is not None:
            self.state_cache.move_to_end(state_id)
            return cached
        tick=self.tick()
        row=self.db.execute('select provenance,raw_gray,exact_hash from visual_states where state_id=?',(state_id,)).fetchone()
        if row is None:raise ValueError('STACK recurrence state not committed')
        payload=json.loads(row[0])['state_payload']
        if hashlib.sha256(row[1]).hexdigest()!=row[2]:raise ValueError('STACK historical source hash mismatch')
        payload['_raw']=row[1]
        self.state_cache[state_id]=payload
        if len(self.state_cache)>128:self.state_cache.popitem(last=False)
        self.counts['historical_state_lookups']+=1
        self.record('historical_evidence_lookup',tick)
        return payload

    def write_metrics(self):
        tick = self.tick()
        distributions = {}
        for name, values in self.timings.items():
            d=values.distribution()
            distributions[name] = dict(count=d['calls'], total_seconds=d['total'],
                avg_seconds=d['mean'],p50_seconds=d['median'],p95_seconds=d['p95'],p99_seconds=d['p99'],
                quantiles_approximate=d['quantiles_approximate'],quantile_capacity=d['quantile_capacity'])
        inference = self.timings['OCR_inference'].total
        confidence = self.confidence.distribution()
        seat_timing = {f'T{t:02d}/{s}':{name:values.distribution()
            for name,values in stages.items()} for (t,s),stages in self.route_timings.items()}
        table_counts = defaultdict(Counter)
        for (table,seat),counts in self.routes_counts.items():
            table_counts[table].update(counts)
        metrics = dict(self.counts, process='RUNNING' if not self.closed else 'CLOSED',
            last_frame=self.last_frame, timing=distributions,
            ocr_rows_per_inference_second=self.counts['ocr_calls']/inference if inference else None,
            confidence_mean=confidence['mean'],confidence_min=confidence['min'],confidence_p50=confidence['median'],
            confidence_quantiles_approximate=confidence['quantiles_approximate'],
            per_seat={f'T{t:02d}/{s}':dict(c) for (t,s),c in self.routes_counts.items()},
            per_table={f'T{t:02d}':dict(c) for t,c in table_counts.items()},per_seat_timing=seat_timing,
            sqlite_bytes=self.path.stat().st_size, queue_depth=len(self.pending),
            pending_states=len(self.timeline.states.items),pending_observations=len(self.timeline.observations.items),
            hot_history_cache_states=len(self.state_cache),exact_index_entries=sum(len(p.index) for p in self.timeline.pools),
            batch_policy='256 observations OR 120 source frames OR 2 wall seconds; initial and final flush mandatory',
            png_transport_calls=0, conflicts=0, identity_status='ACTOR_UNBOUND',
            quality_verdict='STACK_PIPELINE_PARTIAL', known_false_admission='T06/seat2/frame96 Sit out',
            timing_notes='OCR shared batch timings overlap across seats; parent CPU excludes worker CPU, resource sampler records worker CPU/RAM; zero conflicts does not prove actor identity',
            pending_visual_candidates=sum(t.pending is not None for t in self.timeline.tracks))
        temporary = self.output / 'progress.tmp'
        temporary.write_text(json.dumps(metrics,indent=2),'utf-8')
        temporary.replace(self.output / 'progress.json')
        self.record('metrics_write',tick)

    def finish(self):
        if self.closed:
            return
        try:
            # Mandatory final flush includes current last_seen updates even without new states.
            self.flush()
        finally:
            if self.client is not None:
                self.client.close()
            self.receipts.close()
            self.rejected_receipts.close()
            self.db.close()
            self.closed = True
            self.write_metrics()
