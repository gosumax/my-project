"""Isolated pixel-only experiment. Never a production entrypoint.

Exact native row rasters + ordered overlap are a restricted reference, not
independent human ground truth. Unobservable full replacement remains UNKNOWN.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import sys
import time
import zlib
import base64

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import av
import cv2
import numpy as np
from parser_core.chat_rows import physical_spans, exclude_first_visible_span
from parser_core.layout import load_fixed_layout
from parser_core.ram_first_stage import FirstStage, png_compatible_gray


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), 'utf-8')


def capture(video, frames, out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT / 'production_manifest.json').read_text('utf-8'))
    hashes = {p: sha(ROOT / p) for p in manifest['hashes']}
    write_json(out / 'source_before.json', hashes)
    layout = load_fixed_layout(ROOT / 'configs/nine_table_layout.json', ROOT / 'configs/layout_847x404.json', ROOT / 'configs/chat_log_bottom_rois.json')
    sessions = {str(s.slot_id): f'experiment_slot_{s.slot_id:02d}' for s in layout.slots}
    views = {}
    def observer(meta, pixels):
        views[(meta['frame_id'], meta['table_id'], meta['source_span_index'])] = meta['stable_row_id']
    start = time.perf_counter()
    stage = FirstStage(video, sessions, frames, root=ROOT, identity_observer=observer, decode_threads=3)
    new_rows = 0
    for row in stage.rows():
        views[(row.frame_id, row.table_id, row.source_span_index)] = row.stable_row_id
        new_rows += 1
    baseline_seconds = time.perf_counter() - start
    write_json(out / 'baseline_views.json', [[*k, v] for k, v in views.items()])
    previous = {}
    start = time.perf_counter()
    n = 0
    with (out / 'states.jsonl').open('w', encoding='utf-8') as sink, av.open(str(video)) as src:
        stream = src.streams.video[0]
        stream.thread_type = 'FRAME'
        stream.codec_context.thread_count = 3
        for fid, frame in enumerate(src.decode(stream)):
            if fid >= frames:
                break
            rgb = frame.to_ndarray(format='rgb24')
            for slot in layout.slots:
                x1, y1, x2, y2 = slot.chat_box
                crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
                if slot.slot_id in previous and np.array_equal(previous[slot.slot_id], crop):
                    continue
                previous[slot.slot_id] = crop
                gray = png_compatible_gray(crop)
                record = {'frame': fid, 'table': slot.slot_id, 'pts': frame.pts,
                          'time_base': [frame.time_base.numerator, frame.time_base.denominator],
                          'box': list(slot.chat_box), 'shape': list(gray.shape),
                          'sha256': hashlib.sha256(gray.tobytes()).hexdigest(),
                          'rgb_sha256': hashlib.sha256(crop.tobytes()).hexdigest(),
                          'zlib_gray': base64.b64encode(zlib.compress(gray.tobytes())).decode()}
                sink.write(json.dumps(record) + '\n')
                n += 1
    write_json(out / 'capture.json', {'frames': stage.seen_frames, 'states': n, 'baseline_new_ids': new_rows,
               'baseline_seconds_including_decode': baseline_seconds, 'state_capture_seconds': time.perf_counter()-start,
               'baseline_metrics': dict(stage.metrics), 'video': str(video.resolve()), 'video_sha256': sha(video),
               'manifest_sha256': sha(ROOT/'production_manifest.json'), 'decode_threads': 3, 'OCR_calls': 0,
               'scope': 'ISOLATED_FIRST_STAGE_REPLAY_NO_PRODUCTION_LAUNCH'})


def reference(old, new, os, ns):
    """Strict native-raster, multiple ordered anchors; no OCR/canonical resize.

    A shift is known only if at least three exact rows vote for it, with no
    competing translation. Rows at other positions / changed rasters UNKNOWN.
    """
    votes = defaultdict(list)
    for j, b in enumerate(ns):
        for i, a in enumerate(os):
            if a[1]-a[0] != b[1]-b[0]:
                continue
            d = a[0]-b[0]
            if d < 0 or d > old.shape[0]-24:
                continue
            if np.array_equal(old[a[0]:a[1], :-12], new[b[0]:b[1], :-12]):
                votes[d].append((j, i))
    valid = []
    for d, pairs in votes.items():
        if len(pairs) < 3 or len({i for _, i in pairs}) != len(pairs) or len({j for j, _ in pairs}) != len(pairs):
            continue
        ordered = sorted(pairs)
        if [i for _, i in ordered] != sorted(i for _, i in ordered):
            continue
        valid.append((len(pairs), d, dict(pairs)))
    valid.sort(reverse=True, key=lambda x: x[0])
    if not valid or (len(valid)>1 and valid[1][0] >= 3):
        return None
    _, d, mapping = valid[0]
    # All nonmatching rows must be an appended suffix. Otherwise new vs pixel
    # modification cannot be independently resolved from these two images.
    missing = [j for j in range(len(ns)) if j not in mapping]
    if missing and min(missing) <= max(mapping):
        return {'shift': d, 'mapping': mapping, 'new_indices': [], 'count_known': False, 'unknown_indices': missing}
    # A changed bottom row is not necessarily an appended physical row. Reject
    # a suffix if it sits on any old row's translated coordinates.
    if any(abs((os[i][0]+os[i][1]-ns[j][0]-ns[j][1])/2-d)<=3
           for j in missing for i in range(len(os))):
        return {'shift': d, 'mapping': mapping, 'new_indices': [], 'count_known': False, 'unknown_indices': missing}
    return {'shift': d, 'mapping': mapping, 'new_indices': missing, 'count_known': True, 'unknown_indices': []}


def correlation(old, new, fixed=False):
    if fixed:
        # One minimal fix: remove fixed scrollbar/UI columns from comparison.
        old, new = old[:, :-12], new[:, :-12]
    h = old.shape[0]
    # Native-y exhaustive search; at least 48px overlapping evidence.
    scores = []
    for d in range(h-47):
        a, b = old[d:, ::3].astype(np.float32), new[:h-d, ::3].astype(np.float32)
        a -= a.mean(); b -= b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        score = float(np.sum(a*b)/denom) if denom > 0 else -1.
        scores.append(score)
    order = np.argsort(scores)[::-1]
    shift = int(order[0])
    second = max((scores[i] for i in range(len(scores)) if abs(i-shift)>2), default=-1.)
    return {'shift': shift, 'score': scores[shift], 'margin': scores[shift]-second,
            'runner_up_score': second, 'top_shifts': [[int(i), scores[i]] for i in order[:5]],
            'accepted': scores[shift] >= .90 and scores[shift]-second >= .02}


def row_ncc_reference(old,new,os,ns):
    """Independent LOCAL row-order pixel proxy, not a human identity oracle.

    Uses tight-band normalized pixels, local Pearson similarity and >=6-row
    contiguous unique overlap, never the whole-viewport NCC shift. Geometry
    spread is retained; local mutation and unproven suffix remain UNKNOWN.
    """
    if not os or not ns: return None
    def vectors(gray,spans):
        rows=np.array([cv2.resize(gray[a:b,:-12],(108,16),interpolation=cv2.INTER_AREA).ravel() for a,b in spans],np.float64)
        rows-=rows.mean(axis=1,keepdims=True)
        norms=np.linalg.norm(rows,axis=1,keepdims=True)
        return np.divide(rows,norms,out=np.zeros_like(rows),where=norms>0)
    matrix=vectors(new,ns)@vectors(old,os).T
    candidates=[]
    for offset in range(-len(ns)+1,len(os)):
        pairs=[(j,j+offset) for j in range(len(ns)) if 0<=j+offset<len(os)]
        run=[]
        for j,i in pairs:
            delta=(os[i][0]+os[i][1]-ns[j][0]-ns[j][1])/2
            if matrix[j,i]>=.93 and delta>=-2:
                run.append((j,i))
            else:
                if len(run)>=6:candidates.append(run)
                run=[]
        if len(run)>=6:candidates.append(run)
    if not candidates:return None
    candidates.sort(key=lambda r:(len(r),float(np.mean([matrix[j,i] for j,i in r]))),reverse=True)
    best=candidates[0]
    if len(candidates)>1 and len(candidates[1])>=len(best)-1:return None
    mapping=dict(best)
    # Extend same ordered diagonal only on local similarity; don't bridge an
    # unknown changed row into a claimed physical birth.
    offset=best[0][1]-best[0][0]
    for j in range(len(ns)):
        i=j+offset
        if 0<=i<len(os) and matrix[j,i]>=.93:mapping[j]=i
    displacements=[(os[i][0]+os[i][1]-ns[j][0]-ns[j][1])/2 for j,i in mapping.items()]
    if max(displacements)-min(displacements)>3:return None
    shift=float(np.median(displacements))
    missing=[j for j in range(len(ns)) if j not in mapping]
    known=not missing or (min(missing)>max(mapping) and
          all(all(abs((os[i][0]+os[i][1]-ns[j][0]-ns[j][1])/2-shift)>3 for i in range(len(os))) for j in missing))
    return {'shift':shift,'mapping':mapping,'new_indices':missing if known else [],'count_known':known,
            'unknown_indices':[] if known else missing,'reference_type':'LOCAL_ROW_NCC_ORDER_PROXY',
            'row_displacements':displacements,'row_scores':{j:float(matrix[j,i]) for j,i in mapping.items()},
            'displacement_range':max(displacements)-min(displacements)}


def coordinate_map(os, ns, shift, tolerance=2):
    mapping = {}
    used = set()
    for j, b in enumerate(ns):
        distances = [(abs((a[0]+a[1]-b[0]-b[1])/2-shift), i) for i,a in enumerate(os) if i not in used]
        if distances:
            delta, i = min(distances)
            if delta <= tolerance:
                mapping[j] = i; used.add(i)
    return mapping


def evaluate(out, cached_fast=False, tolerant_reference=False):
    cached = {}
    if cached_fast:
        for line in (out/'fast_correlations.jsonl').open(encoding='utf-8'):
            r=json.loads(line); cached[(r['table'],r['after_frame'])]=r['algorithms']
    prefix = 'row_reference_' if tolerant_reference else ('after_support_' if cached_fast else '')
    records = []
    for line in (out/'states.jsonl').open(encoding='utf-8'):
        r = json.loads(line)
        r['gray'] = np.frombuffer(zlib.decompress(base64.b64decode(r.pop('zlib_gray'))), np.uint8).reshape(r['shape']).copy()
        assert hashlib.sha256(r['gray'].tobytes()).hexdigest() == r['sha256']
        r['spans'] = exclude_first_visible_span(physical_spans(r['gray']))[0]
        records.append(r)
    baseline = {(f,t,i): rid for f,t,i,rid in json.loads((out/'baseline_views.json').read_text('utf-8'))}
    # Measurement on first third; evaluation remains fixed on later frames.
    cut = max(r['frame'] for r in records)//3
    gaps = Counter(round((b[0]+b[1]-a[0]-a[1])/2, 1) for r in records if r['frame']<=cut for a,b in zip(r['spans'], r['spans'][1:]))
    step = float(gaps.most_common(1)[0][0])
    stats = {arm: Counter() for arm in ['OLD', 'NCC_RAW', 'NCC_MASK', 'NCC_MASK_APPEND']}
    if cached_fast:
        stats['NCC_SUPPORT'] = Counter(); stats['NCC_STATIONARY'] = Counter()
    shifts = Counter(); fractions = Counter(); reasons = Counter(); coverage = Counter()
    previous = {}; id_states = {}; next_id = Counter(); timings = Counter(); failures = []; transitions = []
    seen_ids = defaultdict(set)
    all_views = []; evidence_cases = []
    for r in records:
        t, fid, new, ns = r['table'], r['frame'], r['gray'], r['spans']
        prev = previous.get(t)
        outputs = {}
        if prev:
            old, os = prev['gray'], prev['spans']
            tick = time.perf_counter(); ref = (row_ncc_reference if tolerant_reference else reference)(old,new,os,ns); timings['reference'] += time.perf_counter()-tick
            if cached_fast:
                raw=cached[(t,fid)]['NCC_RAW']; masked=cached[(t,fid)]['NCC_MASK']
            else:
                tick = time.perf_counter(); raw = correlation(old,new); timings['NCC_RAW'] += time.perf_counter()-tick
                tick = time.perf_counter(); masked = correlation(old,new,True); timings['NCC_MASK'] += time.perf_counter()-tick
            arms=[('NCC_RAW',raw),('NCC_MASK',masked),('NCC_MASK_APPEND',masked)]
            if cached_fast:
                arms.append(('NCC_SUPPORT',cached[(t,fid)]['NCC_SUPPORT']))
                arms.append(('NCC_STATIONARY',cached[(t,fid)]['NCC_SUPPORT']))
            for name, alg in arms:
                mapping = coordinate_map(os,ns,alg['shift']) if alg['accepted'] else {}
                predicted = round(alg['shift']/step) if alg['accepted'] else None
                fallback=None
                if name=='NCC_STATIONARY' and not alg['accepted']:
                    anchors=reference(old,new,os,ns)
                    if anchors and anchors['shift']==0:
                        mapping=anchors['mapping'];fallback='EXACT_STATIONARY_ANCHORS_NO_BIRTH_CLAIM'
                if name in ['NCC_MASK_APPEND','NCC_SUPPORT'] and alg['accepted']:
                    predicted = sum(j not in mapping and (not mapping or j>max(mapping)) for j in range(len(ns)))
                outputs[name] = {'algorithm': alg, 'mapping': mapping, 'predicted': predicted,'fallback':fallback}
            if fid>cut:
                coverage['transitions'] += 1
                if ref:
                    coverage['shift_known'] += 1; shifts[ref['shift']] += 1
                    fractions['near_integer_0.10'] += abs(ref['shift']/step-round(ref['shift']/step)) <= .10
                    if ref['count_known']:
                        coverage['count_known'] += 1; coverage['reference_new_rows'] += len(ref['new_indices'])
                        coverage[f"new_count_{len(ref['new_indices'])}"] += 1
                    else:
                        coverage['count_unknown'] += 1
                else:
                    coverage['shift_unknown'] += 1
            rec = {'table':t,'before_frame':prev['frame'],'after_frame':fid,'row_step_px':step,
                   'before_sha256':prev['sha256'],'after_sha256':r['sha256'],
                   'reference':ref,'algorithms':outputs,'evaluation':fid>cut}
            transitions.append(rec)
            if fid>cut and (ref is None or not ref['count_known']):
                evidence_cases.append((rec,old,new,'REFERENCE','UNKNOWN'))
        for arm in stats:
            old_ids = id_states.get((t,arm), [])
            if arm=='OLD':
                ids = [baseline[(fid,t,j+1)] for j in range(len(ns))]
            else:
                mapping = outputs.get(arm,{}).get('mapping',{})
                ids = []
                for j in range(len(ns)):
                    if j in mapping:
                        ids.append(old_ids[mapping[j]])
                    else:
                        next_id[(t,arm)] += 1; ids.append(f'{arm}:T{t}:R{next_id[(t,arm)]}')
            if prev and fid>cut:
                stats[arm]['visible_observations'] += len(ns)
                if ref:
                    errors = []
                    for j,i in ref['mapping'].items():
                        if ids[j] != old_ids[i]:
                            stats[arm]['broken_row_id'] += 1
                            if ids[j] not in seen_ids[(t,arm)]:
                                stats[arm]['old_as_new'] += 1; stats[arm]['duplicate_births'] += 1
                            elif ids[j] not in old_ids:
                                stats[arm]['historical_id_disagreement'] += 1
                            errors.append({'new_index':j,'expected_old_index':i,'expected_id':old_ids[i],'actual_id':ids[j]})
                    if ref['count_known']:
                        for j in ref['new_indices']:
                            if ids[j] in old_ids:
                                stats[arm]['new_as_old'] += 1; stats[arm]['lost_births'] += 1
                                errors.append({'new_index':j,'expected':'NEW','actual_id':ids[j]})
                        if arm!='OLD':
                            pred = outputs[arm]['predicted']
                            if pred is None:
                                stats[arm]['count_abstained'] += 1
                            else:
                                stats[arm]['count_predictions'] += 1; stats[arm]['predicted_new_rows'] += pred
                                stats[arm]['true_rows_on_predicted_pairs'] += len(ref['new_indices'])
                                stats[arm]['count_errors'] += pred != len(ref['new_indices'])
                                if pred != len(ref['new_indices']):
                                    errors.append({'expected_new_count':len(ref['new_indices']),'predicted_new_count':pred})
                    if errors:
                        alg = outputs.get(arm,{}).get('algorithm')
                        cause = 'UNKNOWN'
                        if alg:
                            if alg['margin'] < .02: cause='NCC_AMBIGUOUS'
                            elif abs(alg['shift']-ref['shift'])>1.5: cause='UNKNOWN'
                            elif ref['shift']==0: cause='LOCAL_CHANGE_WITHOUT_SCROLL'
                            elif abs(ref['shift']/step-round(ref['shift']/step))>.10: cause='UNKNOWN'
                        reasons[(arm,cause)] += 1
                        failures.append({**rec,'arm':arm,'cause':cause,'cause_status':'OBSERVED_ARITHMETIC_FAILURE' if cause=='LOCAL_CHANGE_WITHOUT_SCROLL' else 'HYPOTHESIS_OR_UNKNOWN',
                                         'symptom':'SHIFT_DISAGREEMENT' if alg and abs(alg['shift']-ref['shift'])>1.5 else 'COUNT_OR_ID_DISAGREEMENT','errors':errors})
                        evidence_cases.append((rec,old,new,arm,cause))
                else:
                    stats[arm]['unknown_transition_observations'] += len(ns)
            id_states[(t,arm)] = ids
            seen_ids[(t,arm)].update(ids)
            all_views.append({'frame':fid,'table':t,'arm':arm,'spans':ns,'row_ids':ids})
        previous[t] = r
    for name,data in [('transitions',transitions),('failures',failures),('tracker_views',all_views)]:
        with (out/f'{prefix}{name}.jsonl').open('w',encoding='utf-8') as f:
            for row in data: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    # Every error gets its original native before/after crop; images shared by
    # arms to avoid redundant files. Raw complete states also retained JSONL.
    gallery = out/'cases'; gallery.mkdir(exist_ok=True)
    entries=[]; saved=set()
    for rec,old,new,arm,cause in evidence_cases:
        key=f"T{rec['table']:02d}_{rec['before_frame']:06d}_{rec['after_frame']:06d}"
        if key not in saved:
            cv2.imwrite(str(gallery/f'{key}_before.png'),old)
            cv2.imwrite(str(gallery/f'{key}_after.png'),new)
            saved.add(key)
        entries.append(f'<div><h3>{key} {arm}: {cause}</h3><img src="cases/{key}_before.png"><img src="cases/{key}_after.png"><pre>{html.escape(json.dumps(rec,ensure_ascii=False,indent=2))}</pre></div>')
    (out/f'{prefix}review.html').write_text('<meta charset="utf-8"><style>body{background:#222;color:#eee;font-family:monospace}img{image-rendering:pixelated;width:600px}pre{white-space:pre-wrap}</style><h1>Pixel-only restricted reference; UNKNOWN is not ground truth</h1>'+''.join(entries),'utf-8')
    before=json.loads((out/'source_before.json').read_text('utf-8'))
    changed=[p for p,h in before.items() if sha(ROOT/p)!=h]
    summary={'row_step_px':step,'calibration_end_frame':cut,'row_center_gap_histogram':dict(gaps),
             'reference_shift_histogram':dict(shifts),'integer_ratio':dict(fractions),'coverage':dict(coverage),
             'metrics':{a:dict(s) for a,s in stats.items()},'failure_reasons':{f'{a}:{b}':v for (a,b),v in reasons.items()},
             'timing_seconds':dict(timings),'evidence_error_pairs':len(saved),'production_changed':changed,
             'reference_contract':'EXACT_NATIVE_ROW_RASTERS_MIN3_ORDERED_ANCHORS_UNIQUE_TRANSLATION; APPENDED_SUFFIX_CONDITIONAL; NOT_HUMAN_GT',
             'metrics_contract':'Conditional adjacent-state ID disagreements on exact translated raster anchors. duplicate_births count fresh candidate IDs on old raster anchors, historical recovery separate. lost_births are inherited IDs on geometrically new visible suffix. Not absolute global lost/duplicate physical lines.',
             'verdict':'NOT PROVEN','limitations':['UNKNOWN shifts and inside-row mutations not scored as true births','No independent complete physical-line ground truth','References and arms share physical_spans segmentation; missed bands cannot be measured','No full-video or HH certification']}
    if cached_fast: summary['timing_seconds'] = json.loads((out/'fast_ncc_parity.json').read_text('utf-8'))['timing_seconds']
    if tolerant_reference:
        summary['reference_contract']='LOCAL_ROW_NCC_GEOMETRY_ORDER_PROXY_MIN6_CONTIGUOUS_ROWS_SCORE0.93; CONDITIONAL_GEOMETRIC_SUFFIX; NOT_INDEPENDENT_HUMAN_GT'
        summary['metrics_contract']=summary['metrics_contract'].replace('exact translated raster anchors','local ordered row NCC pixel proxy anchors')
    write_json(out/f'{prefix}summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['capture','evaluate']); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--video',type=Path,default=ROOT/'video/smoke_17.mp4'); p.add_argument('--frames',type=int,default=3000)
    p.add_argument('--cached-fast',action='store_true')
    p.add_argument('--tolerant-reference',action='store_true')
    a=p.parse_args()
    if a.mode=='capture': capture(a.video,a.frames,a.output)
    else: evaluate(a.output,a.cached_fast,a.tolerant_reference)
