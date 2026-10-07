"""Read frozen evidence and emit diagnostic statuses; never classify tracks."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/cross_track_condition_audit_smoke17_20261001'

def main():
    OUT.mkdir(exist_ok=True)
    records = [json.loads(s) for s in (ROOT/'runs/cross_track_linker_smoke17_20261001/relationships.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = []
    for r in records:
        p, f = r['motion'], r['features']
        mapped = p['target_new_index'] is not None
        exact = p['exact_target_content']
        anchors = len(p['neighbor_anchors'])
        adjacent = f['delta_frame']==1 or f['delta_frame']==2 and p['local_loss_bridge']
        causal_frames = [r['old'][k] for k in ('source_frame','first_frame','last_frame')]+[r['new'][k] for k in ('source_frame','first_frame','last_frame')]+p['evidence_frames']
        causal_frames += [v for ep in ('old','new') for span in r[ep]['visible_spans'] for v in span]
        if r['onset']: causal_frames += r['onset']['evidence_frames']
        causal_frames += f['hand_boundary_frames']
        assert max(causal_frames)<=r['decision_frame']==r['new']['first_frame']
        status = lambda value: 'PASS' if value else 'FAIL'
        c = dict(same_table=status(f['same_table']), same_session=status(f['same_session']),
            causal_temporal_order=status(r['old']['first_frame']<r['new']['first_frame'] and r['old']['last_frame']<r['new']['first_frame']),
            immediate_viewport_continuity=status(adjacent), viewport_alignment=status(p['aligned']),
            target_row_match=status(mapped and p['target_new_index']==p['candidate_new_index']),
            unique_row_match=status(p['target_unique']) if exact else 'UNKNOWN',
            same_normalized_text=status(bool(r['old']['normalized_text']) and r['old']['normalized_text']==r['new']['normalized_text']),
            text_uniqueness='UNKNOWN', pixel_similarity_strength='UNKNOWN',
            exact_pixel_content=status(exact) if mapped else 'UNKNOWN',
            coherent_neighbor_movement=status(anchors>0) if mapped else 'UNKNOWN',
            one_neighbor=status(anchors>=1) if mapped else 'UNKNOWN',
            two_neighbors=status(anchors>=2) if mapped else 'UNKNOWN',
            no_visibility_gap=status(all(len(r[ep]['visible_spans'])==1 for ep in ('old','new'))),
            no_simultaneous_visibility=status(not f['causal_visibility_overlap']),
            no_boundary_conflict=status(not f['hand_boundary_frames']), no_epoch_conflict='UNKNOWN',
            causal_snapshot_available='PASS', pixel_evidence_available=status(f['pixel_similarity'] is not None),
            text_evidence_available=status(bool(r['old']['normalized_text']) and bool(r['new']['normalized_text'])))
        blockers = []
        if not p['aligned']: blockers.append('NO_VIEWPORT_ALIGNMENT')
        if not mapped: blockers.append('NO_TARGET_ROW_MATCH')
        if not exact: blockers.append('NO_EXACT_PIXEL_PROOF')
        if not p['target_unique']: blockers.append('NO_UNIQUE_EXACT_ROW_PROOF')
        if anchors<2: blockers.append('NO_2_NEIGHBORS')
        if not adjacent: blockers.append('NO_IMMEDIATE_VIEWPORT_LINK')
        if f['hand_boundary_frames']: blockers.append('BOUNDARY_CONFLICT')
        # Informational observations are not invented classifier vetoes.
        observations = ['VISIBILITY_GAP'] if c['no_visibility_gap']=='FAIL' else []
        # Actual short-circuit of the existing strong expression, including dependency.
        primary = 'NO_VIEWPORT_ALIGNMENT' if not p['aligned'] else 'NO_UNIQUE_EXACT_ROW_PROOF'
        required = ['same_table','same_session','no_simultaneous_visibility','viewport_alignment','target_row_match','unique_row_match','exact_pixel_content','two_neighbors','immediate_viewport_continuity','same_normalized_text','no_boundary_conflict']
        missing = [k for k in required if c[k]!='PASS']
        rows.append(dict(pair_id=r['pair_id'],original_verdict=r['link_status'],decision_frame=r['decision_frame'],
            conditions=c,blocking_reasons=blockers,observations=observations,primary_block_reason=primary,
            returned_reason=r['link_reason'],missing_same_conditions=missing,missing_count=len(missing),
            metrics={k:f[k] for k in ('pixel_similarity','text_similarity','delta_ms','delta_y')},
            causal_evidence_frames=sorted(set(causal_frames))))
    assert len(rows)==590
    neighbor_path=OUT/'independent_neighbors.jsonl'
    if neighbor_path.exists():
        neighbors={r['pair_id']:r for r in (json.loads(line) for line in neighbor_path.read_text(encoding='utf-8').splitlines())}
        for r in rows:
            v=neighbors[r['pair_id']]['target_y_existing_tolerance_pass']
            r['conditions']['y_position_continuity']='UNKNOWN' if v is None else ('PASS' if v else 'FAIL')
            r['independent_neighbor_measurement']=neighbors[r['pair_id']]
    summary=dict(pair_count=len(rows),condition_matrix={k:dict(Counter(r['conditions'][k] for r in rows)) for k in rows[0]['conditions']},
        blocking_reasons=dict(Counter(b for r in rows for b in r['blocking_reasons'])),
        primary_block_reasons=dict(Counter(r['primary_block_reason'] for r in rows)),
        returned_reasons=dict(Counter(r['returned_reason'] for r in rows)),
        near_same=dict(only1=sum(r['missing_count']==1 for r in rows),only2=sum(r['missing_count']==2 for r in rows),
            three_plus=sum(r['missing_count']>=3 for r in rows),histogram=dict(Counter(r['missing_count'] for r in rows))),
        causality='All consumed snapshot and motion frames <= new.first_frame; no terminal lifetimes used',
        status_semantics='UNKNOWN includes prerequisite-skipped and absent predicate; source booleans preserved. Blockers count absent proof, not independently tested failure.',
        unchanged_verdicts=dict(Counter(r['original_verdict'] for r in rows)))
    (OUT/'condition_matrix.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (OUT/'condition_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))

def report():
    s=json.loads((OUT/'condition_summary.json').read_text(encoding='utf-8'))
    d=json.loads((ROOT/'runs/cross_track_condition_inspection_20261001/summary.json').read_text(encoding='utf-8'))
    n=json.loads((OUT/'neighbor_summary.json').read_text(encoding='utf-8'))
    depth=json.loads((OUT/'pixel_change_summary.json').read_text(encoding='utf-8'))
    lines=['# Cross-track condition audit — Smoke17', '',
        'Диагностика замороженных 590 candidate pairs. Verdict каждого трека сохранён: SAME=0, DIFFERENT=0, AMBIGUOUS=590. Правила, thresholds, два соседа, source rows, stable IDs, Jev и edge-gate не изменены.', '',
        'FILE → FUNCTION → CALLER → DATA: `parser_core/chat_rows.py` → `align_upward / match_rows / match_row_blocks` → `ram_first_stage.py:FirstStage.rows` → непронесённая identity / новый UNRESOLVED_IDENTITY. Эксперимент `cross_track_linker.py:motion_proof` повторяет физическое сопоставление, `decide` проверяет conjunction. Это эксперимент, а production определён только PRODUCTION.md и production_manifest.json.', '',
        '## 590 PAIRS CONDITION MATRIX', '',
        'PASS — условие проверено и выполнено; FAIL — проверено и не выполнено; UNKNOWN — проверка пропущена из-за prerequisite либо в действующих правилах нет такой бинарной проверки. Отсутствующего causal snapshot, pixels или text нет: NO_EVIDENCE=0 для всех трёх. UNKNOWN не равен доказанному конфликту.', '',
        '| CONDITION | PASS | FAIL | UNKNOWN |', '|---|---:|---:|---:|']
    for k,v in s['condition_matrix'].items(): lines.append(f"| {k} | {v.get('PASS',0)} | {v.get('FAIL',0)} | {v.get('UNKNOWN',0)} |")
    lines += ['', 'Y continuity дополнительно измерена на tight spans после компенсирования доказанного viewport shift; результаты и определения ниже. Δy padded crop само по себе не является условием отказа.', '',
        'Нулевой target_unique не означает 590 доказанных неоднозначных строк: проверка уникальности запускается только после exact target, а exact target не получен ни разу. Pixel similarity — непрерывная MAE-метрика; отдельного порога PIXEL_NOT_STRONG_ENOUGH в classifier нет. Действующее требование — точное совпадение native content без scrollbar. Text similarity=1 не проверяет уникальность физической строки. TEXT_NOT_UNIQUE и EPOCH_CONFLICT нельзя объявить измеренными blockers: таких доказательств здесь нет.', '',
        '## BLOCKING REASONS', '',
        'Числа означают отсутствие необходимого proof; одна пара имеет несколько причин. Для NO_2_NEIGHBORS 491 проверка gated/skipped и 97 проверены без двух anchors. Поэтому 588 не означает 588 физически пустых viewport.', '',
        '| BLOCK_REASON | COUNT | % от 590 |', '|---|---:|---:|']
    for k,v in sorted(s['blocking_reasons'].items(),key=lambda kv:-kv[1]): lines.append(f'| {k} | {v} | {v/590*100:.2f}% |')
    lines += ['| MISSING_CAUSAL_SNAPSHOT | 0 | 0.00% |','| MISSING_PIXEL_EVIDENCE | 0 | 0.00% |','| MISSING_TEXT_EVIDENCE | 0 | 0.00% |', '',
        'VISIBILITY_GAP: 1 causal-prefix наблюдение (0.17%); это отдельное наблюдение, а не самостоятельный veto в decide. Simultaneous visibility: 0. Epoch endpoint context не записан: 590 UNKNOWN. Исходный epoch первой source row не доказывает epoch последнего viewport.', '',
        '## PRIMARY BLOCKING REASON', '',
        'У classifier нет отдельного primary reason для каждого conjunct: он возвращает 586 INSUFFICIENT_COMBINED_PHYSICAL_CONTINUITY и 4 BOUNDARY_BLOCKS_SAME_WITHOUT_PHYSICAL_ONSET_PROOF. Для first failed conjunct существующего выражения strong порядок ровно такой: aligned → target_unique → exact_target_content → two neighbors.', '',
        '| PRIMARY_BLOCK_REASON в strong | COUNT | % |','|---|---:|---:|',
        '| NO_VIEWPORT_ALIGNMENT | 22 | 3.73% |','| NO_UNIQUE_EXACT_ROW_PROOF | 568 | 96.27% |', '',
        'Этот порядок объясняет выполнение кода, но скрывает prerequisite: при построении proof сначала отсутствует target mapping у 491 пары (83.22%), а у оставшихся 99 (16.78%) mapped target не совпал с raster точно. Именно поэтому uniqueness не проверена. Boundary — отдельная возвращаемая причина для четырёх пар, не подмена first strong conjunct.', '',
        '## NEAR-SAME CASES', '',
        'Подсчёт по 11 необходимым условиям: table, session, no simultaneous visibility, alignment, target mapped to candidate, exact pixels, unique exact row, two neighbors, immediate continuity, nonempty equal normalized text, no boundary. FAIL и UNKNOWN считаются отсутствующим доказательством. Зависимые условия явно считаются отдельно; это не независимые наблюдения.', '',
        f"Missing only 1: **{s['near_same']['only1']}**; missing only 2: **{s['near_same']['only2']}**; missing 3+: **{s['near_same']['three_plus']}**. Histogram: `{s['near_same']['histogram']}`.", '',
        'Случаев «text PASS, exact pixel PASS, time PASS, y PASS, 1 neighbor PASS, только 2nd FAIL»: **0**. Exact pixel PASS=0. Более сильное независимое препятствие: mapped target=99, immediate continuity=359, их пересечение=0. Даже игнорирование exact/unique/neighbors не даёт пару с совместными alignment + candidate mapping + immediate continuity. Это диагностическая проверка conjunction, не альтернативная классификация.', '',
        'Две пары с сохранёнными 2+ anchors имеют 3/4-frame handoff (100/133.333ms), Δy=0 и similarity≈0.999598/0.999930. Обе всё равно без exact target и unique proof, а handoff вне допустимого immediate link. Это ближайшие примеры с соседями; SAME им не присвоен.', '',
        '## NEIGHBOR BOTTLENECK', '',
        'Сохранённые target-associated anchors: 0 — 588; 1 — 0; 2+ — 2. У 491 пары подсчёт пропущен: target mapping отсутствует. Независимые измерения выполнены заново на тех же causal endpoint ROIs, без prerequisite целевого соответствия и без использования для verdict.', '',
        'Числовой итог независимого измерения (machine-readable, определения полей в скрипте; source hashes сохранены в JSON artifact):', '', '```json', json.dumps({k:v for k,v in n.items() if k!='source_hashes'},ensure_ascii=False,indent=2), '```', '',
        'Независимые viewport neighbors не равны anchors блока, содержащего target. Top/bottom position сама по себе не исключает двух соседей: они могут быть на одной стороне. Физическая невозможность установлена только когда в одном endpoint меньше двух остальных eligible строк; исключённая первая visible строка отмечена отдельно. Изменение IDs у визуальных соседей — корреляция. UNRESOLVED_IDENTITY не является фильтром anchors в действующем linker, поэтому нельзя объявить его причиной потери второго соседа.', '',
        'Ровно один strict independent neighbor: `table_deb776c52437ea38b8ca8db5:R000601 → R000619`, frames 16537→16538. Target center residual=1px, но registered row_distance=10.698893 >10 и native raster unequal. Не хватает не только второго соседа: target mapping не проходит pixel-distance gate.', '',
        '## WHY 491 HAVE NO OLD ROW → NEW ROW MAPPING', '',
        'Проверен old target против ВСЕХ new spans, затем воспроизведены sorted(distance+delta), one-to-one greedy assignment и crossed-order reset действующего match_rows. Это объяснение фактической работы существующего кода, без нового порога.', '',
        '| Причина отсутствия mapping | COUNT | % от 491 |','|---|---:|---:|',
        '| Alignment не доказан; match_rows не запущен | 22 | 4.48% |',
        '| Есть строки в пределах center residual ≤7px, но у всех row_distance >10 | 469 | 95.52% |',
        '| Нет new row в допустимой геометрии | 0 | 0.00% |',
        '| Eligible target потерян из-за one-to-one competition | 0 | 0.00% |',
        '| Mapping сброшен из-за crossed order | 0 | 0.00% |', '',
        '**Главный измеренный отказ — pixel distance, а не y-position или конкуренция IDs.** Row distance считается на tight row, с masked scrollbar, resize 256×24 и средней absolute grayscale difference. Это другая метрика, чем сохранённая similarity padded crop: фон/padding могут давать similarity≈0.97 одновременно с distance>10 у текста. Наличие похожего текста не отменяет фактический pixel gate.', '',
        '## WHAT CHANGES IN THE 99 MAPPED RASTERS', '',
        '| Измерение после исключения scrollbar | COUNT |','|---|---:|',
        '| Одинаковые размеры tight raster | 72 |','| Различается высота tight raster / segmentation extent | 27 |',
        '| Есть изменения внутри crop, за пределами outer 2px | 99 |',
        '| Есть изменения active pixels (хотя бы один endpoint >70) | 99 |',
        '| Есть изменения dark background pixels (оба endpoints ≤70) | 99 |',
        '| Pixels пересекают границу intensity >70 | 99 |',
        '| Малый vertical offset −2..2 улучшает MAE, best offset ≠0 | 19 |',
        '| Exact совпадение даже cropped overlap при −2..2px | 0 |', '',
        'Положение согласуется с найденным viewport transform, но native content не равен. У 27 случаев меняется segmentation/crop extent; это не объясняет остальные 72. У всех 99 differences заходят внутрь строки и затрагивают как светлое содержимое, так и тёмный фон; они не ограничены только краями crop. Для 19 случаев небольшой дополнительный сдвиг уменьшает ошибку, но ни разу не восстанавливает exact cropped overlap. Поэтому простой integer chat shift или только border mismatch не объясняет весь отказ.', '',
        'Это доказывает grayscale и support differences, но **не определяет отдельно антиалиасинг, lossy video compression, перерисовку UI или ошибочное физическое соответствие**. Threshold-crossing >70 сам по себе не доказывает изменение символа: небольшая смена оттенка тоже может пересечь 70. Для причинной атрибуции AA/compression нужна отдельная исходная raster/codec/render evidence; здесь её нет. Правила не ослаблены.', '',
        'Полные per-pair dimensions, pixel counts, MAE/max, row/column counts и offset diagnostics находятся в `pixel_change_details.jsonl`; причины каждого из 491 mapping — в `mapping_rejection_details.jsonl`. Descriptor summary:', '',
        '```json',json.dumps({k:v for k,v in depth.items() if k!='source_hashes'},ensure_ascii=False,indent=2),'```','',
        '## PIXEL/TEXT/TIME/Y DISTRIBUTIONS', '',
        'Quantiles: NumPy linear; time — exact PTS new.first − causal old.last, ms; y — signed padded crop Δy, px; pixel — padded crop MAE similarity. Text — normalized SequenceMatcher. Frame gap считается по decoded indices, не по fps. Boundary-near ниже означает наличие separating confirmed boundary внутри old.last < boundary ≤ new.first; произвольное окно «near» не добавлено.', '']
    for name in ('all','gap1','gap_gt1','causal_visibility_gap','causal_boundary_present'):
        g=d['groups'][name]
        lines += [f"### {name} (n={g['count']})", '', '| METRIC | min | p10 | p25 | median | p75 | p90 | max |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for key,v in g['metrics'].items():
            q=v['quantiles']; vals=[v['min'],q['q10'],q['q25'],q['q50'],q['q75'],q['q90'],v['max']]
            lines.append('| '+key+' | '+' | '.join(f'{x:.6f}' for x in vals)+' |')
        lines.append('')
    lines += ['Pixel/text/time/y доступны для 590/590; missing=0. Группа gap>1 содержит 231 пару: 7 gap2, 224 gap>2. Ни один gap2 не получил local_loss_bridge. Высокая average similarity не доказывает точное соответствие glyphs: фон и padding входят в метрику.', '',
        '## WHY DIFFERENT = 0', '',
        '| DIFFERENT witness / observation | COUNT |','|---|---:|',
        '| Different table/session | 0 |','| Causal simultaneous distinct physical positions | 0 |',
        '| Old target proved continuing at another current row | 0 |',
        '| Proven newly exposed bottom onset | 0 |','| Full boundary + onset + two neighbors predicate | 0 |',
        '| Different normalized text | 0 |','| Strong positive physical conflict satisfying current rules | 0 |',
        '| Outside immediate SAME handoff window | 231 |','| Gap >2 frames | 224 |', '',
        'Ни одна фактическая пара не дала simultaneous witness. Если бы distinct rows были simultaneously visible, существующая ветка могла бы доказать DIFFERENT, но для 590 таких фактов нет: hypothetical count нельзя превратить в доказательство. Incompatible pixels/text: текстовых конфликтов 0; exact mismatch / отсутствие exact proof не является доказательством разных строк. Бинарного порога pixel incompatibility нет, поэтому это UNKNOWN, а не 590 DIFFERENT. Максимальный causal gap=86.033s; отдельного time-only DIFFERENT threshold нет. Onset существует у 417, aligned у 403, 2+ onset anchors у 13, newly exposed у 0. Ни длинный gap, ни boundary без physical onset не доказывает новую строку.', '',
        'Эти данные показывают недостаток positive DIFFERENT witnesses. Они не доказывают, что DIFFERENT classifier ошибочно строг: нет независимой ground truth, подтверждающей пропущенные DIFFERENT.', '',
        '## CAUSAL SAFETY', '',
        'Все 590 condition rows проверяют snapshot source/first/last/spans, motion/onset и boundary frames ≤ new.first_frame. Независимые pixels получены только из causal old.last и new.first. Terminal metadata не участвует в conditions, blockers, neighbors или verdict. OCR source frame уже ≤ decision; используется сохранённое исходное чтение, новый OCR не запускается. Это аудит заранее заданного offline candidate set, а не доказательство causal online candidate generation.', '',
        'Source/production/video/frozen linker hashes совпали до/после обоих endpoint replay; глобальный image inventory 296336 не изменился, image writes=0. Provenance limitation: первый neighbor run не имел full-byte pin diagnostic script на старте; добавление неиспользуемой depth-функции происходило во время него. Для depth script SHA зафиксирован во время выполнения и сверён после; это не выдаётся за start-time pin. Loaded neighbor measurement, causal input hashes и frozen classifier не менялись. Independent aggregation подтверждает числовые результаты; это не visual SAME ground truth.', '',
        '## POST_HOC_CHECK', '',
        'Отдельная offline validation по full lifetimes: 4 terminal envelopes interleave, 12 пар имеют visibility gap хотя бы одного полного lifetime; actual co-visibility=0 по полным spans и 54,631 chat states. При causal prefix visibility-gap count=1. Четыре будущих old reappearances не подаются в решение. POST_HOC_CHECK не меняет ни одного verdict.', '',
        '## EDGE-GATE IMPACT', '',
        '**6685 → 6685 UNRESOLVED_EDGE**, safely resolved=0; source rows **8146 → 8146**. Это сохранённая исходная simulation; ничего не применяется. Gate не потребляет experimental relationship metadata. Ни edge-gate, ни эпохи/границы, ни IDs/source rows не переписаны.', '',
        '## VERDICT', '',
        '**EVIDENCE_NOT_ENOUGH** при действующих замороженных правилах. Самого требования второго соседа недостаточно, чтобы объяснить все отказы: раньше не получен доказанный unique exact target, и нет пары с одновременно immediate handoff и mapped candidate. Это не сертификат оптимальности thresholds; RULES_TOO_STRICT без независимого подтверждения физических SAME не доказан.', '',
        '**Чаще всего не хватает доказательства, что конкретная старая строка физически продолжилась именно в конкретной новой строке: её pixels должны совпасть точно, положение — согласоваться с viewport movement, и это соответствие должно быть уникальным в обоих viewport.** Одинакового текста и высокой средней похожести картинки здесь недостаточно. Два согласованных соседа также часто не доказаны, но их проверка обычно не достигается из-за отсутствия target mapping.', '',
        '## ARTIFACTS', '',
        '- `runs/cross_track_condition_audit_smoke17_20261001/condition_matrix.jsonl`: все 590 pairs, statuses, blockers, primary, metrics, missing conditions и causal frames.',
        '- `runs/cross_track_condition_audit_smoke17_20261001/independent_neighbors.jsonl`: независимое измерение neighbors/y на causal ROIs.',
        '- `runs/cross_track_condition_audit_smoke17_20261001/condition_summary.json` / `neighbor_summary.json`: агрегаты и provenance.',
        '- `runs/cross_track_condition_audit_smoke17_20261001/mapping_rejection_details.jsonl` / `pixel_change_details.jsonl` / `pixel_change_summary.json`: подробный ответ на два уточнения.',
        '- `runs/cross_track_condition_inspection_20261001/summary.json`: полные quantiles и conjunctions.',
        '- Новые diagnostic scripts: `experiments/build_cross_track_condition_report.py`, `experiments/audit_cross_track_conditions.py`. Замороженные linker/runner не изменены.', '',
        'Работа остановлена после диагностики. Ни одно правило автоматически не ослаблено.']
    (ROOT/'docs/CROSS_TRACK_CONDITION_AUDIT_SMOKE17_20261001.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    import sys
    report() if '--report' in sys.argv else main()
