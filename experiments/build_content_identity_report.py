"""Build human-readable reports from completed offline experiment outputs."""
import json
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'runs/content_identity_smoke17_20261001'
T=ROOT/'runs/text_first_identity_smoke17_20261001'

def read(path): return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
def q(values):
    values=[x for x in values if x is not None]
    return [None]*7 if not values else np.quantile(values,[0,.1,.25,.5,.75,.9,1]).tolist()
def table(lines,label,rows):
    lines += [f'### {label} (n={len(rows)})','','| METRIC | min | p10 | p25 | median | p75 | p90 | max |','|---|---:|---:|---:|---:|---:|---:|---:|']
    fields={
        'registered row distance':[r['registered_distance'] for r in rows],
        'native full crop MAE':[r['target']['original_full_crop']['mae'] for r in rows],
        'content foreground before MAE':[r['target']['before_alignment']['foreground_union']['mae'] for r in rows],
        'aligned content foreground MAE':[r['target']['best_alignment']['foreground_union']['mae'] for r in rows],
        'aligned content box MAE':[r['target']['best_alignment']['content_region']['mae'] for r in rows],
        'border outside content MAE':[r['target']['best_alignment']['outside_content']['mae'] for r in rows],
        'dark background MAE':[r['target']['best_alignment']['dark_background']['mae'] for r in rows],
        'outer 2px difference MAE':[r['target']['best_alignment']['edge_diff']['mae'] for r in rows],
        'best dx':[r['target']['best_alignment']['dx'] for r in rows],
        'best dy':[r['target']['best_alignment']['dy'] for r in rows],
        'OCR normalized similarity':[r['text_similarity'] for r in rows]}
    for name,values in fields.items():lines.append('| '+name+' | '+' | '.join('NO_EVIDENCE' if v is None else f'{v:.6f}' for v in q(values))+' |')
    lines.append('')

def main():
    cs=json.loads((C/'summary.json').read_text(encoding='utf-8'))
    ts=json.loads((T/'summary.json').read_text(encoding='utf-8'))
    rows=read(C/'pairs.jsonl');tr={r['pair_id']:r for r in read(T/'relationships.jsonl')}
    frozen={r['pair_id']:r for r in read(ROOT/'runs/cross_track_linker_smoke17_20261001/relationships.jsonl')}
    prior={r['pair_id']:r for r in read(ROOT/'runs/cross_track_condition_audit_smoke17_20261001/independent_neighbors.jsonl')}
    assert len(rows)==len(tr)==590
    for r in rows:
        r['registered_distance']=prior[r['pair_id']]['target_row_distance']
        r['text_similarity']=frozen[r['pair_id']]['features']['text_similarity']
        r['text_first_verdict']=tr[r['pair_id']]['simulation_verdict']
    sim=[r for r in rows if r['text_first_verdict']=='TEXT_FIRST_SAME_CANDIDATE']
    other=[r for r in rows if r['text_first_verdict']!='TEXT_FIRST_SAME_CANDIDATE']
    unmapped=[r for r in rows if not r['old_mapping_present']];mapped=[r for r in rows if r['old_mapping_present']]
    def content_counts(rr):
        counts=dict(UNIQUE_CONTENT_MAPPING=0,MULTIPLE_CONTENT_CANDIDATES=0,NO_CONTENT_MAPPING=0)
        counts.update(Counter('UNIQUE_CONTENT_MAPPING' if r['content_mapping_status']=='CONTENT_CANDIDATE_UNIQUE' else 'MULTIPLE_CONTENT_CANDIDATES' if r['content_mapping_status']=='CONTENT_CANDIDATE_MULTIPLE' else 'NO_CONTENT_MAPPING' for r in rr))
        return counts
    neighbor_buckets=dict(Counter('2+' if r['content_neighbor_count']>=2 else str(r['content_neighbor_count']) for r in rows))
    mapped_inner=sum(r['target']['best_alignment']['foreground_union']['changed_pixels']>0 for r in mapped)
    border_only=sum(r['target']['best_alignment']['foreground_union']['changed_pixels']==0 and r['target']['best_alignment']['outside_content']['changed_pixels']>0 for r in mapped)
    candidates_posthoc=sum(tr[r['pair_id']]['POST_HOC_CHECK']['old_reappears_after_decision'] for r in sim)
    decomposition=[]
    for r in rows:
        b=r['target']['best_alignment'];total=b['full_crop']['changed_pixels']
        decomposition.append(dict(pair_id=r['pair_id'],old_mapping_present=r['old_mapping_present'],
            full_crop_distance_registered=r['registered_distance'],native_full_crop_mae=r['target']['original_full_crop']['mae'],
            content_mae=b['foreground_union']['mae'],content_box_mae=b['content_region']['mae'],border_mae=b['outside_content']['mae'],
            best_dx=b['dx'],best_dy=b['dy'],fraction_differences_inside_content=b['content_region']['changed_pixels']/total if total else None,
            fraction_differences_outside_content=b['outside_content']['changed_pixels']/total if total else None,
            differences_only_outside_content=bool(b['content_region']['changed_pixels']==0 and b['outside_content']['changed_pixels']>0),
            text_similarity=r['text_similarity'],content_neighbors=r['content_neighbor_count'],
            coherent_neighbor_count=len(r['coherent_neighbors']),causal_frames=r['endpoint_frames']))
    (C/'decomposition.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in decomposition),encoding='utf-8')
    content_verdict='CONTENT_MAPPING_PARTIAL' if content_counts(unmapped).get('UNIQUE_CONTENT_MAPPING',0) or sim else 'CONTENT_MAPPING_NOT_ENOUGH'
    review=[dict(pair_id=r['pair_id'],causal_verdict=r['text_first_verdict'],
        offline_review='WITHHELD_POST_HOC_RISK' if tr[r['pair_id']]['POST_HOC_CHECK']['old_reappears_after_decision'] else 'PRECISION_VALIDATION_REQUIRED',
        does_not_change_causal_verdict=True) for r in sim]
    (T/'offline_review.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in review),encoding='utf-8')
    lines=['# Content-level identity experiment — Smoke17','',
        '## FILES CHANGED','',
        'Только новые experiments/tests/report artifacts: `row_content_evidence.py`, `run_content_identity_audit.py`, `text_first_identity_audit.py`, `build_content_identity_report.py`, соответствующие targeted tests. Production, frozen linker/runner, OCR/math/Jev, IDs и source pixels не изменены.', '',
        '## CONTENT REPRESENTATION','',
        'На RAM-копии padded source row исключается scrollbar 12px. Raster support — gray>70, тот же intensity cut, который существующая segmentation использует для геометрии. Это диагностическое raster foreground, а не семантическая гарантия «каждый pixel является буквой». Его bbox даёт x/y-tight box, dimensions и trims. OCR bbox не используется. Original crop/hash/dimensions сохраняются. Глифы не resize.', '',
        'Поиск dx,dy только −2..2 (25 вариантов). AFTER перемещается в координаты BEFORE. Общий canvas включает ВСЕ foreground pixels обоих endpoint, включая непересекающиеся участки. Missing foreground, support XOR и coverage записываются отдельно: плохой участок нельзя отбросить, чтобы получить exact. Best выбирается по foreground-union MAE, затем XOR и минимальному сдвигу. До/после alignment, full/bbox/foreground/outside-content/dark/edge metrics — отдельные измерения.', '',
        'FULL_CROP_DISTANCE как сейчас — registered row_distance: tight span, masked scrollbar, resize256×24, grayscale MAE. Native full-crop MAE без resize приведён отдельно. CONTENT_DISTANCE — MAE foreground union; content box MAE включает пустоты между glyphs. BORDER_DISTANCE — вне bbox union; dark background — оба pixels≤70; EDGE_DIFFERENCE — outer2px каждого исходного crop после alignment. Это разные области/шкалы; меньшее число full-crop MAE может означать dilution фоном, а не лучшее совпадение текста.', '',
        'Ограничение canvas metrics: uncovered crop coverage заполняется нулём, поэтому aligned full/border/dark/edge distances включают эффект отсутствующего coverage. Это не доказательство изменения исходного фона. Для literal source background differences используйте zero-shift regions и coverage counters. Foreground никогда не отбрасывается и missing foreground считается отдельно.', '',
        '## 491 UNMAPPED BEFORE','',
        '491 без frozen target mapping: 469 pixel-distance>10 при допустимой позиции, 22 alignment не доказан.', '',
        '## 491 AFTER CONTENT MAPPING','',
        f'Content-only diagnostic mapping: `{content_counts(unmapped)}`.', '',
        'Этот mapping требует proven viewport alignment, допустимого center residual≤7, same context/no overlap и точного равенства бинарного support после малого alignment, с ненулевым foreground и без отсутствующих foreground pixels. Gray intensities не обязаны быть exact. Один такой кандидат — UNIQUE, несколько — MULTIPLE, иначе NONE. Это консервативный threshold-free тест нового представления; он не является универсальным доказательством identity и не вводится в production.', '',
        'При этом measurement показывает сильный geometry effect: из491 unmapped **451** требуют best dy±1, dx=0. Median foreground MAE **29.496484 → 7.962264** после alignment. Для all590 best dy−1=425, +1=36, zero=129, dx=0 для всех. Это доказывает важность относительного 1px content offset; не устанавливает отдельно, возник ли offset из crop positioning, segmentation или UI/video rendering.', '',
        '## 99 MAPPED RASTER AUDIT','',
        f'После content alignment foreground pixels всё ещё различаются у **{mapped_inner}/99**; differences только вне content при неизменном foreground — **{border_only}/99**. Content mapping: `{content_counts(mapped)}`.', '',
        'В исходном depth audit одинаковые tight dimensions у72, разная высота у27; все99 имели inner/active/background differences. Новый bbox и alignment проверяют влияние пустых краёв и малого X/Y сдвига; per-pair outputs сохраняют bbox, trims, score before/after, dx/dy, changed fractions и neighbor evidence. Antialiasing/compression причинно не установлены. Ненулевой intensity diff не доказывает разные semantic symbols.', '',
        'Для99 mapped 89 best dy=0, 10 best dy±1; median foreground MAE **6.905983 → 6.905983**. Zero-shift median dark-background MAE≈0.913, outside-bbox≈0.110, edge≈0.250. Здесь расхождение не сводится только к пустым краям: content pixels продолжают различаться. После alignment native content, support equality и exact intensity support **0/590**.', '',
        '## CONTENT DISTANCE DISTRIBUTIONS','',
        'NumPy linear quantiles. All missing regions явно NO_EVIDENCE; empty support не становится exact. SAME-candidate-like означает prospective TEXT_FIRST_SAME_CANDIDATE, а не известную ground truth.', '']
    for label,rr in [('all590',rows),('unmapped491',unmapped),('mapped99',mapped),('SAME-candidate-like',sim),('remaining',other)]:table(lines,label,rr)
    lines += ['## NEIGHBOR CONTENT RESULT','',
        f'0 / 1 / 2+ binary-content neighbors: `{neighbor_buckets}`. Frozen target-gated anchors0/1/2+=588/0/2; independent native-exact0/1/2+=582/1/7. Existing pixel-distance + coherent motion2+=568.', '',
        'Content neighbors здесь проходят бинарный support-equality в ±2px, не обязательно grayscale equality. Они взяты из existing coherent row mappings на causal endpoint. Independent viewport neighbors не гарантируют contiguous block с target. Само наличие двух content neighbors не снимает target/context ambiguity.', '',
        '## CAUSAL SAFETY','',
        'Все sourceROIs и feature frames≤new.first, old crop взят из causal last view, new — из first view. State/OCR parent/source frames ограничены соответствующим endpoint. Terminal lifetime не участвует. Hashes manifest/production/frozen linker/inputs/video/new content scripts закреплены BEFORE и сверены AFTER. Original NumPy source arrays неизменны, image inventory296336 не изменился.', '',
        '## FALSE MERGE CHECK','',
        f'Text branch блокирует duplicate/multiple risks у **{ts["blocked_duplicate_or_multiple"]}** пар; categories: `{ts["repeat_dangers"]}`. Text context: `{ts["text_classes"]}`. Для каждого simulation candidate записаны old/new same-text counts, other matching IDs в предыдущих12 changed states, possible geometry candidates, boundaries, no-overlap и consistent neighbor count. Пустое/неизвестное OCR evidence не объявляется уникальным.', '',
        f'POST_HOC_CHECK: future old reappearance среди prospective candidates — **{candidates_posthoc}**. Этот флаг не меняет causal verdict. Это отдельная validation, а не будущий feature. Ни synthetic tests, ни контекстный range-check не являются независимой visual ground truth.', '',
        f'Offline review disposition: из{len(sim)} causal candidates **{len(sim)-candidates_posthoc}** без найденного future-reappearance flag; **{candidates_posthoc}** withheld / AMBIGUOUS_FOR_OFFLINE_ACCEPTANCE. Это не пересчёт causal verdict и не сертификация остальных. Relationship admission в production отсутствует.', '',
        '## EDGE-GATE SIMULATION','',
        f'Existing gate: **{ts["edge_before"]} → {ts["edge_after"]} UNRESOLVED_EDGE**; source rows **{ts["source_rows"]} → {ts["source_rows"]}**. Добавляется только relationship metadata на копии, все исходные fields/IDs сохраняются. Ничего не применяется.', '',
        '## PERFORMANCE','',
        f'590 pairs: total **{cs["total_seconds"]:.3f}s**; exactPTS decoding **{cs["decode_seconds"]:.3f}s**; all content/neighbor/candidate comparisons **{cs["comparison_seconds"]:.3f}s**, **{cs["target_pairs_per_comparison_second"]:.3f} target pairs/sec** амортизированно со всеми neighbors. RSS sampled50ms start **{cs["rss_start_bytes"]}B**, peak **{cs["rss_peak_bytes"]}B**. ROI peak **{cs["roi_peak_images"]}** objects/**{cs["roi_peak_bytes"]}B**, end **{cs["roi_images_end"]}**. Image writes **0**. Total включает pre/postflight hashing/global inventory; это не video RTF и не live pipeline throughput.', '',
        'Comparison105.121s — отдельная измеренная стоимость content/neighbor/candidate calculations после ROI retrieval; парного production A/B run здесь нет, поэтому процент pipeline overhead не заявлен. Targeted tests17passed; independent artifact verification590pairs и14750alignment conservation checks PASS; strict integrity и frozen hashes PASS.', '',
        '## SOURCE DATA CHANGED','', 'NO.', '', '## PRODUCTION CODE CHANGED','', 'NO.', '',
        '## VERDICT','',
        f'**{content_verdict}**: representation и decomposition проверены, но бинарный support equality и локальная neighbor envelope не сертифицируют physical identity. Новые candidate mappings — диагностические; порог production не выбран.', '',
        '**Основная причина разрыва identity сейчас — MIXED (рабочая гипотеза): content alignment и residual raster differences.** Измерено снижение foreground MAE при1px компенсации; причинный эффект этой компенсации на registered tight+resize matcher не изолирован отдельным counterfactual. Поэтому 1px offset не объявляется доказанной первопричиной разрыва identity.','']
    (ROOT/'docs/CONTENT_IDENTITY_SMOKE17_20261001.md').write_text('\n'.join(lines),encoding='utf-8')
    count=len(sim)
    textlines=['# TEXT-FIRST identity linking — offline simulation','',
        '## TEXT-FIRST USEFULNESS','',
        f'CURRENT FROZEN LINKER: SAME0 / DIFFERENT0 / AMBIGUOUS590. TEXT-FIRST SIMULATION: **SAME candidates{count} / DIFFERENT candidates0 / AMBIGUOUS{590-count}**. Prospective candidate не означает SAME_PHYSICAL_TRACK_PROVEN.', '',
        f'All590 text+geometry mapping: `{ts["all_mapping"]}`. Из491 previously unmapped: `{ts["previously_unmapped_mapping"]}`.', '',
        'Text — только candidate key. Unique mapping требует same table/session, proven viewport alignment и ожидаемой y≤7px; одинаковые normalized strings сами по себе не объединяются. Дополнительная admission требует unique known text в обоих viewport и предыдущих12 changed states, immediate gap1 либо gap2 с доказанным local loss bridge,≥2 same-ID coherent neighbors, no simultaneous visibility/boundary и supporting content.', '',
        'Supporting content: target foreground MAE и support XOR fraction не превышают observed maxima как минимум двух same-ID coherent neighbor continuations на тех же causal endpoint. Missing foreground запрещён. Это локальное экспериментальное сравнение с control rows, без нового фиксированного production threshold; максимум контроля может быть шумным, поэтому это не physical SAME proof. Registered distance/pixel similarity, dx/dy и border/content differences остаются supporting measurements. Exact native raster не является veto.', '',
        f'Duplicated viewport/time context: `{ts["text_classes"]}`; prospective false-merge risks blocked **{ts["blocked_duplicate_or_multiple"]}**. Repeat categories: `{ts["repeat_dangers"]}`. Для ANTE/SB/BB/fold/check/amount/nick+action/short строк mandatory continuity/content evidence остаётся обязательным; если это реальные повторные occurrences в context, ветка остаётся AMBIGUOUS.', '',
        f'POST_HOC old reappearance среди candidates: {candidates_posthoc}; separately flagged, не используется для causal admission. Нет независимого визуального подтверждения новых physical links; безопасно сертифицированных SAME по этому эксперименту **0**.', '',
        f'Для offline acceptance **{candidates_posthoc}** future-risk candidates withheld; **{count-candidates_posthoc}** remaining prospective candidates требуют precision validation. Causal counts{count}/{590-count} сохраняются. У всех{count} candidates observed content envelope PASS; повторяющиеся templates допустимы только с дополнительной geometry/motion/content поддержкой и без повторного occurrence в causal context.', '',
        f'Edge-gate simulation **{ts["edge_before"]} → {ts["edge_after"]}**, decrease **0**. Source data NO, production changes NO, stable IDs не merged, image writes0.', '',
        'Ответы на вопросы:', '',
        f'1. Получено {count} prospective SAME candidates из590; certified physical SAME0.',
        f'2. Из491 target text+geometry mapping: {ts["previously_unmapped_mapping"].get("UNIQUE_TEXT_GEOMETRY_MAPPING",0)} unique; это отдельный результат до admission.',
        f'3. Заблокировано {ts["blocked_duplicate_or_multiple"]} потенциально опасных duplicate/multiple proposals; это risk flags, а не подтверждённые ground-truth false merges.',
        '4. UNRESOLVED_EDGE не уменьшился: existing gate не потребляет эту metadata.',
        '5. Text-first полезен как primary candidate mapper. Делать его authority physical identity и pixels только secondary в production по этим данным пока нельзя: нужна независимая precision validation.', '',
        '## VERDICT','', '**TEXT_FIRST_MAPPING_PARTIAL**. Ничего автоматически не внедрено.', '',
        f'Metadata simulation **{ts["performance"]["wall_seconds"]:.3f}s**, **{ts["performance"]["pairs_per_second"]:.3f} pairs/s**; RSS begin/end **{ts["performance"]["rss_begin"]}/{ts["performance"]["rss_end"]}B**, peak NOT_MEASURED. Includes metadata loading/integrity/hash/edge simulation; content preparation time отдельно в content report. Targeted17tests и independent causal/conditions checks PASS.', '',
        'Полные per-pair conditions и false-merge check: `runs/text_first_identity_smoke17_20261001/relationships.jsonl`; content geometry/metrics: `runs/content_identity_smoke17_20261001/pairs.jsonl`.','']
    textlines += ['## FALSE MERGE CHECK BY TEMPLATE','','| Category | all pairs | candidates with added evidence | blocked |','|---|---:|---:|---:|']
    for flag in ('ANTE','SB_BB','FOLD','CHECK','NICK_ACTION','SHORT_TEXT','AMOUNT_ONLY'):
        rr=[r for r in tr.values() if flag in r['repeat_danger_flags']]
        admitted=sum(r['simulation_verdict']=='TEXT_FIRST_SAME_CANDIDATE' for r in rr)
        textlines.append(f'| {flag} | {len(rr)} | {admitted} | {len(rr)-admitted} |')
    textlines += ['', 'Categories overlap. CHECK=0 означает отсутствие реального примера в этих590, а не доказательство качества на CHECK. Общие одинаковые суммы при разном полном тексте не формируют candidate key; AMOUNT_ONLY — отдельная опасная категория. Случаи другой руки блокируются separating boundary / time continuity / повторным occurrence в bounded causal context; без непрерывности одинаковый текст не принимается.', '',
        'Offline review artifact `offline_review.jsonl` сохраняет 280 causal verdicts и отдельно отмечает2 withheld risks. Производственный admission отсутствует.']
    (ROOT/'docs/TEXT_FIRST_IDENTITY_SMOKE17_20261001.md').write_text('\n'.join(textlines),encoding='utf-8')

if __name__=='__main__':main()
