"""Summarize frozen experiment receipts, without upgrading proxy evidence."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def main(out):
    def read(name):return json.loads((out/name).read_text('utf-8'))
    cap=read('capture.json'); rule=read('shift_rule_summary.json'); rows=read('row_reference_summary.json')
    fast=read('fast_ncc_parity.json'); flow=read('optical_flow_summary.json')
    zones=read('zonal_summary.json');ocr=read('zonal_ocr_receipt.json');zs=zones['statistics']
    root=Path(__file__).resolve().parents[1]; before=read('source_before.json')
    changed=[p for p,h in before.items() if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
    manifest_changed=hashlib.sha256((root/'production_manifest.json').read_bytes()).hexdigest()!=cap['manifest_sha256']
    receipt={'registered_file_count':len(before),'changed_registered_files':changed,'manifest_changed':manifest_changed,
             'production_changed':bool(changed or manifest_changed),'experiment_sources':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'experiments').glob('vertical*chat*.py')}}
    receipt['experiment_sources']['experiments/vertical_shift_rule_audit.py']=hashlib.sha256((root/'experiments/vertical_shift_rule_audit.py').read_bytes()).hexdigest()
    (out/'final_integrity_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),'utf-8')
    c=rule['calibration'];a=rule['accepted_only_statistics'];s=rule['statistics'];p=rule['positive_transition_statistics']
    track=[]
    for arm,m in rows['metrics'].items():
        track.append(f"| {arm} | {m.get('lost_births',0)} | {m.get('duplicate_births',0)} | {m.get('broken_row_id',0)} | {m.get('historical_id_disagreement',0)} |")
    text=f'''# CHAT vertical pixel-shift experiment — 2026-10-02

Production unchanged. Input: saved Smoke17 video, first {cap['frames']} frames, 9 fixed CHAT ROIs, {cap['states']} changed states. Tracking/count decisions have ZERO OCR inputs. Only after pixel decisions, {ocr['queue_rows']} admitted new-bottom rows were recognized in a separate frozen-queue pass. Baseline is the currently registered `FirstStage.rows`, replayed directly as an isolated component; no alternative production launcher or full HH run.

## SHIFT to new physical rows

**ROW STEP:** modal adjacent center gap {c['modal_gap_px']:.2f} px; calibration-only fit {c['regression_pitch_px']:.5f} px, {c['raw_adjacent_gap_count']} adjacent gaps and {c['unique_table_geometry_states']} unique table geometries. Slope P05/P95 {c['fit_slope_p05_p95'][0]:.5f}/{c['fit_slope_p05_p95'][1]:.5f} px. Independently counted calibration suffix: {c.get('independent_calibration_scroll_samples',0)} transitions, median pitch {c.get('step_from_physical_suffix_median_px',0):.3f} px, P05/P95 {c.get('step_from_physical_suffix_p05_p95_px',[])}. Adjacent glyph-band centers are imperfect proxies for rendered baselines; fit is an estimate, not proof of a rigid 20px lattice.

**SHIFT VALUES:** whole-ROI NCC includes 20, 40–41, 60–61, 81, 102–103, 143 px and larger ambiguous/unsupported jumps. Complete distribution is in `shift_rule_summary.json` and every transition in the CSV.

**TOLERANCE:** ±2 px was tested. Among accepted positive transitions with sufficient bottom context, {a['positive_count_matches_within_tolerance']}/{a['positive_count_established']} fit this tolerance. Maximum residual {rule['maximum_quantization_residual_covered_px']:.5f} px. Thus a universal ±2 px bound was not proven. Center/height jitter and accumulated effective pitch estimation error explain why exactly 40 or 60 is too strict; source video cannot separate antialiasing from compression. Fixed ROI coordinates exclude a configured crop-box jump.

**TESTED TRANSITIONS:** {s['tested_transitions']} evaluation transitions, after calibration frame {c['calibration_end_frame']}. {p['total_positive_NCC_shifts']} have positive NCC shifts. Actual visible suffix count is established conditionally for {s['actual_count_established']}; {s['actual_count_unknown']} remain UNKNOWN.

**CORRECT NEW-ROW COUNT:** accepted NCC plus adequate bottom context: {a['exact_count_matches']}/{a['actual_count_established']}, 0 count mismatches. Positive accepted subset: {a['positive_exact_count_matches']}/{a['positive_count_established']}, {a['positive_actual_new_rows']} visible new rows. This is the decisive positive subset; zero-shift cases do not establish scroll accuracy.

**FAILURES:** unfiltered NCC predictions have {s['count_mismatches']} mismatch; maximum count error {rule['maximum_count_error_covered']}. T09 1470→1471 predicts 1 (SHIFT20), but bottom suffix has 0 new rows. Ten lower bands are byte-identical at unchanged coordinates while the upper seven redraw. Scores: shift20 .724434 vs shift0 .718268, margin .006166. NCC rejects; the acceptance gate removes this error. Neighbor states support local/partial redraw, not a proven animation cause. See the native comparison and the nine-frame montage below.

**UNKNOWN REASONS:** {json.dumps(rule['unknown_actual_reasons'],ensure_ascii=False)}. These are evidence limitations, not automatically algorithm errors. They cannot be labelled by `round(SHIFT/pitch)` without circular reasoning.

**PRODUCTION CHANGED:** {'YES' if receipt['production_changed'] else 'NO'}; registered bytes and manifest checked against capture receipt.

**VERDICT:** `NOT_PROVEN`. The guarded rule is supported on the measured positive subset; global reliability remains unproved because many positive shifts lack an independently located bottom physical row, the reference shares segmentation with both arms, and complete human physical-line ground truth is absent. Global `SHIFT_TO_NEW_ROWS_PROVEN` would overstate this evidence.

## How labels were constructed

For each pair, whole-ROI NCC provides shift and score. Separately, local row pixels establish a unique ordered contiguous overlap of at least six bands. Actual count requires at least six consecutive local pixel matches ending at the OLD BOTTOM band. Bands below its matched current position are counted directly. Neither OCR nor the division rule creates labels. Unknown rows retain empty actual/error cells with an explicit status. Local raster resizing and NCC are a conditional visual reference, not independent human truth; approximate repeated-row matches can still be ambiguous.

The first exact-native-raster reference covered only zero shifts. It was retained as `summary.json` / `transitions.jsonl`, then supplemented by the separately labelled local row proxy. No nonzero quality claim is based on exact same-coordinate equality. Each image stays at native source dimensions, hashes retained in state and case JSON.

## Current tracker versus experimental tracker

These are candidate-ID disagreements on local pixel-reference-covered observations, not globally established physical losses/duplicates. All source pixels remain stored. Lost denotes missed candidate birth; duplicate denotes fresh candidate ID on an old proxy anchor. Historical ID restoration is separate. The 4 OLD disagreements need independent physical-row adjudication before calling them real production defects.

| arm | missed births | fresh IDs on old anchors | broken ID associations | historical disagreement |
|---|---:|---:|---:|---:|
{chr(10).join(track)}

Before → after: scrollbar mask 56→20 fresh-ID disagreements; minimum overlap 48→120px 20→20 (no improvement); exact stationary-anchor fallback 20→11 (partial improvement, 7 above OLD). Suffix arithmetic affects count formula, not identity. For rejected global NCC, the stationary fix retains only exact proven old anchors, makes no new-birth/count claim, and leaves remaining observations uncertain. No tracker was promoted. Original comparison verdict: `NOT PROVEN`.

## User-required GLOBAL_SCROLL gate and bottom-only OCR

Separate shifts are computed for top/middle/bottom thirds. At least two confident zones must agree within2px; any confident contradicting zone vetoes. Motion ≤2px is treated within the jitter/no-row-admission tolerance. GLOBAL alone applies division and admission. LOCAL/UNRESOLVED never allocate new row IDs or OCR rows by this formula. Existing compatible objects KEEP their IDs with explicit UNRESOLVED visible links, preserving the inventory without claiming uncertain physical continuity. Unknown parents are not synthesized.

GLOBAL_SCROLL_COUNT={zs['GLOBAL_SCROLL_COUNT']}, LOCAL_CHANGE_COUNT={zs['LOCAL_CHANGE_COUNT']}, UNRESOLVED={zs['UNRESOLVED_COUNT']}, FALSE_GLOBAL_SCROLL={zs['FALSE_GLOBAL_SCROLL']} on{zs['GLOBAL_SCROLL_COUNT_VALIDATED']} count-covered GLOBAL transitions; {zs['GLOBAL_SCROLL_ACTUAL_COUNT_UNKNOWN']} GLOBAL counts remain UNKNOWN. These scope limits apply to the zero false-global number.

T09 1470→1471: top20, middle0, bottom0 →LOCAL_CHANGE;0 OCR queue rows. Its stationary bottom vetoes the moving top. The CSV lists all zonal shifts/confidence/classification.

OCR: {ocr['OCR_results']}/{ocr['queue_rows']} admitted rows, {ocr['OCR_errors']} errors, {ocr['seconds_including_startup']:.3f}s including startup. WholeCHAT requests0; initial-visible row requests0; carried-old row requests0; repeat queued IDs{zones['all_frames_OCR_repeated_row_ids']}. Queue before/after SHA identical and matched the frozen replay receipt; no OCR result feeds count/identity. {zs['CARRIED_KNOWN_IDS']} existing ID transports during evaluation, {zs.get('CARRIED_WITH_PARENT_UNRESOLVED',0)} with explicitly unresolved parent physical links. New ID objects are created only for N admitted lower bands on GLOBAL. Per-row previous/translated/observed coordinates and inherited IDs are in the zonal tracker receipts.

Zonal computation{zones['timing_seconds']['zonal_ncc_seconds']:.3f}s, separately measured. See `zonal_transitions.csv`, `zonal_tracker_views.jsonl`, `zonal_ocr_queue.jsonl`, `zonal_raw_ocr_results.jsonl`, `zonal_ocr_receipt.json`, and `zonal_review.html`.

## Speed and Optical Flow

Baseline component replay including decode: {cap['baseline_seconds_including_decode']:.3f}s. Baseline matching/alignment/history telemetry: {sum(cap['baseline_metrics'][k] for k in ['match_rows_zero_seconds','align_upward_seconds','match_rows_shifted_seconds','history_row_identity_seconds']):.3f}s. Exhaustive scalar NCC: RAW325.199s, MASK321.144s. FFT computes the SAME overlap Pearson NCC: RAW{fast['timing_seconds']['NCC_RAW']:.3f}s, MASK{fast['timing_seconds']['NCC_MASK']:.3f}s. Across all6266 pairs, no chosen-shift or acceptance differences. Maximum score roundoff {fast['max_best_score_absolute_delta']:.8g}. These are component measurements, not complete production speed or per-arm end-to-end certification; stationary incremental cost was not separately measured.

Optical Flow is a bounded failure-conditioned Farneback probe, not a replacement tracker. Counts: {json.dumps(flow['counts'])}; flow compute {flow['flow_total_seconds']:.3f}s. Large-motion and UNKNOWN cases remain unproven; selecting diagnostic failures is not a general benchmark. Per-pair shifts, median absolute deviations, horizontal motion and NCC comparisons are in `optical_flow_probe.jsonl`.

## Review artifacts

- [Every transition: CSV](shift_to_new_rows.csv)
- [Positive/rejected before-after native crops and JSON](rule_review.html)
- [Representative one / two–three / fast scroll / top disappearance / local mutation controls](row_reference_controls.html)
- [Full tracker disagreement gallery](row_reference_review.html)
- [Machine-readable rule report](shift_rule_summary.json)
- [Integrity receipt](final_integrity_receipt.json)
- [Raw frozen CHAT states](states.jsonl)

![T09 1470→1471: red changed upper bands, green exact stationary lower bands](diagnostic/T09_1470_1471_stationary_suffix.png)

[Nine neighboring states](diagnostic/T09_1469_1477_montage.png) and [native row measurements](diagnostic/T09_1469_1477_row_details.json).

## Limits and tested controls

Observed real data include 1, 2–3 and larger suffixes, top-row disappearance, altered row pixels and the partial-redraw counterexample. Synthetic pixel stress tests cover append-without-scroll, exactly repeated rows, inside-row and bottom-row mutations, full replacement, three new rows and FFT numeric contracts. Exact repeated-text identity is not established through OCR; all-same raster sequences are intentionally ambiguous. Complete missed-band segmentation quality is not measured. No row deletion, OCR correction, hand repair, golden rewrite or production promotion occurred.
'''
    (out/'REPORT.md').write_text(text,'utf-8');print(str((out/'REPORT.md').resolve()))
    short=f'''# Отдельный эксперимент CHAT: итог

**ROW STEP:** около20px; оценка по {c['raw_adjacent_gap_count']} промежуткам — {c['regression_pitch_px']:.5f}px. На33 независимых калибровочных скроллах медиана20px.

**SHIFT VALUES:** 20,40–41,60–61px и более крупные движения. Близость к целому проверяется по каждому переходу в CSV; она сама не доказывает глобальную прокрутку.

**TOLERANCE:** ±2px. На проверенных положительныхGLOBAL —67/68; максимальный остаток2.118px, поэтому универсальный допуск±2px не доказан.

**TESTED TRANSITIONS:**4044; GLOBAL_SCROLL_COUNT={zs['GLOBAL_SCROLL_COUNT']}, LOCAL_CHANGE_COUNT={zs['LOCAL_CHANGE_COUNT']}, UNRESOLVED={zs['UNRESOLVED_COUNT']}.

**CORRECT NEW-ROW COUNT:**{zs['GLOBAL_COUNT_CORRECT']}/{zs['GLOBAL_SCROLL_COUNT_VALIDATED']}GLOBAL с достаточным независимым нижним пиксельным контекстом. Для{zs['GLOBAL_SCROLL_ACTUAL_COUNT_UNKNOWN']}GLOBAL фактическое количествоUNKNOWN; это не заменено результатом деления.

**FALSE_GLOBAL_SCROLL:**{zs['FALSE_GLOBAL_SCROLL']} на покрытой выборке. На непокрытых переходах показатель не установлен.

**FAILURES / REASONS:**T09 1470→1471 — верх20, середина0, низ0; LOCAL_CHANGE,0 новых строк по формуле и0 запросовOCR. Это локальная перерисовка, не доказанная анимация. Общий NCC безgate ошибочно предсказывал1 строку. Другие причиныUNKNOWN:74 коротких/прерванных нижних контекста,66 переходов без уникального пиксельного перекрытия,11 без найденной последней старой строки. Изменение пикселей/границglyph и округление поддерживаются измерениями; отдельно отличить антиалиасинг от видеокомпрессии нельзя. КоординатыROI фиксированы.

**OCR / ROW_ID:**{ocr['OCR_results']}/{ocr['queue_rows']} новых нижних строк обработаны после решенияtracker. ПолныйCHAT, первоначально видимые и перенесённые старые строки вOCR не отправлялись; повторныхID вочереди0. Старые существующиеID переносятся геометрически приGLOBAL; LOCAL не создаёт новыхID, сомнительные физические связи явноUNRESOLVED. Обратного путиOCR→identity нет.

**FIXES:**маскаscrollbar снизила условные ID-разногласия56→20; увеличениеперекрытия120px не помогло; stationary fallback20→11. Новый зональныйgate отделяет локальную перерисовку; KEEP сохраняет существующие объекты. Это условные разногласия, не доказанные глобальные физические дубли. Текущийtracker имеет4 таких разногласия и не изменён. OpticalFlow:0/30 положительных controls в пределах2px; заменаNCC этимprobe не подтверждена.

**SPEED:**FFT-NCC23–24s против321–325s с совпадением всех6266 решений. Зональныйgate{zones['timing_seconds']['zonal_ncc_seconds']:.2f}s; OCR{ocr['seconds_including_startup']:.2f}s. Эти измерения относятся к компонентам, не к полномуproduction.

**PRODUCTION CHANGED:**{'YES' if receipt['production_changed'] else 'NO'}.

**VERDICT:**`NOT_PROVEN`. Правило поддержано на проверенныхGLOBAL, но113GLOBAL и общий физическийground truth ещё не доказаны; production promotion не выполнялась.

[Таблица всех переходов](zonal_transitions.csv) · [ТаблицаSHIFT/STEP/actual/error](shift_to_new_rows.csv) · [Зоны и примеры](zonal_review.html) · [Подробный отчёт](REPORT.md)

![T09: слева1470, справа1471; красныйверхизменился, зелёныйниз неподвижен](diagnostic/T09_1470_1471_stationary_suffix.png)
'''
    (out/'SHORT_REPORT_RU.md').write_text(short,'utf-8')


if __name__=='__main__':main(Path(sys.argv[1]))
