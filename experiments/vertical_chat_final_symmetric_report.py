"""Final report for the strict symmetric branch, preserving prior evidence."""
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.vertical_chat_final_report import main as previous_report

def main(out):
    previous_report(out)
    (out/'BEFORE_SYMMETRY_REPORT.md').write_text((out/'REPORT.md').read_text('utf-8'),'utf-8')
    read=lambda p:json.loads((out/p).read_text('utf-8'))
    rule=read('shift_rule_summary.json');c=rule['calibration'];receipt=read('final_integrity_receipt.json')
    strict=out/'symmetric_v2';summary=read('symmetric_v2/zonal_summary.json');s=summary['statistics']
    ids=read('symmetric_v2/id_transfer_summary.json');ocr=read('symmetric_v2/zonal_ocr_receipt.json')
    rr=list(map(json.loads,(strict/'symmetric_transitions.jsonl').open(encoding='utf-8')))
    gg=[r for r in rr if r['classification']=='GLOBAL_SCROLL'];known=[r for r in gg if r['actual_new_rows'] is not None]
    residual=max(abs(r['SHIFT_PX']-r['symmetry']['N']*r['ROW_STEP_PX']) for r in gg)
    # Post-decision text inventory only, never identity or admission inputs.
    texts={i['item_id']:i.get('raw_text','') for l in (strict/'zonal_raw_ocr_results.jsonl').open(encoding='utf-8') for i in json.loads(l)['results']['results']}
    groups=defaultdict(list)
    for rid,text in texts.items():groups[text].append(rid)
    repeated={text:values for text,values in groups.items() if len(values)>1}
    within=[{'table':r['table'],'frame_from':r['frame_from'],'frame_to':r['frame_to'],'text':text,'ids':values} for r in gg for text,values in repeated.items() if len(set(values)&set(r['old_ids']))>1]
    (strict/'repeated_text_posthoc.json').write_text(json.dumps({'repeated_text_groups':repeated,'same_old_viewport_examples':within,'identity_inputs_from_OCR':0},ensure_ascii=False,indent=2),'utf-8')
    short=f'''**ROW STEP:** около 20 px; оценка {c['regression_pitch_px']:.5f} px по {c['raw_adjacent_gap_count']} промежуткам. На 33 независимо посчитанных калибровочных суффиксах медиана 20 px.

**SHIFT VALUES:** 20, 40–41, 60–61 px; также проверены более крупные движения.

**TOLERANCE:** ±2 px — проверяемый рабочий допуск. В строгом варианте 46/46 commits укладываются, максимальный остаток {residual:.3f} px. В исходной покрытой выборке есть остаток 2.118 px, поэтому универсальность ±2 px не доказана.

**TESTED TRANSITIONS:** 4044 соседних перехода после калибровки. Зональный gate: GLOBAL 181, LOCAL 3850, UNRESOLVED 13. Симметрия соседних GLOBAL: 167 проходят, 14 отклонены как UNRESOLVED. Это diagnostic candidates, а не 167 committed переносов.

Строгий tracker с подтверждённым checkpoint: GLOBAL_SCROLL_COUNT={s['GLOBAL_SCROLL_COUNT']}, LOCAL_CHANGE_COUNT={s['LOCAL_CHANGE_COUNT']}, UNRESOLVED={s['UNRESOLVED_COUNT']}. Здесь frame_from — последний подтверждённый кадр, не обязательно соседний.

**CORRECT NEW-ROW COUNT:** {s['GLOBAL_COUNT_CORRECT']}/{s['GLOBAL_COUNT_VALIDATED']} evaluation GLOBAL с независимым нижним pixel context; ещё {s['GLOBAL_COUNT_UNKNOWN']} UNKNOWN. С калибровкой: {len(known)}/{len(known)} совпадений на 46 commits, 29 UNKNOWN. Максимальная установленная ошибка количества — 0. FALSE_GLOBAL_SCROLL=0 только на покрытой выборке.

**ID_TRANSFER_CORRECT / WRONG / UNRESOLVED:** 95 / 0 / 27 на evaluation; 479 / 0 / 211 с калибровкой. DUPLICATE_ID=0, LOST_ID=0 на проверенных связях. Все 690 расчётных координатных переносов выполняются ровно на SHIFT_PX; физическое pixel-proxy подтверждение считается отдельно. Максимальный остаток наблюдаемых центров glyph — 1.5 px.

**FAILURES:** 14 нарушений симметрии соседних кандидатов; 3875 отказов строгого tracker до проверки симметрии.

**FAILURE REASONS:** 14 изменений количества выделенных видимых полос (17↔18); clipping и подавление первой полосы влияют на inventory. Для строгого tracker — устаревший checkpoint и недостаточный score/margin: 3734 случая с нулём уверенных зон, 141 с одной. T09 1470→1471: верх20, середина0, низ0 → LOCAL_CHANGE, без переноса и OCR.

**OCR:** {ocr['OCR_results']}/{ocr['queue_rows']} новых нижних строк, ошибок {ocr['OCR_errors']}. Старые строки, весь CHAT и LOCAL — 0 запросов. Hash замороженной очереди сохранён. На каждом commit N удалений сверху = N новых объектов снизу = N OCR-строк.

**PRODUCTION CHANGED:** {'YES' if receipt['production_changed'] else 'NO'}. 103 зарегистрированных файла и manifest сохранены; 32 теста прошли.

**VERDICT:** `NOT_PROVEN` для `SHIFT_TO_NEW_ROWS` и `OLD_ROW_ID_TRANSFER_SAFE`. На покрытой выборке арифметика и перенос работают; UNKNOWN не превращены в доказанные связи.

[Полный отчёт](REPORT.md) · [SHIFT/STEP/actual/error](symmetric_v2/shift_to_new_rows.csv) · [20 примеров row_id](symmetric_v2/id_transfer_review.html) · [Все 14 нарушений симметрии и checkpoint-примеры](symmetric_v2/symmetry_failure_review.html)
'''
    detail=f'''# CHAT: итог строгого симметричного эксперимента

{short}

## Данные и независимость проверки

Первые 3000 кадров сохранённого Smoke17-видео, девять фиксированных CHAT ROI, 6275 изменённых состояний. Первые997 кадров — calibration;4044 перехода после неё — evaluation. Frozen states сохраняют исходные pixels/hash/frame/PTS/time_base/ROI. Production lock прочитан, baseline — зарегистрированный FirstStage.rows, изолированный компонент без полного production run.

Whole-ROI NCC выбирает shift. Отдельный local-row raster proxy ищет уникальное упорядоченное перекрытие минимум шести соседних полос. Actual new rows считается непосредственно под найденной последней старой нижней полосой; для этого нужен непрерывный шестистрочный контекст, заканчивающийся OLD BOTTOM. Деление и OCR не создают actual labels. UNKNOWN остаётся пустым actual/error в CSV. Proxy использует ту же сегментацию полос и не заменяет полный человеческий physical ground truth. Точное совпадение на прежних координатах применяется только для stationary controls, никогда для эталона ненулевого scroll.

Калибровка: модальный gap20, fit20.17647 на163 уникальных геометриях; slope P05/P95 {c['fit_slope_p05_p95']}. Independently counted suffix pitch median20, P05/P95 {c['step_from_physical_suffix_p05_p95_px']}. Glyph centers зависят от содержимого строки и не являются строгими baselines. Значения40–41/60–61 согласуются с округлением и накопленным отличием эффективного шага от ровно20. Разделить антиалиасинг и видеокомпрессию по этому видео нельзя. Crop box фиксирован; configured ROI jitter исключён, внутренняя render jitter не установлена.

Исходный accepted whole-NCC positive subset:68/68 correct,110 новых полос;67/68 в пределах±2px. T03 2324→2328: SHIFT103,N5,residual2.118px; видимое число полос меняется, строгая симметрия отклоняет. Unfiltered NCC имеет один установленный count mismatch: T09 1470→1471 предсказывает1 при неподвижном нижнем суффиксе0. 151 actual UNKNOWN:74 коротких/прерванных нижних контекста,66 без уникального перекрытия,11 без найденной нижней старой полосы. [Исходная таблица](shift_to_new_rows.csv), [native before/after](rule_review.html).

## Зоны и симметрия

Top/middle/bottom NCC считаются отдельно: score≥.90,margin≥.02; минимум две уверенные зоны согласованы в пределах2px. Любая уверенная противоречащая зона veto GLOBAL. Движение≤2px не допускает новых строк. T09 1470→1471: top20/middle0/bottom0; десять нижних полос byte-identical на прежних координатах. Whole score20=.724434, score0=.718268, margin=.006166 также rejected. Это локальная перерисовка; причина animation не доказана. [Соседние кадры](diagnostic/T09_1469_1477_montage.png).

Предыдущий adjacent zonal replay:181 GLOBAL,3850 LOCAL,13 UNRESOLVED;68/68 count-covered GLOBAL correct,113UNKNOWN. Его442 OCR rows — сохранённый результат прежнего контракта, не финального strict tracker.

Симметрия проверяет same visible count, Nout=Nin=N и ordered retained centers±2px. На adjacent GLOBAL167/181 PASS;14 UNRESOLVED из-за17↔18 выделенных полос. T01 1861→1871: SHIFT20,N1,old18/new17,Nout2/Nin1. Clipping/threshold и exclude_first_visible_span меняют выбранный inventory; это не доказательство выхода двух целых физических строк. Все14 случаев имеют JSON и before/after. Никакого переноса соседнегоID для них не придумано.

Строгий stateful tracker сравнивает current crop с последним committed crop. Только GLOBAL+symmetry удаляет первыеN объектов из видимого окна (source journal остаётся), переносит retained model positions ровно наSHIFT с теми жеID, создаётN новых bottom objects/OCR. LOCAL/rejected не меняют confirmed IDs,coordinates,counters,queue. Первый inventory seeded безOCR.

Минимальное исправление v1→v2: наблюдаемая симметрия сравнивает RAW observed bounds, а накопленные model coordinates хранятся отдельно. v1 ошибочно смешивал эти координаты:36 геометрических отказов;v2 —0. Evaluation GLOBAL6→8,LOCAL127→161,UNRES3911→3875. v1 сохранён отдельно. Exact model arithmetic — проверка контракта, не независимое доказательство совпадения пикселей.

Все3875 strict UNRESOLVED остановлены до симметрии:3734 с0 accepted zones,141 с1. Median checkpoint gap1585frames;3861 gaps>100,3747>500. T1 checkpoint120→1000 против adjacent993→1000 (LOCAL,scores1/1/1);T2 489→1000 против984→1000 (LOCAL);T8 161→1028 против1027→1028 (GLOBAL20,scores.992/.991/.964). Старый checkpoint теряет текущий content overlap. Совпадение band coordinates само по себе не восстанавливает identity. Безопасная повторная привязка после LOCAL/UNRESOLVED этим экспериментом не доказана. Причины и crops сохранены; решение не маскируется принудительным переносом.

## ID, одинаковый текст и OCR

Evaluation95 correct/27UNKNOWN и все479correct/211UNKNOWN — pairwise local pixel-proxy evidence. Ноль wrong/duplicate/lost относится к покрытым связям, не сертифицирует всю lineage. Вышедшие сверху IDs не считаются LOST. 690model transfers exact; raw glyph-center residual max1.5px. 20 annotated comparisons охватывают1/2/3 строки,fast scroll и no-commit controls. ОдинID имеет одинаковый цвет. Rejected AFTER подписанUNKNOWN: прежниеID на текущие неизвестные pixels не проецируются. Raw images отдельно сохранены.

Post-decision OCR выявил {len(repeated)} групп одинакового текста у различныхID, но двух OCR-known одинаковых строк в одном проверенном OLD viewport не найдено. Поэтому реальные same-text neighbour transfers не считаются доказанными. Synthetic identical rasters сохраняют ambiguity/UNKNOWN. Text никогда не выбирает physical identity. [Post-hoc список](symmetric_v2/repeated_text_posthoc.json).

Final strict queue100 распознана100/100,errors0,{ocr['seconds_including_startup']:.3f}s со startup. Из них14 evaluation,86calibration. Queue SHA {ocr['queue_before_sha256']},after идентичен. WholeCHAT/initial/old/carried/LOCAL requests0,OCR identity inputs0. Worker проверяет pixel hash и response item IDs.

32 targeted tests passed:1/2/3 carry,visible count change,neighbour offset,partial local mutation,commit→LOCAL→commit checkpoint,identical raster ambiguity,FFT parity,frozenqueue mutation/duplicate guards,chat_rows и observer isolation. Reviewer проверил stateful tracker и reporting,существенных замечаний нет.

FFT сохранил все6266 chosen-shift/acceptance decisions,23–24s вместо321–325s scalar. Final zonal compute {summary['timing_seconds']['zonal_ncc_seconds']:.3f}s. Это component timing, не production end-to-end benchmark. Исторические минимальные исправления:scrollbar mask conditional fresh-ID disagreements56→20;overlap120px20→20;stationary fallback20→11. Bounded Farneback probe0/30 positive controls±2px — заменаNCC этимprobe не подтверждена. [Полные предыдущие результаты](BEFORE_SYMMETRY_REPORT.md).

## Вывод

Арифметика количества и симметричный перенос поддержаны на покрытых GLOBAL. Полный physical count на UNKNOWN, real same-text neighbour safety и безопасное восстановление после неподтверждённого изменения остаются недоказанными. Вердикты обоих правил NOT_PROVEN. Production не изменён и не продвигался.

![T09: слева1470,справа1471; верхизменился,десять нижних полос неподвижны](diagnostic/T09_1470_1471_stationary_suffix.png)

[Integrity](final_integrity_receipt.json) · [ID metrics](symmetric_v2/id_transfer_summary.json) · [Checkpoint diagnosis](symmetric_v2/checkpoint_diagnostics.json) · [Adjacent symmetry](symmetric_v2/adjacent_symmetry_audit.json)
'''
    (out/'SHORT_REPORT_RU.md').write_text(short,'utf-8');(out/'REPORT.md').write_text(detail,'utf-8')
    print(str((out/'SHORT_REPORT_RU.md').resolve()))

if __name__=='__main__':main(Path(sys.argv[1]))
