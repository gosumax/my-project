"""Publish final diagnostic verdict, preserving failed v1 evidence."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/start_stack_two_stage_ncc_20261003'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def publish():
    v2=OUT/'candidate_v2';reg=read(v2/'regression20/summary.json');held=read(v2/'heldout_smoke17_new/tester_receipt.json');unreviewed=read(v2/'heldout_smoke17_new/unlabeled_review_summary.json')
    snapshot=v2/'final_repro_source';snapshot.mkdir(exist_ok=True);sources=[]
    for name in ['dealing_two_stage_ncc_20261003.py','run_two_stage_regression_20261003.py','run_two_stage_smoke17_20261003.py','test_dealing_two_stage_20261003.py','report_two_stage_ncc_20261003.py']:
        source=ROOT/'experiments'/name;target=snapshot/name;shutil.copyfile(source,target);sources.append({'source':str(source),'snapshot':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    (v2/'final_source_manifest.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text=f'''# Двухступенчатый NCC Dealing cards — offline

**Вердикт: NOT_PROVEN для финального trigger. Production не изменён. Подключение под feature flag не подготовлено: условие успешной held-out проверки не выполнено.**

Реализована схема `Dealer: Dealing gate → 4-class NCC → fresh current-hand check → once-per-hand latch`. Классификация проверена на regression и новом Smoke17. Независимый источник доказанной свежести физической строки отсутствует; без него корректно сохраняется UNKNOWN и event не выдаётся.

## Фиксация до проверки

V2 config: `candidate_v2/frozen_config.json`; SHA256 `9fc6e263d7b4102e8f4e456ae2db77bace3eaebd091014c762c2b244ff14bc3c` совпадает с preregistration.

- Gate: отдельное подтверждение `Dealer:` **и** `Dealing` в одной физической строке, оба NCC >=0.90. Позиция Dealing привязана к найденному Dealer; нельзя взять совпадение из другой строки/слова.
- Только после gate: `cards / Turn / Flop / River`, общие40×16 неизменённые grayscale templates, bounded horizontal shifts. Margin >=0.20, сохранён прежний диагностический уровень; не выбрана найденная на старых20 separating boundary0.213..0.251.
- Абсолютный word threshold0.95 сохранён отдельно как диагностический признак; не снижен и не служит основанием для относительного ranking.
- Scores — correlation, не вероятность. Шаблоны Smoke14/T01 не дообучались на Smoke17, resize/brightness correction/template bank отсутствуют.
- Отказ gate: статус UNKNOWN, scores/winner/runner-up/margin =null; четырёхклассовый NCC не запускается.

Для прошедших gate сохранены все4scores, winner, runner_up, margin, frame/sourcePTS, table, hand_id или null, row bounds, pixel hash. Для UNKNOWN сохранены причина и gate evidence. Row bounds — y interval в CHAT ROI; x/offsets задаются hash-bound layout/ROI manifests. Ничего не классифицируется принудительно через fallback nearest class после отказа gate.

## Regression20 до нового held-out

| Метрика | Значение |
|---|---:|
| Старые руки / frames / row observations |20 /{reg['frames']} /{reg['rows']}|
| Первая ступень отсекла |{reg['first_stage_excluded']}|
| Дошли до4-class |{reg['four_class_rows']}|
| Запуск4-class после отказа gate |{reg['gate_fail_classifier_calls']}|
| Классификация cards в initial3frames |20HITS /0MISSES|
| Pixel-bound hand-header проверки |40/40|
| Известные ложные Ante pixel hashes |9/9 отсечены gate; stage2calls0|
| Финальный trigger |0HITS /20MISSES /0duplicates|

Последняя строка не является успешной trigger-acceptance: свежесть не доказана. Старые20не использовались для подбора нового margin.

## Новый held-out Smoke17

После regression выполнен ранее не просмотренный диапазон **1801–3600 inclusive**,1800frames, все9tables. Диапазон и конфигурация записаны до просмотра его labels: `candidate_v2/heldout_smoke17_new/new_heldout_preregistration.json`.

Source video SHA256: `27c0ba10544e0f6025e2d563baf7693d36c185cdbbfeb4517d3486116e9f8baf`. Прямой PyAV decode исходного видео; все физические CHAT spans каждого кадра, без допуска/фильтрации production tracker. Старые `physical_rows` используются только для независимых pixel-bound headers/последующей разметки, а не как единственный detector input. Dealing labels не подаются в model или freshness check.

| Требуемая метрика, финальные triggers | Значение |
|---|---:|
| HITS |{held['HITS']}|
| MISSES,11независимо размеченных новых hand groups |{held['MISSES']}|
| FALSE POSITIVES |{held['FALSE_POSITIVES']}|
| UNKNOWN classification row observations |{held['UNKNOWN_ROWS']}|
| UNKNOWN freshness row observations |296718|
| DUPLICATE TRIGGERS |{held['DUPLICATE_TRIGGERS']}|

**Ноль false triggers получен при нуле triggers, поэтому не доказывает пригодность метода.**

| Диагностика ступеней | Значение |
|---|---:|
| Все row observations |296718|
| Первая ступень отсекла |{held['stage1_excluded']}|
| Дошли до4-class |{held['stage2_observations']}|
| Запуск4-class после отказа gate |0|
| Cards classifications, повторные видимости включены |10034|
| Turn /Flop /River classifications |3665 /1662 /1467|
| Новые hand groups с >=1правильной independently-labeled cards classification |11/11|
| Independently-labeled cards rows |25classified /32observed;7misses|
| Все independently-labeled Dealing rows |39gatepass /56observed;17gatemisses|
| Independently-labeled обычные строки |470;gatefalsepass0 /falsecards0|

11/11 — классификация хотя бы одного известного наблюдения группы. Это не recall первого появления, не стабильная тройка и не11events. Initial partial hands не включены в11новых групп; сохраняется UNKNOWN context.

Из10034cards classifications3717заблокированы CURRENT_HAND_UNKNOWN и6317ROW_FRESHNESS_UNKNOWN. Строки Turn/Flop/River — известные иные классы, а не trigger-кандидаты.

## Какие false positives и UNKNOWN остались

В470pixel-bound обычных строках нового held-out ложных пропусков gate нет. Дополнительно9lowest-confidence cards crops, по одной на table, проверены по sourcePTS/pixelSHA и визуально:9/9реальных `Dealer: Dealing cards`,0false.

Но полной truth-разметки detector candidates нет:796unique hashes у10034cards observations;34hashes имеют независимую разметку/ручной контроль. **762hashes /9548observations остаются UNREVIEWED.** Полный candidate FALSE POSITIVE total — UNKNOWN, не0. Девять crops — bounded контроль, не сертификат полного диапазона.

Проверка нового held-out заняла97.611s (включая source verification/decode); это offline время, не production latency. Совпадение526/526source-control row pixels,0ошибок. Production throughput/live OCR savings не измерялись.

## Почему freshness не считается доказанной

Нумерованный hand header принимается только после точного совпадения его source-frame pixels, в причинном порядке и без отката hand number. Initial partial/currenthand неизвестен до header. Нельзя использовать dealer/cards truth как context oracle.

Визуальное suffix/prefix перекрытие3строк доказывает только overlap, не новую физическую строку: исторический/повторный viewport может иметь такой же блок. Поэтому overlap_append_candidate сохраняется, но append_proven=False и freshness UNKNOWN. При пропуске кадров identity сбрасывается. Pixel similarity или NEW_CANDIDATE из старого tracker сами не превращаются в доказательство свежести.

Latch выдаёт не более1event на подтверждённый(table,hand); нужны3последовательных подтверждения одного fresh row track. Unknown row/hand не расходует latch. Продолжающаяся видимость и исчезновение/повторное появление не создают второй event. Эти state-contracts проверены тестами, но end-to-end source producer независимой fresh-row receipt пока не реализован. Именно это блокирует все реальные triggers.

## Сохранённый неудачный V1 прогон

V1 frames0–1800 оставлены без удаления evidence. Whole-prefix NCC0.90 иногда зависел от жирного `Dealer:` и пропускал иной текст. Конкретный false candidate: T03 frame557 `Dealer: rObot bets45 815`, prefix0.911468/cards0.417972/margin0.218696. Он не дал event только из-за неизвестного контекста; это не успех gate.

V2 исправляет общий pattern через конъюнкцию обоих слов при тех же0.90/0.20, а не через подбор порога. Прежняя bet-строка — negative regression: UNKNOWN,4scores=null. Новый held-out иной, его labels не использованы для v2фиксации. Наблюдение V1 не переименовано в независимый held-out дляV2.

## Проверки и безопасное подключение

8targeted tests passed,0failed: skip stage2 on gate fail, unknown geometry/context, latch persistence/reappearance, gaps, историческое visual overlap без свежести. Independent audit: обе исходные проблемы identity-over-gap/overlap-as-proof исправлены; fresh producer остаётся явным ограничением. Current manifest integrity требуется подтвердить отдельным коротким check; production-код/configs/manifest не менялись.

**Подключение под feature flag сейчас заблокировано.** Нужно независимо доказать source-bound свежесть и принадлежность строки текущей руке, затем проверить meaningful final triggers на новом held-out с отрицательными history/reappearance controls. Нельзя получить0FPчерез0admitted events и назвать это успешным acceptance.

Evidence: v2frozen_config, regression20rows/frames/header_pixel_checks/hand_results, heldout rows.jsonl/frames.jsonl/ground_truth.jsonl/triggers.jsonl (имена фактических файлов в run directory), tester_receipt, preregistration, boundedreview/contact_sheet, review.json, final_source_manifest и reproducible source snapshots. Исходные pixels/OCR/UNKNOWN и ранние неудачные results сохранены.
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    verified={'verdict':'NOT_PROVEN','production_changed':False,'feature_flag_connection_ready':False,'frozen_prefix_threshold':.90,'frozen_margin_threshold':.20,'absolute_word_95_unchanged':True,'regression_classification_hits':20,'heldout_window_inclusive':[1801,3600],'final_trigger_HITS':0,'final_trigger_MISSES':11,'final_trigger_FALSE_POSITIVES':0,'final_trigger_DUPLICATE_TRIGGERS':0,'zero_trigger_zero_fp_is_acceptance':False,'classification_UNKNOWN':279890,'freshness_UNKNOWN':296718,'stage1_excluded':279890,'stage2_rows':16828,'known_new_hand_classification_groups':11,'known_cards_rows_classified':25,'known_cards_rows':32,'labeled_non_dealing_false_gate_passes':0,'labeled_non_dealing_rows':470,'all_candidate_false_positive_total':'UNKNOWN','candidate_unreviewed_hashes':762,'candidate_unreviewed_observations':9548,'primary_blocker':'Independent physical-row fresh/current-hand lifecycle proof producer unavailable.'}
    (OUT/'VERIFIED_RESULT.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return str(OUT/'REPORT.md')
if __name__=='__main__':print(publish())
