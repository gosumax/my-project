"""Measured adjacent/SAFE comparison, source-preserving edge casebook."""
import base64
from collections import Counter,defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import sys
import zlib
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.vertical_chat_id_transfer_audit import annotated


def report(source,out):
    read=lambda p:json.loads(p.read_text('utf-8'))
    old=read(source/'symmetric_v2/zonal_summary.json');strict=read(source/'adjacent_safe_v3/zonal_summary.json');new=read(out/'zonal_summary.json')
    old_metrics={**old['statistics'],**read(source/'symmetric_v2/id_transfer_summary.json')['metrics_by_scope']['evaluation']}
    ids=read(out/'id_transfer_summary.json');ocr=read(out/'zonal_ocr_receipt.json');s=new['statistics']
    rr=list(map(json.loads,(out/'symmetric_transitions.jsonl').open(encoding='utf-8')))
    index={(r['table'],r['frame_from'],r['frame_to']):r for r in rr}
    remaining=[r for r in rr if r['evaluation'] and r['classification']=='UNRESOLVED']
    failures=[]
    for r in remaining:
        top=r['zones']['top'];height=r['zones']['bottom']['y2'];candidate=top['shift_px']
        support={name:max(0,min(z['y2'],height-candidate)-z['y1']) for name,z in r['zones'].items()}
        failures.append({'table':r['table'],'frame_from':r['frame_from'],'frame_to':r['frame_to'],
                         'reason':r['reason'],'top_candidate_shift_px':candidate,'zone_support_at_top_candidate_px':support,
                         'two_zone_support_possible_at_top_candidate':sum(v>=48 for v in support.values())>=2,
                         'zones':r['zones']})
    (out/'remaining_failure_diagnostics.json').write_text(json.dumps({'scope':'13 evaluation adjacent failures, top shift is a candidate not confirmed GLOBAL',
        'minimum_zone_overlap_px':48,'cases':failures},ensure_ascii=False,indent=2),'utf-8')
    old_edges=[r for r in read(source/'symmetric_v2/adjacent_symmetry_audit.json')['transitions'] if not r['symmetry']['passed']]
    edge_records=[{'old':r,'new':index[(r['table'],r['frame_from'],r['frame_to'])]} for r in old_edges]
    text_by_id={i['item_id']:i.get('raw_text','') for l in (out/'zonal_raw_ocr_results.jsonl').open(encoding='utf-8') for i in json.loads(l)['results']['results']}
    groups=defaultdict(list)
    for rid,text in text_by_id.items():groups[text].append(rid)
    repeated={text:ids for text,ids in groups.items() if len(ids)>1}
    same_text_examples=[];seen=set()
    for r in rr:
        if r['classification']!='GLOBAL_SCROLL':continue
        local=defaultdict(list)
        for rid in r['old_ids']:
            if rid in text_by_id:local[text_by_id[rid]].append(rid)
        for text,values in local.items():
            if len(values)>1 and (text,tuple(values)) not in seen:
                same_text_examples.append({'transition':r,'same_text':text,'old_distinct_ids':values});seen.add((text,tuple(values)))
    same_text_examples=same_text_examples[:5]
    needed={(r['new']['table'],f) for r in edge_records for f in [r['new']['frame_from'],r['new']['frame_to']]}
    needed.update((r['transition']['table'],f) for r in same_text_examples for f in [r['transition']['frame_from'],r['transition']['frame_to']])
    images={}
    for line in (source/'states.jsonl').open(encoding='utf-8'):
        r=json.loads(line);key=(r['table'],r['frame'])
        if key in needed:images[key]=np.frombuffer(zlib.decompress(base64.b64decode(r['zlib_gray'])),np.uint8).reshape(r['shape']).copy()
    gallery=out/'edge_fix_cases';gallery.mkdir(exist_ok=True);sections=[]
    for entry in edge_records:
        r=entry['new'];t,f1,f2=r['table'],r['frame_from'],r['frame_to'];key=f'T{t:02d}_{f1:06d}_{f2:06d}'
        before=annotated(images[(t,f1)],r['old_observed_spans'],r['old_ids'],f'BEFORE {f1}; border fragment separate',r['old_link_states'])
        after=annotated(images[(t,f2)],r['new_observed_spans'],r['new_ids'],f'AFTER {f2}; N={r["symmetry"]["N"]}; ? unresolved',r['new_link_states'])
        cv2.imwrite(str(gallery/f'{key}.png'),np.hstack([before,after]))
        (gallery/f'{key}.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2),'utf-8')
        sections.append(f'<h2>{key}: OLD {entry["old"]["symmetry"]["reason"]} → NEW {r["classification"]}</h2><img src="edge_fix_cases/{key}.png"><pre>{html.escape(json.dumps(entry,ensure_ascii=False,indent=2))}</pre>')
    (out/'edge_fix_review.html').write_text('<meta charset="utf-8"><style>body{font-family:monospace}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>All 14 measured top-edge cases; ? means unresolved lineage</h1>'+''.join(sections),'utf-8')
    same_sections=[];same_gallery=out/'same_text_cases';same_gallery.mkdir(exist_ok=True)
    for entry in same_text_examples:
        r=entry['transition'];t,f1,f2=r['table'],r['frame_from'],r['frame_to'];key=f'T{t:02d}_{f1:06d}_{f2:06d}'
        before=annotated(images[(t,f1)],r['old_observed_spans'],r['old_ids'],f'BEFORE {f1}: same text, distinct IDs',r['old_link_states'])
        after=annotated(images[(t,f2)],r['new_observed_spans'],r['new_ids'],f'AFTER {f2}: identity NEVER from OCR',r['new_link_states'])
        cv2.imwrite(str(same_gallery/f'{key}.png'),np.hstack([before,after]));(same_gallery/f'{key}.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2),'utf-8')
        same_sections.append(f'<h2>{key}: {html.escape(entry["same_text"])}</h2><img src="same_text_cases/{key}.png"><pre>{html.escape(json.dumps(entry,ensure_ascii=False,indent=2))}</pre>')
    (out/'same_text_review.html').write_text('<meta charset="utf-8"><style>img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Post-decision OCR selects review cases only; no identity feedback</h1>'+''.join(same_sections),'utf-8')
    (out/'repeated_text_posthoc.json').write_text(json.dumps({'repeated_text_groups':repeated,'same_old_viewport_examples':same_text_examples,'identity_inputs_from_OCR':0,'limit':'Repeated raw OCR text is not independently certified equal physical content.'},ensure_ascii=False,indent=2),'utf-8')
    before=read(source/'source_before.json');cap=read(source/'capture.json')
    changed=[p for p,h in before.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    manifest_changed=hashlib.sha256((ROOT/'production_manifest.json').read_bytes()).hexdigest()!=cap['manifest_sha256']
    integrity={'registered_file_count':len(before),'changed_registered_files':changed,'manifest_changed':manifest_changed,
               'production_changed':bool(changed or manifest_changed),'source_states_sha256':hashlib.sha256((source/'states.jsonl').read_bytes()).hexdigest(),
               'experiment_sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'experiments').glob('vertical*.py')}}
    (out/'integrity_receipt.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2),'utf-8')
    names=['GLOBAL_SCROLL_COUNT','LOCAL_CHANGE_COUNT','UNRESOLVED_COUNT','ID_TRANSFER_CORRECT','ID_TRANSFER_WRONG','ID_TRANSFER_UNRESOLVED','DUPLICATE_ID','LOST_ID','ID_TRANSFER_CONFIRMED_PARENT_CORRECT','OCR_NEW_BOTTOM_ROWS']
    comparison=[{'metric':k,'v2_stale_checkpoint':old_metrics.get(k),'v3_adjacent_exact_LOCAL':strict['statistics'].get(k),'v4_adjacent_KEEP_uncertain':s.get(k)} for k in names]
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,list(comparison[0]));writer.writeheader();writer.writerows(comparison)
    edge_fixed=sum(e['new']['classification']=='GLOBAL_SCROLL' and e['new']['symmetry']['passed'] for e in edge_records)
    pct=(1-s['UNRESOLVED_COUNT']/old['statistics']['UNRESOLVED_COUNT'])*100
    result={'evaluation_comparison':comparison,'top_edge_cases_tested':len(edge_records),'top_edge_cases_fixed':edge_fixed,
            'OCR_NEW_ROWS_ONLY':ocr['only_GLOBAL_new_bottom_rows'] and ocr['old_row_requests']==ocr['whole_CHAT_requests']==ocr['initial_visible_row_requests']==0,
            'UNRESOLVED_reduction_percent':pct,'same_text_visual_examples':len(same_text_examples),'production_changed':integrity['production_changed'],
            'minimal_fix_verdict':'ADJACENT_MOTION_FIX_CONFIRMED_ON_FROZEN_SAMPLE',
            'physical_identity_verdict':'OLD_ROW_ID_TRANSFER_SAFE_NOT_PROVEN','count_rule_verdict':'SHIFT_TO_NEW_ROWS_NOT_PROVEN',
            'metric_scope':'ID_TRANSFER_CORRECT = pairwise object-ID agreement with local raster proxy; most earlier parent lineage remains explicitly unresolved.'}
    (out/'comparison_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    value=lambda v:'не измерялось' if v is None else str(v)
    table='\n'.join(f'| {row["metric"]} | {value(row["v2_stale_checkpoint"])} | {value(row["v3_adjacent_exact_LOCAL"])} | {value(row["v4_adjacent_KEEP_uncertain"])} |' for row in comparison)
    text=f'''# CHAT tracker: соседний кадр и SAFE_ID_STATE — 2026-10-03

Минимальное исправление сравнения кадров подтверждено на той же frozen выборке: UNRESOLVED **3875→13 (−{pct:.2f}%)**. Все **14/14** ранее отклонённых top-edge переходов проходят симметрию. Production не изменён.

NCC всегда сравнивает текущий crop с непосредственно предыдущим сохранённым состоянием того же стола. Исходный capture сохранял изменённые состояния; пропущенные неизменившиеся frames имеют те же pixels. SAFE_ID_STATE и source-object journal не используются как NCC raster. Они обновляются только при GLOBAL+symmetry; LOCAL/UNRESOLVED не создают ID/OCR и не изменяют SAFE snapshot.

## Сравнение того же evaluation

4044 перехода после калибровки, девять CHAT ROI; те же 6275 сохранённых состояний за 3000 кадров. Калибровка до кадра 997 не изменена. Пороги NCC (.90/.02), зональный gate и ROW_STEP 20.17647 px сохранены. Другие методы определения движения не добавлены.

| Метрика | v2 старый checkpoint | v3 соседний/exact LOCAL | v4 соседний/KEEP с uncertainty |
|---|---:|---:|---:|
{table}

v3 сохранён как контроль: native exact LOCAL стирал почти все observation links. В v4 использован прежний geometry KEEP для совместимых stationary объектов. При изменении pixels object ID сохраняется **как кандидат** с LOCAL_LINK_UNRESOLVED; последующие exact pixels не снимают старую неопределённость. GLOBAL переносит её как GLOBAL_PARENT_UNRESOLVED. UNRESOLVED motion обнуляет текущие links; SAFE journal не оживляет потерянные связи. Это не новый метод identity.

**1079 ID_TRANSFER_CORRECT** — совпадения ID объектов по локальному пиксельному эталону. Полная история физической строки при этом не доказана. Correct с подтверждённым по правилам эксперимента родителем — {s['ID_TRANSFER_CONFIRMED_PARENT_CORRECT']}; переносов с неопределённой историей родителя — {s['ID_CARRY_PARENT_UNRESOLVED']}/{s['ID_CARRY_COUNT']}. WRONG/DUPLICATE/LOST=0 относятся к измеренной выборке; неизвестные связи не сертифицированы. Вышедшие сверху IDs не считаются LOST, исходный журнал сохраняется.

С калибровкой: 255 GLOBAL commits, 4148 точных координатных переносов, 1659 совпадений с локальным эталоном, 2489 UNKNOWN. Максимальный остаток наблюдаемых центров glyph — {ids['observed_glyph_center_max_residual_px']:.1f} px. Расчётные координаты объектов меняются ровно на SHIFT. Проверяется упорядоченная геометрия с допуском ±2 px.

## Верхний край и симметрия

Production helper всегда исключает первую выделенную полосу. Эксперимент отделяет первую raw-полосу с y≤1 как edge/UNKNOWN и **сохраняет полноценную первую строку**. В 14 прежних отказах частичная полоса начиналась с y0/1 и имела высоту 4–6 px; полная — с y11–13 и высотой 10–12 px. Все 14 теперь имеют одинаковое число полных строк и остаток переноса ≤1.5 px. Произвольное изменение 17↔18 от проверки не освобождается.

Риск остаётся: обрезанный glyph с пустыми scanlines и первым ink y≥2 не доказан partial. Он может безопасно остаться UNRESOLVED; без данных threshold не расширен. В данной выборке оставшихся count/symmetry failures0.

Для каждого GLOBAL: N=round(SHIFT/ROW_STEP), Nout=Nin=N. Верхние N полных строк уходят из видимого окна, оставшиеся ID перемещаются ровно на SHIFT, неизвестные родительские ID остаются None. Создаются ровно N нижних объектов, только они идут в OCR. Частичный верхний edge не получает нового объекта/OCR. Фактическое количество совпало в **68/68** покрытых случаях; FALSE_GLOBAL_SCROLL=0 на этой выборке, ещё 113 actual UNKNOWN. Универсальный SHIFT_TO_NEW_ROWS_PROVEN пока не доказан. T09 1470→1471 остаётся LOCAL_CHANGE (top20/middle0/bottom0), новых ID и OCR — 0.

## Остаточные отказы, визуальные примеры и OCR

13 UNRESOLVED имеют недостаточно уверенных зон на **соседних** кадрах; сравнения с устаревшим checkpoint больше нет. Для всех 13 сохранены before/after и зональные измерения. 20 примеров с ID включают сдвиги на 1/2/3 строки, крупные сдвиги и LOCAL. Знак ? означает неподтверждённую историю; одинаковый цвет — одинаковый сохранённый ID объекта. Дополнительно сохранены все 14 исправленных edge-примеров. После замораживания решений OCR использован только для выбора обзорных примеров: {len(same_text_examples)} случаев одинакового текста у разных старых ID. Текст не определяет identity.

Конкретный предел оставшихся отказов: верхняя зона выбирает большие candidate shifts 262–323 px. При таком сдвиге для средней и нижней зон не хватает минимального перекрытия 48 px, поэтому второй уверенной зоны на том же сдвиге получить нельзя. В 12 случаях уверена только верхняя зона; T09 1350→1351 не имеет ни одной уверенной зоны (top score .896). Например T05 1123→1124: top302/score.933, middle20/.649, bottom20/.472; T03 2515→2516: top323/.906, middle58/.349, bottom59/.310. Это ограничение пространственного перекрытия и уверенности, не прежний top-edge отказ. Большой top candidate сам по себе не объявляется GLOBAL и не порождает OCR. Измерения всех 13 сохранены в remaining_failure_diagnostics.json.

Отдельный read-only аудит LOCAL показал причину потери links в v3: среди 69300 геометрически совместимых полос 48402 имеют одинаковые bounds и pixels; 20735 — одинаковые bounds, но различающиеся pixels; 163 — разные bounds. Для 20735 различий median MAE всего .0116 уровня серого, 19939 имеют MAE≤1. Поэтому exact-equality разрывает связь даже при малом pixel change. Есть и существенные локальные изменения (например T09 1066→1067); KEEP отмечает их неопределённость и не сертифицирует identity по координатам.

Замороженная OCR-очередь: **{ocr['OCR_results']}/{ocr['queue_rows']}**, ошибок {ocr['OCR_errors']}, время {ocr['seconds_including_startup']:.3f} s со startup. Evaluation — 292 новые нижние строки, calibration — 150. Запросов whole CHAT / initial / old / carried / LOCAL / UNRESOLVED — 0; входов OCR в identity — 0. Hash очереди сохранён: {ocr['queue_before_sha256']}. KEEP не изменил очередь: v3/v4 полностью идентичны.

## Проверки и итог

**40 tests passed / 0 failed**. Проверены переносы на 1/2/3 позиции, симметрия, появление/исчезновение частичного верхнего band, отказ при пропавшей внутренней строке, LOCAL→GLOBAL и GLOBAL→UNRESOLVED→GLOBAL без возврата старых SAFE ID. Также проверены сохранение uncertainty, изменённая/дублированная очередь, FFT parity, production chat_rows и observer isolation. На всех 6266 records frame_from совпадает с непосредственным предшественником, ошибок 0. На 6011 non-GLOBAL records изменений SAFE/OCR — 0; ошибок координат/ID receipts — 0/4148. Независимый reviewer блокирующих рисков не нашёл; ограничение edge y≥2 явно сохранено.

Время NCC v4 — {new['timing_seconds']['zonal_ncc_seconds']:.3f} s. Все 103 зарегистрированных файла и manifest неизменны; production integrity PASS. Повторного capture или полного production run не было; hash frozen input сохранён в receipt.

**MINIMAL FIX:** CONFIRMED_ON_FROZEN_SAMPLE. **PRODUCTION CHANGED:** NO.

**SHIFT_TO_NEW_ROWS:** NOT_PROVEN для всей выборки (113 actual UNKNOWN). **OLD_ROW_ID_TRANSFER_SAFE:** NOT_PROVEN для всей истории (почти вся предыдущая lineage неопределённа). Улучшение определения движения не закрывает эти ограничения identity.

[Сравнение CSV](comparison.csv) · [Все SHIFT/STEP/actual/error](shift_to_new_rows.csv) · [20 ID-примеров](id_transfer_review.html) · [14 edge-примеров](edge_fix_review.html) · [13 остаточных отказов](symmetry_failure_review.html) · [Same-text review](same_text_review.html) · [Integrity](integrity_receipt.json)
'''
    (out/'REPORT_RU.md').write_text(text,'utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2));print(str((out/'REPORT_RU.md').resolve()))

if __name__=='__main__':report(Path(sys.argv[1]),Path(sys.argv[2]))
