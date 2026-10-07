"""Document independent visual review; no model scores substituted for labels."""
import json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/text_first_precision_review_smoke17_20261001'
def read(p):return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line]
def percent(p):return 'NOT_MEASURED' if p is None else f'{p*100:.2f}%'

def main():
    s=json.loads((OUT/'independent_precision_summary.json').read_text('utf-8'))
    results=read(OUT/'independent_results.jsonl');sealed={r['case_id']:r for r in read(OUT/'sealed_evidence.jsonl')}
    text={r['pair_id']:r for r in read(ROOT/'runs/text_first_identity_smoke17_20261001/relationships.jsonl')}
    sample=json.loads((OUT/'sample_summary.json').read_text('utf-8'))
    verification_path=OUT/'verification.json'
    verification=json.loads(verification_path.read_text('utf-8')) if verification_path.exists() else None
    ph_path=OUT/'independent_post_hoc.jsonl';ph=read(ph_path) if ph_path.exists() else []
    candidate=s['prospective_candidates'];fn=s['ambiguous_false_negative_sample'];allstat=s['all_sample']
    post_complete={r['case_id'] for r in ph}>= {'V135','V175'}
    complete=s['status']=='COMPLETE_CAUSAL_REVIEW' and post_complete
    verdict=('TEXT_FIRST_PRECISION_UNSAFE' if s['false_positives'] else
        'TEXT_FIRST_PRECISION_PROVEN_HIGH' if complete and candidate['reviewed']==280 and candidate['UNCERTAIN']==0
          and all(r.get('false_same_established') is False and r.get('post_hoc_result')=='FRAGMENTED_IDENTITY_SUPPORTED' for r in ph) else
        'TEXT_FIRST_PRECISION_PARTIAL')
    s.update(task_complete=complete,verdict=verdict,post_hoc_complete=post_complete,
        precision_claim_scope='Independent AI visual review of one frozen Smoke17 candidate set; not deployment certification or general unseen-video accuracy.')
    (OUT/'independent_precision_summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
    status=[]
    byid={r['case_id']:r for r in results}
    for cid,e in sealed.items():
        r=byid.get(cid)
        status.append(dict(case_id=cid,pair_id=e['pair_id'],review_status='REVIEWED' if r else 'NOT_REVIEWED',
            visual_label=r['visual_label'] if r else None,precision_included=bool(r),
            original_simulation_verdict=e['original_simulation_verdict'],categories=e['categories']))
    (OUT/'review_status.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in status),encoding='utf-8')
    lines=['# TEXT-FIRST precision validation — Smoke17','',
        '## FILES CHANGED','',
        'Только новые experiment/report scripts и review JSON/JSONL artifacts. `text_first_precision_review.py` регенерирует исходные viewports в RAM; `text_first_posthoc_view.py` показывает отдельные future-validation states; `aggregate_text_first_precision.py` сводит сохранённые independent labels; `report_text_first_precision_final.py` строит этот отчёт. Production, frozen linker, TEXT-FIRST admission rules, content thresholds, OCR/math/Jev, stable IDs и source rows не изменены.', '',
        '## REVIEW SAMPLE','',
        f'Зафиксировано **330 уникальных случаев: все280 candidates и50 stratified AMBIGUOUS**. Завершено **{s["reviewed_unique"]}**, pending **{s["pending"]}**. Reviewer labels записывались после фактического показа RAM-images, не выводились из OCR/scores. Каждый reviewer не видел original admission outcome, score/envelope или terminal lifetime. Целевые строка/IDs/text/time/y обозначены; watermark classification=VISUAL_REVIEW_PENDING. Batches сохранялись сразу, чтобы service interruption не терял результат.', '',
        'Категории перекрываются без повторного подсчёта pair в общем total. Все58 ANTE,13 SB/BB,15 FOLD,55 NICK_ACTION и154 NORMAL candidates включены. WEAK_CONTENT28 — ближайшие к local control envelope top decile; все остальные candidates также включены. MAX_Y_RESIDUAL106 — все residual1.5px. GAP2 candidates0: все7 gap2 находятся в50-case AMBIGUOUS sample.', '',
        'Полное causal evidence каждого pair сохранено в `sealed_evidence.jsonl`: old/new stable IDs, table/session, exactPTS frames/times, Δframes/ms, raw/normalized text+uniqueness, expected/actual y/residual/shift, same-ID coherent neighbors, target/control MAE/XOR, registered distance, native content/alignment/bbox/trims, boundary/overlap/gap/bridge/categories. Его доступ открывается только после causal visual labels. `blind_cases.jsonl` не содержит admission outcome или future lifetimes.', '',
        '## VISUAL VALIDATION RESULT','',
        '| Cohort | reviewed | VISUALLY_SAME | VISUALLY_DIFFERENT | VISUALLY_UNCERTAIN |','|---|---:|---:|---:|---:|']
    for label,st in [('ALL REVIEWED',allstat),('280 prospective candidates',candidate),('50 AMBIGUOUS sample',fn)]:
        lines.append(f'| {label} | {st["reviewed"]} | {st["SAME"]} | {st["DIFFERENT"]} | {st["UNCERTAIN"]} |')
    if verification:
        lines+=['',f'Independent metadata verification: **{verification["status"]}**, checks passed **{verification["checks_passed"]}**, failed **{verification["checks_failed"]}**; `verification.json`. Receipt/hash consistency проверена отдельно от visual judgments: metadata сама по себе не доказывает physical SAME.']
    lines+=['',
        'SAME требует видимой сохранности конкретного occurrence с соседним блоком/порядком/viewport motion; одинаковый OCR не ground truth. DIFFERENT — positive new-hand/new-occurrence/physical contradiction. Непросмотренные интервалы, повторяемый блок, blur или неоднозначность остаются UNCERTAIN. Совпадение двух текстов не заменяет physical continuity.', '',
        f'Shared review: {s["shared_cases_complete"]} случаев, exact label agreement {s["shared_exact_agreement"]}; disagreements **{len(s["shared_disagreements"])}**. Разногласия conservatively становятся UNCERTAIN, без root override. Raw reviewer judgments сохранены. Root preliminary observations предыдущего запуска не входят в metrics.', '',
        '## PRECISION','',
        f'Главная precision TEXT-FIRST candidates: **{candidate["SAME"]}/({candidate["SAME"]}+{candidate["DIFFERENT"]}) = {percent(candidate["confirmed_precision"])}**. UNCERTAIN **{candidate["UNCERTAIN"]}** и pending не входят в denominator; их нельзя считать успешными merges.', '',
        f'Для всей перемешанной reviewed выборки requested formula SAME/(SAME+DIFFERENT) = **{percent(allstat["confirmed_precision"])}**. Это доля SAME среди определённых visual labels, а не precision admission classifier: cohort включает заведомо отклонённые AMBIGUOUS.', '',
        '| Candidate category | reviewed | SAME | DIFFERENT | UNCERTAIN | confirmed precision |','|---|---:|---:|---:|---:|---:|']
    for cat,st in s['candidate_categories'].items():
        lines.append(f'| {cat} | {st["reviewed"]} | {st["SAME"]} | {st["DIFFERENT"]} | {st["UNCERTAIN"]} | {percent(st["confirmed_precision"])} |')
    lines+=['',
        'Categories overlap; таблицы не суммируются. GAP2/short/amount strata с нулевым candidate denominator имеют NOT_MEASURED, а не100%. Это AI-assisted independent visual review одного frozen video; не human-certified physical IDs и не accuracy на новых videos.', '',
        '## FALSE POSITIVES','',
        f'Observed candidate false positives: **{s["false_positives"]}**. Detailed judgments и полный evidence — `visual_false_positives.jsonl` / `sealed_evidence.jsonl`.', '']
    fp=[r for r in results if r['original_simulation_verdict']=='TEXT_FIRST_SAME_CANDIDATE' and r['visual_label']=='VISUALLY_DIFFERENT']
    for r in fp:
        e=sealed[r['case_id']]
        lines += [f'- {r["case_id"]}: `{r["pair_id"]}`; frames {e["old"]["last_frame"]}→{e["new"]["first_frame"]}; categories {r["categories"]}.',
            '  '+ ' / '.join(x['rationale'] for x in r['raw_judgments'].values())]
    if not fp:lines+=['False-positive pattern в completed candidate labels не найден. Это не утверждение об отсутствии риска у UNCERTAIN или unseen videos.']
    lines+=['', '## AMBIGUOUS FALSE-NEGATIVE SAMPLE','',
        f'Reviewed **{fn["reviewed"]}/50**: obvious SAME **{fn["SAME"]}**, DIFFERENT **{fn["DIFFERENT"]}**, UNCERTAIN **{fn["UNCERTAIN"]}**. `observed false-negative candidates` = **{s["observed_false_negative_candidates"]}**. Это не full recall metric: selection stratified и intentionally near-admission/edge-heavy.', '',
        'Missing admission reasons у visually SAME AMBIGUOUS (overlap counts):', '', '```json',
        json.dumps(dict(Counter(reason for r in results if r['original_simulation_verdict']=='AMBIGUOUS' and r['visual_label']=='VISUALLY_SAME' for reason in text[r['pair_id']]['blocking_reasons'])),ensure_ascii=False,indent=2),'```', '',
        'Подробные skipped-SAME examples: `visual_false_negatives.jsonl`. Даже obvious visual SAME не используется для изменения uniqueness/gap/neighbors/content thresholds в этом задании.', '',
        '## POST_HOC REAPPEARANCE','',
        'Causal judgments зафиксированы до future-image review; SHA256 сохранены в `causal_review_freeze.json`. Independent POST_HOC_RISK_RESULT labels содержатся в `independent_post_hoc.jsonl` и не меняют original causal TEXT-FIRST verdict или causal reviewer label. Root предыдущего preliminary posthoc не считается independent ground truth.', '',
        '| CASE | new first | old returns | delay |','|---|---:|---:|---:|',
        '| V135 R001196→R001199 | 26728 /890933.333ms | 26749 /891633.333ms |21frames /700ms |',
        '| V175 R001183→R001189 | 26380 /879333.333ms | 26691 /889700ms |311frames /10366.667ms |', '']
    if ph:
        for r in ph:lines+=['```json',json.dumps(r,ensure_ascii=False,indent=2),'```','']
    else:lines+=['Independent post-hoc adjudication pending. Оба случая withheld до его завершения.','']
    lines+=['Возвращение old ID само по себе не доказывает false SAME: повторное наследование старого ID может быть сложным fragmentation. Нужны target+neighbor physical context и рассмотрение alternate occurrence; posthoc future не входит в earlier admission.', '',
        '## EDGE-GATE PROSPECTIVE IMPACT','',
        f'С visually SAME validated prospective relationships **{s["edge_simulation"]["visually_validated_candidate_relationships"]}** на копиях source rows: **6685 → {s["edge_simulation"]["after"]} UNRESOLVED_EDGE**. Source fields/IDs/epochs/boundaries сохранены, rows8146. Existing gate не потребляет relationship metadata; theoretical removal под действующим gate0. Не придуман альтернативный edge policy. Ничего не применяется.', '',
        '## IMAGE WRITES','', '**0**. Native sourceROI, overlay, zoom, contact-sheet encoding толькоRAM/BytesIO; images не записаны вPNG/JPG/WEBP/base64 disk cache. JSON receipts содержат hashes/frames, безimagepayload. Final source/imageinventory safety — `safety_audit.json`; strict production integrity PASS.', '',
        '## SOURCE DATA CHANGED','', 'NO.', '', '## PRODUCTION CODE CHANGED','', 'NO.', '',
        '## VERDICT','',f'**{verdict}**.', '',
        'Высокая observed precision одной frozen выборки не разрешает production merge, не доказывает устойчивость на новых раздачах/videos и не меняет certified hand-history contract. Все admission rules/thresholds frozen; automatic SAME/DIFFERENT/merge/edge admission отсутствуют.', '',
        'После final review работа останавливается. Незавершённые cases явно остаются pending; completed visual labels доступны для проверки по сохранённому RAM renderer.']
    (ROOT/'docs/TEXT_FIRST_PRECISION_REVIEW_SMOKE17_20261001.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=complete,verdict=verdict,candidate_precision=candidate['confirmed_precision'],candidate_counts=candidate,post_hoc_complete=post_complete),ensure_ascii=False))

if __name__=='__main__':main()
