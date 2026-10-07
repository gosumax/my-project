# TEXT-FIRST precision validation — Smoke17

## FILES CHANGED

Только новые experiment/report scripts и review JSON/JSONL artifacts. `text_first_precision_review.py` регенерирует исходные viewports в RAM; `text_first_posthoc_view.py` показывает отдельные future-validation states; `aggregate_text_first_precision.py` сводит сохранённые independent labels; `report_text_first_precision_final.py` строит этот отчёт. Production, frozen linker, TEXT-FIRST admission rules, content thresholds, OCR/math/Jev, stable IDs и source rows не изменены.

## REVIEW SAMPLE

Зафиксировано **330 уникальных случаев: все280 candidates и50 stratified AMBIGUOUS**. Завершено **330**, pending **0**. Reviewer labels записывались после фактического показа RAM-images, не выводились из OCR/scores. Каждый reviewer не видел original admission outcome, score/envelope или terminal lifetime. Целевые строка/IDs/text/time/y обозначены; watermark classification=VISUAL_REVIEW_PENDING. Batches сохранялись сразу, чтобы service interruption не терял результат.

Категории перекрываются без повторного подсчёта pair в общем total. Все58 ANTE,13 SB/BB,15 FOLD,55 NICK_ACTION и154 NORMAL candidates включены. WEAK_CONTENT28 — ближайшие к local control envelope top decile; все остальные candidates также включены. MAX_Y_RESIDUAL106 — все residual1.5px. GAP2 candidates0: все7 gap2 находятся в50-case AMBIGUOUS sample.

Полное causal evidence каждого pair сохранено в `sealed_evidence.jsonl`: old/new stable IDs, table/session, exactPTS frames/times, Δframes/ms, raw/normalized text+uniqueness, expected/actual y/residual/shift, same-ID coherent neighbors, target/control MAE/XOR, registered distance, native content/alignment/bbox/trims, boundary/overlap/gap/bridge/categories. Его доступ открывается только после causal visual labels. `blind_cases.jsonl` не содержит admission outcome или future lifetimes.

## VISUAL VALIDATION RESULT

| Cohort | reviewed | VISUALLY_SAME | VISUALLY_DIFFERENT | VISUALLY_UNCERTAIN |
|---|---:|---:|---:|---:|
| ALL REVIEWED | 330 | 313 | 8 | 9 |
| 280 prospective candidates | 280 | 280 | 0 | 0 |
| 50 AMBIGUOUS sample | 50 | 33 | 8 | 9 |

Independent metadata verification: **PASS**, checks passed **773**, failed **0**; `verification.json`. Receipt/hash consistency проверена отдельно от visual judgments: metadata сама по себе не доказывает physical SAME.

SAME требует видимой сохранности конкретного occurrence с соседним блоком/порядком/viewport motion; одинаковый OCR не ground truth. DIFFERENT — positive new-hand/new-occurrence/physical contradiction. Непросмотренные интервалы, повторяемый блок, blur или неоднозначность остаются UNCERTAIN. Совпадение двух текстов не заменяет physical continuity.

Shared review: 20 случаев, exact label agreement 19; disagreements **1**. Разногласия conservatively становятся UNCERTAIN, без root override. Raw reviewer judgments сохранены. Root preliminary observations предыдущего запуска не входят в metrics.

## PRECISION

Главная precision TEXT-FIRST candidates: **280/(280+0) = 100.00%**. UNCERTAIN **0** и pending не входят в denominator; их нельзя считать успешными merges.

Для всей перемешанной reviewed выборки requested formula SAME/(SAME+DIFFERENT) = **97.51%**. Это доля SAME среди определённых visual labels, а не precision admission classifier: cohort включает заведомо отклонённые AMBIGUOUS.

| Candidate category | reviewed | SAME | DIFFERENT | UNCERTAIN | confirmed precision |
|---|---:|---:|---:|---:|---:|
| NORMAL | 154 | 154 | 0 | 0 | 100.00% |
| ANTE | 58 | 58 | 0 | 0 | 100.00% |
| SB_BB | 13 | 13 | 0 | 0 | 100.00% |
| FOLD | 15 | 15 | 0 | 0 | 100.00% |
| NICK_ACTION | 55 | 55 | 0 | 0 | 100.00% |
| GAP2 | 0 | 0 | 0 | 0 | NOT_MEASURED |
| WEAK_CONTENT | 28 | 28 | 0 | 0 | 100.00% |
| MAX_Y_RESIDUAL | 106 | 106 | 0 | 0 | 100.00% |
| POST_HOC_RISK | 2 | 2 | 0 | 0 | 100.00% |
| SHORT_TEXT | 0 | 0 | 0 | 0 | NOT_MEASURED |
| AMOUNT_ONLY | 0 | 0 | 0 | 0 | NOT_MEASURED |

Categories overlap; таблицы не суммируются. GAP2/short/amount strata с нулевым candidate denominator имеют NOT_MEASURED, а не100%. Это AI-assisted independent visual review одного frozen video; не human-certified physical IDs и не accuracy на новых videos.

## FALSE POSITIVES

Observed candidate false positives: **0**. Detailed judgments и полный evidence — `visual_false_positives.jsonl` / `sealed_evidence.jsonl`.

False-positive pattern в completed candidate labels не найден. Это не утверждение об отсутствии риска у UNCERTAIN или unseen videos.

## AMBIGUOUS FALSE-NEGATIVE SAMPLE

Reviewed **50/50**: obvious SAME **33**, DIFFERENT **8**, UNCERTAIN **9**. `observed false-negative candidates` = **33**. Это не full recall metric: selection stratified и intentionally near-admission/edge-heavy.

Missing admission reasons у visually SAME AMBIGUOUS (overlap counts):

```json
{
  "SUPPORTING_CONTENT": 24,
  "TEXT_UNIQUE": 9,
  "IMMEDIATE_CONTINUITY": 7,
  "UNIQUE_GEOMETRY": 2
}
```

Подробные skipped-SAME examples: `visual_false_negatives.jsonl`. Даже obvious visual SAME не используется для изменения uniqueness/gap/neighbors/content thresholds в этом задании.

## POST_HOC REAPPEARANCE

Causal judgments зафиксированы до future-image review; SHA256 сохранены в `causal_review_freeze.json`. Independent POST_HOC_RISK_RESULT labels содержатся в `independent_post_hoc.jsonl` и не меняют original causal TEXT-FIRST verdict или causal reviewer label. Root предыдущего preliminary posthoc не считается independent ground truth.

| CASE | new first | old returns | delay |
|---|---:|---:|---:|
| V135 R001196→R001199 | 26728 /890933.333ms | 26749 /891633.333ms |21frames /700ms |
| V175 R001183→R001189 | 26380 /879333.333ms | 26691 /889700ms |311frames /10366.667ms |

```json
{
  "case_id": "V135",
  "review_phase": "POST_HOC_CHECK",
  "shown_frames": [
    26727,
    26728,
    26729,
    26730,
    26731,
    26732,
    26733,
    26734,
    26735,
    26736,
    26740,
    26749
  ],
  "post_hoc_result": "FRAGMENTED_IDENTITY_SUPPORTED",
  "false_same_established": false,
  "rationale": "The riverpom shows glyph line remains immediately below kisunya shows/Full house555TT/TT cards and immediately above Three of a kind555/AJ cards. It moves from row15 to row13 at the causal scroll, stays at row13 through all intermediate changed states, then moves to row9 at old-ID return with those neighbors intact. The same 555 flop, 5557 turn and 1600187 all-in/call block remain ordered; the 55577 river is appended before the later kisunya win4538374 block. This visually supports one retained occurrence across fragmented IDs.",
  "possible_alternate_explanation": "An indistinguishable repeated history would require the same cards, bet amounts, show descriptions and ordered neighbors; no fresh dealing or hand boundary appears in the reviewed interval.",
  "limitations": "Grayscale compression softens glyph edges. Reviewed all reported changed states through first old-ID return plus causal endpoints, rather than each unchanged frame; this supports continuity without proving tracker internals.",
  "causal_judgments_unchanged": true
}
```

```json
{
  "case_id": "V175",
  "review_phase": "POST_HOC_CHECK",
  "shown_frames": [
    26379,
    26380,
    26460,
    26480,
    26520,
    26640,
    26689,
    26690,
    26691
  ],
  "post_hoc_result": "FRAGMENTED_IDENTITY_SUPPORTED",
  "false_same_established": false,
  "rationale": "The riverpom calls446000 line stays directly between kisunya bets446000 and Dealing Turn/5557 board. It moves row15 to row13 at the causal scroll, remains row13 in all intermediate changed states, then moves row6 at old-ID return with the same neighbor block. The 555 flop, kisunya call88000, ordered preceding folds and following kisunya all-in1600187 remain intact; return adds riverpom call1600187 and the two show blocks below, without replacing the target occurrence.",
  "possible_alternate_explanation": "A duplicated identical betting history remains theoretically possible, but the reviewed states contain no new dealing/hand boundary or contrary neighbor movement and retain the same unusual board and all-in amount.",
  "limitations": "Grayscale compression softens glyph edges. All reported changed states through first old-ID return and causal endpoints were reviewed; unchanged frames were not separately displayed. This is visual support rather than proof of tracker internal cause.",
  "causal_judgments_unchanged": true
}
```

Возвращение old ID само по себе не доказывает false SAME: повторное наследование старого ID может быть сложным fragmentation. Нужны target+neighbor physical context и рассмотрение alternate occurrence; posthoc future не входит в earlier admission.

## EDGE-GATE PROSPECTIVE IMPACT

С visually SAME validated prospective relationships **280** на копиях source rows: **6685 → 6685 UNRESOLVED_EDGE**. Source fields/IDs/epochs/boundaries сохранены, rows8146. Existing gate не потребляет relationship metadata; theoretical removal под действующим gate0. Не придуман альтернативный edge policy. Ничего не применяется.

## IMAGE WRITES

**0**. Native sourceROI, overlay, zoom, contact-sheet encoding толькоRAM/BytesIO; images не записаны вPNG/JPG/WEBP/base64 disk cache. JSON receipts содержат hashes/frames, безimagepayload. Final source/imageinventory safety — `safety_audit.json`; strict production integrity PASS.

## SOURCE DATA CHANGED

NO.

## PRODUCTION CODE CHANGED

NO.

## VERDICT

**TEXT_FIRST_PRECISION_PROVEN_HIGH**.

Высокая observed precision одной frozen выборки не разрешает production merge, не доказывает устойчивость на новых раздачах/videos и не меняет certified hand-history contract. Все admission rules/thresholds frozen; automatic SAME/DIFFERENT/merge/edge admission отсутствуют.

После final review работа останавливается. Незавершённые cases явно остаются pending; completed visual labels доступны для проверки по сохранённому RAM renderer.
