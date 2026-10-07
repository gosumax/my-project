# TEXT-FIRST identity linking — offline simulation

## TEXT-FIRST USEFULNESS

CURRENT FROZEN LINKER: SAME0 / DIFFERENT0 / AMBIGUOUS590. TEXT-FIRST SIMULATION: **SAME candidates280 / DIFFERENT candidates0 / AMBIGUOUS310**. Prospective candidate не означает SAME_PHYSICAL_TRACK_PROVEN.

All590 text+geometry mapping: `{'UNIQUE_TEXT_GEOMETRY_MAPPING': 566, 'NO_TEXT_GEOMETRY_MAPPING': 24}`. Из491 previously unmapped: `{'UNIQUE_TEXT_GEOMETRY_MAPPING': 467, 'NO_TEXT_GEOMETRY_MAPPING': 24}`.

Text — только candidate key. Unique mapping требует same table/session, proven viewport alignment и ожидаемой y≤7px; одинаковые normalized strings сами по себе не объединяются. Дополнительная admission требует unique known text в обоих viewport и предыдущих12 changed states, immediate gap1 либо gap2 с доказанным local loss bridge,≥2 same-ID coherent neighbors, no simultaneous visibility/boundary и supporting content.

Supporting content: target foreground MAE и support XOR fraction не превышают observed maxima как минимум двух same-ID coherent neighbor continuations на тех же causal endpoint. Missing foreground запрещён. Это локальное экспериментальное сравнение с control rows, без нового фиксированного production threshold; максимум контроля может быть шумным, поэтому это не physical SAME proof. Registered distance/pixel similarity, dx/dy и border/content differences остаются supporting measurements. Exact native raster не является veto.

Duplicated viewport/time context: `{'TEXT_UNIQUE': 373, 'TEXT_DUPLICATED_IN_TIME_WINDOW': 216, 'TEXT_DUPLICATED_IN_VIEWPORT': 1}`; prospective false-merge risks blocked **217**. Repeat categories: `{'NICK_ACTION': 128, 'ANTE': 86, 'SB_BB': 18, 'FOLD': 51, 'SHORT_TEXT': 11, 'AMOUNT_ONLY': 2}`. Для ANTE/SB/BB/fold/check/amount/nick+action/short строк mandatory continuity/content evidence остаётся обязательным; если это реальные повторные occurrences в context, ветка остаётся AMBIGUOUS.

POST_HOC old reappearance среди candidates: 2; separately flagged, не используется для causal admission. Нет независимого визуального подтверждения новых physical links; безопасно сертифицированных SAME по этому эксперименту **0**.

Для offline acceptance **2** future-risk candidates withheld; **278** remaining prospective candidates требуют precision validation. Causal counts280/310 сохраняются. У всех280 candidates observed content envelope PASS; повторяющиеся templates допустимы только с дополнительной geometry/motion/content поддержкой и без повторного occurrence в causal context.

Edge-gate simulation **6685 → 6685**, decrease **0**. Source data NO, production changes NO, stable IDs не merged, image writes0.

Ответы на вопросы:

1. Получено 280 prospective SAME candidates из590; certified physical SAME0.
2. Из491 target text+geometry mapping: 467 unique; это отдельный результат до admission.
3. Заблокировано 217 потенциально опасных duplicate/multiple proposals; это risk flags, а не подтверждённые ground-truth false merges.
4. UNRESOLVED_EDGE не уменьшился: existing gate не потребляет эту metadata.
5. Text-first полезен как primary candidate mapper. Делать его authority physical identity и pixels только secondary в production по этим данным пока нельзя: нужна независимая precision validation.

## VERDICT

**TEXT_FIRST_MAPPING_PARTIAL**. Ничего автоматически не внедрено.

Metadata simulation **8.276s**, **71.294 pairs/s**; RSS begin/end **21794816/324481024B**, peak NOT_MEASURED. Includes metadata loading/integrity/hash/edge simulation; content preparation time отдельно в content report. Targeted17tests и independent causal/conditions checks PASS.

Полные per-pair conditions и false-merge check: `runs/text_first_identity_smoke17_20261001/relationships.jsonl`; content geometry/metrics: `runs/content_identity_smoke17_20261001/pairs.jsonl`.

## FALSE MERGE CHECK BY TEMPLATE

| Category | all pairs | candidates with added evidence | blocked |
|---|---:|---:|---:|
| ANTE | 86 | 58 | 28 |
| SB_BB | 18 | 13 | 5 |
| FOLD | 51 | 15 | 36 |
| CHECK | 0 | 0 | 0 |
| NICK_ACTION | 128 | 55 | 73 |
| SHORT_TEXT | 11 | 0 | 11 |
| AMOUNT_ONLY | 2 | 0 | 2 |

Categories overlap. CHECK=0 означает отсутствие реального примера в этих590, а не доказательство качества на CHECK. Общие одинаковые суммы при разном полном тексте не формируют candidate key; AMOUNT_ONLY — отдельная опасная категория. Случаи другой руки блокируются separating boundary / time continuity / повторным occurrence в bounded causal context; без непрерывности одинаковый текст не принимается.

Offline review artifact `offline_review.jsonl` сохраняет 280 causal verdicts и отдельно отмечает2 withheld risks. Производственный admission отсутствует.