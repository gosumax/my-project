# Smoke17: deterministic Logic Engine audit

## ROOT CAUSE SUMMARY

Смешивание столов исключено: до любых исправлений 199/199 рук имели ровно один исходный table_id/session_id; девять отдельных replay совпали с ALL199/199. NEW повторно проходит тот же контроль. Сравнение использует одинаковый глобально доказанный CALL=ADDED контракт: локальная повторная калибровка требует witnesses минимум с трёх столов и создала бы конфигурационное различие.

FILE → FUNCTION → CALLER → DATA: `ram_first_stage.py:FirstStage.rows` выставляет uncertain scroll origin → production runner сохраняет physical provenance → experimental `assemble_hands` разделяет `(table_id, session_id)` и вызывает `assemble_messages` → `validate_hand` до OLD continue(DEAL) применял blanket identity gate. Отдельно `parse_text` не распознавал explicit/wrapped server notices; OLD validator превращал UNKNOWN_EVENT в MISSING_ACTION. Missing board/hole evidence помечалось CARD_LOGIC_CONFLICT; eligibility выводилась из неполного ledger. Исправлены только эти экспериментальные правила/контракты verdict, источники сохранены.

## TABLE ISOLATION: ALL vs T01–T09

| Table | Hands | ALL Logic PASS | Alone Logic PASS | Mismatch |
|---|---:|---:|---:|---:|
| T01 | 23 | 0 | 0 | 0 |
| T02 | 20 | 0 | 0 | 0 |
| T03 | 20 | 0 | 0 | 0 |
| T04 | 17 | 0 | 0 | 0 |
| T05 | 19 | 0 | 0 | 0 |
| T06 | 19 | 1 | 1 | 0 |
| T07 | 23 | 0 | 0 | 0 |
| T08 | 29 | 1 | 1 | 0 |
| T09 | 29 | 0 | 0 | 0 |

Каждая рука с hand_id/table_id/source_rows/unique_table_ids_inside_hand: [OLD manifest](<D:/Новый парсер/runs/logic_engine_audit_smoke17_20261001/table_isolation/all_hand_table_manifest.json>), [NEW manifest](<D:/Новый парсер/runs/logic_engine_audit_smoke17_20261001/final/all_hand_table_manifest.json>). Все списки unique_table_ids_inside_hand имеют длину1; нарушений0.

## 22 MATH-PASS HANDS

Фиксированный OLD pool:22; complete18, incomplete4. BEFORE0/22, AFTER2/22. По 18 complete: см. строки complete; пул не пересобирался после изменения. Четыре неполных окна нельзя считать acceptance complete hands. Два хвоста имеют net=payout=0.

| Table | Hand / epoch | Complete OLD | Action NEW | Card NEW | Final NEW | Причина / evidence |
|---|---|---|---|---|---|---|
| T04 | UNKNOWN /63 | False | FAIL | PASS | INCOMPLETE_START | Частичный хвост до первого наблюдаемого hand marker, без blinds/payout. Равенство 0=0 не доказывает полноценную руку. |
| T06 | UNKNOWN /99 | False | FAIL | PASS | INCOMPLETE_START | Частичный хвост с неизвестным карточным OCR, без начала/blinds/payout. Math0=0 не свидетельство полной руки. |
| T02 | 2876720584 /26 | True | FAIL | NOT_VERIFIABLE | LOGIC_FAIL | Перед FLOP kirill_petaev: 40 000 против bet160 000, unacted. kiril_petaev sits out не доказывает alias. Пayout на трёх картах с несколькими live; повтор NEW_HAND остаётся ambiguous. |
| T03 | 2876719109 /44 | True | FAIL | NOT_VERIFIABLE | LOGIC_FAIL | olleggi/olleggì остаются разными наблюдаемыми именами; INVALID_CALL 36 000, повтор folds, неполные round responses и нераспознанные карточные строки. |
| T03 | 2876720313 /46 | True | FAIL | PASS | LOGIC_FAIL | Повтор BB без amount, stale forced contribution и повторные fold. Доказательства для удаления финансовых/actor строк нет. |
| T03 | 2876727402 /57 | True | FAIL | PASS | LOGIC_FAIL | Повтор SB без amount, повторные fold, OCR is All-in0/main potO/left table. Подозрение stale не позволяет mask. |
| T04 | 2876718521 /64 | False | FAIL | PASS | INCOMPLETE_START | Нет полной forced-blind последовательности; raise-to30 000 не повышает текущий bet/contribution; повтор fold. |
| T04 | 2876721764 /68 | True | NOT_VERIFIABLE | PASS | UNRESOLVED_HAND | Повтор Starting a new hand внутри окна; uncertain NEW_HAND/DEAL. Boundary meaning не доказан, monetary action sequence отдельно от certification. |
| T04 | 2876727536 /75 | True | FAIL | PASS | LOGIC_FAIL | Повтор explicit header стал warning; повтор badpenguin folds остаётся невозможным в текущем player state. |
| T05 | 2876723277 /88 | False | FAIL | NOT_VERIFIABLE | INCOMPLETE_START | posts Small / Blind40 000 не собраны как SB; повтор ante, неизвестный main pot, два live до payout без board. Полнота действий не доказана. |
| T06 | 2876718617 /100 | True | PASS | PASS | FULLY_VALID_HAND | Wrapped/truncated reconnect и blind-level notices — информационные события; они сохраняются. Betting checks не заменяются предполагаемыми действиями. |
| T06 | 2876722366 /106 | True | FAIL | PASS | LOGIC_FAIL | BB без amount; повтор NEW_HAND; cards fragment. Без восстановления из наблюдений остаётся gap. |
| T06 | 2876722787 /107 | True | NOT_VERIFIABLE | PASS | UNRESOLVED_HAND | Payout81 100 точно равен ceiling81 100 с folded dead money. Повтор NEW_HAND R000277 остаётся AMBIGUOUS boundary; не sidepot bug. |
| T06 | 2876724008 /109 | True | FAIL | NOT_VERIFIABLE | LOGIC_FAIL | Повтор start markers, split joined-table notice, TURN без подтверждённых карт и OCR830J040. Missing cards — unknown, не contradiction. |
| T06 | 2876724664 /110 | True | FAIL | PASS | LOGIC_FAIL | UNCALLED_RETURN vesnushka R000439 без amount и повтор start marker; никакое число не придумывалось. |
| T06 | 2876727749 /114 | True | NOT_VERIFIABLE | PASS | UNRESOLVED_HAND | Blind-level notices перестали быть действиями; повтор NEW_HAND остаётся unresolved boundary. |
| T06 | 2876728636 /116 | True | FAIL | NOT_VERIFIABLE | LOGIC_FAIL | Повтор NEW_HAND, FLOP без карт, raw724029 неизвестен. Карточная неопределённость сохраняется. |
| T08 | 2876718614 /141 | True | PASS | PASS | FULLY_VALID_HAND | Единственный OLD blocker: unresolved origin на DEAL R000018. Эта строка не меняет state/player/amount/cards/boundary; сохраняется warning. |
| T08 | 2876721851 /150 | True | FAIL | PASS | LOGIC_FAIL | Повтор truncated SB без amount; stale SB; return sezonnoe без amount. Подтверждённого duplicate mask нет. |
| T08 | 2876724417 /157 | True | FAIL | PASS | LOGIC_FAIL | Повтор SB etobilodavno без amount. Отсутствие численного evidence не устранено. |
| T08 | 2876725694 /160 | True | FAIL | PASS | LOGIC_FAIL | raise-to12 000 не выше current bet; rimi13 имеет12 000, но unacted перед FLOP; split joined-table/cards fragments. |
| T09 | 2876726613 /191 | True | FAIL | PASS | LOGIC_FAIL | raise-to160 000 не выше bet; kisunya777 unacted перед FLOP; повтор folds riverpomojet и неизвестный the main pot. |

Полная матрица, отсортированная по first failure: [JSON](<D:/Новый парсер/runs/logic_engine_audit_smoke17_20261001/final/math_pass_22_matrix.json>). Включает start/end, all failures, affected source IDs, timestamps и весь raw text. [Классификация каждого OLD failure](<D:/Новый парсер/runs/logic_engine_audit_smoke17_20261001/final/per_failure_classification.json>) содержит A/B/C, исходный issue, raw/timestamp и фактическое состояние BEFORE/AFTER.

Трассы actual validator: [OLD654 message steps](<D:/Новый парсер/runs/logic_engine_audit_smoke17_20261001/old_traces/traces.jsonl>), [NEW](<D:/Новый парсер/runs/logic_engine_audit_smoke17_20261001/final/new_traces.jsonl>); индивидуальные JSON/MD OLD и JSON NEW лежат рядом. Покрыты также continue branches, включая NEW_HAND/DEAL/UNKNOWN/TABLE_INFO. Это sys.settrace реального engine, не реконструкция альтернативным валидатором.

## FALSE-POSITIVE RULES FOUND / LOGIC ENGINE BUGS

- ROW provenance больше не равен betting FAIL. Unresolved action/money/card/boundary и строки financial inference evidence остаются CRITICAL certification blockers. Только harmless informational origin становится NONCRITICAL warning.
- Explicit reconnect/limits notices, включая подтверждённые multiline continuation, становятся TABLE_INFO/BLIND_LEVEL. Неизвестные остальные строки остаются MISSING_ACTION. Notice не означает check/fold/call; round closure проверяется как прежде.
- Отсутствующие карты дают NOT_VERIFIABLE, наблюдаемое противоречие — CONFLICT. Карты/sidepot/provenance не подменяют отдельный action verdict. Cards PASS означает consistency доступных данных, не полноту hand evidence.
- Повтор того же explicit hand ID сохраняется как redundant metadata warning. Повтор ненумерованного NEW_HAND не признан stale: AMBIGUOUS completeness blocker.
- Eligibility на неполном ledger: NOT_VERIFIABLE, вместо доказанного sidepot conflict. Денежная консервация проверяется независимо. Старый ceiling не изменён: среди22 нет доказанного counterexample.

## TRUE POKER CONFLICTS FOUND / UPSTREAM DIRTY-TIMELINE CONFLICTS

TRUE_POKER_CONFLICT означает несовместимость наблюдаемой parsed sequence с записанным BEFORE state, а не доказательство незаконной реальной игры. Примеры: действие после recorded fold, call36 000 при другом required delta, raise-to не выше bet. При dirty source эти нарушения могут быть upstream artefacts; ничьи имена не объединялись.

UPSTREAM: missing forced blinds/amounts/actions/cards, incomplete capture tails, неоднозначные repeated NEW_HAND, repeated actor/financial observations. T02#2876720584 сохраняет unacted kirill_petaev40k против160k; похожее имя kiril_petaev в sits-out не является identity evidence. Math0 не доказывает полноту последовательности.

RAW vs CLEANED: [все22 сравнения](<D:/Новый парсер/runs/logic_engine_audit_smoke17_20261001/final/raw_vs_cleaned_candidates.json>). Временная маска разрешена только для повторного того же explicit hand identifier, не влияющего на деньги. Финансовые/NEW_HAND подозрения остаются AMBIGUOUS. Candidate masks adopted=0; observed rows никогда не удалялись.

## OLD failure codes in fixed22 (hands containing code)

| Code | Hands |
|---|---:|
| ROW_IDENTITY_UNRESOLVED | 20 |
| CHAT_OVERLAP_DUPLICATE | 15 |
| MISSING_ACTION | 12 |
| ACTION_AFTER_FOLD | 6 |
| MISSING_AMOUNT | 6 |
| CARD_LOGIC_CONFLICT | 5 |
| BETTING_ROUND_INCOMPLETE | 4 |
| INCOMPLETE_START | 4 |
| STALE_BLIND_OR_ANTE | 4 |
| INVALID_RAISE | 3 |
| OLD_WIN | 3 |
| HAND_BOUNDARY_CONFLICT | 2 |
| INCOMPLETE_END | 2 |
| MISSING_UNCALLED_RETURN | 2 |
| INVALID_CALL | 1 |

Прочие перечисленные в задании codes:0 в этом пуле. Частоты не считаются автоматически root causes; классификация дана отдельно.

## SMOKE17 BEFORE / AFTER

| Metric | OLD | NEW |
|---|---:|---:|
| candidate_windows | 199 | 199 |
| complete | 160 | 160 |
| logic_pass | 0 | 2 |
| math_pass | 22 | 22 |
| card_pass | 81 | 88 |
| fully_valid | 0 | 2 |

| Exclusive final status | OLD | NEW |
|---|---:|---:|
| CARD_CONFLICT | 101 | 10 |
| FULLY_VALID_HAND | 0 | 2 |
| INCOMPLETE_END | 7 | 7 |
| INCOMPLETE_START | 32 | 32 |
| LOGIC_FAIL | 59 | 141 |
| MATH_FAIL | 0 | 4 |
| UNRESOLVED_HAND | 0 | 3 |

Статусы были разделены; рост LOGIC_FAIL среди exclusive statuses связан также с переводом отсутствующих карт из CARD_CONFLICT в NOT_VERIFIABLE. Это не ухудшение action sequence. Подробнее: [summary](<D:/Новый парсер/runs/logic_engine_audit_smoke17_20261001/final/summary.json>). Experiment FULLY_VALID_HAND не является production HH certification или independent visual acceptance.

## TOP REMAINING FAILURE REASONS

| Code | Hands NEW |
|---|---:|---:|
| ROW_IDENTITY_UNRESOLVED | 197 |
| MATH_DELTA | 177 |
| MISSING_ACTION | 159 |
| SIDE_POT_NOT_VERIFIABLE | 151 |
| CHAT_OVERLAP_DUPLICATE | 135 |
| UNCALLED_RETURN_NOT_VERIFIABLE | 132 |
| CARD_EVIDENCE_MISSING | 111 |
| BETTING_ROUND_INCOMPLETE | 109 |
| OLD_WIN | 99 |
| STALE_BLIND_OR_ANTE | 89 |
| MISSING_AMOUNT | 84 |
| INVALID_CALL | 56 |

## FILES CHANGED / TESTS ADDED

`parser_core/poker_hand_assembly.py`, `parser_core/poker_logic_engine.py` — experimental only. Диагностика: `experiments/audit_poker_logic_engine.py`, `experiments/replay_logic_audit.py`, `experiments/build_logic_audit_report.py`. Новые регрессии: `tests/test_poker_logic_layered_verdicts.py` (11 tests): harmless DEAL origin, critical monetary origin, missing vs contradictory cards, wrong winner retained, incomplete-ledger eligibility, same explicit header, ambiguous NEW_HAND, anchored/multiline server notices, missing response not hidden, inference financial dependency, wrapped/truncated pause preservation.

Исходные59 tests и исходные math/card/Jev/replay файлы сохранены по hash. Все199 денежных ledger и player states совпали OLD/NEW. Production76/76 hashes совпали. Source8146/8146 fields/order сохранены; source SHA2b7822db0fa4c14706a8ef8e576b5942e22275cd28c29b62c5174720ff566a5b неизменен. Перестройка информационных сообщений не меняет source order.

## PERFORMANCE BEFORE/AFTER

Warmup1 исключён;20×199=3980 hands на arm, tracing выключен. Четыре arms разделяют изменение validator и информационной assembly.

| Same input | Engine | Logic mean ms | Logic p95 ms | Math mean ms | Math p95 ms | Wall s |
|---|---|---:|---:|---:|---:|---:|
| frozen199_OLD | OLD | 0.3151 | 0.8350 | 0.0618 | 0.0850 | 1.6752 |
| frozen199_NEW | NEW | 0.3966 | 0.9773 | 0.0725 | 0.1172 | 2.1350 |
| new199_OLD | OLD | 0.3215 | 0.8542 | 0.0625 | 0.0904 | 1.7054 |
| new199_NEW | NEW | 0.3344 | 0.8426 | 0.0620 | 0.0856 | 1.8014 |

NEW/OLD wall: frozen199 ×1.2745, NEW199 ×1.0563. Многократной деградации нет; это ограниченный sequential microbenchmark, не полноценное latency SLA. Frozen NEW arm имеет отдельный max12.48ms; wall варьирует между arms. Видео FPS/RTF и Jev latency здесь NOT_MEASURED. [Benchmark+hashes](<D:/Новый парсер/runs/logic_engine_audit_smoke17_20261001/final_benchmark/summary.json>).

## FULL PYTEST RESULT

`python -m pytest tests/test_poker_logic_math_jev.py tests/test_poker_logic_layered_verdicts.py -q`: exit0,70 passed0.48s. `python -m pytest -q`: exit0,495 passed +7 subtests59.10s, wall59.864s. Strict production integrity: exit0 PASS. [Test receipts](<D:/Новый парсер/runs/logic_engine_audit_smoke17_20261001/final_tests/latest_full_summary.json>).

## PRODUCTION CODE CHANGED

NO. Jev/TLS/API не изменялись и не запускались. Видео не декодировалось. Promotion отсутствует.

## VERDICT

LOGIC_ENGINE_PARTIALLY_FIXED

Доказанные ложные verdict исправлены; оставшиеся отказы сохранены с фактическими source/state причинами. Неполные/повторные/неоднозначные данные остаются blockers; никакая строка не удалена ради PASS.
