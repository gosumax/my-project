# FIXED Smoke17: FIRST_LOGIC_BREAK + Math + Jev MCP

Дата: 2026-10-01. Только experimental слой. Production не менялся.

## FILES CHANGED

- parser_core/poker_logic_engine.py — последовательная остановка на первом action/completeness break, полный rollback состояния; default legacy API сохранён.
- parser_core/poker_diagnostics.py — DIAGNOSTIC verdict, trace, независимый observed Math, отдельные Cards, строгая повторная проверка authorized candidate.
- parser_core/poker_math_engine.py — полный observed ledger и финансовые snapshots каждой строки; суммы не придумываются.
- parser_core/poker_hand_assembly.py — только доказанные общие grammar/continuation patterns; исходные строки не редактируются.
- parser_core/jev_poker_resolver.py — MCP jev_decide, lossless ID dictionary, diagnostic packet, latency ошибок, безопасные multiline bindings.
- experiments/replay_first_break_fixed.py; experiments/finalize_first_break_review.py — replay, table isolation, source evidence и данные девяти листов.
- tests/test_poker_first_break.py — 43 regression cases.
- outputs/logic_math_fixed_smoke17/build.mjs — воспроизводимый artifact-tool XLSX builder.

## FIXED INPUT

`D:\Новый парсер\runs\temporal_overlap_boundary_smoke17_20261001\ordered_timeline.jsonl`
SHA256: `9f67b9b470bfbf333341e8c7155baeea5bd7a4a04fbe3c116f3fb4c5257ad77a`. Physical rows: 7954. BROKEN 7640 не использовался.
Зафиксированный source snapshot/boundary inventory: runs/logic_math_fixed_smoke17_20261001/baseline/.
Во всех девяти столах source row IDs и порядок сохранены. 7954/7954 source rows представлены в assembled messages; timeline SHA не изменился.

## SEGMENT COUNTS

196 = 189 numbered markers + 7 leading visible prefixes. All 196 hands: unique_table_ids_inside_hand = 1.
189 numbered starts; после исправления grammar — 182 observed start/end pairs. Это структурное наличие, а не доказательство legal completion.
Frozen assembler имел 181 pair и один внутренний missing-end T07 #2876726168. Source R000609/R000610 содержит `_rinrus_wins 16 900 from the main pot`: исправление разделителя nickname/verb распознало уже существующий payout. False boundary-conflict исчез без переноса строк между руками.
Остались 7 последних numbered segments без end: T01 #2876729234, T02 #2876729386, T03 #2876729418, T04 #2876729188, T05 #2876728919, T06 #2876729176, T09 #2876729498.
Для каждого segment boundary evidence находится в HAND_SUMMARY: source_rows, boundary_start, boundary_next, boundary_end, boundary_conflicts_current. Полный frozen inventory — baseline/boundary_inventory.json.

### Семь unnumbered

Все начинаются при первом видимом наблюдении стола, до первого numbered marker. Unnumbered interior/trailing segments: 0. Классификация: LEADING_VISIBLE_PREFIX; partial first context поддержан порядком source. Нельзя доказать по одному prefix, что это ровно одна полная poker-hand или исключить скрытые boundaries; bleed/wrong assembly не объявлены proven. PROVEN_CAUSE = UNKNOWN.

| Table | First/last source row | Frames | Next numbered hand |
|---|---|---:|---|
| T01 | R000001 → R000034 | 0 → 181 | 2876718828 |
| T02 | R000001 → R000046 | 0 → 490 | 2876718950 |
| T03 | R000001 → R000039 | 0 → 864 | 2876719109 |
| T04 | R000001 → R000004 | 0 → 0 | 2876718521 |
| T05 | R000001 → R000045 | 0 → 443 | 2876718933 |
| T06 | R000001 → R000001 | 0 → 0 | 2876718617 |
| T09 | R000001 → R000043 | 0 → 671 | 2876719058 |

## FIRST LOGIC BREAK RESULTS

Logic PASS = 2; FAIL = 193; NOT_VERIFIABLE = 1. FIRST_BREAK records = 194: 193 observed-message breaks + 1 virtual end-of-segment check.
Нельзя приписать виртуальный INCOMPLETE_END последней допустимой строке: source_row_ids пусты, observed = END_OF_SEGMENT. После первого observed break trace сохраняет AFTER_BREAK / NOT_EVALUATED_AFTER_FIRST_BREAK; state_after первого break равен последнему доверенному state_before.
Публичный слой выдаёт DIAGNOSTIC, не запрещает обработку/сохранение observed rows и не экспортирует HH. Независимый full observed Math продолжает считать все строки. Legacy validate_hand(default) сохраняет старый интерфейс для совместимости и resolver proof-gates; его action cascade не выдаётся как новые первичные диагнозы.

## RULE TRIGGERS

| RULE_TRIGGER | First-break hands |
|---|---:|
| MISSING_BLINDS_BEFORE_ACTION | 5 |
| CHAT_OVERLAP_DUPLICATE | 79 |
| MISSING_ACTION | 43 |
| INVALID_RAISE | 6 |
| MISSING_AMOUNT | 11 |
| OLD_WIN | 41 |
| BETTING_ROUND_INCOMPLETE | 2 |
| INVALID_CALL | 1 |
| ACTION_AFTER_FOLD | 4 |
| STALE_BLIND_OR_ANTE | 1 |
| INCOMPLETE_END | 1 |

## PROVEN ROOT CAUSES

Для текущего FIRST_BREAK каждой из 194 проблемных рук причина остаётся UNKNOWN. Это не означает, что source-проверка доказала реальную потерю действия или суммы.
Отдельно доказаны и исправлены 183 source-message groups в 62 руках: MULTILINE_NOT_ASSEMBLED = 55, PARSER_MISCLASSIFICATION = 128. 238 old-message comparison records относятся к 183 уникальным новым source groups; они не являются 238 независимыми причинами.
Примеры доказанных patterns: numeric + and is All-in; split and is / All-in; nickname underscore перед known verb; finished the / tournament и joined/left the / table; bare Limits prefix + numeric level; Dealing cards●; денежная сумма перед terminal ●.
FILE → FUNCTION → CALLER → DATA: poker_hand_assembly.parse_text/assemble_messages → assemble_hands → replay_first_break_fixed.run → сохранённые FIXED physical rows. Соседние continuation rows объединяются только в одном table/session/epoch.
Source corrections хранят raw rows, old parsed fields, new parsed fields, source IDs, FIRST_BREAK before/after. Evidence: final_diagnostics_v2/source_repair_evidence.json и HAND_SUMMARY.source_proven_repairs_in_hand. Full-hand cause не объявляется proven только потому, что parser pattern исправлен или delta поменялась.
Оборванные posts Small без Blind остаются UNKNOWN. Repeated DEAL/ANTE/NEW_HAND — rule trigger; одинаковый текст сам по себе не доказывает stale row.
Классификация оставшихся проблемных рук A–G: A=0, B=0, C=0, D=0, E=0, F=0, G=194. Это counts доказанных причин текущих нерешённых FIRST_BREAK, а не отрицание исправленных parser/assembly patterns. Подтверждённых REAL_SOURCE_ACTION_MISSING/REAL_SOURCE_AMOUNT_MISSING нет.

## MATH RESULTS

Math PASS = 15; FAIL = 172; NOT_VERIFIABLE = 9. Numeric delta=0 = 24, но 9 из этих ledger не имеют достаточной field certainty. FULLY_VALID = 2.
Полные player/street contributions, returns, payouts, missing fields и first suspicious action сохранены. Suspicious action не объявлена точкой возникновения delta без source proof. Prefix net-minus-paid — snapshot, не conservation failure: в prefix_math отсутствует ложный PASS/FAIL, conservation_test_applicable=False.
Observed ledger совпадает с существующим Math на тех же текущих assembled observations: 196/196, net_total/payout_total/delta mismatches = 0.
Exact delta distribution целиком сохранена в summary.json; каждая сумма delta — в HAND_SUMMARY и MATH_FAILS. Ненулевые строки не удалены ради delta=0.

## MISSING AMOUNT RESULTS

107 observed money actions без суммы. В предыдущем MCP diagnostic snapshot было 156: 40 source amounts восстановлены из известных terminal-glyph patterns и ещё 9 из полной split all-in suffix. Эти цифры относятся к parser fields, не к выдуманным финансовым фактам.
Отдельный MISSING_AMOUNTS содержит все 107 строк, player/street/source IDs; для недоказанной суммы expected_amount_source=NONE, candidate_amount пустая, PROVEN_CAUSE=UNKNOWN. MATH_FAILS содержит только 172 FAIL, а 9 NOT_VERIFIABLE остаются явно видимы в summary/table stats/missing-amount view.

## JEV RESULTS

MCP URL: https://www.jevai.org/api/mcp. initialize PASS: jevai 0.1.0, protocol 2025-03-26; tools/list PASS: 6 tools, включая jev_decide. Tools schemas сохранены в jev_mcp_tools.json.
Bearer credential хранится в JEV_API_KEY пользовательского окружения, не в коде/workbook/отчёте. Существующий mcp_servers.jev в Codex config использует bearer_token_env_var.
Фактические model calls: 3 MCP calls + 1 service-default probe; successful responses=0, accepted hypotheses=0. Все вернули “The request credentials or model access were rejected.” При default model ответ тот же. Handshake/tools discovery не доказывают доступ к модели.
Диагностические пакеты содержат first break, state, observed ledger, missing fields, physical ±10 context и boundary evidence. Lossless dictionary уменьшает повторение длинных source IDs; исходная рука не обрезается для лимита сервиса. В предыдущем MCP snapshot 28 запросов отклонены локально по 32KiB; это НЕ model calls.
JEV_CASES показывает 4 реальные попытки с latency, snapshot и ошибкой. После последних source-only parser fixes новых model calls не было, поскольку доступ к модели уже отклонён; прежние ответы не выданы за проверку нового packet.
Evidence gates сейчас поддерживают safe stale duplicate, observed blind amount с raw level proof и multiline. Остальные разрешённые пользователем классы остаются UNSUPPORTED_OR_UNPROVEN_HYPOTHESIS до отдельного доказательства; INVENT_NEW_ACTION запрещён. Любой candidate должен точно совпасть с authorized transformation и сохранить original source rows, затем проверяется с самого начала через Logic + Math + Cards.

## T01–T09 RESULTS

| Table | Segments | Numbered | Prefix | Complete start/end | Logic PASS | Logic FAIL | Math PASS | Math FAIL | Fully valid | Single-table parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| T01 | 22 | 21 | 1 | 20 | 0 | 22 | 0 | 22 | 0 | PASS |
| T02 | 18 | 17 | 1 | 16 | 0 | 18 | 1 | 17 | 0 | PASS |
| T03 | 20 | 19 | 1 | 18 | 0 | 20 | 1 | 17 | 0 | PASS |
| T04 | 17 | 16 | 1 | 15 | 0 | 17 | 4 | 12 | 0 | PASS |
| T05 | 19 | 18 | 1 | 17 | 0 | 19 | 1 | 18 | 0 | PASS |
| T06 | 19 | 18 | 1 | 17 | 0 | 19 | 6 | 11 | 0 | PASS |
| T07 | 23 | 23 | 0 | 23 | 0 | 23 | 0 | 23 | 0 | PASS |
| T08 | 29 | 29 | 0 | 29 | 2 | 27 | 2 | 24 | 2 | PASS |
| T09 | 29 | 28 | 1 | 27 | 0 | 28 | 0 | 28 | 0 | PASS |

Во всех single-table replay переиспользован один frozen cross-table CALL=ADDED contract (5 source witnesses / 4 tables). На одном столе не переобучался threshold “3 tables”, иначе это было бы сравнением разных контрактов. Все диагностические результаты 196/196 совпали.

## EXCEL PATH

D:\Новый парсер\outputs\logic_math_fixed_smoke17\logic_math_audit_fixed_smoke17.xlsx
Все 9 обязательных листов; header freeze, filters, числовые суммы, raw/state wrap, красный FIRST_BREAK, без merged cells и изображений. hand_key связывает строки между листами. Исходные state/source JSON доступны в ячейках и сохранённом hands.jsonl; длинные строки можно раскрыть в Excel.

## TESTS ADDED / TARGETED / FULL PYTEST

43 cases в test_poker_first_break.py покрывают missing blinds, observed missing amount, blind/call/all-in/return continuations, fold/all-in/check/call/raise rules, stale/repeated events, old payout, boundary/identity, partial first/last, first break + rollback/no cascade, full candidate revalidation/no hallucination, table isolation, MCP request/error contract.
Targeted: 113 passed, 0 failed, 0.70s. Full pytest: 538 passed + 7 subtests passed, 0 failed, 60.05s. Production video/capture не запускались.
Independent read-only reviewer нашёл и подтвердил исправления arbitrary candidate edits/masks, post-break card cascade, missed repeated DEAL, cross-epoch multiline, rejected legitimate cross-row blind evidence. Финальный audit: новых существенных findings нет.

## PERFORMANCE

Saved-row replay + all-table isolation: 12.381s. First-break layer mean 5.536ms/hand; max 40.005ms. Первое измерение до parser fixes: mean 5.042ms. Prefix lengths изменились, поэтому это наблюдение, не изолированный speedup claim.
Полный observed Math/Card проход и source/report generation входят в replay wall; XLSX render/export и network probes измеряются отдельно и не выдаются за скорость Logic.

## PRODUCTION INTEGRITY

76/76 зарегистрированных SHA256 совпадают с production_manifest.json. PRODUCTION.md, manifest, raw FIXED timeline не изменялись. Default official pipeline/run_production.py не затронут.
PRODUCTION CODE CHANGED = NO.

## VERDICT

PARTIAL_REPAIR_ONLY. FIRST_LOGIC_BREAK diagnostics воспроизведены, общие source grammar bugs исправлены, Excel построен. Все нерешённые hand causes сохранены UNKNOWN. Jev MCP подключён на уровне сервера, model access пока отклонён. HH не публиковался.

## REPRODUCE

```powershell
python experiments/replay_first_break_fixed.py --output runs/fresh_first_break_fixed
python -m pytest tests/test_poker_first_break.py tests/test_poker_logic_math_jev.py tests/test_poker_logic_layered_verdicts.py -q
python -m pytest -q
```
Для новых model calls после восстановления доступа: тот же replay с --jev-live --jev-mcp и новым output. Не следует повторять failed calls, пока доступ сервиса не изменился.
