# Официальный TypeSafe Jev: подключение и проверка 2026-10-01

Подключение работает. Проверены реальные API-запросы, stdio MCP и весь подходящий пул FIXED Smoke17.

## Подключение

- Официальный API: `https://api.typesafe.ai/v1/systemone`, Bearer authentication.
- Запрошен `jev-latest`; все 180 poker-ответов содержат фактическую модель `jev-1.13.0`.
- Ключ хранится в Windows User environment `TYPESAFE_API_KEY`. Код, Codex config, Excel и отчёты не содержат значение ключа.
- Старый `jevai.org` — другой сервис. Его прежние ошибки доступа не характеризуют официальный TypeSafe API. Новый ключ этому сервису не передаётся.
- Codex server `jev` заменён на локальный stdio-адаптер `scripts/jev_typesafe_mcp.py`. Он вызывает официальный API и предоставляет инструмент `jev_decide`, поддерживающий Choice, Noul, Score.
- MCP-адаптер создан в проекте. Это не заявление о наличии официального TypeSafe MCP endpoint.
- Реальный отдельный процесс прошёл `initialize`, notification, `tools/list`, `tools/call`. Один запрос с тремя типами вопросов вернул CALL, Noul 0.94 и Score 2.0.
- В текущем чате каталог native tools ещё не перезагружался. Для появления инструмента в Codex нужно перезапустить MCP server в Settings либо приложение. Прямой replay уже использовал работающий API.

Официальные документы: [TypeSafe API](https://docs.typesafe.ai/api), [Codex MCP](https://developers.openai.com/codex/mcp).

## FIXED Smoke17

Источник: `runs/temporal_overlap_boundary_smoke17_20261001/ordered_timeline.jsonl`.
SHA256: `9f67b9b470bfbf333341e8c7155baeea5bd7a4a04fbe3c116f3fb4c5257ad77a`.

7954 физических строк; 196 сегментов: 189 numbered, 7 unnumbered. Start и end наблюдаются у 182 сегментов. Из них 2 уже FULLY_VALID, остальные 180 отправлены Jev. Каждый сегмент содержит строки одного стола; все T01–T09 isolated replay совпали с общим replay.

| Проверка | Результат |
|---|---:|
| Реальные poker API calls | 180 |
| Успешные structured responses | 180 |
| API failures | 0 |
| Старые 32 KiB preflight rejections | 0 |
| UNRESOLVED | 138 |
| LIKELY_STALE_DUPLICATE | 41 |
| CARD_EVIDENCE_CONFLICT | 1 |
| Принятые candidate repairs | 0 |
| Средняя latency API call, включая Windows transport | 1755.943 ms |
| Время полного replay с изоляцией столов и API | 330.287 s |
| Input / output tokens, только 180 poker calls | 3736695 / 8541 |

41 duplicate-гипотеза отклонена: отсутствует требуемое сочетание совпадающего pixel fingerprint и evidence rows. Повторённый текст не доказывает stale duplicate. Это rejection criterion, а не доказанная первопричина ошибки руки. Исходные строки сохранены, delta не подгонялся.

Текущий resolver задаёт закрытый Choice между заранее построенными source-linked кандидатами: повторённое сообщение или blind amount из явного observed blind level. Он не создаёт произвольные новые действия, identity/boundary repairs или полный HH. Поэтому успешное подключение не означает исправление всех исходных ошибок.

Итоговая диагностика осталась прежней: `FULLY_VALID=2`, `LOGIC_FAIL=182`, `MISSING_AMOUNT=11`, `MATH_FAIL=1`. Текущая первопричина остальных проблемных рук остаётся `UNKNOWN` без source proof.

Во время live replay отдельный процесс использовал прежние reporting labels: явный UNRESOLVED попал под `INVALID_TARGET_ROWS`, а transport подписывался `REST_NATIVE_DECISIONS`. Эти labels исправлены в текущем коде. Итоговая Excel-view пересчитала только labels offline, сохранив старые live labels в отдельных колонках. Нативные ответы и оригинальный live replay не перезаписаны. Новых API calls при финализации не было.

## Проверки кода и сохранности

- Целевые четыре файла: 128 pytest PASS.
- Единственный полный прогон: 550 pytest PASS + 7 subtests; последние дополнительные регрессии проверены целевым прогоном.
- Независимый audit выявил redirect forwarding и неверный provider label; оба исправлены.
- Verified TLS, без отключения certificate checks. Redirects запрещены в Windows и urllib transports.
- Production integrity PASS, все 76 зарегистрированных хэшей сохранены.
- Исходный FIXED hash сохранён. Проверка выходных данных на значение ключа не обнаружила совпадений.

## Воспроизведение

```powershell
python experiments/replay_first_break_fixed.py --output runs/logic_math_fixed_smoke17_20261001/typesafe_live_fresh --jev-live --jev-typesafe
```

Нужна новая output directory. Этот saved-row replay экспериментальный и не заменяет `run_production.py`.

Реальные результаты: `runs/logic_math_fixed_smoke17_20261001/typesafe_live_20261001/` (`summary.json`, `hands.jsonl`, `jev_cache/`, `connection_report.json`). MCP probe: `runs/logic_math_fixed_smoke17_20261001/typesafe_mcp_probe_20261001.json`.

Excel: `outputs/logic_math_fixed_smoke17/logic_math_audit_fixed_smoke17.xlsx`. Все девять вкладок сохранены; JEV_CASES содержит 180 реальных запросов, фактическую модель, confidence, evidence, token usage, latency и причину непринятия.
