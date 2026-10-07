# FILES CHANGED

- `experiments/capture_temporal_evidence.py` — новый lossless evidence-only replay.
- `experiments/analyze_temporal_evidence.py` — новый аналитический слой.
- `tests/test_temporal_evidence_analysis.py` — 20 новых regression tests; вместе с существующими targeted tests: 29 passed.
- `outputs/temporal_context_smoke17_20261001/build.mjs` — новый builder отдельной review-книги.
- Этот отчёт; новые артефакты в `runs/temporal_context_smoke17_20261001/` и `outputs/temporal_context_smoke17_20261001/`.

Предшествующие dirty/untracked изменения сохранены. Frozen/vendor, manifest, production source, resolver, hand reconstruction и mathematics не изменены. Промежуточные результаты анализа сохранены в `analysis_v1/` и `analysis_v2/`; они не являются итоговым результатом.

# EXISTING TEMPORAL FEATURES AUDIT

Статусы относятся к исходному production до этой задачи. Единственный production: PRODUCTION.md + production_manifest.json, запуск run_production.py. README содержит исторические пути, не заменяющие manifest.

| feature | exists-partial-missing | path |
|---|---|---|
| source_frame | EXISTS | `parser_core/ram_first_stage.py:35` RamRow.frame_id; decode enumerate в FirstStage.rows; `scripts/run_smoke14_ram.py:367`, timeline.frame_id; XLSX.frame_id |
| timestamp_ms | PARTIAL | Эквивалент video_time в секундах: `ram_first_stage.py:105`; timeline.video_time и XLSX.video_time. Numeric milliseconds отсутствовали |
| real video PTS | PARTIAL | RamRow.source_pts/time_base_num/time_base_den; `scripts/run_smoke14_ram.py:330,341` physical/OCR JSONL; timeline/XLSX сохраняли только float video_time |
| table_id | EXISTS | RamRow.table_id; timeline/XLSX.table_id, `scripts/run_smoke14_ram.py:367` |
| session_id | EXISTS | `scripts/run_smoke14_ram.py:69` source_identity: source hash + layout/config/slot; RamRow.session_id, physical.table_session_id, timeline.session_id |
| roi_id | EXISTS | timeline.source_roi_change_id=`session:frame:chat`; physical.source_span_index / timeline.row_index; spatial y1 в physical JSONL |
| crop_id | EXISTS | `parser_core/parallel_ocr.py:13` row_key=`table:frame:span`, scoped by source/session; physical.row_key и source hash/session. Отдельный crop_id не нужен |
| row_id | PARTIAL | physical.row_id / timeline.stable_row_id; XLSX не экспортировал stable_row_id |
| track_id | PARTIAL | Функциональный эквивалент stable_row_id: `ram_first_stage.py:166` перенос immediate identity; history contiguous blocks; `:234` identity_observer. Production подавляет inherited observations, lifespan не экспортируется |
| first_seen_ms | PARTIAL | Timestamp emitted row представляет первое видимое наблюдение stable_row_id; явный numeric first_seen и censoring отсутствовали |
| last_seen_ms | MISSING | Production не сериализует lifespan; observer вызывается только в changed viewport |
| observation_count | MISSING | Только aggregate metrics, не per-track count |
| pixel/image hash | EXISTS | RamRow.pixel_sha256, physical/timeline.row_pixel_sha256; raw grayscale bytes/base64 в physical_rows.jsonl |
| similarity previous crop | PARTIAL | `parser_core/chat_rows.py:93,137,189` align_upward / canonical_row_distance / match_row_blocks; ROI unchanged через np.array_equal (`ram_first_stage.py:116`), но per-view similarity не сериализуется |
| OCR confidence | EXISTS | raw_ocr_response.score; timeline/XLSX.confidence, `scripts/run_smoke14_ram.py:372` |
| normalized OCR text | EXISTS | timeline/XLSX.normalized_text=raw.strip(), `scripts/run_smoke14_ram.py:371`; это trim, не семантическая правка raw OCR |
| canonical event / duplicate relationship | PARTIAL | Physical stable identity, normalize/group_rows/message_id и card projection; explicit cross-track duplicate/canonical relationship отсутствует в production. Экспериментальные overlap proposals и Jev hypothesis_options не являются production решениями |

FILE → FUNCTION → CALLER → DATA: `run_production.py:main` → registered `scripts/run_smoke14_ram.py:main` → `FirstStage.rows` → RamRow → ParallelRowOCR по row_key → normalize/group_rows/card projection → ordered_timeline.jsonl → registered XLSX builder. Production не вызывает resolver/logic/math. Сохранённый JSONL отдельно читают экспериментальные poker_hand_assembly / poker_logic_engine / poker_math_engine / Jev. Эта задача только читает их текущие результаты и deterministic hypothesis_options, не выполняет Jev requests и не меняет решения.

# WHAT WAS ADDED

Только отдельный evidence-only слой, без promotion. stable_row_id остаётся row/track authority: дублирующие track_id/row_id/source_frame/ocr_raw поля в enriched timeline не добавлены. Их эквиваленты frame_id/stable_row_id/raw_ocr_text/normalized_text/confidence уже существуют. row_key перенесён из physical evidence для прямого lookup.

- `timestamp_ms`, `timestamp_text`, source_pts и рациональная time_base в lossless enriched timeline.
- `first_seen_ms`, `last_seen_ms`, observation_count, explicit_view_count, visibility_gap_count, left/right censoring, chat_line_y1.
- `track_views.jsonl`: существующие ID, source frame/PTS, row position, hash и 1−MAE/255 между последовательными одинаково shaped views; это диагностический score, без threshold/merge решения.
- `tracks.jsonl`: disjoint visibility spans. Unchanged ROI frames наследуют предыдущий viewport до следующего изменения, включая пустое состояние. Не интерполируются gaps. observation_count считает decoded frames visible; explicit_view_count считает changed-state observations.
- Research labels, source-linked confirmed starts, persistent tails, duplicate candidate comparisons, missing windows и card-crop parent temporal contexts.

First seen — первое **видимое наблюдение**, а не timestamp server action. INITIAL_VISIBLE и UNRESOLVED_IDENTITY не получают выдуманную историю до начала записи. Встроенный tracker переносит ID только по своим существующим доказательствам; новые IDs не объединяются из-за совпадения текста/времени.

Timestamp берётся из `PTS × time_base × 1000`, не frame/fps. Все 28 222 кадра имеют PTS, time_base=1/15360; отрицательных/равных временных переходов 0. Межкадровый delta: 33.333…–133.333…ms: фиксированное frame/fps не гарантирует исходное время. Отличие numeric ms от сохранённого production video_time×1000 не более 1.164e−10ms (float representation). Raw PTS/base остаются точной authority. OCR completion order не используется. XLSX round-trip проверяется отдельно в workbook_verification.json.

# SMOKE17 TEMPORAL RESULT

Baseline: реальный полный `runs/production_smoke17_20261001_013909`, PARTIAL_VISUAL, 28 222 frames, 8 146 physical/OCR/timeline rows. Source video SHA256: `27c0ba10544e0f6025e2d563baf7693d36c185cdbbfeb4517d3486116e9f8baf`. Baseline timeline SHA256: `2b7822db0fa4c14706a8ef8e576b5942e22275cd28c29b62c5174720ff566a5b`.

Парный A/B использует фиксированные реальные baseline OCR/cards. Новый full capture-only replay собрал 950 047 явных views и 54 631 changed chat states за 1 215.977s, включая отдельный PTS decode. Это стоимость диагностики, не новый production performance benchmark. Все 8 146 source row IDs, pixels, hashes, PTS/time_base проверены на equality; source inventory/global/per-table timeline order и исходные поля сохранены. Повторный OCR/model запуск не выполнялся.

Итоговая enriched timeline: 8 146/8 146, SHA256 `2ce7e2972f0ff5d93b3a33e95ec0b922c5f1e846188d6953289cb9583484c7c7`.

7 732 baseline rows имели UNRESOLVED/UNKNOWN/ERROR/UNREADABLE. Из них 1 606 получили ограниченную temporal attribution: 122 earlier-context tail + 1 484 current; 6 126 остались ambiguous. Статус OCR/identity/logic этих строк не исправлен. Это независимая временная диагностика, не independent visual certification.

90 confirmed starts получены **существующим offline edge_gate правилом**: explicit safe NEW_HAND или contiguous forced bets + DEAL. Hand number/header сам по себе не подтверждает start. Boundary recognition остаётся RAW_UNREVIEWED. Unconfirmed numbered markers блокируют attribution; other tables/sessions не участвуют. Header перед единственным confirmed start своего next candidate не создаёт скрытую дополнительную boundary; сам prefix остаётся ambiguous.

Доказанные ошибки hand assignment: 0. Это отсутствие доказательства в этом эксперименте, а не доказательство правильности всех assignments.

# CURRENT_HAND SEPARATION RESULT

Единица — исходная physical timeline row, относительная к её существующему candidate и confirmed boundaries:

| class | rows |
|---|---:|
| PREVIOUS_HAND_TAIL | 142 |
| CURRENT_HAND | 1578 |
| NEXT_HAND_HEAD | 0 |
| AMBIGUOUS | 6426 |
| TOTAL | 8146 |

PREVIOUS означает доказанно более ранний **видимый контекст** до start; номер непосредственно предыдущей руки не утверждается. 296 отдельных track/start pairs показывают старые tracks, которые впервые видны до confirmed start и продолжают быть видны после него. Это 296 unique tracks, не 296 новых source events; собственный header candidate перед его forced-bet start исключён из этого результата.

NEXT=0: на реальных candidate windows не доказано переназначение к следующей руке; synthetic regression подтверждает обработку new track после следующего start. Ненумерованный interior segment не приравнивается к leading visible prefix. Last interval без next confirmed start остаётся ambiguous; payout не заменяет отсутствующее доказательство полной руки.

# DUPLICATE RESULT

590 unique candidate pairs: 585 same confirmed temporal interval, 5 different intervals; 0 автоматически доказанных duplicate, 590 ambiguous. Для каждой пары сохранены Δtimestamp, same/different stable track, ROI position, pixel hash/similarity, OCR similarity, interval и overlap visibility. Разные IDs и одинаковый текст/близкие pixels не доказывают ни duplicate, ни новое poker событие.

Актуальный deterministic resolver generator на saved trace conflicts и 180 saved resolver requests дал 134 duplicate options (с повторениями), 69 unique resolver pairs. Все сохранены отдельно с target/evidence row IDs и происхождением. Пересечения с прежними overlap proposals не удваивают unique pair count. Jev decisions не принимались. Остальные пары — legacy experiment proposals, не текущие resolver decisions.

# MISSING EVENT RESULT

59 saved FIRST_BREAK missing-code windows локализованы по source frame/row position между ближайшими безопасно parsed соседями. Low-confidence/unresolved IDs и visible tracks доступны для каждого окна. Sameframe PTS не расширяет window до всех строк viewport. Endpoint basis: safe parsed RAW_UNREVIEWED, не human-certified; отсутствующий endpoint явно открыт.

Диагностический source — текущий saved fixed replay на 7 954 rows; поиск evidence выполняется по **всем исходным 8 146 rows**. Это coverage FIRST_BREAK missing codes, не всех ledger issues. Отсутствующие события восстановлены: 0. Физическое наличие crop в window не доказывает, что это нужное отсутствующее poker action; top first-visible span по прежнему не получает downstream row/track.

# TESTS

`python -X utf8 -m pytest tests/test_temporal_evidence_analysis.py tests/test_identity_observer_isolation.py tests/test_row_identity_history.py tests/test_row_identity_overlap.py tests/test_parallel_ocr.py -q`: exit 0, **29 passed**, 0.61s.

`python scripts/check_production_integrity.py`: exit 0, **PASS**, strict registered hashes unchanged. Independent read-only audit: initial false-prefix/window/isolation findings corrected; final real-row checks found 0 current/tail labels with unresolved identity or visibility gaps. Full source replay verified pixels/IDs/PTS for 8 146/8 146 and completed 28 222/28 222 frames. Full pytest / new production golden smoke не требуются для evidence-only isolated files и не запускались; production promotion не выполнялась.

XLSX round-trip **PASS**: 8 146 rows, 67 columns, 1 104 303 field comparisons, 0 mismatches, numeric round-trip error=0; 24 438 PTS/time_base integer cells сохранены точно. Global/per-table order/IDs, frame/video_time и RESULT counts совпали. Все 15 sheet previews просмотрены; clipped duplicate sources/missing-code ячейки исправлены форматированием. Workbook SHA256: `2361532f2f2d543dbcf6734060880209c990cb20e059e5d9665bd2b66cb9d80c`. Receipt: `outputs/temporal_context_smoke17_20261001/workbook_verification.json`.

# PRODUCTION CODE CHANGED

**NO**.

# VERDICT

**TEMPORAL_HAND_SEPARATION_PARTIAL**

`IS_TEMPORAL_EVIDENCE_ENOUGH_FOR_EDGE_GATE: PARTIAL`

Temporal evidence помогает локализовать viewport rows/пропуски и наблюдать persistent old tracks, но не обеспечивает надёжное отделение всех рук. Не хватает двух сигналов: (1) доказанная физическая связь cross-track IDs после потери identity; (2) confirmed start/end и onset bounds для censored/unknown viewport rows. Source time само по себе не заменяет эти доказательства. Автоматическое удаление/переназначение строк в resolver/logic/math не внедрено.
