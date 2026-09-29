# Новый парсер PokerDom

Рабочая реализация ТЗ от 27.09.2026. Текущее состояние: **этап 0 частично выполнен; этап 1 начат только как сбор исходных доказательств**. Полных историй рук и новых обученных моделей пока нет.

## Уже работает

- `scripts/build_inventory.py` копирует три выбранных веса и фиксирует SHA-256 исходников, копий и справочных каталогов в `provenance/manifest.json`.
- `scripts/build_coverage.py` строит `docs/coverage.csv` из готовых каталогов: 35 типов сообщений, 12 типов сущностей и 14 ролей сумм. `CATALOG_ONLY` означает отсутствие независимой визуальной проверки.
- `scripts/capture_video.py` читает видео через PyAV, сохраняет исходные PTS с рациональной временной базой, кадры в SQLite и два заданных ROI в PNG. Повторный запуск с теми же входами не добавляет логических наблюдений; конфликт хеша или координат вызывает ошибку.
- `scripts/ocr_chat_pilot.py` сохраняет исходное чтение Tesseract полного чат-кропа, версию движка и параметры. Результат имеет статус `RAW_UNREVIEWED`, не создаёт событие; повторное чтение той же версии не дублируется.
- `scripts/export_diagnostics.py` выгружает наблюдения и последние OCR-версии в CSV с ссылками на PNG, без утверждения о полноте HH.
- `scripts/track_chat_rows.py`, `scripts/assemble_messages.py`, `scripts/normalize_messages.py`, `scripts/reduce_hands.py` строят одностольные строки, сообщения, события-кандидаты и только `PARTIAL` окна рук. Решения трекера и интерпретации версионируются; никакой кандидат не становится полным HH автоматически.
- `parser_core/journal.py` задаёт схему для наблюдений, OCR, физических строк, сообщений, версий событий и рук, экспортов и разрывов. Схема последующих этапов не означает, что эти этапы реализованы.
- `scripts/capture_nine_tables.py` читает видео одним декодером через фиксированный layout и отдельную калибровку нижней границы чата, записывает девять независимых сессий слотов, PTS и исходные PNG стола/чата. Идентичность столов остаётся `UNRESOLVED`; это эксперимент для раскладки 2560×1440.

## Пробный захват одного стола

Используется Python 3.13 с пакетами `av` и `Pillow`. Координаты ниже взяты из старой конфигурации для `smoke_01.mkv` и действуют только для этой раскладки.

```powershell
python scripts/capture_video.py 'D:\Проэкты\Парсер пд v1.0\video\smoke_01.mkv' `
  --table-roi 0,0,1408,1080 --chat-roi 1410,233,1888,948 `
  --layout-id 8max_1920x1080_legacy_roi --output runs\smoke01_fixed_roi_v2 --max-frames 5
python -m unittest discover -s tests -v
python scripts/ocr_chat_pilot.py runs\smoke01_fixed_roi_v2 --max-observations 5
python scripts/export_diagnostics.py runs\smoke01_fixed_roi_v2
```

Один вызов для текущего частичного пайплайна:

```powershell
python scripts/run_one_table.py --video 'D:\Проэкты\Парсер пд v1.0\video\smoke_01.mkv' `
  --table-roi 0,0,1408,1080 --chat-roi 1410,233,1888,948 `
  --layout-id 8max_1920x1080_legacy_roi `
  --start-frame 1600 --max-frames 100 --output runs\new_one_table_pilot
```

Для повторной обработки уже сохранённых доказательств: `python scripts/run_one_table.py --output runs\new_one_table_pilot --skip-capture`. Команда сохраняет отчёт `pilot_run.json` с версиями этапов и фактическим диапазоном кадров. Это пилот без сертификации HH.

Следующий более длинный пилот: [docs/STAGE1_PILOT.md](docs/STAGE1_PILOT.md), данные в `runs/smoke01_chat_300_v2`. Для него применены `--chat-only --max-frames 300`, затем `track_chat_rows.py`, OCR с `--roi-type CHAT_ROW`, сборка сообщений, нормализация и частичный reducer.

Отдельная проверка резкого скролла и воспроизводимый A/B на сохранённых входах: [docs/DENSE_SCROLL_AB.md](docs/DENSE_SCROLL_AB.md). Текущая версия трекера — `row_track_v9`; старые журналы сохранены для сравнения. Первая видимая полоска каждого чата исключается из строк OCR и отдельно регистрируется в журнале.

Сборщик сообщений `message_assembly_v5` отделяет каждую строку `Дилер:` и присоединяет строки без нового явного начала в пределах той же эпохи чата. Пустой OCR не исчезает внутри предыдущего действия. Экспорт `event_fragments_diagnostics.csv` связывает каждый фрагмент события с raw OCR, PTS и SHA PNG; версии сообщений в скопированных журналах сохранены для аудита. История исправления начала сообщения: [docs/MESSAGE_ASSEMBLY_REPLAY_20260928.md](docs/MESSAGE_ASSEMBLY_REPLAY_20260928.md).

`candidate_ru_v4` различает малый/большой блайнд и обычную ставку при OCR-пунктуации и слитной сумме, распознаёт «делает чек»; нечитаемый тип блайнда остаётся `UNKNOWN_EVENT`. История исправления блайндов: [docs/BLIND_NORMALIZER_REPLAY_20260928.md](docs/BLIND_NORMALIZER_REPLAY_20260928.md).

Пустой первичный OCR строки получает один повтор с увеличением PNG до 3×; первичное чтение сохранено отдельной ревизией. События из повторного чтения остаются `UNRESOLVED` / `OCR_RETRY_UNVERIFIED`. Проверка на Smoke10 и Smoke11: [docs/OCR_RETRY_REPLAY_20260928.md](docs/OCR_RETRY_REPLAY_20260928.md).

Очередь разметки из существующих каталогов: [docs/ANNOTATION_QUEUE.md](docs/ANNOTATION_QUEUE.md). Она содержит 300 сообщений, 30 переходов и 12 окон рук для визуальной проверки; независимым тестом не является.

## Девять столов: ограниченный пилот

```powershell
python scripts/run_nine_tables.py --video 'D:\Проэкты\V2.0 PD\video\smoke_11.mp4' `
  --output runs\smoke11_new_pilot --max-frames 5
python scripts/run_nine_tables.py --output runs\smoke11_new_pilot --skip-capture
```

Первая команда декодирует один источник и запускает строки → OCR → сообщения → кандидаты событий → частичные окна рук → диагностический CSV. Вторая повторяет только производные этапы из сохранённых PNG и журнала. Текущий проверенный пилот и визуальные ROI: [docs/NINE_TABLE_ROI_RERUN.md](docs/NINE_TABLE_ROI_RERUN.md). [Ранний пилот](docs/NINE_TABLE_PILOT.md) сохранён как исторический. Никаких проверенных HH этот путь пока не создаёт.

Без `--max-frames` декодируются все кадры; такой запуск может занять много времени и места. При ограничении источник получает статус `PARTIAL`. Папка `runs/smoke01_fixed_roi_v2` содержит проверенный пробный захват. Папки `runs/smoke01_capture_pilot` и `runs/smoke01_fixed_roi` — ранние пробы, не используемые как эталон.

## Что нужно дальше

Потоковый запуск с PNG и SQLite на SSD `C:`:

```powershell
python scripts/run_nine_tables_streaming.py --video video/smoke_13.mp4 --max-frames 12148
```

Каталог данных по умолчанию задаётся в `configs/run_storage_local.json`:
`C:/ParserData/runs`. `--output` переопределяет его. Один декодер и четыре
потока сохранения ROI работают одновременно с накопительным трекером строк,
пятью Tesseract-процессами и одной моделью VL. Повторно используются только
PNG с полностью одинаковыми пикселями; записи наблюдений, PTS и хеши
сохраняются для каждого кадра. `pilot_run.json`, `tracking_progress.json`,
`ocr_progress.json` и `resource_samples.csv` содержат прогресс и замеры.
Первые три этапа перекрываются, поэтому их времена нельзя складывать как
общее время прогона. Статус результата остаётся `PARTIAL_PIPELINE`.

Инструменты заморозки независимого визуального корпуса, аудита ручной разметки
и сравнения текущих версий сообщений/событий описаны в
[docs/VISUAL_CORPUS_20260928.md](docs/VISUAL_CORPUS_20260928.md). Smoke12
сохранён как сырой test-кандидат без OCR и без доказанной независимости
сессии; текущие результаты качества — `NOT_MEASURED`.

Непрерывное минутное окно Smoke12 для слота 01 (все 1800 кадров, исходный
чат и стол, без OCR) заморожено для визуальной разметки. Пользователь
подтвердил отдельную игровую сессию; новая версия манифеста отмечена
`independent_test=true`, но качество ещё `NOT_MEASURED`. Итог и просмотрщик:
[docs/SMOKE12_CONTINUOUS_SLOT1_20260928.md](docs/SMOKE12_CONTINUOUS_SLOT1_20260928.md).

См. [docs/STAGE0_STATUS.md](docs/STAGE0_STATUS.md), [docs/ANNOTATION_PROTOCOL.md](docs/ANNOTATION_PROTOCOL.md), [docs/TEST_PROTOCOL.md](docs/TEST_PROTOCOL.md). Следующие блокеры: визуально проверенный независимый эталон, реестр изменчивой геометрии и замены стола в слоте, чтение всех полей и строгие проверки полных рук. До их проверки нельзя объявлять качество девяти столов, точный HH или realtime.
