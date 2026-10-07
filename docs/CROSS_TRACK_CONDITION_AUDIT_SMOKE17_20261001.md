# Cross-track condition audit — Smoke17

Диагностика замороженных 590 candidate pairs. Verdict каждого трека сохранён: SAME=0, DIFFERENT=0, AMBIGUOUS=590. Правила, thresholds, два соседа, source rows, stable IDs, Jev и edge-gate не изменены.

FILE → FUNCTION → CALLER → DATA: `parser_core/chat_rows.py` → `align_upward / match_rows / match_row_blocks` → `ram_first_stage.py:FirstStage.rows` → непронесённая identity / новый UNRESOLVED_IDENTITY. Эксперимент `cross_track_linker.py:motion_proof` повторяет физическое сопоставление, `decide` проверяет conjunction. Это эксперимент, а production определён только PRODUCTION.md и production_manifest.json.

## 590 PAIRS CONDITION MATRIX

PASS — условие проверено и выполнено; FAIL — проверено и не выполнено; UNKNOWN — проверка пропущена из-за prerequisite либо в действующих правилах нет такой бинарной проверки. Отсутствующего causal snapshot, pixels или text нет: NO_EVIDENCE=0 для всех трёх. UNKNOWN не равен доказанному конфликту.

| CONDITION | PASS | FAIL | UNKNOWN |
|---|---:|---:|---:|
| same_table | 590 | 0 | 0 |
| same_session | 590 | 0 | 0 |
| causal_temporal_order | 590 | 0 | 0 |
| immediate_viewport_continuity | 359 | 231 | 0 |
| viewport_alignment | 568 | 22 | 0 |
| target_row_match | 99 | 491 | 0 |
| unique_row_match | 0 | 0 | 590 |
| same_normalized_text | 590 | 0 | 0 |
| text_uniqueness | 0 | 0 | 590 |
| pixel_similarity_strength | 0 | 0 | 590 |
| exact_pixel_content | 0 | 99 | 491 |
| coherent_neighbor_movement | 2 | 97 | 491 |
| one_neighbor | 2 | 97 | 491 |
| two_neighbors | 2 | 97 | 491 |
| no_visibility_gap | 589 | 1 | 0 |
| no_simultaneous_visibility | 590 | 0 | 0 |
| no_boundary_conflict | 586 | 4 | 0 |
| no_epoch_conflict | 0 | 0 | 590 |
| causal_snapshot_available | 590 | 0 | 0 |
| pixel_evidence_available | 590 | 0 | 0 |
| text_evidence_available | 590 | 0 | 0 |
| y_position_continuity | 566 | 2 | 22 |

Y continuity дополнительно измерена на tight spans после компенсирования доказанного viewport shift; результаты и определения ниже. Δy padded crop само по себе не является условием отказа.

Нулевой target_unique не означает 590 доказанных неоднозначных строк: проверка уникальности запускается только после exact target, а exact target не получен ни разу. Pixel similarity — непрерывная MAE-метрика; отдельного порога PIXEL_NOT_STRONG_ENOUGH в classifier нет. Действующее требование — точное совпадение native content без scrollbar. Text similarity=1 не проверяет уникальность физической строки. TEXT_NOT_UNIQUE и EPOCH_CONFLICT нельзя объявить измеренными blockers: таких доказательств здесь нет.

## BLOCKING REASONS

Числа означают отсутствие необходимого proof; одна пара имеет несколько причин. Для NO_2_NEIGHBORS 491 проверка gated/skipped и 97 проверены без двух anchors. Поэтому 588 не означает 588 физически пустых viewport.

| BLOCK_REASON | COUNT | % от 590 |
|---|---:|---:|
| NO_EXACT_PIXEL_PROOF | 590 | 100.00% |
| NO_UNIQUE_EXACT_ROW_PROOF | 590 | 100.00% |
| NO_2_NEIGHBORS | 588 | 99.66% |
| NO_TARGET_ROW_MATCH | 491 | 83.22% |
| NO_IMMEDIATE_VIEWPORT_LINK | 231 | 39.15% |
| NO_VIEWPORT_ALIGNMENT | 22 | 3.73% |
| BOUNDARY_CONFLICT | 4 | 0.68% |
| MISSING_CAUSAL_SNAPSHOT | 0 | 0.00% |
| MISSING_PIXEL_EVIDENCE | 0 | 0.00% |
| MISSING_TEXT_EVIDENCE | 0 | 0.00% |

VISIBILITY_GAP: 1 causal-prefix наблюдение (0.17%); это отдельное наблюдение, а не самостоятельный veto в decide. Simultaneous visibility: 0. Epoch endpoint context не записан: 590 UNKNOWN. Исходный epoch первой source row не доказывает epoch последнего viewport.

## PRIMARY BLOCKING REASON

У classifier нет отдельного primary reason для каждого conjunct: он возвращает 586 INSUFFICIENT_COMBINED_PHYSICAL_CONTINUITY и 4 BOUNDARY_BLOCKS_SAME_WITHOUT_PHYSICAL_ONSET_PROOF. Для first failed conjunct существующего выражения strong порядок ровно такой: aligned → target_unique → exact_target_content → two neighbors.

| PRIMARY_BLOCK_REASON в strong | COUNT | % |
|---|---:|---:|
| NO_VIEWPORT_ALIGNMENT | 22 | 3.73% |
| NO_UNIQUE_EXACT_ROW_PROOF | 568 | 96.27% |

Этот порядок объясняет выполнение кода, но скрывает prerequisite: при построении proof сначала отсутствует target mapping у 491 пары (83.22%), а у оставшихся 99 (16.78%) mapped target не совпал с raster точно. Именно поэтому uniqueness не проверена. Boundary — отдельная возвращаемая причина для четырёх пар, не подмена first strong conjunct.

## NEAR-SAME CASES

Подсчёт по 11 необходимым условиям: table, session, no simultaneous visibility, alignment, target mapped to candidate, exact pixels, unique exact row, two neighbors, immediate continuity, nonempty equal normalized text, no boundary. FAIL и UNKNOWN считаются отсутствующим доказательством. Зависимые условия явно считаются отдельно; это не независимые наблюдения.

Missing only 1: **0**; missing only 2: **0**; missing 3+: **590**. Histogram: `{'4': 454, '5': 112, '6': 20, '7': 2, '3': 2}`.

Случаев «text PASS, exact pixel PASS, time PASS, y PASS, 1 neighbor PASS, только 2nd FAIL»: **0**. Exact pixel PASS=0. Более сильное независимое препятствие: mapped target=99, immediate continuity=359, их пересечение=0. Даже игнорирование exact/unique/neighbors не даёт пару с совместными alignment + candidate mapping + immediate continuity. Это диагностическая проверка conjunction, не альтернативная классификация.

Две пары с сохранёнными 2+ anchors имеют 3/4-frame handoff (100/133.333ms), Δy=0 и similarity≈0.999598/0.999930. Обе всё равно без exact target и unique proof, а handoff вне допустимого immediate link. Это ближайшие примеры с соседями; SAME им не присвоен.

## NEIGHBOR BOTTLENECK

Сохранённые target-associated anchors: 0 — 588; 1 — 0; 2+ — 2. У 491 пары подсчёт пропущен: target mapping отсутствует. Независимые измерения выполнены заново на тех же causal endpoint ROIs, без prerequisite целевого соответствия и без использования для verdict.

Числовой итог независимого измерения (machine-readable, определения полей в скрипте; source hashes сохранены в JSON artifact):

```json
{
  "pairs": 590,
  "strict_independent_neighbor_buckets": {
    "0": 582,
    "2+": 7,
    "1": 1
  },
  "threshold_matched_neighbor_buckets": {
    "2+": 568,
    "0": 22
  },
  "center_coherent_threshold_neighbor_buckets": {
    "2+": 568,
    "0": 22
  },
  "viewport_edges": 145,
  "physically_less_than_two_neighbors": 0,
  "pairs_with_changed_unresolved_neighbor": 71,
  "pairs_with_two_threshold_neighbors_but_less_than_two_same_id_neighbors": 0,
  "strict_neighbors_changed_unresolved": 0,
  "literal_unresolved_identity_veto_pairs": 0,
  "target_y_pass": 566,
  "target_y_fail": 2,
  "target_y_unknown": 22,
  "target_distance_pass": 114,
  "target_native_exact": 0,
  "native_exact_among_gap1": 0,
  "replay_seconds": 235.45990869996604,
  "total_seconds": 357.7832216999959,
  "roi_requests": 1180,
  "decoded_seek_frames": 10088,
  "roi_peak_images": 3,
  "roi_peak_bytes": 260820,
  "roi_images_at_end": 0,
  "image_writes": 0,
  "image_inventory_count": 296336,
  "hashes_unchanged": true,
  "original_verdict_counts": {
    "AMBIGUOUS": 590
  },
  "causal_safety": "Only endpoint snapshots and ROIs <= new first frame. Origin labels are fixed first-observation metadata. Terminal lifetimes are not inspected.",
  "caveat": "Independent neighbors need not share a contiguous target block and cannot replace the frozen decision gate. Changed UNRESOLVED IDs are correlation only, not a gate veto.",
  "diagnostic_script_sha256_at_finish": "90aee8bdd5f651ae44412b1ee5599b35925307cff56f79dd10236f8e6bc522ca",
  "diagnostic_script_capture": "No full script start-byte hash captured. Loaded measure function remained unchanged; unused depth diagnostics were added to file during first replay. Source/production/linker hashes were captured before and after.",
  "strict_one_neighbor_cases": [
    {
      "pair_id": "table_deb776c52437ea38b8ca8db5:R000601->table_deb776c52437ea38b8ca8db5:R000619",
      "endpoint_frames": [
        16537,
        16538
      ],
      "target_center_residual": 1.0,
      "target_row_distance": 10.698893229166666,
      "target_native_exact": false
    }
  ]
}
```

Независимые viewport neighbors не равны anchors блока, содержащего target. Top/bottom position сама по себе не исключает двух соседей: они могут быть на одной стороне. Физическая невозможность установлена только когда в одном endpoint меньше двух остальных eligible строк; исключённая первая visible строка отмечена отдельно. Изменение IDs у визуальных соседей — корреляция. UNRESOLVED_IDENTITY не является фильтром anchors в действующем linker, поэтому нельзя объявить его причиной потери второго соседа.

Ровно один strict independent neighbor: `table_deb776c52437ea38b8ca8db5:R000601 → R000619`, frames 16537→16538. Target center residual=1px, но registered row_distance=10.698893 >10 и native raster unequal. Не хватает не только второго соседа: target mapping не проходит pixel-distance gate.

## WHY 491 HAVE NO OLD ROW → NEW ROW MAPPING

Проверен old target против ВСЕХ new spans, затем воспроизведены sorted(distance+delta), one-to-one greedy assignment и crossed-order reset действующего match_rows. Это объяснение фактической работы существующего кода, без нового порога.

| Причина отсутствия mapping | COUNT | % от 491 |
|---|---:|---:|
| Alignment не доказан; match_rows не запущен | 22 | 4.48% |
| Есть строки в пределах center residual ≤7px, но у всех row_distance >10 | 469 | 95.52% |
| Нет new row в допустимой геометрии | 0 | 0.00% |
| Eligible target потерян из-за one-to-one competition | 0 | 0.00% |
| Mapping сброшен из-за crossed order | 0 | 0.00% |

**Главный измеренный отказ — pixel distance, а не y-position или конкуренция IDs.** Row distance считается на tight row, с masked scrollbar, resize 256×24 и средней absolute grayscale difference. Это другая метрика, чем сохранённая similarity padded crop: фон/padding могут давать similarity≈0.97 одновременно с distance>10 у текста. Наличие похожего текста не отменяет фактический pixel gate.

## WHAT CHANGES IN THE 99 MAPPED RASTERS

| Измерение после исключения scrollbar | COUNT |
|---|---:|
| Одинаковые размеры tight raster | 72 |
| Различается высота tight raster / segmentation extent | 27 |
| Есть изменения внутри crop, за пределами outer 2px | 99 |
| Есть изменения active pixels (хотя бы один endpoint >70) | 99 |
| Есть изменения dark background pixels (оба endpoints ≤70) | 99 |
| Pixels пересекают границу intensity >70 | 99 |
| Малый vertical offset −2..2 улучшает MAE, best offset ≠0 | 19 |
| Exact совпадение даже cropped overlap при −2..2px | 0 |

Положение согласуется с найденным viewport transform, но native content не равен. У 27 случаев меняется segmentation/crop extent; это не объясняет остальные 72. У всех 99 differences заходят внутрь строки и затрагивают как светлое содержимое, так и тёмный фон; они не ограничены только краями crop. Для 19 случаев небольшой дополнительный сдвиг уменьшает ошибку, но ни разу не восстанавливает exact cropped overlap. Поэтому простой integer chat shift или только border mismatch не объясняет весь отказ.

Это доказывает grayscale и support differences, но **не определяет отдельно антиалиасинг, lossy video compression, перерисовку UI или ошибочное физическое соответствие**. Threshold-crossing >70 сам по себе не доказывает изменение символа: небольшая смена оттенка тоже может пересечь 70. Для причинной атрибуции AA/compression нужна отдельная исходная raster/codec/render evidence; здесь её нет. Правила не ослаблены.

Полные per-pair dimensions, pixel counts, MAE/max, row/column counts и offset diagnostics находятся в `pixel_change_details.jsonl`; причины каждого из 491 mapping — в `mapping_rejection_details.jsonl`. Descriptor summary:

```json
{
  "pairs": 590,
  "mapping_missing": 491,
  "mapping_reasons": {
    "PIXEL_DISTANCE_OVER_10_FOR_POSITION_ELIGIBLE_ROWS": 469,
    "ALIGNMENT_NOT_PROVEN": 22
  },
  "mapped_native_mismatch": 99,
  "shape_equal": 72,
  "shape_different": 27,
  "pure_vertical_translation_within_2px": 0,
  "best_nonzero_vertical_offset": 19,
  "inner_pixels_changed_pairs": 99,
  "active_pixels_changed_pairs": 99,
  "background_pixels_changed_pairs": 99,
  "intensity_class_changed_pairs": 99,
  "hashes_unchanged": true,
  "image_writes": 0,
  "image_inventory_count": 296336,
  "replay_seconds": 240.63490909989923,
  "total_seconds": 372.2030826000264,
  "roi_requests": 1180,
  "decoded_seek_frames": 10088,
  "roi_peak_images": 3,
  "roi_images_at_end": 0,
  "caveat": "Existing thresholds only; offset search and gray-level descriptors never enter verdicts. No antialiasing/compression causal attribution.",
  "descriptor_quantiles": {
    "overlap_mae": {
      "min": 0.00774932614555256,
      "p10": 0.7135970888571842,
      "p25": 1.850237437265739,
      "median": 2.6355218855218854,
      "p75": 6.316015897973445,
      "p90": 10.439847068819033,
      "max": 12.213254035683942
    },
    "overlap_max": {
      "min": 2.0,
      "p10": 7.0,
      "p25": 28.5,
      "median": 70.0,
      "p75": 92.5,
      "p90": 168.0,
      "max": 194.0
    },
    "overlap_changed_pixels": {
      "min": 17.0,
      "p10": 826.6,
      "p25": 917.5,
      "median": 1113.0,
      "p75": 1302.5,
      "p90": 1603.2,
      "max": 1828.0
    },
    "tight_center_residual": {
      "min": 0.0,
      "p10": 0.0,
      "p25": 0.0,
      "median": 0.0,
      "p75": 0.5,
      "p90": 0.5,
      "max": 1.0
    },
    "registered_row_distance": {
      "min": 0.0068359375,
      "p10": 0.5423177083333334,
      "p25": 1.8667805989583333,
      "median": 2.2574869791666665,
      "p75": 4.478841145833334,
      "p90": 6.176074218750001,
      "max": 9.542317708333334
    }
  },
  "changed_fraction_quantiles": {
    "min": 0.0057277628032345014,
    "p10": 0.3713566630865167,
    "p25": 0.40222984562607206,
    "median": 0.48274410774410775,
    "p75": 0.5694468481981463,
    "p90": 0.7450882658359294,
    "max": 0.8226415094339623
  },
  "position_eligible_min_distance_quantiles": {
    "min": 10.014485677083334,
    "p10": 10.181868489583334,
    "p25": 10.588541666666666,
    "median": 11.301595052083334,
    "p75": 12.213704427083334,
    "p90": 13.572428385416666,
    "max": 33.112141927083336
  },
  "overlap_only_offset_exact_pairs": 0,
  "diagnostic_script_sha256": "90aee8bdd5f651ae44412b1ee5599b35925307cff56f79dd10236f8e6bc522ca",
  "diagnostic_script_capture": "Depth file stayed unchanged since launch; SHA captured during replay and verified at end. Frozen production/linker/input files were hashed before and after.",
  "shape_pairs": {
    "[11, 214] -> [12, 214]": 8,
    "[9, 216] -> [10, 216]": 1,
    "[10, 214] -> [10, 214]": 4,
    "[11, 214] -> [11, 214]": 8,
    "[11, 212] -> [11, 212]": 37,
    "[12, 214] -> [12, 214]": 2,
    "[14, 214] -> [14, 214]": 3,
    "[9, 214] -> [10, 214]": 6,
    "[11, 216] -> [11, 216]": 8,
    "[9, 212] -> [10, 212]": 4,
    "[9, 216] -> [9, 216]": 1,
    "[10, 212] -> [9, 212]": 3,
    "[10, 216] -> [10, 216]": 2,
    "[10, 212] -> [10, 212]": 5,
    "[11, 212] -> [9, 212]": 2,
    "[11, 212] -> [10, 212]": 1,
    "[11, 214] -> [10, 214]": 1,
    "[11, 218] -> [11, 218]": 1,
    "[14, 212] -> [14, 212]": 1,
    "[10, 214] -> [9, 214]": 1
  }
}
```

## PIXEL/TEXT/TIME/Y DISTRIBUTIONS

Quantiles: NumPy linear; time — exact PTS new.first − causal old.last, ms; y — signed padded crop Δy, px; pixel — padded crop MAE similarity. Text — normalized SequenceMatcher. Frame gap считается по decoded indices, не по fps. Boundary-near ниже означает наличие separating confirmed boundary внутри old.last < boundary ≤ new.first; произвольное окно «near» не добавлено.

### all (n=590)

| METRIC | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| pixel_similarity | 0.933544 | 0.957846 | 0.967525 | 0.972746 | 0.978059 | 0.992163 | 0.999930 |
| text_similarity | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| delta_ms | 33.333333 | 33.333333 | 33.333333 | 33.333333 | 4900.000000 | 19676.666667 | 86033.333333 |
| delta_y | -307.000000 | -162.000000 | -80.000000 | -41.000000 | -21.000000 | -1.000000 | 201.000000 |

### gap1 (n=359)

| METRIC | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| pixel_similarity | 0.933544 | 0.957714 | 0.966546 | 0.970139 | 0.975232 | 0.979108 | 0.996132 |
| text_similarity | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| delta_ms | 33.333333 | 33.333333 | 33.333333 | 33.333333 | 33.333333 | 33.333333 | 133.333333 |
| delta_y | -263.000000 | -42.000000 | -41.000000 | -21.000000 | -21.000000 | -1.000000 | 1.000000 |

### gap_gt1 (n=231)

| METRIC | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| pixel_similarity | 0.944211 | 0.962433 | 0.972331 | 0.977322 | 0.992071 | 0.995908 | 0.999930 |
| text_similarity | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| delta_ms | 66.666667 | 266.666667 | 2183.333333 | 9333.333333 | 20066.666667 | 34833.333333 | 86033.333333 |
| delta_y | -307.000000 | -243.000000 | -162.000000 | -124.000000 | -40.000000 | -20.000000 | 201.000000 |

### causal_visibility_gap (n=1)

| METRIC | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| pixel_similarity | 0.970896 | 0.970896 | 0.970896 | 0.970896 | 0.970896 | 0.970896 | 0.970896 |
| text_similarity | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| delta_ms | 2266.666667 | 2266.666667 | 2266.666667 | 2266.666667 | 2266.666667 | 2266.666667 | 2266.666667 |
| delta_y | 201.000000 | 201.000000 | 201.000000 | 201.000000 | 201.000000 | 201.000000 | 201.000000 |

### causal_boundary_present (n=4)

| METRIC | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| pixel_similarity | 0.940938 | 0.953768 | 0.973014 | 0.984219 | 0.985198 | 0.986036 | 0.986595 |
| text_similarity | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| delta_ms | 33.333333 | 33.333333 | 33.333333 | 1450.000000 | 2866.666667 | 2866.666667 | 2866.666667 |
| delta_y | -263.000000 | -262.700000 | -262.250000 | -90.500000 | 81.000000 | 81.000000 | 81.000000 |

Pixel/text/time/y доступны для 590/590; missing=0. Группа gap>1 содержит 231 пару: 7 gap2, 224 gap>2. Ни один gap2 не получил local_loss_bridge. Высокая average similarity не доказывает точное соответствие glyphs: фон и padding входят в метрику.

## WHY DIFFERENT = 0

| DIFFERENT witness / observation | COUNT |
|---|---:|
| Different table/session | 0 |
| Causal simultaneous distinct physical positions | 0 |
| Old target proved continuing at another current row | 0 |
| Proven newly exposed bottom onset | 0 |
| Full boundary + onset + two neighbors predicate | 0 |
| Different normalized text | 0 |
| Strong positive physical conflict satisfying current rules | 0 |
| Outside immediate SAME handoff window | 231 |
| Gap >2 frames | 224 |

Ни одна фактическая пара не дала simultaneous witness. Если бы distinct rows были simultaneously visible, существующая ветка могла бы доказать DIFFERENT, но для 590 таких фактов нет: hypothetical count нельзя превратить в доказательство. Incompatible pixels/text: текстовых конфликтов 0; exact mismatch / отсутствие exact proof не является доказательством разных строк. Бинарного порога pixel incompatibility нет, поэтому это UNKNOWN, а не 590 DIFFERENT. Максимальный causal gap=86.033s; отдельного time-only DIFFERENT threshold нет. Onset существует у 417, aligned у 403, 2+ onset anchors у 13, newly exposed у 0. Ни длинный gap, ни boundary без physical onset не доказывает новую строку.

Эти данные показывают недостаток positive DIFFERENT witnesses. Они не доказывают, что DIFFERENT classifier ошибочно строг: нет независимой ground truth, подтверждающей пропущенные DIFFERENT.

## CAUSAL SAFETY

Все 590 condition rows проверяют snapshot source/first/last/spans, motion/onset и boundary frames ≤ new.first_frame. Независимые pixels получены только из causal old.last и new.first. Terminal metadata не участвует в conditions, blockers, neighbors или verdict. OCR source frame уже ≤ decision; используется сохранённое исходное чтение, новый OCR не запускается. Это аудит заранее заданного offline candidate set, а не доказательство causal online candidate generation.

Source/production/video/frozen linker hashes совпали до/после обоих endpoint replay; глобальный image inventory 296336 не изменился, image writes=0. Provenance limitation: первый neighbor run не имел full-byte pin diagnostic script на старте; добавление неиспользуемой depth-функции происходило во время него. Для depth script SHA зафиксирован во время выполнения и сверён после; это не выдаётся за start-time pin. Loaded neighbor measurement, causal input hashes и frozen classifier не менялись. Independent aggregation подтверждает числовые результаты; это не visual SAME ground truth.

## POST_HOC_CHECK

Отдельная offline validation по full lifetimes: 4 terminal envelopes interleave, 12 пар имеют visibility gap хотя бы одного полного lifetime; actual co-visibility=0 по полным spans и 54,631 chat states. При causal prefix visibility-gap count=1. Четыре будущих old reappearances не подаются в решение. POST_HOC_CHECK не меняет ни одного verdict.

## EDGE-GATE IMPACT

**6685 → 6685 UNRESOLVED_EDGE**, safely resolved=0; source rows **8146 → 8146**. Это сохранённая исходная simulation; ничего не применяется. Gate не потребляет experimental relationship metadata. Ни edge-gate, ни эпохи/границы, ни IDs/source rows не переписаны.

## VERDICT

**EVIDENCE_NOT_ENOUGH** при действующих замороженных правилах. Самого требования второго соседа недостаточно, чтобы объяснить все отказы: раньше не получен доказанный unique exact target, и нет пары с одновременно immediate handoff и mapped candidate. Это не сертификат оптимальности thresholds; RULES_TOO_STRICT без независимого подтверждения физических SAME не доказан.

**Чаще всего не хватает доказательства, что конкретная старая строка физически продолжилась именно в конкретной новой строке: её pixels должны совпасть точно, положение — согласоваться с viewport movement, и это соответствие должно быть уникальным в обоих viewport.** Одинакового текста и высокой средней похожести картинки здесь недостаточно. Два согласованных соседа также часто не доказаны, но их проверка обычно не достигается из-за отсутствия target mapping.

## ARTIFACTS

- `runs/cross_track_condition_audit_smoke17_20261001/condition_matrix.jsonl`: все 590 pairs, statuses, blockers, primary, metrics, missing conditions и causal frames.
- `runs/cross_track_condition_audit_smoke17_20261001/independent_neighbors.jsonl`: независимое измерение neighbors/y на causal ROIs.
- `runs/cross_track_condition_audit_smoke17_20261001/condition_summary.json` / `neighbor_summary.json`: агрегаты и provenance.
- `runs/cross_track_condition_audit_smoke17_20261001/mapping_rejection_details.jsonl` / `pixel_change_details.jsonl` / `pixel_change_summary.json`: подробный ответ на два уточнения.
- `runs/cross_track_condition_inspection_20261001/summary.json`: полные quantiles и conjunctions.
- Новые diagnostic scripts: `experiments/build_cross_track_condition_report.py`, `experiments/audit_cross_track_conditions.py`. Замороженные linker/runner не изменены.

Работа остановлена после диагностики. Ни одно правило автоматически не ослаблено.
