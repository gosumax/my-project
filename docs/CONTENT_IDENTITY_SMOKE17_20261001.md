# Content-level identity experiment — Smoke17

## FILES CHANGED

Только новые experiments/tests/report artifacts: `row_content_evidence.py`, `run_content_identity_audit.py`, `text_first_identity_audit.py`, `build_content_identity_report.py`, соответствующие targeted tests. Production, frozen linker/runner, OCR/math/Jev, IDs и source pixels не изменены.

## CONTENT REPRESENTATION

На RAM-копии padded source row исключается scrollbar 12px. Raster support — gray>70, тот же intensity cut, который существующая segmentation использует для геометрии. Это диагностическое raster foreground, а не семантическая гарантия «каждый pixel является буквой». Его bbox даёт x/y-tight box, dimensions и trims. OCR bbox не используется. Original crop/hash/dimensions сохраняются. Глифы не resize.

Поиск dx,dy только −2..2 (25 вариантов). AFTER перемещается в координаты BEFORE. Общий canvas включает ВСЕ foreground pixels обоих endpoint, включая непересекающиеся участки. Missing foreground, support XOR и coverage записываются отдельно: плохой участок нельзя отбросить, чтобы получить exact. Best выбирается по foreground-union MAE, затем XOR и минимальному сдвигу. До/после alignment, full/bbox/foreground/outside-content/dark/edge metrics — отдельные измерения.

FULL_CROP_DISTANCE как сейчас — registered row_distance: tight span, masked scrollbar, resize256×24, grayscale MAE. Native full-crop MAE без resize приведён отдельно. CONTENT_DISTANCE — MAE foreground union; content box MAE включает пустоты между glyphs. BORDER_DISTANCE — вне bbox union; dark background — оба pixels≤70; EDGE_DIFFERENCE — outer2px каждого исходного crop после alignment. Это разные области/шкалы; меньшее число full-crop MAE может означать dilution фоном, а не лучшее совпадение текста.

Ограничение canvas metrics: uncovered crop coverage заполняется нулём, поэтому aligned full/border/dark/edge distances включают эффект отсутствующего coverage. Это не доказательство изменения исходного фона. Для literal source background differences используйте zero-shift regions и coverage counters. Foreground никогда не отбрасывается и missing foreground считается отдельно.

## 491 UNMAPPED BEFORE

491 без frozen target mapping: 469 pixel-distance>10 при допустимой позиции, 22 alignment не доказан.

## 491 AFTER CONTENT MAPPING

Content-only diagnostic mapping: `{'UNIQUE_CONTENT_MAPPING': 0, 'MULTIPLE_CONTENT_CANDIDATES': 0, 'NO_CONTENT_MAPPING': 491}`.

Этот mapping требует proven viewport alignment, допустимого center residual≤7, same context/no overlap и точного равенства бинарного support после малого alignment, с ненулевым foreground и без отсутствующих foreground pixels. Gray intensities не обязаны быть exact. Один такой кандидат — UNIQUE, несколько — MULTIPLE, иначе NONE. Это консервативный threshold-free тест нового представления; он не является универсальным доказательством identity и не вводится в production.

При этом measurement показывает сильный geometry effect: из491 unmapped **451** требуют best dy±1, dx=0. Median foreground MAE **29.496484 → 7.962264** после alignment. Для all590 best dy−1=425, +1=36, zero=129, dx=0 для всех. Это доказывает важность относительного 1px content offset; не устанавливает отдельно, возник ли offset из crop positioning, segmentation или UI/video rendering.

## 99 MAPPED RASTER AUDIT

После content alignment foreground pixels всё ещё различаются у **99/99**; differences только вне content при неизменном foreground — **0/99**. Content mapping: `{'UNIQUE_CONTENT_MAPPING': 0, 'MULTIPLE_CONTENT_CANDIDATES': 0, 'NO_CONTENT_MAPPING': 99}`.

В исходном depth audit одинаковые tight dimensions у72, разная высота у27; все99 имели inner/active/background differences. Новый bbox и alignment проверяют влияние пустых краёв и малого X/Y сдвига; per-pair outputs сохраняют bbox, trims, score before/after, dx/dy, changed fractions и neighbor evidence. Antialiasing/compression причинно не установлены. Ненулевой intensity diff не доказывает разные semantic symbols.

Для99 mapped 89 best dy=0, 10 best dy±1; median foreground MAE **6.905983 → 6.905983**. Zero-shift median dark-background MAE≈0.913, outside-bbox≈0.110, edge≈0.250. Здесь расхождение не сводится только к пустым краям: content pixels продолжают различаться. После alignment native content, support equality и exact intensity support **0/590**.

## CONTENT DISTANCE DISTRIBUTIONS

NumPy linear quantiles. All missing regions явно NO_EVIDENCE; empty support не становится exact. SAME-candidate-like означает prospective TEXT_FIRST_SAME_CANDIDATE, а не известную ground truth.

### all590 (n=590)

| METRIC | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| registered row distance | 0.006836 | 2.339681 | 10.130412 | 11.023031 | 11.812785 | 13.406283 | 20.408854 |
| native full crop MAE | 0.017857 | 1.998330 | 5.594856 | 6.949668 | 8.281172 | 10.749245 | 16.946272 |
| content foreground before MAE | 0.011095 | 6.932868 | 23.896262 | 28.546847 | 32.258255 | 37.148899 | 51.697368 |
| aligned content foreground MAE | 0.011095 | 1.763688 | 6.805535 | 7.740381 | 12.680114 | 14.569673 | 25.396552 |
| aligned content box MAE | 0.014411 | 1.376794 | 3.641010 | 4.280330 | 6.797919 | 7.564623 | 11.630936 |
| border outside content MAE | 0.022422 | 0.134772 | 5.604077 | 6.193897 | 8.036747 | 10.524516 | 12.852652 |
| dark background MAE | 0.024020 | 0.917408 | 4.460803 | 4.963532 | 5.793248 | 6.840820 | 7.765017 |
| outer 2px difference MAE | 0.038377 | 0.280022 | 9.806476 | 9.897503 | 9.972518 | 10.083283 | 11.060448 |
| best dx | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| best dy | -1.000000 | -1.000000 | -1.000000 | -1.000000 | 0.000000 | 0.000000 | 1.000000 |
| OCR normalized similarity | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

### unmapped491 (n=491)

| METRIC | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| registered row distance | 0.250814 | 10.130534 | 10.518962 | 11.264648 | 12.175700 | 13.532227 | 20.408854 |
| native full crop MAE | 0.270465 | 5.182840 | 6.029729 | 7.456305 | 8.566398 | 10.775391 | 16.946272 |
| content foreground before MAE | 1.588040 | 23.621239 | 27.236029 | 29.496484 | 33.598064 | 37.261426 | 51.697368 |
| aligned content foreground MAE | 0.039801 | 1.889371 | 7.018044 | 7.962264 | 12.817706 | 14.602544 | 25.396552 |
| aligned content box MAE | 0.033573 | 1.470007 | 3.659934 | 4.613973 | 6.855942 | 7.587862 | 11.630936 |
| border outside content MAE | 0.102133 | 5.578112 | 5.855053 | 6.909237 | 8.520842 | 10.722628 | 12.852652 |
| dark background MAE | 0.178196 | 4.405834 | 4.669897 | 5.073155 | 6.101535 | 6.911009 | 7.560145 |
| outer 2px difference MAE | 0.125000 | 9.797440 | 9.857463 | 9.916418 | 9.986445 | 10.094225 | 11.060448 |
| best dx | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| best dy | -1.000000 | -1.000000 | -1.000000 | -1.000000 | -1.000000 | 0.000000 | 1.000000 |
| OCR normalized similarity | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

### mapped99 (n=99)

| METRIC | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| registered row distance | 0.006836 | 0.542318 | 1.866781 | 2.257487 | 4.478841 | 6.176074 | 9.542318 |
| native full crop MAE | 0.017857 | 0.550279 | 1.565542 | 1.998605 | 5.041482 | 5.788616 | 8.847935 |
| content foreground before MAE | 0.011095 | 1.265028 | 5.508610 | 6.905983 | 9.309600 | 19.167649 | 29.393715 |
| aligned content foreground MAE | 0.011095 | 1.265028 | 5.508610 | 6.905983 | 8.177764 | 13.850122 | 25.382289 |
| aligned content box MAE | 0.014411 | 1.123924 | 3.332381 | 4.048694 | 4.550032 | 7.293437 | 11.243233 |
| border outside content MAE | 0.022422 | 0.032608 | 0.065753 | 0.111233 | 0.299885 | 1.387264 | 12.290720 |
| dark background MAE | 0.024020 | 0.417495 | 0.770751 | 0.913043 | 1.377402 | 2.910344 | 7.765017 |
| outer 2px difference MAE | 0.038377 | 0.159586 | 0.185268 | 0.250000 | 0.469027 | 3.008139 | 10.471125 |
| best dx | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| best dy | -1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| OCR normalized similarity | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

### SAME-candidate-like (n=280)

| METRIC | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| registered row distance | 10.014486 | 10.166569 | 10.533610 | 11.197266 | 11.951375 | 13.512744 | 17.204915 |
| native full crop MAE | 1.333147 | 5.343003 | 6.197952 | 7.576336 | 8.235343 | 10.752930 | 16.381305 |
| content foreground before MAE | 2.445409 | 26.087367 | 27.339698 | 29.086319 | 32.201248 | 36.411685 | 47.456995 |
| aligned content foreground MAE | 0.220758 | 1.829933 | 7.018376 | 7.933322 | 12.423994 | 14.030336 | 18.431250 |
| aligned content box MAE | 0.175659 | 1.427990 | 3.601270 | 4.479062 | 6.786200 | 7.473589 | 11.630936 |
| border outside content MAE | 0.111538 | 5.826520 | 5.866484 | 7.088474 | 9.146103 | 10.909186 | 12.414179 |
| dark background MAE | 1.085460 | 4.533214 | 4.739082 | 5.134011 | 6.523358 | 7.063670 | 7.560145 |
| outer 2px difference MAE | 0.266593 | 9.813159 | 9.855597 | 9.916417 | 9.985182 | 10.064908 | 11.060448 |
| best dx | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| best dy | -1.000000 | -1.000000 | -1.000000 | -1.000000 | -1.000000 | -0.900000 | 1.000000 |
| OCR normalized similarity | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

### remaining (n=310)

| METRIC | min | p10 | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| registered row distance | 0.006836 | 1.906136 | 4.104004 | 10.451742 | 11.758586 | 13.242562 | 20.408854 |
| native full crop MAE | 0.017857 | 1.503364 | 3.552407 | 5.968750 | 8.464216 | 10.427076 | 16.946272 |
| content foreground before MAE | 0.011095 | 2.731434 | 7.303107 | 27.531532 | 32.356325 | 37.558806 | 51.697368 |
| aligned content foreground MAE | 0.011095 | 1.717779 | 6.412295 | 7.597149 | 13.086434 | 15.291154 | 25.396552 |
| aligned content box MAE | 0.014411 | 1.343792 | 3.683425 | 4.237485 | 6.874757 | 7.691030 | 11.293890 |
| border outside content MAE | 0.022422 | 0.076451 | 0.237355 | 5.855944 | 6.955677 | 8.869506 | 12.852652 |
| dark background MAE | 0.024020 | 0.781051 | 1.145838 | 4.659191 | 5.300437 | 6.177045 | 7.765017 |
| outer 2px difference MAE | 0.038377 | 0.195782 | 0.385845 | 9.878732 | 9.959999 | 10.101444 | 10.996302 |
| best dx | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| best dy | -1.000000 | -1.000000 | -1.000000 | -1.000000 | 0.000000 | 0.000000 | 1.000000 |
| OCR normalized similarity | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## NEIGHBOR CONTENT RESULT

0 / 1 / 2+ binary-content neighbors: `{'0': 567, '1': 11, '2+': 12}`. Frozen target-gated anchors0/1/2+=588/0/2; independent native-exact0/1/2+=582/1/7. Existing pixel-distance + coherent motion2+=568.

Content neighbors здесь проходят бинарный support-equality в ±2px, не обязательно grayscale equality. Они взяты из existing coherent row mappings на causal endpoint. Independent viewport neighbors не гарантируют contiguous block с target. Само наличие двух content neighbors не снимает target/context ambiguity.

## CAUSAL SAFETY

Все sourceROIs и feature frames≤new.first, old crop взят из causal last view, new — из first view. State/OCR parent/source frames ограничены соответствующим endpoint. Terminal lifetime не участвует. Hashes manifest/production/frozen linker/inputs/video/new content scripts закреплены BEFORE и сверены AFTER. Original NumPy source arrays неизменны, image inventory296336 не изменился.

## FALSE MERGE CHECK

Text branch блокирует duplicate/multiple risks у **217** пар; categories: `{'NICK_ACTION': 128, 'ANTE': 86, 'SB_BB': 18, 'FOLD': 51, 'SHORT_TEXT': 11, 'AMOUNT_ONLY': 2}`. Text context: `{'TEXT_UNIQUE': 373, 'TEXT_DUPLICATED_IN_TIME_WINDOW': 216, 'TEXT_DUPLICATED_IN_VIEWPORT': 1}`. Для каждого simulation candidate записаны old/new same-text counts, other matching IDs в предыдущих12 changed states, possible geometry candidates, boundaries, no-overlap и consistent neighbor count. Пустое/неизвестное OCR evidence не объявляется уникальным.

POST_HOC_CHECK: future old reappearance среди prospective candidates — **2**. Этот флаг не меняет causal verdict. Это отдельная validation, а не будущий feature. Ни synthetic tests, ни контекстный range-check не являются независимой visual ground truth.

Offline review disposition: из280 causal candidates **278** без найденного future-reappearance flag; **2** withheld / AMBIGUOUS_FOR_OFFLINE_ACCEPTANCE. Это не пересчёт causal verdict и не сертификация остальных. Relationship admission в production отсутствует.

## EDGE-GATE SIMULATION

Existing gate: **6685 → 6685 UNRESOLVED_EDGE**; source rows **8146 → 8146**. Добавляется только relationship metadata на копии, все исходные fields/IDs сохраняются. Ничего не применяется.

## PERFORMANCE

590 pairs: total **437.364s**; exactPTS decoding **219.584s**; all content/neighbor/candidate comparisons **105.121s**, **5.613 target pairs/sec** амортизированно со всеми neighbors. RSS sampled50ms start **333803520B**, peak **482250752B**. ROI peak **3** objects/**260820B**, end **0**. Image writes **0**. Total включает pre/postflight hashing/global inventory; это не video RTF и не live pipeline throughput.

Comparison105.121s — отдельная измеренная стоимость content/neighbor/candidate calculations после ROI retrieval; парного production A/B run здесь нет, поэтому процент pipeline overhead не заявлен. Targeted tests17passed; independent artifact verification590pairs и14750alignment conservation checks PASS; strict integrity и frozen hashes PASS.

## SOURCE DATA CHANGED

NO.

## PRODUCTION CODE CHANGED

NO.

## VERDICT

**CONTENT_MAPPING_PARTIAL**: representation и decomposition проверены, но бинарный support equality и локальная neighbor envelope не сертифицируют physical identity. Новые candidate mappings — диагностические; порог production не выбран.

**Основная причина разрыва identity сейчас — MIXED (рабочая гипотеза): content alignment и residual raster differences.** Измерено снижение foreground MAE при1px компенсации; причинный эффект этой компенсации на registered tight+resize matcher не изолирован отдельным counterfactual. Поэтому 1px offset не объявляется доказанной первопричиной разрыва identity.
