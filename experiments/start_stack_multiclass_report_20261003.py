"""Compile audited multi-template diagnostic results, no pipeline edits."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'runs/start_stack_frame_ledger_wide_smoke14_20261003'
DEST=OUT/'multi_template_classification'
def read(name):return json.loads((DEST/name).read_text(encoding='utf-8'))
def rows(name):return [json.loads(s) for s in (DEST/name).read_text(encoding='utf-8').splitlines()]
def publish():
    hands=rows('hand_results.jsonl');misses=rows('six_miss_results.jsonl');receipt=read('tester_validation_receipt.json');gap=read('tester_margin_separation_diagnostic.json')
    text='''# Multi-template NCC: cards / Turn / Flop / River

Offline only. Same20hands,1224sourceframes,22083physical-row appearances. Production unchanged; absolute threshold0.95 preserved as a separate diagnostic flag. No relative margin threshold promoted.

## Method and saved data

All four templates use an untouched40x16 grayscale canvas containing only the rightmost class word. Dealer/Dealing pixels are excluded. The canvas contains each complete word plus background padding; the foreground ends bycolumn28. No resize, intensity correction, local T04 template, or threshold adjustment.

For every physical row, all four NCC scores are computed within that same row with a bounded horizontal shift. Winner/runner-up/margin therefore never compare words from different CHAT rows. Raw scores, shifts, row bounds, source frame/PTS, file hashes and pixel hashes are retained in `row_scores.jsonl`. `frame_scores.jsonl` contains1224frame records with all4scores and margins: independently bound current row for initial DEAL triples; otherwise a clearly labelled best-cards row diagnostic. A frame can contain multiple different classes simultaneously, so the row records remain authoritative.

Current-hand/fresh-row identity comes from independent offline review. Ranking a word does not establish a new hand. Repeated appearances are observations, not new triggers. Existing once-per-hand latch remains separately tested; this report's20relative classifications are not20production-admitted trigger events.

## HITS / MISSES / FALSE POSITIVES

HITS means the independently bound current DEAL row wins as `cards` in all3initial consecutive frames.

| Decision diagnostic | HITS | MISSES | False card row appearances | False source frames |
|---|---:|---:|---:|---:|
| Absolute cards score >=0.95 |14|6|0|0|
| Relative winner only, no margin gate |20|0|UNKNOWN total; >=204 proven|>=153 proven|
| Relative winner and margin >=0.05 |20|0|UNKNOWN total; >=204 proven|>=153 proven|
| Relative winner and margin >=0.10 |20|0|UNKNOWN total; >=204 proven|>=153 proven|
| Relative winner and margin >=0.20 |20|0|204|153|
| Relative winner and margin >=0.30 |14|6|0|0|

The204false row appearances come from9unique pixels across3hand windows: `izold23 posts Ante` on T03 and `irek008/ravvin2 posts Ante` on T04. These are classically outside all four dealing-word classes, yet a forced nearest-class classifier chooses `cards`. All129candidate hashes with margin>=0.20 were independently reviewed:120cards and9non-cards. All97hashes at>=0.30 are covered by that same review. All746row appearances passing the unchanged absolute0.95 criterion are covered by the same pixel truth with0false/0unreviewed (`absolute_95_pixel_check.json`).

Lower-margin totals remain explicitly UNKNOWN:1729/1207/579unique candidate hashes at margins0/.05/.10 were not exhaustively reviewed. Neither raw argmax nor a score margin is a calibrated probability or safe admission rule.

## Six previous misses

Each row is the first independently observed DEAL frame. Scores below are from the common40x16canvas; they differ from the previous30x16single-word experiment (for example T04's old~0.915 becomes~0.923). Min margin covers all3initialframes. In all six, Turn is runner-up; cards wins all3frames, while absolute0.95fails all3.

| Table / hand | Frame | cards | Turn | Flop | River | Margin first | Min margin triple |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for r in misses:
        f=r['initial_frames'][0];s=f['scores']
        text+=f"|{r['table']} /{r['hand_id']}|{f['frame']}|{s['cards']:.6f}|{s['Turn']:.6f}|{s['Flop']:.6f}|{s['River']:.6f}|{f['margin']:.6f}|{r['margin_min']:.6f}|\n"
    text+='''
T04both: current row physically exists at correct row/word position, but frozen T01 raster scores below absolute0.95. Four-class ranking retains a large cards-vs-Turn difference. T05both: the same absolute rejection with cards still highest; the two frame vectors are almost identical. T06both: lowest absolute cards scores (~0.910), yet margin remains~0.251. No row recovery, aliasing, or synthetic evidence is used to turn these observations into classifications. These results diagnose threshold rejection versus label ranking; they do not identify the upstream font/renderer/compression cause of raster variation.

## Is confident relative classification possible near0.915?

Within this sample, yes: correct class ranking survives the raster variation. But safe handling requires an UNKNOWN option for non-dealing rows and independently confirmed row/hand context.

'''
    text+=f"Highest reviewed non-cards margin: **{gap['verified_non_cards_max_margin']:.9f}**. Lowest margin among60fresh DEAL observations: **{gap['fresh_cards_min_margin']:.9f}**. Gap: **{gap['posthoc_initial_rows_separation_gap']:.9f}**. Lowest margin among all reviewed complete cards candidates: **{gap['all_reviewed_cards_candidates_min_margin']:.9f}**, still above the reviewed non-cards maximum.\n\n"
    text+='''The observed separating interval is a post-hoc result on these20hands. It suggests a margin gate could separate this sample without lowering absolute0.95, but no value was installed or promoted. Threshold selection and independent held-out validation remain separate work. The classification evidence supports further testing, not a production claim of20/20with0FP.

## All20hands

| Table / hand | Absolute0.95 triple | Relative cards triple | Min margin |
|---|---|---|---:|
'''
    for r in hands:text+=f"|{r['table']} /{r['hand_id']}|{'HIT' if r['absolute_cards_3_frames'] else 'MISS'}|{'HIT' if r['relative_cards_3_frames'] else 'MISS'}|{r['margin_min']:.6f}|\n"
    text+='''
## Negative controls and validation

Real observed Flop/Turn/River rows select their corresponding class, with margins0.280215/0.295092/0.280215. They are template-calibration sources, not independent held-out negatives. The Ante rows above are the decisive independent open-set counterexample.

Run exit0; scan23.3547s, NCC10.5648s. Five targeted tests passed. Eight registered source-file checks passed; every TABLE read also verifies its journal file hash. All22083four-score calculations were verified to be row-local. Independent review found no material contract issue within the diagnostic scope.

Evidence: `template_manifest.json`, `frame_scores.jsonl`, `row_scores.jsonl`, `hand_results.jsonl`, `six_miss_results.jsonl`, `tester_validation_receipt.json`, `tester_margin_separation_diagnostic.json`, reviewed high-margin rows/contact pages and `review.json`. Original detector evidence is preserved. Live row freshness, live speed/OCR savings and global false-positive rates are unproved.
'''
    path=OUT/'REPORT_MULTICLASS.md';path.write_text(text,encoding='utf-8')
    dump={'HITS_relative_current_rows':20,'MISSES_relative_current_rows':0,'HITS_absolute_95':14,'MISSES_absolute_95':6,'margin_020_false_physical_row_appearances':204,'margin_020_false_source_frames':153,'margin_030_HITS':14,'margin_030_MISSES':6,'margin_030_false_positives':0,'winner_only_false_positive_total':'UNKNOWN; >=204physicalrowappearances verified','absolute_threshold_unchanged':.95,'margin_threshold_promoted':False,'production_changed':False,'source_frames':1224,'physical_rows':22083,'six_misses_resolved_as_class_rank':6,'score_margin_is_probability':False}
    (DEST/'VERIFIED_RESULT.json').write_text(json.dumps(dump,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return path
if __name__=='__main__':print(publish())
