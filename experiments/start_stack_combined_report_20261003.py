"""Publish isolated experimental evidence without promoting production."""
import json,html
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'runs/start_stack_frame_ledger_wide_smoke14_20261003'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def publish():
    clean=read('clean_forced_window/summary.json');identity=read('seat_identity_reclassification/summary.json')
    ncc=read('ncc_dealing_cards_trigger/summary.json');cards=read('card_deal_delay/summary.json')
    text=f'''# START_STACK: offline Smoke14 evidence

20 reviewed hands, 9 tables, 156 occupied seat/hand opportunities. Production unchanged; manifest integrity PASS. This report supersedes the acceptance interpretation of earlier phase summaries, while retaining their raw evidence.

## Nearest future frame after 681

| Observation | Frame | Delay from 681 |
|---|---:|---:|
| First dealt cards in motion, manual pixel review | 682 | +1 |
| Two cardbacks visible at all 8 seats, still moving | 688 | +7 |
| All occupied card ROIs match settled reference, NCC >=0.95 | 692 | +11 |
| Three consecutive matches available, 692/693/694 | 694 | +13 |
| Earlier conservative stack/hand checkpoint | 728 | +47 |
| First strict post-checkpoint numeric triple in prior run | 729/730/731 | +48/+49/+50 |

At 30fps, +11 frames = 0.367 seconds and confirmation +13 = 0.433 seconds. The frame numbers refer to actual database observations, including reused PNG paths; filenames alone are not frame identity.

Across 20 hands, settled-reference triple starts occur +{cards['all_occupied_start_offset_min']}..+{cards['all_occupied_start_offset_max']} frames from the recorded CHAT deal observation (median {cards['all_occupied_start_offset_median']}); confirmation at the third frame is +9..+22. Nine calibration hands and eleven later same-slot holdouts all detected. This measures visual template alignment, not exact animation completion or an authoritative hand boundary. Calibration uses later reviewed reference crops; it is an offline diagnostic.

## Experimental rule

1. NCC `Dealing cards` starts pending RAM capture of nick/stack ROIs immediately. TABLE detection is the fallback when CHAT NCC misses.
2. Check card ROIs every source frame. The observed range suggests dense checking through at least +22 frames, allowing confirmation of a triple beginning at +20. Elapsed time only schedules checks; it never opens a window.
3. Require current new-hand evidence, forced bets, known occupied seats and visual dealt-pair confirmation. Three consecutive positive card observations confirm at the third frame. A stale CHAT line cannot reopen a hand/window or increment occupancy epoch.
4. Accept numeric stack only with confidence >=0.90 and 3 equal consecutive readings strictly after confirmation and strictly before the first global CALL/BET/RAISE/ALL_IN. CHECK/FOLD do not close.
5. START_STACK = displayed stack + actual ANTE + actual SB/BB. Never include later action/refund/win money. Any observed monetary action earlier than the checkpoint leaves the window closed.

There is no universal fixed +N-frame DEAL_COMPLETE rule. Card visibility also does not guarantee that an ANTE/SB/BB overlay has already revealed the numeric stack. Live causal confirmation and action-onset latency are not implemented or measured by this experiment.

## Clean forced-only sample

| Metric | Count |
|---|---:|
| HANDS WITH CLEAN WINDOW | {clean['hands_with_clean_window']} |
| SEATS OBSERVED | {clean['seats_observed']} |
| STABLE STACKS | {clean['stable_stacks']} |
| START_STACK RECOVERED | {clean['counts']['START_STACK_RECOVERED']} |
| NO_STACK_BEFORE_FIRST_ACTION | 2 |
| IDENTITY_UNKNOWN / FORCED_BET_UNKNOWN / TIMING_UNKNOWN / CONFLICT | 0 / 0 / 0 / 0 |
| Later monetary events used | 0 |

The two missing numbers are T03 seat4 SITOUT in both selected hands. Forced evidence is explicit manual TABLE/CHAT review with own-frame hashes and physical blind seats, not inferred from stack equality.

Verdict: 154 forced-only reconstructions are supported in the reviewed-checkpoint → first source-visible monetary CHAT interval. The earlier script label `START_STACK_FORCED_WINDOW_PROVEN_ON_SAMPLE` has this restricted meaning. It does not prove hidden monetary onset, the earliest stable stack after the newly found earlier card boundary, production readiness or complete hand reconstruction. CHECK/FOLD preserve the window; same-frame boundary ordering is rejected.

## Seat identity and row identity

Player IDs use table + seat + observed occupancy epoch. OCR nickname variants label the seat and never create an epoch. Epoch1 means first observed occupancy, not known historical occupant identity. Cross-hand gaps remain UNKNOWN; confirmed vacancy at T05 seat1 frame3693 is retained.

| Money-row identity class | Count |
|---|---:|
| PLAYER_RESOLVED_ROW_RESOLVED | 0 |
| PLAYER_RESOLVED_ROW_UNRESOLVED | 309 |
| PLAYER_UNRESOLVED | 0 |
| BOTH_UNRESOLVED | 22 |

Nonexclusive blockers: player binding22, physical row duplicate/lineage risk331, unknown grammar0, amount ambiguity13, money events sharing numeric observation frame16. Identical content or append geometry alone does not prove row lifecycle. None of the331 money messages is silently admitted as row-resolved. These later rows do not block the forced-only checkpoint calculation.

Full frame ledger remains NOT_PROVEN after seat reclassification: MULTI_STATE_EXACT0 / CONFLICT0 / LEDGER_UNKNOWN108 / TIMING_UNKNOWN48, usable observations0. Diagnostic24 exact and17 conflicting groups are excluded from acceptance. Cross-hand78 continuity opportunities remain UNKNOWN.

## CHAT NCC trigger

One fixed threshold0.95 and literal template were retained. Raw hits18/20; hits after recorded current-hand header14/20; pre-hand raw hits10. Both T04 hands miss. Actual trigger capture covers138 of154 baseline stable triples;16 lack trigger capture. Baseline recovery154 must not be reported as NCC recovery154.

Matching cost {ncc['opencv_match_ms_per_saved_table']:.3f} ms per saved TABLE image over {ncc['frames_scanned']} frames. Matching alone took {ncc['opencv_matching_seconds']:.3f}s; saved-file decode/hash is measured separately. The46..51-frame lead is to the conservative reviewed checkpoint, not earliest card completion or measured OCR delay. Live OCR wait reduction is NOT_MEASURED. Intermediate capture image files written:0.

## Provenance and checks

Off-by-one audit24 checks found report-stage mixing, no TABLE/ROI frame offset: original OCR728/3106 differs from fresh post-checkpoint triples729..731/3107..3109. Prior artifacts preserved in prior_report_evidence. All source pixels/OCR and UNKNOWN states retained.

18 targeted tests passed. The pre-checkpoint monetary-action reopening regression is fixed and independently reviewed. 154/154 numeric triples passed source hash, confidence, strict boundary and forced-only arithmetic checks; all4 frozen input hashes unchanged. Independent review has no remaining material findings.

See `clean_forced_window/tester_execution_validation.json`, `ncc_dealing_cards_trigger/tester_execution_validation.json`, `final_review.json`, `card_deal_delay/hand_results.jsonl`, `card_deal_delay/frame_scores.jsonl`, and per-phase summaries for numerical checks and raw provenance. No production promotion performed.
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    (OUT/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>START_STACK offline</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.5 Segoe UI}pre{white-space:pre-wrap}</style><pre>'+html.escape(text)+'</pre>',encoding='utf-8')
    return str(OUT/'REPORT.md')
if __name__=='__main__':print(publish())
