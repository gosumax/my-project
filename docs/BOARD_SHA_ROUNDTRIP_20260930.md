# BOARD SHA round-trip: saved Smoke14 replay

## Identity contract

`parser_core/visual_crops.py` hashes the exact decoded RGB pixels of a BOARD
crop as `board_sha`; the PNG bytes retain a separate `crop_sha256`. A BOARD
record also carries `board_evidence_id`, table/session/frame/row, street, ROI,
source table and video hashes. The live VL request carries both hashes and
`board_evidence_id`. The worker verifies the input file and pixel hash, then
echoes all three identity fields. The client and association code check the
echo before parsing cards or projecting to the physical timeline row. An
identical image at another frame has its own evidence ID. Repeated IDs and
replayed response IDs are rejected.

The old neighboring chat tile remains separate evidence. It cannot project a
BOARD into the timeline or Excel. HAND_ID and FLOP/TURN/RIVER continuity are
checked after crop/response identity. An explicit NEW_HAND marker clears the
active HAND_ID.

## Saved evidence result

Source: `runs/smoke14_parallel_vl_full_20260929`. The original 106 BOARD
crop files all exist and match their stored PNG SHA. None of their 106 legacy
responses contains an echoed BOARD SHA or evidence ID; none contains a strict
full rank-and-suit card sequence. Therefore those responses cannot prove the
new round-trip and cannot be accepted by replay.

Five BOARD crops from the earlier `NO_VALID_ADJACENT_CARD_TILE` group were
resent to the current VL model using the new request contract. The raw
responses are in `runs/board_sha_vl_probe_v2_20260930.jsonl`. Four returned
`OK` with verified SHA/evidence ID and unreadable card sequences. One returned
`TOKEN_LIMIT`. No response supplied a complete card sequence. This is a
bounded model probe, not an accuracy measurement.

Replay with those five responses:

`runs/smoke14_board_sha_replay_v3_20260930/board_replay_summary.json`

| Measure | Earlier neighbor-based replay | Strict BOARD replay |
|---|---:|---:|
| BOARD evidence | 106 | 106 |
| Accepted BOARD | 41 | 0 |
| Unresolved BOARD | 65 | 106 |
| BOARD no valid neighboring tile | 60 | 0 |
| BOARD response SHA/evidence mismatch | not checked | 101 legacy responses |
| New SHA-bound response but cards unreadable | not applicable | 4 |
| New response unavailable (`TOKEN_LIMIT`) | not applicable | 1 |

The 41 previous accepted readings came from adjacent chat row tiles, not from
the exact BOARD crop. Their removal is intentional fail-closed behavior under
the new identity requirement. No BOARD card is accepted from matching text or
a neighboring tile alone.

The replay preserved the original 2,691 physical rows, their source identity,
frame and row order, raw OCR and pixel hashes. The generated XLSX has 2,691
data rows on `ALL_EVENTS`, with `board_evidence_id` and `board_sha` columns.
Both columns are empty on the physical rows because no BOARD passed all gates.
There are no five recovered accepted examples to prove; the remaining work is
to obtain exact BOARD crop responses with complete ranks and suits, then run
the same replay and inspect timeline and workbook cells.

This evidence proves the identity mechanism on four returned responses and
row parity on saved Smoke14. It does not prove an end-to-end accepted BOARD
round-trip or improved BOARD coverage. No new full-video run was performed.
