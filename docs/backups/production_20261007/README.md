# Production backup - 2026-10-07

Requested local checkpoint before resuming the FULL_HH_PRODUCTION task.
GitHub origin: https://github.com/gosumax/my-project.git. Remote upload was not
completed: HTTPS/Git connections timed out. The user authorized a local Git
commit/bundle and resumption of the original HH task.

The production definition is [PRODUCTION.md](../../../PRODUCTION.md) and
[production_manifest.json](../../../production_manifest.json).
Snapshot is an integration candidate, NOT_PROMOTED; HH>=95% is NOT_PROVEN.
Official short_v1: 2101 frames, all33 modules ON, short18/18 PASS, errors0,
physical496/OCR496/timeline520, PNG0, certifiedHH0. Code changed after this run.
Latest focused regressions:118 passed. Earlier full pytest:1454 passed/178 failed;
the same178 failures reproduced with the saved baseline. A new full suite and
fresh short_v2/native10100 remain necessary. Offline v5 reanalysis:5 complete
contexts,4 eligible reconstructions,0 proven START_STACK joins; not a video run.
Saved FirstStage ON replay was interrupted at user request (T07 frame8012);
it is partial evidence, not a full704/legacy1461 acceptance.

Included: project code, configs, tests/fixtures, models, documentation/provenance,
and434 small runtime dependencies otherwise ignored under runs/ (inventory below).
Existing training-template PNGs are runtime assets; production PNG output stays0.
Not included: input videos, full captures/OCR/SQLite run artifacts, generated
reports/logs, local .codex settings/caches, external Python/OCR/Node installations.
These remain locally available. Refer to dependency_inventory.json for exact
runtime file hashes and external prerequisites. Configs currently contain absolute
Windows paths: restoration at another location needs explicit path adaptation.
No runtime/config path or recognition algorithm was changed for this backup.

.gitattributes disables text newline conversion so Git preserves hash-pinned bytes.
Source integrity passed before this checkpoint. The Git bundle is created outside
versioned paths after committing and includes the complete local Git history.
