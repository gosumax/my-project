"""Register a reviewed source chronology correction; no golden promotion."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'production_manifest.json';raw=path.read_bytes();m=json.loads(raw)
changed=['PRODUCTION.md','parser_core/canonical_events.py','parser_core/poker_math_engine.py',
    'tests/test_forced_snapshot_chronology.py','tests/fixtures/forced_chronology_short_v6.json.gz']
unexpected=[p for p,h in m['hashes'].items() if p not in changed and hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
if unexpected:raise ValueError(('UNRELATED_MANIFEST_CHANGE',unexpected))
baseline=ROOT/'runs/full_hh_production_20261006/forced_chronology_tests_v1/before_registration_manifest.json'
if not baseline.exists():baseline.write_bytes(raw)
if changed[3] not in m['tests']:m['tests'].append(changed[3])
for p in changed:m['hashes'][p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
m['full_hh_production']['forced_snapshot_chronology']={
    'rule':'COMPLETE_FORCED_PREFIX_OBSERVED_BY_FIRST_POSTFORCED_NUMBER_V1',
    'authority':'REPRODUCED_ROSTER_AND_FIRST_NUMBER; FRAME_PTS_ORDER; UNCHANGED_EPOCH',
    'raw_vision_fact':'RETAINED','quality':'NOT_PROVEN','promotion':'NOT_PROMOTED'}
path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n','utf-8')
print('Registered reviewed candidate hashes',len(m['hashes']))
