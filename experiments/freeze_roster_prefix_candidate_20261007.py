"""Register the FULL-only roster proof candidate without golden promotion."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'production_manifest.json'
m=json.loads(path.read_text('utf-8'))
additions={'modules':['parser_core/nickname_roster_binding.py'],
           'tests':['tests/test_nickname_roster_binding.py']}
for field,names in additions.items():
    for name in names:
        if name not in m[field]:m[field].append(name)
changed=['PRODUCTION.md','parser_core/seat_forced_binding.py','parser_core/canonical_events.py',
         'scripts/run_smoke14_ram.py','tests/fixtures/roster_prefix_smoke19.json',
         'tests/fixtures/roster_prefix_smoke19_live_v4.json']
changed+=sum(additions.values(),[])
unexpected=[p for p,sha in m['hashes'].items() if p not in changed and hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=sha]
if unexpected:raise ValueError(('UNRELATED_MANIFEST_CHANGE',unexpected))
for p in changed:m['hashes'][p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
m['full_hh_production']['roster_prefix_binding']={
    'rule':'UNIQUE_LITERAL_PREFIX_COMPLETE_ROSTER_V1',
    'authority':'SOURCE_BOUND_COMPLETE_ANTE_ROSTER_AND_UNIQUE_INTERSECTION_OF_LITERAL_LABEL_CONSTRAINTS',
    'raw_nicknames':'PRESERVED_WITHOUT_ALIASES',
    'quality':'NOT_PROVEN','promotion':'NOT_PROMOTED'}
path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n','utf-8')
print('Registered candidate hashes',len(m['hashes']))
