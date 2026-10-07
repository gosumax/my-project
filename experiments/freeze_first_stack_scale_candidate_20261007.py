"""Exact registration of a reviewed FULL-only same-source OCR supplement."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'production_manifest.json';raw=path.read_bytes();m=json.loads(raw)
additions={'modules':['parser_core/first_stack_ocr.py'],'tests':['tests/test_first_stack_ocr.py']}
changed=['PRODUCTION.md','parser_core/start_stack_live.py','parser_core/start_stack_visibility.py',
         'tests/fixtures/first_stack_scaled_smoke19.json']+sum(additions.values(),[])
unexpected=[p for p,h in m['hashes'].items() if p not in changed and hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
if unexpected:raise ValueError(('UNRELATED_MANIFEST_CHANGE',unexpected))
baseline=ROOT/'runs/full_hh_production_20261006/stack_representation_v1/before_registration_manifest.json'
if not baseline.exists():baseline.write_bytes(raw)
for field,names in additions.items():
    for name in names:
        if name not in m[field]:m[field].append(name)
for p in changed:m['hashes'][p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
m['full_hh_production']['same_source_stack_scale']={
    'rule':'SAME_SOURCE_NEAREST2_NUMERIC_AGREEMENT_V1','raw_primary':'IMMUTABLE',
    'authority':'SAME_VALUE_AT_UNCHANGED_CONFIDENCE; EXACT_SOURCE_AND_FIRST_VISIBLE_TRIPLE_REQUIRED',
    'quality':'NOT_PROVEN','promotion':'NOT_PROMOTED'}
path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n','utf-8')
print('Registered reviewed candidate hashes',len(m['hashes']))
