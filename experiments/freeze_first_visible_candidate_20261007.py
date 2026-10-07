"""Register this authorized FULL-only candidate; never promote a golden build."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'production_manifest.json'
m=json.loads(path.read_text('utf-8'))
additions={'modules':['parser_core/start_stack_visibility.py'],
           'allowed_dependencies':['models/forced_stack_overlays.json'],
           'tests':['tests/test_start_stack_visibility.py']}
for field,names in additions.items():
    for name in names:
        if name not in m[field]:m[field].append(name)
changed=['PRODUCTION.md','parser_core/start_stack_live.py','parser_core/poker_math_engine.py',
         'parser_core/poker_reconstruction_layer.py','parser_core/stack_evidence.py',
         'parser_core/full_hh_profile.py','scripts/run_smoke14_ram.py',
         'tests/fixtures/first_visible_stack_smoke19.json']
changed+=sum(additions.values(),[])
unexpected=[p for p,sha in m['hashes'].items() if p not in changed and hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=sha]
if unexpected:raise ValueError(('UNRELATED_MANIFEST_CHANGE',unexpected))
for p in changed:m['hashes'][p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
m['full_hh_production']['first_visible_stack_contract']={
    'rule':'POKERDOM_FIRST_VISIBLE_POST_FORCED_V1',
    'authority':'USER_CLIENT_RULE_20261007; POSITIVE_FORCED_OVERLAY_FADE_AND_SOURCE_BOUND_FIRST_NUMERIC_TRIPLE',
    'source_join':'EXACT_RGB_REPRODUCES_DIRECT_AND_SQLITE_ROIS_GRAY_FRAME_PTS_SESSION',
    'timing':'OBSERVED_46_AND_51_FRAMES; NO_UNIVERSAL_FIXED_DELAY',
    'quality':'NOT_PROVEN', 'promotion':'NOT_PROMOTED'}
path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n','utf-8')
print('Registered candidate hashes',len(m['hashes']))
