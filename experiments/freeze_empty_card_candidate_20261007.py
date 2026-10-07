"""Register reviewed FULL-only source evidence; never a golden promotion."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'production_manifest.json'
raw=path.read_bytes();m=json.loads(raw)
additions={
    'modules':['parser_core/empty_seat_evidence.py','parser_core/table_card_back_evidence.py'],
    'tests':['tests/test_empty_seat_evidence.py','tests/test_table_card_back_evidence.py'],
    'allowed_dependencies':['models/empty_seat_buttons.json','models/table_card_back_supplement.json']}
changed=['PRODUCTION.md','parser_core/start_stack_live.py','parser_core/seat_forced_binding.py',
    'scripts/run_smoke14_ram.py','parser_core/seat_occupancy.py','parser_core/nickname_roster_binding.py',
    'tests/test_start_stack_live.py','tests/test_seat_occupancy.py','tests/test_nickname_roster_binding.py',
    'tests/fixtures/roster_prefix_smoke19_live_v4.json','tests/fixtures/empty_button_smoke19.json',
    'tests/fixtures/table_card_back_smoke19.json']
changed += [p.relative_to(ROOT).as_posix() for p in (ROOT/'tests/fixtures/native_empty_deal').iterdir() if p.is_file()]
changed += sum(additions.values(),[])
unexpected=[p for p,h in m['hashes'].items() if p not in changed and hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
if unexpected:raise ValueError(('UNRELATED_MANIFEST_CHANGE',unexpected))
baseline=ROOT/'runs/full_hh_production_20261006/empty_button_tests_v2/before_registration_manifest.json'
if not baseline.exists():baseline.write_bytes(raw)
for field,names in additions.items():
    for name in names:
        if name not in m[field]:m[field].append(name)
for p in changed:m['hashes'][p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
m['full_hh_production']['native_empty_card_completion']={
    'rule':'COMPLETE_EMPTY_BUTTON_AND_FROZEN_CARDS_V1',
    'supplement':'POSITIVE_NATIVE_CARD_BACK_SUPPLEMENT_V1; ONLY_ABSENT_FROZEN_TEMPLATE',
    'authority':'THREE_JOINT_SOURCE_FRAMES; CURRENT_OCCUPANCY_ONLY; NO_ACTOR_AUTHORITY',
    'frozen_model_unchanged':True,'quality':'NOT_PROVEN','promotion':'NOT_PROMOTED'}
path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n','utf-8')
print('Registered reviewed candidate hashes',len(m['hashes']))
