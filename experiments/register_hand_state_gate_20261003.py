from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1];path=root/'production_manifest.json';m=json.loads(path.read_text('utf-8'))
m['production_version']='production-candidate-hand-state-gate-off-20261003'
for key,rel in [('modules','parser_core/hand_state_marker_gate.py'),('tests','tests/test_hand_state_marker_gate.py')]:
    if (root/rel).is_file() and rel not in m[key]:m[key].append(rel)
m['stages']['hand_state_marker_gate']={'file':'parser_core/hand_state_marker_gate.py','symbol':'HandStateMarkerGate.select'}
for rel in m['modules']+m['configs']+m['tests']+m['allowed_dependencies']+['PRODUCTION.md']:
    m['hashes'][rel]=hashlib.sha256((root/rel).read_bytes()).hexdigest()
m['dealing_start_stack']['marker_gate']='HAND_STATE; no CHAT lineage requirement'
path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n','utf-8')
