"""Explicit OFF candidate registration; no golden freeze/promotion."""
from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[1]
path=root/'production_manifest.json';m=json.loads(path.read_text('utf-8'))
m['production_version']='production-candidate-header-admission-off-20261003'
for key,rel in [('modules','parser_core/hand_header_admission.py'),
                ('tests','tests/test_hand_header_admission.py')]:
    if (root/rel).is_file() and rel not in m[key]:m[key].append(rel)
for rel in ['models/hand_header_prefix/Hand_hash.png','models/hand_header_prefix/hash.png','models/hand_header_prefix/model.json']:
    if rel not in m['allowed_dependencies']:m['allowed_dependencies'].append(rel)
m['stages']['hand_header_admission']={'file':'parser_core/hand_header_admission.py','symbol':'HandHeaderAdmission.observe_frame'}
if 'hand_header_ocr.jsonl' not in m['outputs']:m['outputs'].append('hand_header_ocr.jsonl')
for rel in m['modules']+m['configs']+m['tests']+m['allowed_dependencies']+['PRODUCTION.md']:
    m['hashes'][rel]=hashlib.sha256((root/rel).read_bytes()).hexdigest()
m['dealing_start_stack']['header_admission_baseline']='runs/header_admission_fix_20261003/before'
path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n','utf-8')
