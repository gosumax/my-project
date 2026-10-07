"""One-time source-preserving extraction; not a runtime entrypoint."""
import ast
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
before = ROOT / 'runs/stack_pipeline_integration_20261005/before'
before.mkdir(parents=True, exist_ok=False)
manifest = json.loads((ROOT / 'production_manifest.json').read_text('utf-8'))
for name in list(manifest['hashes']) + ['production_manifest.json', 'PRODUCTION.md']:
    target = before / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, target)

def extract(path, names):
    source = (ROOT / path).read_text('utf-8')
    tree = ast.parse(source)
    return '\n\n'.join(ast.get_source_segment(source, n) for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names)

header = '''"""Frozen STACK algorithms extracted without threshold or recognition changes."""
import base64, hashlib, json, re, subprocess, sys, threading, queue, time
from collections import Counter, deque
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
MASK_CONFIG = ROOT / 'configs/stack_sitout_mask.json'
VERSION = 'sitout-fullword-mask-v1'
'''
chunks = [extract('experiments/stack_visual_tracker/tracker.py', {'metrics','different','Track'}),
    extract('experiments/stack_digit_gate/gate.py', {'segments','normalized','DigitGate'}),
    extract('experiments/stack_numeric_timeline/sitout_mask.py', {'SitOutMask','excluding_gate_class'}),
    extract('experiments/stack_numeric_timeline/timeline.py', {'RawTiny','Timeline','numeric'})]
# Only module resolution changes: use the identical class already extracted above.
chunks[2] = chunks[2].replace("    import sys\n    sys.path.insert(0,str(ROOT/'experiments/stack_digit_gate'))\n    from gate import DigitGate\n", '')
(ROOT / 'parser_core/stack_timeline_algorithms.py').write_text(header + '\n\n'.join(chunks) + '\n', 'utf-8')
shutil.copyfile(ROOT / 'experiments/stack_numeric_timeline/raw_worker.py', ROOT / 'parser_core/raw_worker.py')
for source, target in [
    ('runs/stack_digit_gate_20261004_dataset_v1/gate_config_dev.json','configs/stack_digit_gate.json'),
    ('runs/stack_roi_refine_20261004_v3/layout_stack_refined.json','configs/stack_roi_refined.json'),
    ('experiments/stack_numeric_timeline/sitout_mask.json','configs/stack_sitout_mask.json')]:
    shutil.copyfile(ROOT / source, ROOT / target)
receipt = dict(source_preserving_extraction=True, baseline=str(before),
    configuration_changes=False, numeric_threshold=.95,
    known_gate_false_admission='Smoke19 frame96 T06 seat2 Sit out; retained unchanged')
(before / 'integration_receipt.json').write_text(json.dumps(receipt, indent=2), 'utf-8')
