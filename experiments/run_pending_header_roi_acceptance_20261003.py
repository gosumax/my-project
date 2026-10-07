"""Fresh official bounded ON acceptance; frozen source receipt, no promotion."""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/deal_complete_fix_chat_review_20261003'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    candidate = OUT / 'acceptance_candidate'
    candidate.mkdir(exist_ok=False)
    manifest = json.loads((ROOT / 'production_manifest.json').read_text('utf-8'))
    hashes = {}
    for name in list(manifest['hashes']) + ['production_manifest.json']:
        src = ROOT / name
        dst = candidate / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        hashes[name] = sha(src)
    (candidate / 'frozen_hashes.json').write_text(json.dumps(hashes, indent=2), 'utf-8')
    prior = json.loads((ROOT / 'runs/seat_forced_binding_20261003/acceptance_ON_exit_receipt.json').read_text('utf-8'))
    command = prior['command'].copy()
    command[0] = sys.executable
    run = OUT / 'acceptance_ON'
    if run.exists():
        raise RuntimeError('FRESH_OUTPUT_REQUIRED')
    command[command.index('--output') + 1] = str(run.relative_to(ROOT))
    started = time.perf_counter()
    with (OUT / 'acceptance_ON.stdout.log').open('w', encoding='utf-8') as stdout, (OUT / 'acceptance_ON.stderr.log').open('w', encoding='utf-8') as stderr:
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    changed = [name for name, value in hashes.items() if sha(ROOT / name) != value]
    receipt = dict(command=command, exit_code=result.returncode, wall_seconds=time.perf_counter()-started,
                   registered_hashes_unchanged=not changed, changed_registered_files=changed,
                   manifest_sha256=hashes['production_manifest.json'], scope='OFFICIAL_ON_5401_PENDING_HEADER_ROI_NO_PROMOTION')
    (OUT / 'acceptance_ON_exit_receipt.json').write_text(json.dumps(receipt, indent=2), 'utf-8')
    print(json.dumps(receipt))
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
