"""Bounded official ON acceptance; preserves candidate bytes and full logs."""
import sys,json,hashlib,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/start_stack_live_path_20261003'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=json.loads((ROOT/'production_manifest.json').read_text('utf-8'))
    candidate=OUT/'connection_candidate';candidate.mkdir(exist_ok=False)
    hashes={}
    for name in list(manifest['hashes'])+['production_manifest.json','PRODUCTION.md']:
        p=ROOT/name;dest=candidate/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);hashes[name]=sha(p)
    (candidate/'frozen_hashes.json').write_text(json.dumps(hashes,indent=2),'utf-8')
    run=OUT/'connection_acceptance_ON'
    if run.exists():raise RuntimeError('FRESH_OUTPUT_REQUIRED')
    cmd=[sys.executable,'run_production.py','--video','video/smoke_17.mp4','--frames','5401','--output',str(run.relative_to(ROOT)),
      '--ocr-workers','2','--visual-workers','2','--card-workers','2','--node','C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe',
      '--enable-dealing-cards-hand-state-trigger','--enable-start-stack-capture','--dealing-start-stack-acceptance']
    started=time.perf_counter()
    with (OUT/'connection_acceptance_ON.stdout.log').open('w',encoding='utf-8') as stdout,(OUT/'connection_acceptance_ON.stderr.log').open('w',encoding='utf-8') as stderr:
        result=subprocess.run(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
    changed=[p for p,v in hashes.items() if sha(ROOT/p)!=v]
    receipt={'command':cmd,'exit_code':result.returncode,'wall_seconds':time.perf_counter()-started,
      'registered_hashes_unchanged':not changed,'changed_registered_files':changed,'scope':'OFFICIAL_ON_ACCEPTANCE_5401_FRAMES_NO_PROMOTION'}
    (OUT/'connection_acceptance_ON_exit_receipt.json').write_text(json.dumps(receipt,indent=2),'utf-8')
    print(json.dumps(receipt));return result.returncode
if __name__=='__main__':sys.exit(main())
