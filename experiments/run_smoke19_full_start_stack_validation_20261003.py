"""Full official Smoke19 ON validation wrapper; no production changes."""
import hashlib,json,os,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/smoke19_full_start_stack_validation_20261003'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def main():
    meta=json.loads((OUT/'input_metadata.json').read_text('utf-8'))
    assert meta['actual_decoded_frames']>0
    assert Path(meta['path']).stat().st_size==meta['size_bytes']
    with Path(meta['path']).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==meta['sha256']
    check=subprocess.run([sys.executable,'scripts/check_production_integrity.py'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'launch_integrity.log').write_text(check.stdout+check.stderr,'utf-8');assert check.returncode==0
    flags=json.loads((ROOT/'configs/dealing_start_stack.json').read_text('utf-8'))
    assert flags['enable_dealing_cards_hand_state_trigger'] is False and flags['enable_start_stack_capture'] is False
    manifest=json.loads((ROOT/'production_manifest.json').read_text('utf-8'));prior=json.loads((ROOT/'runs/historical_forced_binding_integration_20261003/acceptance_candidate/frozen_hashes.json').read_text('utf-8'))
    assert all(sha(ROOT/n)==h for n,h in prior.items()),'SMOKE17_FROZEN_BYTES_CHANGED'
    candidate=OUT/'acceptance_candidate';candidate.mkdir(exist_ok=False);hashes={}
    for name in list(manifest['hashes'])+['production_manifest.json']:
        src=ROOT/name;dst=candidate/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);hashes[name]=sha(src)
    (candidate/'frozen_hashes.json').write_text(json.dumps(hashes,indent=2),'utf-8')
    old=json.loads((ROOT/'runs/historical_forced_binding_integration_20261003/acceptance_ON_exit_receipt.json').read_text('utf-8'))
    command=old['command'].copy();command[0]=sys.executable;command[command.index('--video')+1]=meta['relative_path'];command[command.index('--frames')+1]=str(meta['actual_decoded_frames']);command[command.index('--output')+1]=str((OUT/'acceptance_ON').relative_to(ROOT));assert not (OUT/'acceptance_ON').exists()
    started=utc();clock=time.perf_counter();receipt=dict(scope='FULL_OFFICIAL_SMOKE19_ON_NO_PROMOTION',command=command,input=meta,started_utc=started,wrapper_pid=os.getpid(),default_flags=flags,manifest_sha256=hashes['production_manifest.json'])
    (OUT/'launch_receipt.json').write_text(json.dumps(receipt,indent=2),'utf-8')
    with (OUT/'stdout.log').open('w',encoding='utf-8') as stdout,(OUT/'stderr.log').open('w',encoding='utf-8') as stderr:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr);receipt['production_pid']=child.pid;(OUT/'launch_receipt.json').write_text(json.dumps(receipt,indent=2),'utf-8');exitcode=child.wait()
    changed=[n for n,h in hashes.items() if sha(ROOT/n)!=h]
    receipt.update(finished_utc=utc(),exit_code=exitcode,wrapper_wall_seconds=time.perf_counter()-clock,registered_hashes_unchanged=not changed,changed_registered_files=changed)
    final=OUT/'acceptance_ON/final_summary.json'
    if final.exists():
        data=json.loads(final.read_text('utf-8'));receipt['pipeline_metrics']={k:data.get(k,'NOT_MEASURED') for k in ['status','frames','wall_seconds','rtf','peak_pipeline_rss_mb','errors','retry_count','png_files','png_bytes']}
    else:receipt['pipeline_metrics']='NOT_MEASURED'
    (OUT/'exit_receipt.json').write_text(json.dumps(receipt,indent=2),'utf-8');return exitcode
if __name__=='__main__':sys.exit(main())
