"""Explicit CHAT promotion snapshots/candidate registration, never runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[1]
CHECKS=ROOT/'runs/chat_tracker_promotion_20261003_checks'

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def snapshot():
    target=ROOT/'provenance/experiments/chat_tracker_promotion_20261003/before'
    target.mkdir(parents=True,exist_ok=False)
    m=json.loads((ROOT/'production_manifest.json').read_text('utf-8'))
    hashes={p:sha(ROOT/p) for p in m['hashes']}
    if hashes!=m['hashes']:raise ValueError('SOURCE_CHANGED_BEFORE_PROMOTION_SNAPSHOT')
    for p in list(hashes)+['production_manifest.json']:
        dst=target/'files'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dst)
    engines={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'parser_core').rglob('poker*.py')}
    engines.update({p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'parser_core/poker_reconstruction').rglob('*.py')})
    (target/'source_before.json').write_text(json.dumps(hashes,indent=2),'utf-8')
    (target/'poker_engine_before.json').write_text(json.dumps(engines,indent=2),'utf-8')
    subprocess.run(['git','diff','--binary','--output='+str(target/'tracked_dirty.patch')],cwd=ROOT,check=True)
    (target/'git_status.txt').write_bytes(subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT))
    print(str(target))

def candidate():
    CHECKS.mkdir(parents=True,exist_ok=True)
    m=json.loads((ROOT/'production_manifest.json').read_text('utf-8'))
    m['production_version']='production-candidate-chat-tracker-v4-20261003'
    m['modules']=sorted(set(m['modules'])|{'parser_core/chat_tracker.py'})
    m['tests']=sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'tests').glob('test_*.py'))
    m['stages']['row_tracking']={'file':'parser_core/chat_tracker.py','symbol':'AdjacentChatTracker.process'}
    m['chat_tracker']={'version':'adjacent-safe-v4','row_step_px':20.17647058823529,
        'score_threshold':.90,'margin_threshold':.02,'zone_min_overlap_px':48,'tolerance_px':2,
        'frozen_reference':'outputs/vertical_chat_shift_20261002/adjacent_safe_v4',
        'identity_safety':'PROVISIONAL/UNKNOWN lineage retained; no global physical identity certification',
        'initial_visible_OCR':False,'non_GLOBAL_OCR':False}
    paths=set(m['hashes'])|set(m['modules'])|set(m['tests'])
    m['hashes']={p:sha(ROOT/p) for p in sorted(paths)}
    (ROOT/'production_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n','utf-8')
    shutil.copy2(ROOT/'production_manifest.json',CHECKS/'candidate_manifest.json')
    print(m['production_version'])

def engine_receipt():
    target=ROOT/'provenance/experiments/chat_tracker_promotion_20261003/before'
    before=json.loads((target/'poker_engine_before.json').read_text('utf-8'))
    changed=[p for p,h in before.items() if sha(ROOT/p)!=h]
    result={'poker_engine_files_checked':len(before),'poker_engine_changed_files':changed}
    (CHECKS/'poker_engine_integrity.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result));assert not changed

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['snapshot','candidate','engines']);a=p.parse_args()
    {'snapshot':snapshot,'candidate':candidate,'engines':engine_receipt}[a.mode]()
