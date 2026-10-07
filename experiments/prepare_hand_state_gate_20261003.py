from pathlib import Path
import json,hashlib,shutil,subprocess
root=Path(__file__).resolve().parents[1];out=root/'runs/hand_state_marker_gate_20261003/before';out.mkdir(parents=True,exist_ok=False)
m=json.loads((root/'production_manifest.json').read_text('utf-8'));hashes={}
for rel in list(m['hashes'])+['production_manifest.json']:
    dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,dst)
    hashes[rel]=hashlib.sha256((root/rel).read_bytes()).hexdigest()
(out/'hashes.json').write_text(json.dumps(hashes,indent=2),'utf-8')
for args,name in [(['status','--short'],'git_status.txt'),(['diff','--binary'],'worktree.patch'),(['diff','--cached','--binary'],'staged.patch')]:
    (out/name).write_bytes(subprocess.run(['git',*args],cwd=root,capture_output=True).stdout)
