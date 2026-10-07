"""Snapshot before header-only changes; extract a documented prefix model input."""
import sys,json,hashlib,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import av,cv2,numpy as np
from parser_core.layout import load_fixed_layout
from parser_core.ram_first_stage import png_compatible_gray
from parser_core.chat_rows import physical_spans,padded_row_span
out=ROOT/'runs/header_admission_fix_20261003';out.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'production_manifest.json').read_text('utf-8'))
before=out/'before';before.mkdir(exist_ok=True)
hashes={}
for rel in list(manifest['hashes'])+['PRODUCTION.md','production_manifest.json']:
    target=before/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    hashes[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
(before/'hashes.json').write_text(json.dumps(hashes,indent=2),'utf-8')
for args,name in [(['status','--short'],'git_status.txt'),(['diff','--binary'],'worktree.patch'),(['diff','--cached','--binary'],'staged.patch')]:
    (before/name).write_bytes(subprocess.run(['git',*args],cwd=ROOT,capture_output=True).stdout)
layout=load_fixed_layout(ROOT/'configs/nine_table_layout.json',ROOT/'configs/layout_847x404.json',ROOT/'configs/chat_log_bottom_rois.json')
with av.open(str(ROOT/'video/smoke_17.mp4')) as source:
    for fid,frame in enumerate(source.decode(video=0)):
        if fid!=181:continue
        rgb=frame.to_ndarray(format='rgb24');x1,y1,x2,y2=layout.slots[0].chat_box;gray=png_compatible_gray(rgb[y1:y2,x1:x2])
        spans=physical_spans(gray)
        for index,name in [(4,'hand'),(6,'hash')]:
            a,b=padded_row_span(spans,index,len(gray));pixels=gray[a:b].copy()
            cv2.imencode('.png',pixels)[1].tofile(str(out/(name+'_source.png')))
        break
