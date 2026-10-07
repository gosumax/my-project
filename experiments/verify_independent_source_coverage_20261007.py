"""Frame-by-frame pixel audit independent of parser row admission and segments."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import zlib

import av
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from parser_core.layout import load_fixed_layout
from parser_core.ram_first_stage import png_compatible_gray


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=10100)
    args=parser.parse_args()
    if not 1<=args.frames<=10100:raise ValueError('BOUNDED_FRAME_LIMIT')
    args.output.mkdir(parents=True,exist_ok=False)
    audit=ROOT/'runs/chat_sequence_audit_20261006/source_reconstruction/source_pixels.sqlite'
    db=sqlite3.connect(audit.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    old={t:{} for t in range(1,10)}
    for r in db.execute('select table_id,frame,pts,time_base,sha256,height,width from chats order by table_id,frame'):
        old[r['table_id']][r['frame']]=dict(r)
    layout=load_fixed_layout(ROOT/'configs/nine_table_layout.json',ROOT/'configs/layout_847x404.json',ROOT/'configs/chat_log_bottom_rois.json')
    new=sqlite3.connect(args.output/'unrepresented_source_pixels.sqlite')
    new.execute('create table chats(table_id integer,frame integer,pts integer,time_base text,sha256 text,height integer,width integer,gray_zlib blob,primary key(table_id,frame))')
    started=time.perf_counter();checks=0;equal=0;changed=0;at_saved=0;saved_match=0;missing=0
    previous={};nearest={};previous_unrepresented={};per_table={t:dict(checks=0,equal_saved_projection=0,unrepresented=0) for t in old}
    video=ROOT/'video/smoke_19.mp4'
    with (args.output/'per_frame_hashes.jsonl').open('w',encoding='utf-8') as ledger,av.open(str(video)) as container:
        stream=container.streams.video[0];stream.thread_type='FRAME';stream.codec_context.thread_count=3
        decoded=0
        for frame_id,frame in enumerate(container.decode(video=0)):
            if frame_id>=args.frames:break
            rgb=frame.to_ndarray(format='rgb24');tb=[frame.time_base.numerator,frame.time_base.denominator]
            if rgb.shape!=(layout.source_height,layout.source_width,3):raise ValueError('SOURCE_VIDEO_GEOMETRY_CONFLICT')
            for slot in layout.slots:
                t=slot.slot_id;x,y,X,Y=slot.chat_box
                gray=png_compatible_gray(rgb[y:Y,x:X]);digest=hashlib.sha256(gray.tobytes()).hexdigest()
                reference=old[t].get(frame_id)
                if reference is not None:
                    nearest[t]=reference;at_saved+=1
                    exact=(reference['sha256']==digest and [reference['height'],reference['width']]==list(gray.shape) and
                           reference['pts']==frame.pts and json.loads(reference['time_base'])==tb)
                    saved_match+=int(exact)
                reference=nearest.get(t)
                match=bool(reference and reference['sha256']==digest and [reference['height'],reference['width']]==list(gray.shape))
                same_previous=previous.get(t)==digest;changed+=int(not same_previous)
                row=dict(table_id=t,source_frame=frame_id,source_pts=frame.pts,time_base=tb,roi=list(slot.chat_box),
                    pixel_sha256=digest,shape=list(gray.shape),same_as_previous_frame=same_previous,
                    nearest_saved_frame=reference['frame'] if reference else None,equals_saved_pixels=match)
                ledger.write(json.dumps(row)+'\n')
                if not match:
                    missing+=1
                    if previous_unrepresented.get(t)!=digest:
                        new.execute('insert into chats values(?,?,?,?,?,?,?,?)',(t,frame_id,frame.pts,json.dumps(tb),digest,*gray.shape,zlib.compress(gray.tobytes())))
                    previous_unrepresented[t]=digest
                else:previous_unrepresented.pop(t,None)
                checks+=1;equal+=int(match);previous[t]=digest
                per_table[t]['checks']+=1;per_table[t]['equal_saved_projection']+=int(match);per_table[t]['unrepresented']+=int(not match)
            decoded=frame_id+1
            if decoded%500==0:
                new.commit();ledger.flush()
                progress=dict(frames=decoded,checks=checks,equal_saved_projection=equal,unrepresented=missing,seconds=time.perf_counter()-started)
                (args.output/'progress.json').write_text(json.dumps(progress),'utf-8')
                print(json.dumps(progress),flush=True)
    new.commit();new_count=new.execute('select count(*) from chats').fetchone()[0];new.close();db.close()
    result=dict(scope='INDEPENDENT_SOURCE_PIXEL_COVERAGE_NOT_SEMANTIC_EVENT_GROUND_TRUTH',
        frames=decoded,requested_frames=args.frames,table_frame_checks=checks,
        expected_checks=args.frames*9,exact_changed_source_count=changed,
        saved_source_frames_encountered=at_saved,saved_source_frame_pts_hash_matches=saved_match,
        per_frame_equal_nearest_saved=equal,unrepresented_frames=missing,new_changed_views=new_count,
        saved_pixel_coverage_proven=decoded==10100 and checks==90900 and equal==checks and saved_match==at_saved,
        semantic_universe_proven=False,per_table=per_table,seconds=time.perf_counter()-started,
        video_sha256=hashlib.file_digest(video.open('rb'),'sha256').hexdigest(),
        source_database_sha256=hashlib.file_digest(audit.open('rb'),'sha256').hexdigest(),
        implementation_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ('parser_core/layout.py','parser_core/ram_first_stage.py','configs/layout_847x404.json','configs/nine_table_layout.json','configs/chat_log_bottom_rois.json')})
    (args.output/'summary.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
