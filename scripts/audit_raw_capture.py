"""Read-only completeness, provenance and PNG integrity audit for a RAW capture."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def audit(run: Path, first: int, last: int, expected_rois: tuple[str,...]) -> dict:
    run=run.resolve(strict=True)
    with sqlite3.connect((run/'journal.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        frames=db.execute('SELECT frame_id,source_pts,time_base_num,time_base_den FROM frames ORDER BY frame_id').fetchall()
        if [row[0] for row in frames]!=list(range(first,last+1)):
            raise ValueError('Missing, repeated or out-of-range frame IDs')
        pts=[row[1] for row in frames]
        if any(x is None for x in pts) or any(b<=a for a,b in zip(pts,pts[1:])):
            raise ValueError('PTS missing or non-monotonic')
        rates={row[2:] for row in frames}
        if len(rates)!=1:
            raise ValueError('Frame time base changed')
        observations=db.execute('''SELECT frame_id,roi_type,crop_path,crop_sha256,bbox_json
            FROM observations ORDER BY frame_id,roi_type''').fetchall()
        by_frame=Counter(frame for frame,*_ in observations)
        if len(observations)!=(last-first+1)*len(expected_rois) or any(by_frame[frame]!=len(expected_rois) for frame in range(first,last+1)):
            raise ValueError('Incomplete ROI set')
        kinds={frame:set() for frame in range(first,last+1)}
        for frame,kind,*_ in observations:
            if kind in kinds[frame]:
                raise ValueError('Repeated ROI')
            kinds[frame].add(kind)
        if any(values!=set(expected_rois) for values in kinds.values()):
            raise ValueError('Missing ROI type')
        if db.execute('PRAGMA integrity_check').fetchone()!=('ok',) or db.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('SQLite integrity failure')
        derived={name:db.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0] for name in
                 ('recognitions','physical_rows','row_decisions','messages','event_revisions','hand_revisions','exports')}
        if any(derived.values()):
            raise ValueError('RAW capture contains derived predictions')
        source=db.execute('SELECT path,sha256,size_bytes,state FROM sources').fetchone()
    image_sizes=Counter()
    for frame,kind,relative,digest,bbox_json in observations:
        path=(run/relative).resolve(strict=True)
        if not path.is_relative_to(run) or sha256(path)!=digest:
            raise ValueError(f'PNG hash differs at {frame}/{kind}')
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width,height=image.size
        bbox=json.loads(bbox_json)
        if [width,height]!=[bbox[2]-bbox[0],bbox[3]-bbox[1]]:
            raise ValueError(f'PNG size differs from ROI at {frame}/{kind}')
        image_sizes[f'{kind}:{width}x{height}']+=1
    source_path=Path(source[0]).resolve(strict=True)
    if source_path.stat().st_size!=source[2] or sha256(source_path)!=source[1]:
        raise ValueError('Source video hash/size differs')
    return dict(run=str(run),frame_count=len(frames),first_frame=first,last_frame=last,
                first_pts=pts[0],last_pts=pts[-1],time_base=list(next(iter(rates))),
                strictly_increasing_pts=True,roi_count=len(observations),
                png_hash_size_and_decode_verified=len(observations),image_sizes=dict(image_sizes),
                source_sha256=source[1],source_size=source[2],source_state=source[3],
                derived_counts=derived,sqlite_integrity='ok',status='RAW_EVIDENCE_AUDITED')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--first',type=int,required=True)
    parser.add_argument('--last',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.run,args.first,args.last,('chat','table'))
    output=args.output.resolve()
    if not output.is_relative_to(Path(__file__).resolve().parents[1]) or output.exists():
        parser.error('Select new report path inside workspace')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('frame_count','roi_count','png_hash_size_and_decode_verified','sqlite_integrity','status')},ensure_ascii=False))


if __name__=='__main__':
    main()
