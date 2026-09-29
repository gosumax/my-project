"""Build hash-verified contact sheets from a frozen changed-chat review queue."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queue', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-sheet', type=int, default=12)
    args = parser.parse_args()
    queue = args.queue.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or args.per_sheet < 1:
        parser.error('Choose a new output directory and positive sheet size')
    with (queue/'changed_chat_frames.csv').open(encoding='utf-8-sig',newline='') as stream:
        entries=list(csv.DictReader(stream))
    if not entries:
        parser.error('Empty changed-chat queue')
    output.mkdir(parents=True)
    manifest=[]
    for offset in range(0,len(entries),args.per_sheet):
        group=entries[offset:offset+args.per_sheet]
        images=[]
        for item in group:
            path=Path(item['chat_png']).resolve(strict=True)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item['chat_sha256']:
                raise ValueError(f'Changed chat PNG: {path}')
            with Image.open(path) as raw:
                images.append(raw.convert('RGB'))
        columns=min(4,len(group))
        rows=(len(group)+columns-1)//columns
        width=max(im.width for im in images)
        height=max(im.height for im in images)
        sheet=Image.new('RGB',(columns*(width+16),rows*(height+38)), 'white')
        draw=ImageDraw.Draw(sheet)
        for position,(item,im) in enumerate(zip(group,images)):
            x=(position%columns)*(width+16)
            y=(position//columns)*(height+38)
            draw.text((x+4,y+3),f"{item['unit_id']} frame {item['frame_id']}",fill='black')
            sheet.paste(im,(x+4,y+28))
        filename=f'contact_{offset//args.per_sheet+1:03d}.png'
        target=output/filename
        sheet.save(target)
        manifest.append(dict(file=filename,frames=[int(x['frame_id']) for x in group],
                             unit_ids=[x['unit_id'] for x in group],
                             sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (output/'index.json').write_text(json.dumps(dict(queue=str(queue),
        queue_summary_sha256=hashlib.sha256((queue/'queue_summary.json').read_bytes()).hexdigest(),
        selected_changed_frames=len(entries),per_sheet=args.per_sheet,sheets=manifest,
        status='REVIEW_AID_NOT_ANNOTATION'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(selected_changed_frames=len(entries),sheets=len(manifest)),ensure_ascii=False))


if __name__=='__main__':
    main()
