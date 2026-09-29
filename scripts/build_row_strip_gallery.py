"""Build a browser gallery of every captured chat-row strip without duplicating PNGs."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.annotation import sha256, write_json
from parser_core.chat_rows import (ROW_TRACKER_VERSION, exclude_first_visible_span,
                                   padded_row_span, physical_spans)


def build(run: Path, output: Path) -> dict:
    run, output = run.resolve(strict=True), output.resolve(strict=True)
    with sqlite3.connect((run / "journal.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("""SELECT o.frame_id,json_extract(s.geometry_json,'$.slot_id'),
            o.crop_path,o.crop_sha256,f.source_pts,f.time_base_num,f.time_base_den
            FROM observations o JOIN table_sessions s USING(table_session_id)
            JOIN frames f USING(source_id,frame_id) WHERE o.roi_type='chat'
            ORDER BY o.frame_id,json_extract(s.geometry_json,'$.slot_id')""").fetchall()
    frames: dict[int, dict] = {}
    total_strips = 0
    for frame_id, slot, relative, digest, pts, num, den in rows:
        path = (run / relative).resolve(strict=True)
        if not path.is_relative_to(run) or sha256(path) != digest:
            raise ValueError(f"Raw chat PNG differs from journal: {path}")
        with Image.open(path) as image:
            gray = np.asarray(image.convert("L"))
        spans = physical_spans(gray)
        included, suppressed = exclude_first_visible_span(spans)
        strips = [[index, *padded_row_span(spans, index, gray.shape[0])]
                  for index in range(len(suppressed), len(spans))]
        total_strips += len(strips)
        frame = frames.setdefault(frame_id, {"frame_id": frame_id,
            "pts_seconds": round(pts * num / den, 3) if pts is not None else None,
            "slots": {}})
        if str(slot) in frame["slots"]:
            raise ValueError(f"Duplicate chat ROI: {frame_id}, {slot}")
        frame["slots"][str(slot)] = {"image": os.path.relpath(path, output).replace("\\", "/"),
            "sha256": digest, "width": gray.shape[1], "height": gray.shape[0],
            "suppressed": [list(x) for x in suppressed], "strips": strips}
    if not frames or any(set(x["slots"]) != {str(n) for n in range(1, 10)}
                         for x in frames.values()):
        raise ValueError("Every captured frame must have all nine chat ROIs")
    manifest = {"status": "VISUAL_ROW_STRIP_REVIEW_ONLY", "tracker_version": ROW_TRACKER_VERSION,
        "source_run": str(run), "frames": list(frames.values()), "row_strips": total_strips,
        "first_visible_row": "EXCLUDED_AND_LISTED_AS_SUPPRESSED", "ocr": "NONE"}
    write_json(output / "row_strip_gallery.json", manifest)
    data = json.dumps(manifest["frames"], ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    page = """<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smoke 12 — полоски строк чата</title>
<style>
body{margin:0;background:#17191d;color:#eee;font:15px system-ui,sans-serif}
header{position:sticky;top:0;background:#24272d;padding:10px 16px;z-index:2}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button,select{font:inherit;padding:6px 9px;background:#343943;color:#fff;border:1px solid #687080;border-radius:5px}
button{cursor:pointer} #meta{color:#cbd0d9;margin-top:7px}
main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;padding:8px}
article{background:white;color:black;padding:5px;min-width:0}
article canvas{display:block;max-width:100%;height:auto}
@media(max-width:900px){main{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:600px){main{grid-template-columns:1fr}}
</style><header><div class="bar"><button id="prev">←</button>
<label>Кадр <select id="frame"></select></label><button id="next">→</button></div>
<div id="meta"></div></header><main id="grid"></main>
<script id="data" type="application/json">__DATA__</script><script>
const frames=JSON.parse(document.getElementById('data').textContent);
const pick=document.getElementById('frame');
for(let i=0;i<frames.length;i++){const option=document.createElement('option');option.value=i;
option.textContent=`${frames[i].frame_id} (${frames[i].pts_seconds??'?'} с)`;pick.append(option)}
function draw(){const f=frames[Number(pick.value)],grid=document.getElementById('grid');
document.getElementById('meta').textContent=`Кадр ${f.frame_id}, PTS ${f.pts_seconds??'?'} с. Полоски из исходных PNG; верхняя видимая строка исключена. ${Number(pick.value)+1}/${frames.length}`;
grid.replaceChildren();for(let n=1;n<=9;n++){const s=f.slots[n],box=document.createElement('article');
const title=document.createElement('div');title.textContent=`Стол ${String(n).padStart(2,'0')} · ${s.strips.length} полосок`;
box.append(title);const canvas=document.createElement('canvas');canvas.width=s.width*2+40;
canvas.height=Math.max(20,s.strips.reduce((h,r)=>h+Math.max(37,(r[2]-r[1])*2+5),5));
box.append(canvas);grid.append(box);
const ctx=canvas.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height);
const img=new Image();img.onload=()=>{let y=3;for(let i=0;i<s.strips.length;i++){
const [index,top,bottom]=s.strips[i];ctx.fillStyle='black';
ctx.font='13px sans-serif';ctx.fillText(String(index).padStart(2,'0'),3,y+18);
ctx.imageSmoothingEnabled=false;ctx.drawImage(img,0,top,s.width,bottom-top,38,y,s.width*2,(bottom-top)*2);
y+=Math.max(37,(bottom-top)*2+5)}};
img.src=s.image}}
pick.onchange=draw;document.getElementById('prev').onclick=()=>{pick.selectedIndex=Math.max(0,pick.selectedIndex-1);draw()};
document.getElementById('next').onclick=()=>{pick.selectedIndex=Math.min(frames.length-1,pick.selectedIndex+1);draw()};
draw();</script></html>""".replace("__DATA__", data)
    target = output / "row_strip_gallery.html"
    target.write_text(page, encoding="utf-8")
    return {"gallery": str(target), "frames": len(frames), "raw_chat_rois": len(rows),
            "row_strips": total_strips}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
