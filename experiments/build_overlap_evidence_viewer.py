"""Final evidence HTML for an experimental OLD/NEW removal review; no image files."""
import argparse
import base64
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.temporal_overlap_dedup import read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    meta = json.loads((args.run / 'NEW/experiment_summary.json').read_text('utf-8'))
    old = {r['row_id']: r for r in read_jsonl(Path(meta['old']) / 'physical_rows.jsonl')}
    rows = []
    audit_path = args.run / 'independent_audit.json'
    audit = json.loads(audit_path.read_text('utf-8')) if audit_path.exists() else {}
    lost = {r['old_row_id'] for r in audit.get('real_rows_lost', [])}

    def png(data, shape):
        pixels = np.frombuffer(base64.b64decode(data), np.uint8).reshape(shape)
        ok, encoded = cv2.imencode('.png', pixels)
        if not ok:
            raise ValueError('RAM evidence encode failed')
        return 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii')

    for r in read_jsonl(args.run / 'NEW/removed_duplicates.jsonl'):
        physical = old[r['old_row_id']]
        prior = r['matched_previous_observation']
        rows.append({'table': r['table_id'], 'frame': r['frame_id'], 'time_ms': r['timestamp_ms'],
            'text': ('ОШИБКА: реальная строка новой руки — ' if r['old_row_id'] in lost else '') + r['text'], 'current_id': r['old_row_id'], 'previous_id': r['matched_previous_row_id'],
            'previous_frame': prior['frame_id'], 'previous_source_id': prior['stable_row_id'],
            'current_image': png(physical['row_pixels_base64'], physical['pixel_shape']),
            'previous_image': png(prior['row_pixels_base64'], prior['pixel_shape']),
            'mae': r['native_overlap_mae'], 'ncc': r['min_glyph_ncc'],
            'anchors': r['anchor_row_ids'], 'pairs': r['overlap_pairs'],
            'current_sha': r['current_pixel_sha256'], 'previous_sha': r['previous_pixel_sha256']})
    payload = json.dumps(rows, ensure_ascii=False).replace('<', '\\u003c')
    html = '''<!doctype html><meta charset="utf-8"><title>Overlap evidence review</title>
<style>body{font:16px system-ui;margin:25px;background:#f5f5f5}main{max-width:1150px;margin:auto}select,input,button{font:inherit;padding:8px}article{background:white;padding:20px;margin:18px 0;border-radius:8px}img{display:block;image-rendering:pixelated;height:64px;margin:8px 0;max-width:100%}code{overflow-wrap:anywhere}pre{white-space:pre-wrap;font-size:12px}label{display:block;margin-top:15px}</style>
<main><h1>Проверка решений dedup</h1><p>Исходные пиксели OLD и выбранное более раннее изображение. Пометка ОШИБКА обозначает подтверждённую потерю реальной строки новой руки. Остальные удаления не являются сертификатом безопасности. Этот файл только показывает evidence; решения и входы не изменяются.</p>
<select id="table"><option value="">Все столы</option></select> <input id="search" placeholder="Текст / frame / row ID" size="45"><p id="count"></p><section id="rows"></section></main>
<script>const data=PAYLOAD;const select=document.querySelector('#table');for(const t of [...new Set(data.map(r=>r.table))].sort()){const o=document.createElement('option');o.value=o.textContent=t;select.append(o)}
function add(parent,tag,text){const e=document.createElement(tag);e.textContent=text;parent.append(e);return e}
function draw(){const q=document.querySelector('#search').value.toLowerCase();const rows=data.filter(r=>(!select.value||r.table===select.value)&&JSON.stringify([r.text,r.frame,r.current_id,r.previous_id]).toLowerCase().includes(q));document.querySelector('#count').textContent=rows.length+' из '+data.length+' удалений';const out=document.querySelector('#rows');out.replaceChildren();for(const r of rows){const a=document.createElement('article');out.append(a);add(a,'h2',r.table+' · frame '+r.frame+' · '+r.time_ms.toFixed(1)+' ms');add(a,'p',r.text);add(a,'label','Повторное наблюдение — '+r.current_id);const cur=document.createElement('img');cur.src=r.current_image;a.append(cur);add(a,'label','Прежнее наблюдение frame '+r.previous_frame+' — '+r.previous_source_id);const prev=document.createElement('img');prev.src=r.previous_image;a.append(prev);add(a,'p','MAE '+r.mae.toFixed(4)+' · min glyph NCC '+r.ncc.toFixed(4)+' · anchors '+r.anchors.length);const d=document.createElement('details');a.append(d);add(d,'summary','Identity / SHA / overlap');add(d,'pre',JSON.stringify({previous_identity:r.previous_id,current_sha:r.current_sha,previous_sha:r.previous_sha,anchors:r.anchors,overlap:r.pairs},null,2))}}
select.onchange=draw;document.querySelector('#search').oninput=draw;draw();</script>'''.replace('PAYLOAD', payload)
    (args.run / 'evidence_review.html').write_text(html, 'utf-8')
    print(json.dumps({'reviewed_records': len(rows), 'path': str((args.run/'evidence_review.html').resolve())}))


if __name__ == '__main__':
    main()
