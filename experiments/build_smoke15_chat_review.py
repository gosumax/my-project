"""Self-contained visual review of sampled Smoke15 CHAT rows; no GT guesses."""

import base64
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/smoke15_chat_card_sample_v2_20260930"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, default=OUT)
    ap.add_argument("--focus-card", default=None)
    args = ap.parse_args()
    out = args.sample
    sample = out / "candidates.jsonl"
    triage = out / "visual_triage_not_gt.jsonl"
    sample_sha = hashlib.sha256(sample.read_bytes()).hexdigest()
    rows = [json.loads(s) for s in triage.read_text(encoding="utf-8").splitlines() if s]
    if args.focus_card:
        rows = [row for row in rows if any(
            any(candidate["card"] == args.focus_card for candidate in suggestion["nearest_classes"])
            for suggestion in row["suggestions_not_gt"])]
    seen = set()
    display = []
    for row in rows:
        if row["chat_row_sha256"] in seen:
            continue
        seen.add(row["chat_row_sha256"])
        path = Path(row["chat_row_png"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["chat_row_sha256"]:
            raise ValueError(f"Source PNG changed: {path}")
        display.append({"png": row["chat_row_png"], "sha256": row["chat_row_sha256"],
                        "image": "data:image/png;base64," + base64.b64encode(raw).decode("ascii"),
                        "time_s": row["decoded_time_s"], "table_id": row["table_id"],
                        "source_span_index": row["source_span_index"],
                        "card_count": len(row["suggestions_not_gt"]),
                        "rare_priority": bool(row["rare_best"])})
    payload = json.dumps(display, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = r'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Smoke15: карты в чате</title><style>
*{box-sizing:border-box}body{margin:0;background:#101723;color:#e9eff6;font:16px system-ui,sans-serif}header{position:sticky;top:0;background:#1d2a3b;border-bottom:1px solid #456078;padding:12px;z-index:2}
main{max-width:1000px;margin:auto;padding:12px}h1{font-size:22px;margin:0 0 6px}p{margin:6px 0}button{font:inherit;background:#26784d;color:white;border:0;border-radius:6px;padding:9px 14px;cursor:pointer}
select,input{font:17px ui-monospace,Consolas,monospace;background:#0d1825;color:#fff;border:1px solid #69809c;border-radius:5px;padding:8px}.item{border:1px solid #37506d;border-radius:7px;background:#1a2737;padding:10px;margin:9px 0}.row{display:flex;gap:14px;align-items:center;flex-wrap:wrap}.crop{image-rendering:pixelated;width:900px;max-width:100%;background:black}.meta{font-size:13px;color:#b7c7d7}.priority{border-color:#d3a25a}.error{color:#ff9b9b}
</style></head><body><header><h1>Smoke15: проверка карт в строках чата</h1>
<p>__COUNT__ разных строк из выборки. Поля пустые: подсказки сходства не считаются ответами. Первая видимая строка чата исключена, как в production.</p>
<div class="row"><label>Показать <select id="filter"><option value="priority">Редкие карты — приоритет</option><option value="all">Все строки</option><option value="filled">Заполненные</option></select></label><button id="download">Скачать мои решения</button><span id="count"></span></div><p id="error" class="error"></p>
</header><main id="list"></main><script id="data" type="application/json">__DATA__</script><script>
const rows=JSON.parse(document.getElementById('data').textContent),manifestSha='__SHA__';
const key='smoke15-chat-card-review:'+manifestSha;let edits={};try{edits=JSON.parse(localStorage.getItem(key)||'{}')}catch{edits={}}
const list=document.getElementById('list'),error=document.getElementById('error'),count=document.getElementById('count');
function update(){let n=Object.values(edits).filter(x=>x&&x.status==='VERIFIED').length;count.textContent=`${n} размечено из ${rows.length}`}
function render(){list.replaceChildren();let f=document.getElementById('filter').value;let visible=rows.filter(r=>f==='all'||f==='priority'&&r.rare_priority||f==='filled'&&edits[r.sha256]?.status==='VERIFIED');for(const [i,r] of visible.entries()){
 const item=document.createElement('section');item.className='item'+(r.rare_priority?' priority':'');
 const meta=document.createElement('p');meta.className='meta';meta.textContent=`${i+1}/${visible.length} · ${r.time_s.toFixed(1)} с · стол ${r.table_id} · строка ${r.source_span_index} · детектор: ${r.card_count} областей (может ошибаться) · SHA ${r.sha256}`;item.append(meta);
 const line=document.createElement('div');line.className='row';const img=document.createElement('img');img.className='crop';img.src=r.image;img.alt='Исходная CHAT строка Smoke15';line.append(img);item.append(line);
 const label=document.createElement('label');label.textContent='Карты слева направо: ';const input=document.createElement('input');input.value=edits[r.sha256]?.ground_truth||'';input.placeholder='4d 3s 9c 6d';input.dataset.sha=r.sha256;label.append(input);item.append(label);
 input.onchange=()=>{let gt=input.value.trim();if(gt)edits[r.sha256]={status:'VERIFIED',ground_truth:gt,source_png:r.png};else delete edits[r.sha256];localStorage.setItem(key,JSON.stringify(edits));update()};list.append(item)
 }update()}
document.getElementById('filter').onchange=render;
document.getElementById('download').onclick=()=>{error.textContent='';let output=[];for(const r of rows){let d=edits[r.sha256];if(!d||d.status!=='VERIFIED')continue;let gt=d.ground_truth,cards=gt.split(' ');if(!gt||cards.length<1||cards.length>5||cards.some(c=>!(/^[2-9TJQKA][cdhs]$/).test(c))||new Set(cards).size!==cards.length||cards.join(' ')!==gt){error.textContent=`Проверьте формат карт у SHA ${r.sha256}`;return}output.push({chat_row_sha256:r.sha256,source_png:r.png,ground_truth:gt})}
let result={schema:'poker.smoke15-chat-review.v1',sample_manifest_sha256:manifestSha,exported_at:new Date().toISOString(),rows:output};let url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=url;a.download='smoke15_chat_card_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
render();
</script></body></html>'''
    target = out / (f"review_{args.focus_card}.html" if args.focus_card else "review.html")
    rendered = page.replace("__DATA__", payload).replace("__SHA__", sample_sha)
    rendered = rendered.replace("__COUNT__", str(len(display)))
    if args.focus_card:
        rendered = rendered.replace('<option value="all">', '<option value="all" selected>')
    target.write_text(rendered, encoding="utf-8")
    print(json.dumps({"page": str(target), "unique_rows": len(display),
                      "rare_priority_rows": sum(r["rare_priority"] for r in display),
                      "sample_manifest_sha256": sample_sha}, ensure_ascii=False))


if __name__ == "__main__":
    main()
