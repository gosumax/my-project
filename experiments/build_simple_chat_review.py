"""Single-page correction sheet for the 150 visual drafts and two disagreements."""

import base64
import csv
import hashlib
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/paddle_vl15_poker_sft_20260930"
AUDIT = OUT / "chat_card_correction_table.csv"
PAGE = OUT / "simple_review.html"


def main():
    with AUDIT.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream, delimiter=";"))
    records = [r for r in records if r["audit_result"] in
               {"USER_GT_DISAGREES_WITH_VISUAL", "PROVISIONAL_VISUAL_DRAFT"}]
    if len(records) != 152:
        raise ValueError(f"Expected 2 disagreements and 150 drafts; found {len(records)}")
    manifest_sha = hashlib.sha256((OUT / "candidates_manifest.jsonl").read_bytes()).hexdigest()
    for row in records:
        png = Path(row["trimmed_png"])
        row["image_data_url"] = "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode("ascii")
        row["image_sha256"] = hashlib.sha256(png.read_bytes()).hexdigest()
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = r'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Проверка карт чата</title><style>
*{box-sizing:border-box}body{margin:0;background:#101722;color:#e8edf5;font:17px system-ui,sans-serif}
header{position:sticky;top:0;background:#1d2a3a;border-bottom:1px solid #47617c;padding:14px;z-index:2}
h1{font-size:23px;margin:0 0 8px}p{margin:5px 0}button{font:inherit;background:#277c50;color:#fff;border:0;border-radius:7px;padding:10px 15px;cursor:pointer}
main{max-width:1100px;margin:18px auto;padding:0 10px}.item{border:1px solid #344a62;border-radius:8px;padding:11px;margin:9px 0;background:#1b2635}
.item.dispute{border:2px solid #e7a55f}.line{display:flex;align-items:center;gap:13px;flex-wrap:wrap}.crop{image-rendering:pixelated;max-width:100%;background:#080b10}
input{font:18px ui-monospace,Consolas,monospace;background:#0e1926;color:#fff;border:1px solid #7596b6;border-radius:5px;padding:8px;width:270px}
.small{font-size:13px;color:#aab9cb}.warn{color:#ffc486}.error{color:#ff8c8c}.ok{color:#92d8a7}
</style></head><body>
<header><h1>Проверьте карты на картинках</h1>
<p>Если мой ответ правильный — ничего не нажимайте. Если неправильный — исправьте поле рядом с картинкой.</p>
<p><button id="download">Готово — скачать мои исправления</button> <span id="count" class="small"></span></p>
<p id="error" class="error"></p></header><main id="list"></main>
<script id="data" type="application/json">__PAYLOAD__</script><script>
const rows=JSON.parse(document.getElementById('data').textContent),manifestSha='__MANIFEST_SHA__';
const key='poker-chat-simple-review:'+manifestSha;let edits={};try{edits=JSON.parse(localStorage.getItem(key)||'{}')}catch{edits={}}
const list=document.getElementById('list'),count=document.getElementById('count'),error=document.getElementById('error');
for(const [i,r] of rows.entries()){
  const item=document.createElement('section');item.className='item'+(r.audit_result==='USER_GT_DISAGREES_WITH_VISUAL'?' dispute':'');
  const title=document.createElement('div');title.className='small';title.textContent=`${i+1}/152 · ${r.source_video} · кадр ${r.frame_id} · стол ${r.table_id} · ${r.event_type}`;item.append(title);
  if(r.user_gt){const previous=document.createElement('p');previous.className='warn';previous.textContent=`Ваш старый ответ: ${r.user_gt}. Моё чтение: ${r.visual_gt}. Проверьте масть.`;item.append(previous)}
  const line=document.createElement('div');line.className='line';
  const img=document.createElement('img');img.className='crop';img.src=r.image_data_url;img.width=630;img.alt='Карты из сохранённой строки чата';line.append(img);
  const input=document.createElement('input');input.type='text';input.spellcheck=false;input.value=edits[r.candidate_id]??r.visual_gt;input.dataset.id=r.candidate_id;input.setAttribute('aria-label','Карты');
  input.addEventListener('input',()=>{edits[r.candidate_id]=input.value;localStorage.setItem(key,JSON.stringify(edits));count.textContent=`Исправлено полей: ${Object.keys(edits).length}`});line.append(input);item.append(line);
  const note=document.createElement('div');note.className='small';note.textContent=r.candidate_id;item.append(note);list.append(item);
}
count.textContent=`Исправлено полей: ${Object.keys(edits).length}`;
document.getElementById('download').onclick=()=>{
  error.textContent='';const allowed=/^[2-9TJQKA][cdhs]$/;const output=[];
  for(const r of rows){const input=list.querySelector(`input[data-id="${r.candidate_id}"]`),gt=input.value.trim(),cards=gt.split(' ');
    if(!gt||cards.length>5||cards.some(c=>!allowed.test(c))||new Set(cards).size!==cards.length||cards.join(' ')!==gt){error.textContent=`Проверьте формат карт у ${r.candidate_id}`;input.scrollIntoView({behavior:'smooth',block:'center'});input.focus();return}
    output.push({candidate_id:r.candidate_id,ground_truth:gt,derived_image_sha256:r.image_sha256});}
  const result={schema:'poker.chat-card-corrections.v1',manifest_sha256:manifestSha,exported_at:new Date().toISOString(),rows:output};
  const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='chat_card_corrections.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
</script></body></html>'''
    PAGE.write_text(page.replace("__PAYLOAD__", payload).replace("__MANIFEST_SHA__", manifest_sha), encoding="utf-8")
    print(json.dumps({"page": str(PAGE), "items": len(records), "bytes": PAGE.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
