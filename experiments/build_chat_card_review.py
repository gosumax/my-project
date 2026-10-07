"""Build a self-contained, offline human review page for saved CHAT-row crops."""

import base64
import csv
import hashlib
import html
import io
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/paddle_vl15_poker_sft_20260930"
MANIFEST = OUT / "candidates_manifest.jsonl"
PAGE = OUT / "review.html"


def main():
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    audit_path = OUT / "chat_card_visual_audit.csv"
    audit = {}
    if audit_path.exists():
        with audit_path.open(encoding="utf-8-sig", newline="") as stream:
            audit = {r["candidate_id"]: r for r in csv.DictReader(stream, delimiter=";")}
    for row in rows:
        row["visual_draft_gt"] = audit.get(row["candidate_id"], {}).get("visual_gt")
        row["audit_result"] = audit.get(row["candidate_id"], {}).get("audit_result")
        path = Path(row["artifact_path"])
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Image changed after candidate inventory: {path}")
        with Image.open(io.BytesIO(data)) as opened:
            image = opened.convert("RGB")
        image = image.crop((0, 0, image.width - 12, image.height))
        display = io.BytesIO()
        image.save(display, format="PNG")
        row["data_url"] = "data:image/png;base64," + base64.b64encode(display.getvalue()).decode("ascii")
        row["review_kind"] = "CARD_SEQUENCE"
        row["review_display_size"] = [image.width, image.height]
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Poker CHAT card review</title>
<style>
:root{font-family:system-ui,sans-serif;color:#e9edf5;background:#10151e}*{box-sizing:border-box}
body{margin:0}header{position:sticky;top:0;z-index:2;background:#1a2331;padding:12px 18px;border-bottom:1px solid #35445a;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button,select,input{font:inherit}button{background:#2c405b;color:white;border:1px solid #587394;border-radius:6px;padding:7px 10px;cursor:pointer}button:hover{background:#415b7b}
main{max-width:1100px;margin:18px auto;padding:0 14px}.card{background:#1b2635;border:1px solid #364b62;border-radius:10px;padding:18px}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:10px 0}.crop{max-width:100%;image-rendering:pixelated;background:#080b0e;border:1px solid #788ca7;padding:12px}
.label{color:#aebbd0;font-size:13px}.mono,code{font-family:ui-monospace,monospace}code{overflow-wrap:anywhere}
.hint{color:#eac56c}.warn{color:#f39e9e}.good{color:#9ce0b1}.meta{display:grid;grid-template-columns:170px 1fr;gap:6px 14px;margin:16px 0}
input[type=text]{background:#101a27;color:#fff;border:1px solid #647d98;border-radius:5px;padding:8px;min-width:270px;width:50%}
.status{font-weight:700}.small{font-size:13px}.bar{height:8px;background:#344357;border-radius:8px;overflow:hidden}.bar>div{height:100%;background:#5dc18a}
</style></head><body>
<header><strong>CHAT card review</strong><span id="counter"></span>
<button id="prev">← Назад</button><button id="next">Вперёд →</button>
<select id="filter"><option value="visual_drafts" selected>Мои 150 черновиков</option><option value="disagreement">Расхождения</option><option value="compatible">Размер production</option><option value="all">Все</option><option value="pending">Без решения</option><option value="verified">VERIFIED</option><option value="rejected">REJECTED</option></select>
<button id="export">Скачать review_progress.json</button><label class="small">Загрузить прогресс <input type="file" id="import" accept="application/json"></label>
</header><main><div class="bar"><div id="bar"></div></div><p class="small">Изображения сохранены внутри этой страницы. Решения автоматически сохраняются в браузере; экспортируйте JSON для надёжного переноса. Стрелки ←/→: навигация; V: проверить; R: отклонить; Ctrl+S: экспорт.</p>
<section class="card"><div class="row"><span id="status" class="status"></span><span id="priority"></span></div>
<div class="row"><img id="crop" class="crop" alt="Сохранённый crop строки чата"></div>
<p id="geometry"></p><div class="meta" id="meta"></div>
<div class="row"><strong>Метка: последовательность карт</strong><label>Карты <input id="gt" type="text" spellcheck="false" autocomplete="off" placeholder="Qc 6s 4h"></label></div>
<p>Моё чтение изображения: <code id="visual" class="good"></code> <span id="audit_note" class="warn"></span></p>
<p>Подсказка OCR/VL: <code id="proposal" class="hint"></code> <span class="small">Подсказка не считается GT.</span></p>
<p id="error" class="warn"></p>
<div class="row"><button id="verify">✓ VERIFIED</button><button id="reject">✕ REJECTED</button><button id="clear">Сбросить решение</button></div>
</section></main>
<script id="manifest" type="application/json">__PAYLOAD__</script><script>
const MANIFEST_SHA='__SHA__';const rows=JSON.parse(document.getElementById('manifest').textContent);
const key='poker-chat-review:'+MANIFEST_SHA;let saved={},drafts={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch{saved={}}try{drafts=JSON.parse(localStorage.getItem(key+':drafts')||'{}')}catch{drafts={}}
let index=0;const $=id=>document.getElementById(id);const allowed=/^[2-9TJQKA][cdhs]$/;
function data(r){return saved[r.candidate_id]||{status:'UNREVIEWED',verified_gt:'',kind:'CARD_SEQUENCE'}}
function persist(){localStorage.setItem(key,JSON.stringify(saved))}
function visible(){return rows.map((r,i)=>i).filter(i=>{let r=rows[i],d=data(r),f=$('filter').value;return f==='all'||f==='visual_drafts'&&r.audit_result==='PROVISIONAL_VISUAL_DRAFT'||f==='disagreement'&&r.audit_result==='USER_GT_DISAGREES_WITH_VISUAL'||f==='pending'&&d.status==='UNREVIEWED'||f==='verified'&&d.status==='VERIFIED'||f==='rejected'&&d.status==='REJECTED'||f==='compatible'&&r.production_raw_geometry})}
function navigate(delta){let list=visible();if(!list.length)return;let position=list.indexOf(index);index=list[(position+delta+list.length)%list.length]??list[0];render()}
function setMeta(name,value){let a=document.createElement('div');a.className='label';a.textContent=name;let b=document.createElement('code');b.textContent=value??'UNKNOWN';$('meta').append(a,b)}
function render(){let r=rows[index],d=data(r),list=visible();$('counter').textContent=`${index+1}/${rows.length} · ${Object.values(saved).filter(x=>x.status==='VERIFIED').length} VERIFIED · ${Object.values(saved).filter(x=>x.status==='REJECTED').length} REJECTED`;
$('bar').style.width=(100*(Object.values(saved).filter(x=>x.status!=='UNREVIEWED').length/rows.length)).toFixed(1)+'%';
$('status').textContent=d.status;$('status').className='status '+(d.status==='VERIFIED'?'good':d.status==='REJECTED'?'warn':'');$('priority').textContent='priority '+r.review_priority;
$('crop').src=r.data_url;$('crop').style.width=Math.min(r.review_display_size[0]*3,1000)+'px';$('crop').style.height='auto';
$('geometry').innerHTML=r.production_raw_geometry?`<span class="good">Исходный CHAT_ROW ${r.width}×${r.height}; production trim 12 px: ${r.review_display_size.join('×')}.</span>`:`<span class="warn">Legacy геометрия ${r.width}×${r.height}; показано после trim 12 px. Не допускается в train без точного повторного crop из сохранённого источника.</span>`;
$('meta').replaceChildren();[['source',r.source_video],['project',r.source_project],['frame / table / row',`${r.frame_id} / ${r.table_id??'UNKNOWN'} / ${r.row_id??'UNKNOWN'}`],['event / street',`${r.event_type??'UNKNOWN'} / ${r.street??'UNKNOWN'}`],['hand',r.hand_id],['OCR',r.ocr_text],['VL',r.existing_prediction],['image SHA',r.image_sha256],['path',r.artifact_path]].forEach(x=>setMeta(...x));
$('gt').value=d.status==='UNREVIEWED'?(drafts[r.candidate_id]?.gt||r.visual_draft_gt||''):(d.verified_gt||'');$('visual').textContent=r.visual_draft_gt||'нет';$('audit_note').textContent=r.audit_result==='USER_GT_DISAGREES_WITH_VISUAL'?'Ваш ответ отличается от моего чтения — проверьте этот кадр.':'';$('proposal').textContent=r.proposed_gt||'нет';$('error').textContent='';}
function validate(text){let cards=text.split(' ');if(!text||cards.join(' ')!==text||cards.some(c=>!allowed.test(c))||new Set(cards).size!==cards.length||cards.length>5)return 'Нужны 1–5 разных карт в формате rank+suit через один пробел.';return null}
function decision(status){let r=rows[index],gt=$('gt').value.trim();if(status==='VERIFIED'){let error=validate(gt);if(error){$('error').textContent=error;return}}
let prior=saved[r.candidate_id]?.kind;saved[r.candidate_id]={status,verified_gt:status==='VERIFIED'?gt:'',kind:'CARD_SEQUENCE',prior_kind:prior&&prior!=='CARD_SEQUENCE'?prior:null,reviewed_at:new Date().toISOString(),image_sha256:r.image_sha256};delete drafts[r.candidate_id];localStorage.setItem(key+':drafts',JSON.stringify(drafts));persist();render();if(status!=='UNREVIEWED')navigate(1)}
function exportProgress(){let payload={schema:'poker.chat-card-review.v1',manifest_sha256:MANIFEST_SHA,exported_at:new Date().toISOString(),decisions:saved};let blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});let url=URL.createObjectURL(blob);let a=document.createElement('a');a.href=url;a.download='review_progress.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
$('prev').onclick=()=>navigate(-1);$('next').onclick=()=>navigate(1);$('verify').onclick=()=>decision('VERIFIED');$('reject').onclick=()=>decision('REJECTED');$('clear').onclick=()=>decision('UNREVIEWED');$('export').onclick=exportProgress;
$('gt').oninput=()=>{let r=rows[index];drafts[r.candidate_id]={gt:$('gt').value};localStorage.setItem(key+':drafts',JSON.stringify(drafts))};
$('filter').onchange=()=>{let a=visible();if(a.length)index=a[0];render()};$('import').onchange=async e=>{let file=e.target.files[0];if(!file)return;try{let x=JSON.parse(await file.text());if(x.manifest_sha256!==MANIFEST_SHA)throw Error('Манифест изменился');for(let [id,d] of Object.entries(x.decisions||{})){let r=rows.find(v=>v.candidate_id===id);if(!r||d.image_sha256!==r.image_sha256)throw Error('Неверный candidate/hash');if(d.status==='VERIFIED'&&validate(d.verified_gt))throw Error('Неверный GT: '+id)}saved=x.decisions;persist();render()}catch(err){$('error').textContent=String(err)}};
document.addEventListener('keydown',e=>{if(e.ctrlKey&&e.key.toLowerCase()==='s'){e.preventDefault();exportProgress();return}if(['INPUT','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')navigate(-1);if(e.key==='ArrowRight')navigate(1);if(e.key.toLowerCase()==='v')decision('VERIFIED');if(e.key.toLowerCase()==='r')decision('REJECTED')});let initial=visible();if(initial.length)index=initial[0];render();
</script></body></html>"""
    page = page.replace("__PAYLOAD__", payload).replace("__SHA__", manifest_sha)
    PAGE.write_text(page, encoding="utf-8")
    progress = OUT / "review_progress.json"
    if not progress.exists():
        progress.write_text(json.dumps({"schema": "poker.chat-card-review.v1",
                                        "manifest_sha256": manifest_sha,
                                        "decisions": {}}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"html": str(PAGE), "candidates": len(rows), "bytes": PAGE.stat().st_size,
                      "manifest_sha256": manifest_sha}))


if __name__ == "__main__":
    main()
