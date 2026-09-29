"""Build a blind, frame-by-frame viewer for a frozen visual corpus."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parser_core.annotation import VERSION, local_path, sha256


def build(corpus: Path, output: Path) -> dict:
    corpus = corpus.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise ValueError("Choose a new viewer path")
    manifest_path = corpus / "manifest.json"
    digest = sha256(manifest_path)
    if (corpus / "manifest.sha256").read_text(encoding="ascii").strip() != digest:
        raise ValueError("Frozen manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence = {item["evidence_id"]: item for item in manifest["evidence"]}
    if len(evidence) != len(manifest["evidence"]):
        raise ValueError("Duplicate evidence ID")
    units = []
    image_count = 0
    for unit in manifest["units"]:
        frames = {}
        for eid in unit["evidence_ids"]:
            item = evidence[eid]
            if item["roi_type"] not in {"chat", "table"}:
                continue
            image_path = local_path(corpus, item["local"])
            if sha256(image_path) != item["sha256"]:
                raise ValueError(f"Frozen PNG hash changed: {eid}")
            frame = frames.setdefault(item["frame_id"], {"frame_id": item["frame_id"],
                "source_pts": item["source_pts"], "time_base": item["time_base"]})
            if item["roi_type"] in frame:
                raise ValueError(f"Duplicate ROI: {unit['unit_id']}:{item['frame_id']}")
            frame[item["roi_type"]] = {"evidence_id": eid,
                "image": os.path.relpath(image_path, output.parent).replace("\\", "/"),
                "sha256": item["sha256"], "size": item["size"]}
            image_count += 1
        if not frames or any("chat" not in f or "table" not in f for f in frames.values()):
            raise ValueError(f"Incomplete chat/table evidence: {unit['unit_id']}")
        units.append({"unit_id": unit["unit_id"], "table_session_id": unit["table_session_id"],
                      "split": unit["split"], "frames": [frames[k] for k in sorted(frames)]})
    if not units:
        raise ValueError("Frozen corpus has no review units")
    payload = json.dumps({"schema_version": VERSION, "manifest_sha256": digest,
                          "units": units}, ensure_ascii=False,
                         separators=(",", ":")).replace("<", "\\u003c")
    page = r'''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Визуальная проверка исходных кадров</title>
<style>
body{margin:0;background:#15181d;color:#f4f5f7;font:15px system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:#262b32;padding:10px 14px}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button,select,input{font:inherit;padding:5px;background:#373e49;color:white;border:1px solid #7b8798}
button{cursor:pointer} input[type=number]{width:7em} input[type=range]{width:min(35vw,420px)}
#meta,#box{margin-top:6px;color:#d2d8e1;overflow-wrap:anywhere}
main{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:10px}
section{background:#fff;color:#111;padding:8px;min-width:0}
canvas{display:block;max-width:100%;height:auto;image-rendering:pixelated;cursor:crosshair}
#chat{width:min(100%,500px)}
.editor{margin:10px;background:#282e37;padding:12px}.editor h2{font-size:18px;margin:5px 0 10px}
.editor h3{font-size:16px;margin:12px 0 5px}.editor .controls{margin:6px 0}
.editor input[type=text]{min-width:10em}.editor label{display:inline-flex;gap:4px;align-items:center}
.editor button{margin-right:5px}.editor ul{max-height:220px;overflow:auto;padding-left:20px}
.fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}
.fields label{display:flex;flex-wrap:wrap}.fields input{flex:1;min-width:9em}
.hint{color:#cad1db} #editor-status{color:#ffe493}
@media(max-width:900px){main{grid-template-columns:1fr}}
</style><header><div class="controls">
<label>Окно <select id="unit"></select></label>
<button id="back10">−10</button><button id="back">←</button>
<input id="position" type="range" min="0" value="0">
<button id="next">→</button><button id="next10">+10</button>
<label>Кадр <input id="frame" type="number" min="0"></label>
</div><div id="meta"></div><div id="box">Выделите строку на изображении чата; координаты будут показаны здесь.</div>
</header><main><section><strong>Исходный чат</strong><canvas id="chat"></canvas></section>
<section><strong>Исходный стол</strong><canvas id="table"></canvas></section></main>
<div class="editor"><h2>Черновик визуальной разметки</h2>
<p class="hint">Только то, что видно на PNG. OCR-подсказок нет. Черновик всегда
сохраняется как UNREVIEWED; полную проверку покрытия и подпись выполняют отдельно.</p>
<div class="controls"><label>Состояние текста <select id="row-state">
<option>KNOWN</option><option>UNKNOWN</option><option>AMBIGUOUS</option></select></label>
<label>Текст строки <input id="row-text" type="text" size="45"></label>
<button id="add-row">Добавить выделенную строку</button></div>
<h3>Строки окна: выберите фрагменты одного сообщения</h3><ul id="rows"></ul>
<div class="controls"><label>Полнота <select id="completeness"><option>UNKNOWN</option>
<option>PARTIAL</option><option>COMPLETE</option></select></label>
<label>Сценарии через запятую <input id="scenarios" type="text" placeholder="normal, wrapped_amount"></label></div>
<div class="fields" id="fields"></div>
<div class="controls"><button id="add-message">Собрать сообщение из выбранных строк</button>
<span id="editor-status"></span></div><h3>Сообщения окна</h3><ul id="messages"></ul>
<div class="controls"><button id="download">Скачать черновик JSON</button>
<label>Загрузить черновик <input id="import" type="file" accept="application/json,.json"></label></div>
</div>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const unitPick=document.getElementById('unit'),pos=document.getElementById('position');
const frameInput=document.getElementById('frame'),meta=document.getElementById('meta');
const boxText=document.getElementById('box'),chatCanvas=document.getElementById('chat');
let unit=0,index=0,drag=null,current=null;
let draft=data.units.map(u=>({unit_id:u.unit_id,review_status:'UNREVIEWED',
reviewer:null,review_time:null,coverage_complete:false,rows:[],messages:[]}));
let selectedRows=new Set(),dirty=false;
const allowedScenarios=new Set(['normal','wrapped_amount','identical_consecutive',
'initial_partial','clipped_row','large_scroll','all_in','return','side_pot',
'show_cards','unreadable','table_move','table_replace','occlusion',
'scale_change','hidden_chat']);
const fieldNames=['actor','action','amount','role','cards'];
for(const name of fieldNames){const wrap=document.createElement('label');wrap.textContent=name+' ';
const state=document.createElement('select');state.id='state-'+name;
for(const value of ['UNKNOWN','KNOWN','NONE','AMBIGUOUS']){const opt=document.createElement('option');
opt.value=value;opt.textContent=value;state.append(opt)}
const input=document.createElement('input');input.id='value-'+name;input.type='text';
input.placeholder=name==='cards'?'["Ah","Ks"]':'значение';wrap.append(state,input);
document.getElementById('fields').append(wrap)}
data.units.forEach((u,i)=>{const o=document.createElement('option');o.value=i;
o.textContent=`${u.unit_id} · ${u.frames.length} кадров · ${u.split}`;unitPick.append(o)});
function imageToCanvas(kind,item){const canvas=document.getElementById(kind),ctx=canvas.getContext('2d');
canvas.width=item.size[0];canvas.height=item.size[1];ctx.fillStyle='#ddd';
ctx.fillRect(0,0,canvas.width,canvas.height);const img=new Image();
img.onload=()=>{if(data.units[unit].frames[index][kind].evidence_id!==item.evidence_id)return;
ctx.drawImage(img,0,0);if(kind==='chat'&&current)drawBox()};img.src=item.image}
function drawBox(){if(!current)return;const ctx=chatCanvas.getContext('2d');
ctx.strokeStyle='#ffde00';ctx.lineWidth=2;ctx.strokeRect(current[0],current[1],
current[2]-current[0],current[3]-current[1])}
function show(){const u=data.units[unit];index=Math.max(0,Math.min(index,u.frames.length-1));
const f=u.frames[index];pos.max=u.frames.length-1;pos.value=index;frameInput.value=f.frame_id;
const t=f.time_base,seconds=f.source_pts===null?'?':(f.source_pts*t[0]/t[1]).toFixed(3);
meta.textContent=`${u.unit_id} · кадр ${f.frame_id} (${index+1}/${u.frames.length}) · время кадра ${seconds} с (PTS ${f.source_pts} × ${t[0]}/${t[1]}) · chat evidence ${f.chat.evidence_id} · manifest SHA-256 ${data.manifest_sha256}`;
current=null;boxText.textContent='Выделите строку на изображении чата; координаты будут показаны здесь.';
imageToCanvas('chat',f.chat);imageToCanvas('table',f.table)}
function point(ev){const r=chatCanvas.getBoundingClientRect();return [
Math.max(0,Math.min(chatCanvas.width,Math.round((ev.clientX-r.left)*chatCanvas.width/r.width))),
Math.max(0,Math.min(chatCanvas.height,Math.round((ev.clientY-r.top)*chatCanvas.height/r.height)))]}
chatCanvas.addEventListener('pointerdown',ev=>{drag=point(ev);chatCanvas.setPointerCapture(ev.pointerId)});
chatCanvas.addEventListener('pointerup',ev=>{if(!drag)return;const end=point(ev),f=data.units[unit].frames[index];
current=[Math.min(drag[0],end[0]),Math.min(drag[1],end[1]),
Math.max(drag[0],end[0]),Math.max(drag[1],end[1])];drag=null;
if(current[0]===current[2]||current[1]===current[3]){current=null;return}
boxText.textContent=`${f.chat.evidence_id} bbox ${JSON.stringify(current)} · локальные пиксели исходного chat PNG`;
imageToCanvas('chat',f.chat)});
function status(message){document.getElementById('editor-status').textContent=message}
function renderDraft(){const rows=document.getElementById('rows'),messages=document.getElementById('messages');
rows.replaceChildren();messages.replaceChildren();const currentDraft=draft[unit];
for(const row of currentDraft.rows){const li=document.createElement('li'),checkbox=document.createElement('input');
checkbox.type='checkbox';checkbox.checked=selectedRows.has(row.row_id);
checkbox.onchange=()=>{if(checkbox.checked)selectedRows.add(row.row_id);else selectedRows.delete(row.row_id)};
li.append(checkbox,document.createTextNode(` ${row.row_id} · ${row.visual_text.value??row.visual_text.state} · ${row.anchors[0].evidence_id} `));
const edit=document.createElement('button');edit.textContent='Править текст';
edit.onclick=()=>{const state=document.createElement('select');
for(const name of ['KNOWN','UNKNOWN','AMBIGUOUS']){const option=document.createElement('option');
option.value=name;option.textContent=name;state.append(option)}
state.value=row.visual_text.state;
const input=document.createElement('input');input.type='text';input.value=row.visual_text.value??'';
input.size=45;const save=document.createElement('button');save.textContent='Сохранить';
save.onclick=()=>{const value=input.value.trim();if(state.value==='KNOWN'&&!value){
status('Для KNOWN нужен прочитанный текст');return}
row.visual_text={state:state.value,value:state.value==='KNOWN'?value:null};
dirty=true;renderDraft()};const cancel=document.createElement('button');cancel.textContent='Отмена';
cancel.onclick=renderDraft;li.replaceChildren(checkbox,state,input,save,cancel);input.focus()};
const remove=document.createElement('button');remove.textContent='Удалить';
remove.onclick=()=>{if(currentDraft.messages.some(m=>m.row_ids.includes(row.row_id))){status('Строка уже входит в сообщение');return}
currentDraft.rows=currentDraft.rows.filter(r=>r.row_id!==row.row_id);selectedRows.delete(row.row_id);
dirty=true;renderDraft()};li.append(edit,remove);rows.append(li)}
for(const message of currentDraft.messages){const li=document.createElement('li');
li.textContent=`${message.message_id} · ${message.row_ids.join(', ')} · ${message.fields.action.value??message.fields.action.state} `;
const editScenario=document.createElement('button');editScenario.textContent='Править сценарий';
editScenario.onclick=()=>{const input=document.createElement('input');input.type='text';
input.value=message.scenarios.join(', ');input.size=35;
const save=document.createElement('button');save.textContent='Сохранить';
save.onclick=()=>{const scenarios=input.value.split(',').map(s=>s.trim()).filter(Boolean);
if(!scenarios.length||scenarios.some(s=>!allowedScenarios.has(s))){
status('Укажите сценарии из протокола');return}
message.scenarios=[...new Set(scenarios)];dirty=true;renderDraft()};
const cancel=document.createElement('button');cancel.textContent='Отмена';
cancel.onclick=renderDraft;li.replaceChildren(input,save,cancel);input.focus()};
const remove=document.createElement('button');remove.textContent='Удалить';remove.onclick=()=>{
currentDraft.messages=currentDraft.messages.filter(m=>m.message_id!==message.message_id);
dirty=true;renderDraft()};li.append(editScenario,remove);messages.append(li)}
status(`${currentDraft.rows.length} строк, ${currentDraft.messages.length} сообщений в этом окне`)}
document.getElementById('add-row').onclick=()=>{if(!current){status('Сначала выделите строку на PNG чата');return}
const state=document.getElementById('row-state').value,value=document.getElementById('row-text').value.trim();
if(state==='KNOWN'&&!value){status('Для KNOWN нужен прочитанный текст');return}
const rows=draft[unit].rows,used=new Set(rows.map(r=>r.row_id));let n=rows.length+1;
while(used.has(`${data.units[unit].unit_id}_r${String(n).padStart(5,'0')}`))n++;
const rowId=`${data.units[unit].unit_id}_r${String(n).padStart(5,'0')}`;
rows.push({row_id:rowId,anchors:[{evidence_id:data.units[unit].frames[index].chat.evidence_id,
bbox:current.slice()}],visual_text:{state,value:state==='KNOWN'?value:null}});
selectedRows.add(rowId);document.getElementById('row-text').value='';current=null;dirty=true;
imageToCanvas('chat',data.units[unit].frames[index].chat);renderDraft()};
document.getElementById('add-message').onclick=()=>{const currentDraft=draft[unit];
const rowIds=currentDraft.rows.filter(r=>selectedRows.has(r.row_id)).map(r=>r.row_id);
if(!rowIds.length){status('Выберите одну или несколько строк');return}
if(rowIds.some(id=>currentDraft.messages.some(m=>m.row_ids.includes(id)))){
status('Одна строка уже входит в другое сообщение');return}
const scenarios=document.getElementById('scenarios').value.split(',').map(s=>s.trim()).filter(Boolean);
if(!scenarios.length||scenarios.some(s=>!allowedScenarios.has(s))){status('Укажите сценарии из протокола');return}
const fields={};for(const name of fieldNames){const state=document.getElementById('state-'+name).value;
const raw=document.getElementById('value-'+name).value.trim();let value=null;
if(state==='KNOWN'){if(!raw){status(`Поле ${name}: KNOWN требует значение`);return}
if(name==='cards'){try{value=JSON.parse(raw)}catch{status('Карты: нужен JSON-массив');return}
if(!Array.isArray(value)){status('Карты: нужен JSON-массив');return}}else value=raw}
fields[name]={state,value}}
const used=new Set(currentDraft.messages.map(m=>m.message_id));let n=currentDraft.messages.length+1;
while(used.has(`${data.units[unit].unit_id}_m${String(n).padStart(5,'0')}`))n++;
const order=currentDraft.messages.length?Math.max(...currentDraft.messages.map(m=>m.order))+1:0;
currentDraft.messages.push({message_id:`${data.units[unit].unit_id}_m${String(n).padStart(5,'0')}`,
row_ids:rowIds,order,completeness:document.getElementById('completeness').value,
scenarios:[...new Set(scenarios)],fields});selectedRows.clear();dirty=true;renderDraft()};
document.getElementById('download').onclick=()=>{const result={schema_version:data.schema_version,
manifest_sha256:data.manifest_sha256,units:draft};const blob=new Blob(
[JSON.stringify(result,null,2)+'\n'],{type:'application/json'});const url=URL.createObjectURL(blob);
const a=document.createElement('a');a.href=url;a.download=`annotations.draft.${data.manifest_sha256.slice(0,8)}.json`;
a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);dirty=false;status('Черновик скачан; статус остаётся UNREVIEWED')};
function draftShapeOkay(value){if(value.schema_version!==data.schema_version||
value.manifest_sha256!==data.manifest_sha256||!Array.isArray(value.units)||
value.units.length!==data.units.length)return false;
return value.units.every((u,i)=>{const chats=new Map(data.units[i].frames.map(f=>
[f.chat.evidence_id,f.chat.size]));
return u.unit_id===data.units[i].unit_id&&u.review_status==='UNREVIEWED'&&
u.coverage_complete===false&&u.reviewer===null&&u.review_time===null&&
Array.isArray(u.rows)&&Array.isArray(u.messages)&&u.rows.every(r=>
typeof r.row_id==='string'&&r.visual_text&&typeof r.visual_text.state==='string'&&
Array.isArray(r.anchors)&&r.anchors.length>0&&r.anchors.every(a=>{
const size=chats.get(a.evidence_id),b=a.bbox;return size&&Array.isArray(b)&&b.length===4&&
b.every(Number.isInteger)&&0<=b[0]&&b[0]<b[2]&&b[2]<=size[0]&&
0<=b[1]&&b[1]<b[3]&&b[3]<=size[1]}))&&u.messages.every(m=>
typeof m.message_id==='string'&&Array.isArray(m.row_ids)&&
m.fields&&fieldNames.every(name=>m.fields[name]&&
typeof m.fields[name].state==='string'))})}
document.getElementById('import').onchange=async ev=>{const file=ev.target.files[0];if(!file)return;
try{const value=JSON.parse(await file.text());if(!draftShapeOkay(value))
throw Error('Черновик другого корпуса, повреждён или уже подтверждён');
const previous=draft;draft=value.units;selectedRows.clear();
try{renderDraft()}catch(err){draft=previous;renderDraft();throw err}
dirty=false;status('Черновик загружен')}
catch(err){status('Не удалось загрузить: '+err.message)}ev.target.value=''};
unitPick.onchange=()=>{unit=Number(unitPick.value);index=0;selectedRows.clear();show();renderDraft()};
pos.oninput=()=>{index=Number(pos.value);show()};
frameInput.onchange=()=>{const i=data.units[unit].frames.findIndex(f=>f.frame_id===Number(frameInput.value));
if(i>=0)index=i;show()};
for(const [id,delta] of [['back10',-10],['back',-1],['next',1],['next10',10]])
document.getElementById(id).onclick=()=>{index+=delta;show()};
document.onkeydown=ev=>{if(ev.target.tagName==='INPUT'||ev.target.tagName==='SELECT')return;
if(ev.key==='ArrowLeft'){index--;show()}if(ev.key==='ArrowRight'){index++;show()}};
window.addEventListener('beforeunload',ev=>{if(dirty){ev.preventDefault();ev.returnValue=''}});
show();renderDraft();</script></html>'''.replace("__DATA__", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return {"viewer": str(output), "units": len(units),
            "frames": sum(len(x["frames"]) for x in units), "verified_png": image_count,
            "status": "BLIND_VISUAL_REVIEW_ONLY"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.corpus, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
