"""Freeze a label-blind, source-linked Smoke16 CHAT-card review sample."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/card_train_smoke16_test_20260930"
SAMPLE = OUT / "smoke16_sample"

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def write_once(path: Path, data: bytes) -> None:
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Frozen artifact conflict: {path}")
    path.write_bytes(data)

def main() -> None:
    rows = [json.loads(line) for line in (SAMPLE / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    sample_summary = json.loads((SAMPLE / "summary.json").read_text(encoding="utf-8"))
    unique = {}
    for row in rows:
        unique.setdefault(row["row_pixel_sha256"], row)
    selected = []
    # Twelve time strata, five rows each. Choice uses metadata/hash alone, no card identity or model output.
    for time_bin in range(12):
        candidates = [r for r in unique.values() if int(r["requested_time_s"] // 75) == time_bin]
        candidates.sort(key=lambda r: digest(("smoke16-blind-v1:" + r["row_pixel_sha256"]).encode()))
        selected.extend(candidates[:5])
    if len(selected) != 60:
        raise ValueError(f"Expected 60 frozen review rows, found {len(selected)}")
    selected.sort(key=lambda r: (r["requested_time_s"], r["table_id"], r["row_pixel_sha256"]))
    manifest = []
    display = []
    for index, row in enumerate(selected):
        path = Path(row["chat_row_png"])
        raw = path.read_bytes()
        if digest(raw) != row["chat_row_sha256"]:
            raise ValueError(f"Source SHA mismatch: {path}")
        item = {**row, "blind_test_index": index, "selection_rule": "12x75s_time_bins_5_hashranked_unique_rows_v1"}
        manifest.append(item)
        display.append({"sha256": row["chat_row_sha256"], "png": str(path),
                        "image": "data:image/png;base64," + base64.b64encode(raw).decode(),
                        "time_s": row["decoded_time_s"], "table": row["table_id"],
                        "span_index": row["source_span_index"], "detected_spans": len(row["detected_card_spans_inclusive"])})
    payload = ("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in manifest) + "\n").encode("utf-8")
    write_once(OUT / "smoke16_blind_review_manifest.jsonl", payload)
    manifest_sha = digest(payload)
    metadata = {"status": "GT_PENDING", "source_video_sha256": sample_summary["source_video_sha256"],
                "sample_manifest_sha256": digest((SAMPLE / "candidates.jsonl").read_bytes()),
                "blind_review_manifest_sha256": manifest_sha, "rows": len(manifest),
                "selection_rule": "12x75s_time_bins_5_hashranked_unique_rows_v1",
                "ground_truth_source": "HUMAN_VISUAL_REVIEW_REQUIRED",
                "test_usage": "EXCLUDED_FROM_TRAIN_VALIDATION_CHECKPOINT_SELECTION"}
    write_once(OUT / "smoke16_blind_freeze.json", (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    data_js = json.dumps(display, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = '''<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Smoke16 blind CARD review</title><style>
body{font:16px system-ui;background:#111b27;color:#edf3f9;margin:0}header{position:sticky;top:0;background:#213248;padding:12px}main{max-width:1100px;margin:auto;padding:12px}.item{background:#203146;border:1px solid #526a84;margin:10px 0;padding:12px;border-radius:8px}img{width:900px;max-width:100%;image-rendering:pixelated;background:black}input{font:18px Consolas,monospace;width:250px;padding:7px}button{padding:8px 14px;font:inherit}.meta{color:#bbcfdf;font-size:13px}#error{color:#ff9f9f}
</style></head><body><header><strong>Smoke16: независимая визуальная разметка CHAT карт</strong><p>60 строк. Введите видимые карты слева направо, например «4d 3s 9c 6d». При сомнительных границах оставьте UNRESOLVED. Подсказок модели нет.</p><button id="download">Скачать решения</button> <span id="count"></span><p id="error"></p></header><main id="list"></main><script id="data" type="application/json">__DATA__</script><script>
const rows=JSON.parse(document.getElementById('data').textContent), manifestSha='__SHA__', key='smoke16-blind-review:'+manifestSha;
let edits={};try{edits=JSON.parse(localStorage.getItem(key)||'{}')}catch{};const list=document.getElementById('list'),error=document.getElementById('error');
function update(){document.getElementById('count').textContent=`${Object.keys(edits).length}/${rows.length} решений`}
rows.forEach((r,i)=>{let box=document.createElement('section');box.className='item';let m=document.createElement('p');m.className='meta';m.textContent=`${i+1}/60 · ${r.time_s.toFixed(1)} с · стол ${r.table} · исходная строка ${r.span_index} · детектор ${r.detected_spans} областей · SHA ${r.sha256}`;box.append(m);let im=document.createElement('img');im.src=r.image;im.alt='Исходная CHAT строка';box.append(im);let p=document.createElement('p');let input=document.createElement('input');input.placeholder='4d 3s 9c 6d';input.value=edits[r.sha256]?.ground_truth||'';let skip=document.createElement('button');skip.textContent='UNRESOLVED';skip.onclick=()=>{edits[r.sha256]={status:'UNRESOLVED',source_png:r.png};input.value='';localStorage.setItem(key,JSON.stringify(edits));update()};input.onchange=()=>{let gt=input.value.trim();if(gt)edits[r.sha256]={status:'VERIFIED',ground_truth:gt,source_png:r.png};else delete edits[r.sha256];localStorage.setItem(key,JSON.stringify(edits));update()};p.append(input,skip);box.append(p);list.append(box)});update();
document.getElementById('download').onclick=()=>{error.textContent='';let out=[];for(const r of rows){let d=edits[r.sha256];if(!d)continue;if(d.status==='VERIFIED'){let cards=d.ground_truth.split(' ');if(cards.length<1||cards.length>5||cards.some(c=>!(/^[2-9TJQKA][cdhs]$/).test(c))||new Set(cards).size!==cards.length||cards.join(' ')!==d.ground_truth){error.textContent=`Проверьте SHA ${r.sha256}`;return}}out.push({chat_row_sha256:r.sha256,...d})}let result={schema:'poker.smoke16-blind-chat-review.v1',blind_review_manifest_sha256:manifestSha,rows:out};let url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=url;a.download='smoke16_blind_chat_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></body></html>'''.replace("__DATA__", data_js).replace("__SHA__", manifest_sha)
    write_once(OUT / "smoke16_review.html", html.encode("utf-8"))
    print(json.dumps(metadata, ensure_ascii=False))

if __name__ == "__main__":
    main()
