"""Quality-only test: 100 simultaneous independent CHAT card-sequence requests."""
import asyncio
import base64
import collections
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from parser_core.visual_cards import parse_cards

PROMPT = '''This is a tiny crop of a poker CHAT message containing small playing-card
icons. Read every visible card from left to right. Read both the rank and the
suit from the pixels. Use ranks 2-9,T,J,Q,K,A and suits c,d,h,s for clubs,
diamonds, hearts, spades. Return exactly one JSON object {"cards":["3h","8c","4s"]} with
the cards actually visible, in order. The example is format only. If a card is
unreadable, use "UNKNOWN" for that card. No guesses, explanations or markdown.'''
PARAMS = {"model": "qwen3.7-flash", "temperature": 0, "seed": 37,
          "max_tokens": 2048, "reasoning_effort": "low",
          "response_format": {"type": "json_object"}, "stream": False}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dataset():
    folder = ROOT / "runs/paddle_vl15_poker_sft_20260930"
    old = [json.loads(s) for s in (folder / "verified_chat_card_sequences.jsonl").read_text(encoding="utf-8").splitlines()]
    board_ids = {r["candidate_id"] for r in old if r.get("source_user_kind") == "BOARD" and r.get("street") in ("FLOP", "TURN", "RIVER")}
    combined = [json.loads(s) for s in (folder / "combined_verified_chat_card_rows.jsonl").read_text(encoding="utf-8").splitlines()]
    buckets = collections.defaultdict(list)
    seen = set()
    for r in combined:
        cards = r["ground_truth"].split()
        if len(cards) not in (3,4,5) or r["task_kind"] != "CHAT_CARD_SEQUENCE":
            continue
        if r["record_id"] not in board_ids and r.get("gt_status") != "USER_VERIFIED_SMOKE15":
            continue
        p = Path(r["model_input_png"])
        if not p.is_file() and r.get("source_video") == "smoke_15.mp4":
            p = ROOT / "runs/smoke15_chat_card_sample_5s_20260930/production_trim12_verified" / (r["source_sha256"] + ".png")
        data = p.read_bytes()
        if sha(data) != r["model_input_sha256"]:
            raise ValueError("CHAT input SHA mismatch")
        if sha(data) in seen:
            continue
        from PIL import Image
        with Image.open(p) as im:
            width,height = im.size
        if height != 16 or width > 230 or r["right_trim_px"] != 12:
            raise ValueError("Not the required small CHAT crop geometry")
        seen.add(sha(data))
        street = {3:"FLOP",4:"TURN",5:"RIVER"}[len(cards)]
        buckets[street].append({**r, "input_path": str(p.resolve()), "crop_sha": sha(data),
                               "street": street, "street_basis": "verified BOARD cards count",
                               "width": width, "height": height, "expected_cards": cards})
    rows = []
    while len(rows) < 100 and any(buckets.values()):
        for street in ("FLOP","TURN","RIVER"):
            if buckets[street] and len(rows) < 100:
                rows.append(buckets[street].pop(0))
    if not rows:
        raise ValueError("No suitable unique CHAT crops")
    return rows


async def run(output, check_only, rows_override=None):
    output.mkdir(parents=True, exist_ok=False)
    rows = dataset() if rows_override is None else rows_override
    (output / "inputs").mkdir()
    (output / "responses").mkdir()
    (output / "payloads").mkdir()
    payloads = []
    for i,r in enumerate(rows):
        data = Path(r["input_path"]).read_bytes()
        (output / "inputs" / f"{i:03d}_{r['crop_sha']}.png").write_bytes(data)
        body = {**PARAMS, "messages": [{"role":"user", "content":[
            {"type":"text","text":PROMPT},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(data).decode()}}]}]}
        payload = json.dumps(body,ensure_ascii=False).encode()
        payloads.append(payload)
        r["request_id"] = f"chat_quality_{i:03d}_{r['crop_sha'][:12]}"
        r["payload_sha"] = sha(payload)
        (output / "payloads" / (r["request_id"]+".json")).write_bytes(payload)
    manifest = {"experiment":"CHAT_CARD_QUALITY_INDEPENDENT_SIMULTANEOUS_REQUESTS",
        "concurrency":len(rows),"requests":len(rows),"unique_crops":len({r['crop_sha'] for r in rows}),"image_count_per_request":1,"retries":0,"warmup":0,
        "prompt":PROMPT,"prompt_sha":sha(PROMPT.encode()),"parameters":PARAMS,
        "script_sha":sha(Path(__file__).read_bytes()), "source_manifest":str(ROOT / "runs/paddle_vl15_poker_sft_20260930/combined_verified_chat_card_rows.jsonl"),
        "preprocessing":"existing saved CHAT row trim12 PNG bytes unchanged; no resize or table crops",
        "accuracy_reference":"existing human-verified diagnostic chat sequence labels, frozen before API; not independent production acceptance",
        "street_counts":dict(collections.Counter(r["street"] for r in rows)),"selected":rows,"production_changed":False}
    save(output / "manifest.json",manifest)
    if check_only:
        print(json.dumps({"check":"PASS","images":len(rows),"streets":manifest["street_counts"],"first":rows[0]["input_path"]}),flush=True)
        return
    import httpx
    key=os.environ["AITUNNEL_API_KEY"]
    gate=asyncio.Event()
    active=peak=0
    starts=[]
    seen_ids=set()
    logs=(output / "requests.jsonl").open("w",encoding="utf-8",buffering=1)
    async with httpx.AsyncClient(headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},
        timeout=httpx.Timeout(90,connect=10), limits=httpx.Limits(max_connections=len(rows),max_keepalive_connections=len(rows))) as client:
        async def request(i,r):
            nonlocal active,peak
            await gate.wait()
            start=time.perf_counter();starts.append(start);active+=1;peak=max(peak,active)
            rec={"request_id":r["request_id"],"crop_id":r["record_id"],"crop_sha":r["crop_sha"],
                "start_time":datetime.now(timezone.utc).isoformat(),"http_status":None,"raw_response":None,
                "parsed_cards":None,"expected_cards":r["expected_cards"],"format_error":None,"transport_error":None,
                "server_request_id":None,"duplicate_server_id":False,"finish_reason":None,"token_usage":None,
                "retry_count":0,"image_count":1,"payload_sha":r["payload_sha"]}
            try:
                response=await asyncio.wait_for(client.post("https://api.aitunnel.ru/v1/chat/completions",
                    content=payloads[i],headers={"X-Client-Request-Id":r["request_id"]}),timeout=90)
                finished=time.perf_counter()
                rec["http_status"]=response.status_code
                (output / "responses" / (r["request_id"]+".json")).write_bytes(response.content)
                try:
                    obj=response.json()
                    if response.status_code!=200: raise ValueError("HTTP error")
                    rec["server_request_id"]=obj.get("id")
                    if rec["server_request_id"] in seen_ids and rec["server_request_id"]:
                        rec["duplicate_server_id"]=True
                    if rec["server_request_id"]:seen_ids.add(rec["server_request_id"])
                    rec["token_usage"]=obj.get("usage")
                    rec["finish_reason"]=obj["choices"][0]["finish_reason"]
                    rec["raw_response"]=obj["choices"][0]["message"]["content"]
                    content=json.loads(rec["raw_response"])
                    if not isinstance(content,dict) or set(content)!={"cards"} or not isinstance(content["cards"],list) or not all(isinstance(c,str) for c in content["cards"]):
                        raise ValueError("invalid cards JSON")
                    rec["parsed_cards"]=parse_cards(" ".join(content["cards"]),len(r["expected_cards"]),r["street"])
                except (ValueError,KeyError,IndexError,TypeError,AttributeError) as exc:
                    rec["format_error"]=str(exc)
            except (httpx.HTTPError,asyncio.TimeoutError) as exc:
                finished=time.perf_counter();rec["transport_error"]=type(exc).__name__
            finally:active-=1
            rec["finish_time"]=datetime.now(timezone.utc).isoformat();rec["latency_s"]=finished-start
            rec["exact_cards"]=rec["parsed_cards"]==rec["expected_cards"] and rec["finish_reason"]=="stop" and not rec["duplicate_server_id"]
            logs.write(json.dumps(rec,ensure_ascii=False)+"\n")
            return rec
        tasks=[asyncio.create_task(request(i,r)) for i,r in enumerate(rows)]
        await asyncio.sleep(0)
        wall_start=time.perf_counter();gate.set()
        results=await asyncio.gather(*tasks)
        wall=time.perf_counter()-wall_start
    logs.close()
    summary={"status":"COMPLETED","requests":len(rows),"peak_inflight":peak,"launch_spread_s":max(starts)-min(starts),
        "wall_s":wall,"exact_cards":sum(r["exact_cards"] for r in results),"total":len(rows),
        "format_errors":sum(r["format_error"] is not None for r in results),
        "transport_errors":sum(r["transport_error"] is not None for r in results),
        "http_statuses":dict(collections.Counter(str(r["http_status"]) for r in results)),
        "finish_reasons":dict(collections.Counter(str(r["finish_reason"]) for r in results)),
        "parsed_count":sum(r["parsed_cards"] is not None for r in results),
        "duplicate_server_ids":sum(r["duplicate_server_id"] for r in results),
        "cost_rub":sum((r["token_usage"] or {}).get("cost_rub",0) for r in results),"retries":0,
        "street_accuracy":{s:{"exact":sum(r["exact_cards"] for r,row in zip(results,rows) if row["street"]==s),
                              "total":sum(row["street"]==s for row in rows)} for s in ("FLOP","TURN","RIVER")}}
    by_crop=collections.defaultdict(list)
    for r in results:by_crop[r['crop_sha']].append(r)
    summary['unique_crops']=len(by_crop)
    summary['unique_all_attempts_exact']=sum(all(r['exact_cards'] for r in rs) for rs in by_crop.values())
    summary['repeat_differing_cards']=sum(len({json.dumps(r['parsed_cards']) for r in rs})>1 for rs in by_crop.values() if len(rs)>1)
    save(output / "summary.json",summary)
    save(output / "results.json",results)
    lines=["# Qwen: small CHAT card crops, quality test","",f"Exact cards: {summary['exact_cards']}/{len(rows)}.",
        f"{len(rows)} independent requests; peak inflight {peak}; launch spread {summary['launch_spread_s']:.3f}s; wall {wall:.3f}s.",
        f"HTTP: {summary['http_statuses']}; format errors {summary['format_errors']}; transport errors {summary['transport_errors']}; retries0.",
        "Reference: existing human-verified diagnostic labels; production unchanged.","",
        "| # | Crop | Expected | Qwen raw | Parsed | Exact |","|---|---|---|---|---|---|"]
    for i,(row,r) in enumerate(zip(rows,results)):
        raw=str(r["raw_response"]).replace("|","\\|").replace("\n"," ")
        lines.append(f"|{i}|![chat](inputs/{i:03d}_{row['crop_sha']}.png)|{' '.join(r['expected_cards'])}|{raw}|{r['parsed_cards']}|{r['exact_cards']}|")
    (output / "report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    output=Path(sys.argv[1]);asyncio.run(run(output,"--check-only" in sys.argv))
