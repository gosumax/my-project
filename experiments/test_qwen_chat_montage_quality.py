"""Five one-image montage requests; original CHAT pixels pasted without resize."""
import asyncio
import base64
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.test_qwen_chat_board_quality100 import dataset, PARAMS
from parser_core.visual_cards import parse_cards

PROMPT='''The image contains numbered rows of tiny poker CHAT card icons.
Read every visible card in every numbered row independently, left to right.
Use ranks 2-9,T,J,Q,K,A and suits c,d,h,s (clubs,diamonds,hearts,spades).
Return exactly one JSON object {"rows":[{"row":1,"cards":["3h","8c","4s"]}]}.
Include every numbered row once in row order. The example illustrates format only.
Read rank and suit from pixels; do not borrow cards from other rows. Use "UNKNOWN"
for an unreadable card. No prose or markdown.'''

def sha(data):return hashlib.sha256(data).hexdigest()
def save(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')

async def run(out):
    out.mkdir(parents=True,exist_ok=False)
    rows=dataset()[:50]
    singles=json.loads((ROOT/'runs/qwen_chat_board_quality66_20260930_v2/results.json').read_text(encoding='utf-8'))
    baseline={r['crop_sha']:r for r in singles}
    font=ImageFont.load_default()
    manifests=[]
    for count in (10,20,30,40,50):
        # Gray matching native row background, 42px label margin and 8px gaps.
        width=42+max(r['width'] for r in rows[:count]);height=24*count
        canvas=Image.new('RGB',(width,height),(85,85,85));draw=ImageDraw.Draw(canvas)
        members=[]
        for i,r in enumerate(rows[:count]):
            p=Path(r['input_path']);data=p.read_bytes()
            assert sha(data)==r['crop_sha']
            with Image.open(p) as opened:im=opened.convert('RGB')
            x,y=42,i*24+4
            canvas.paste(im,(x,y));draw.text((4,y),str(i+1),fill='white',font=font)
            assert canvas.crop((x,y,x+im.width,y+im.height)).tobytes()==im.tobytes()
            members.append({**r,'montage_row':i+1,'paste_box':[x,y,x+im.width,y+im.height]})
        path=out/f'montage_{count}.png';canvas.save(path)
        manifests.append({'rows':count,'path':str(path.resolve()),'sha256':sha(path.read_bytes()),'size':[width,height],'members':members})
    params={**PARAMS,'max_tokens':8192}
    manifest={'experiment':'CHAT_CROP_MONTAGE_QUALITY','production_changed':False,'requests':5,'image_count_per_request':1,
        'sizes':[10,20,30,40,50],'prompt':PROMPT,'parameters':params,
        'parameter_note':'8192 output tokens fixed for all5 sizes to accommodate50 row JSON; previous single test used2048',
        'preprocessing':'paste existing trim12 CHAT PNG RGB pixels at native size; 42px labels margin,24px row pitch; no resize',
        'script_sha':sha(Path(__file__).read_bytes()),'ground_truth':'saved human-verified diagnostic labels; never included in payload',
        'selection':'nested first50 from previous66; order roundrobin street','montages':manifests}
    save(out/'manifest.json',manifest)
    import httpx
    key=os.environ['AITUNNEL_API_KEY'];results=[]
    async with httpx.AsyncClient(headers={'Authorization':'Bearer '+key},timeout=httpx.Timeout(120,connect=10)) as client:
        for montage in manifests:
            count=montage['rows'];data=Path(montage['path']).read_bytes()
            payload={**params,'messages':[{'role':'user','content':[{'type':'text','text':PROMPT},
                {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data).decode()}}]}]}
            save(out/f'request_{count}.json',payload)
            started=time.perf_counter();raw=None;finish=None;usage=None;status=None;error=None;serverid=None;parsed={}
            try:
                response=await asyncio.wait_for(client.post('https://api.aitunnel.ru/v1/chat/completions',json=payload),timeout=120)
                (out/f'response_{count}.json').write_bytes(response.content);status=response.status_code
                obj=response.json()
                if status!=200:raise ValueError('HTTP error')
                serverid=obj.get('id');usage=obj.get('usage');choice=obj['choices'][0]
                raw=choice['message']['content'];finish=choice['finish_reason']
                decoded=json.loads(raw)
                if not isinstance(decoded,dict) or set(decoded)!={'rows'} or not isinstance(decoded['rows'],list):
                    raise ValueError('invalid rows JSON')
                for item in decoded['rows']:
                    if not isinstance(item,dict) or set(item)!={'row','cards'} or type(item['row'])!=int or item['row'] in parsed or not isinstance(item['cards'],list) or not all(isinstance(c,str) for c in item['cards']):
                        raise ValueError('invalid or duplicate row identity')
                    parsed[item['row']]=item['cards']
                if set(parsed)!=set(range(1,count+1)):raise ValueError('missing or extra numbered rows')
            except (httpx.HTTPError,asyncio.TimeoutError,ValueError,KeyError,TypeError,IndexError,AttributeError) as exc:
                error=type(exc).__name__+': '+str(exc);parsed={}
            wall=time.perf_counter()-started
            outputs=[]
            for member in montage['members']:
                row=member['montage_row'];cards=parsed.get(row)
                normalized=parse_cards(' '.join(cards),len(member['expected_cards']),member['street']) if cards is not None else None
                expected=member['expected_cards'];single=baseline[member['crop_sha']]['parsed_cards']
                outputs.append({'row':row,'crop_sha':member['crop_sha'],'input_path':member['input_path'],
                    'expected':expected,'raw_cards':cards,'parsed_cards':normalized,'single_image_cards':single,
                    'exact':normalized==expected and finish=='stop' and not error,'single_parity':normalized==single and normalized is not None})
            record={'montage_rows':count,'wall_s':wall,'http_status':status,'server_request_id':serverid,
                'raw_response':raw,'finish_reason':finish,'token_usage':usage,'error':error,'retry_count':0,
                'exact':sum(o['exact'] for o in outputs),'total':count,'single_parity':sum(o['single_parity'] for o in outputs),
                'outputs':outputs}
            results.append(record)
            with (out/'requests.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            save(out/'summary.json',{'status':'RUNNING','results':results})
            print(json.dumps({k:v for k,v in record.items() if k not in ('outputs','raw_response','token_usage')},ensure_ascii=False),flush=True)
    summary={'status':'COMPLETED','results':results,'cost_rub':sum((r['token_usage'] or {}).get('cost_rub',0) for r in results)}
    save(out/'summary.json',summary)
    lines=['# Qwen CHAT montage quality','', 'Original CHAT pixels pasted at native size.5 independent one-image requests, no retries. Diagnostic human labels.',
        '', '| Rows/image | Exact | Single-image parity | Wall s | Finish | Error |','|---:|---:|---:|---:|---|---|']
    for r in results:lines.append(f"|{r['total']}|{r['exact']}/{r['total']}|{r['single_parity']}/{r['total']}|{r['wall_s']:.3f}|{r['finish_reason']}|{r['error']}|")
    for r in results:
        count=r['total'];lines.extend(['',f'## {count} rows','',f'![montage](<{(out/f"montage_{count}.png").resolve()}>)','', '| Row | Expected | Actual | Exact |','|---|---|---|---|'])
        for o in r['outputs']:lines.append(f"|{o['row']}|{' '.join(o['expected'])}|{o['parsed_cards']}|{o['exact']}|")
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':asyncio.run(run(Path(sys.argv[1])))
