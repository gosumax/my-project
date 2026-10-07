"""202 accepted verified small CHAT card-sequences, each its own parallel request."""
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.test_qwen_chat_board_quality100 import run

def rows():
    manifest=ROOT/'runs/more_verified_chat_board_crops_20260930/all_chat_card_sequences_manifest.jsonl'
    result=[]
    for r in [json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines()]:
        data=Path(r['input_path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=r['input_sha']:raise ValueError('CHAT input changed')
        with Image.open(r['input_path']) as im:width,height=im.size
        if height!=16:raise ValueError('not smallCHAT row')
        cards=r['ground_truth'].split()
        result.append({**r,'crop_sha':r['input_sha'],'width':width,'height':height,
            'expected_cards':cards,'street':{3:'FLOP',4:'TURN',5:'RIVER'}[len(cards)],
            'street_basis':'card-count bucket only; actualevent not used in quality assessment'})
    if len(result)!=202 or len({r['crop_sha'] for r in result})!=202:raise ValueError('202unique required')
    return result

if __name__=='__main__':
    out=Path(sys.argv[1]);asyncio.run(run(out,'--check-only' in sys.argv,rows()))
    p=out/'manifest.json';m=json.loads(p.read_text(encoding='utf-8'))
    m['launcher_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    m['quality_scope']='allverified3..5cardCHAT sequences irrespectiveofevent; noBOARDtablecrops'
    p.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
