"""Inventory saved verified CHAT sequences; distinguish proven BOARD from unknown kind."""
import collections
import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/paddle_vl15_poker_sft_20260930'
OUT=ROOT/'runs/more_verified_chat_board_crops_20260930'

def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
def sha(b):return hashlib.sha256(b).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    old={r['candidate_id']:r for r in read(SOURCE/'verified_chat_card_sequences.jsonl')}
    previous={r['crop_sha'] for r in json.loads((ROOT/'runs/qwen_chat_board_quality66_20260930_v2/manifest.json').read_text(encoding='utf-8'))['selected']}
    accepted=[];unknown=[];conflicts=[];seen=set()
    for r in read(SOURCE/'combined_verified_chat_card_rows.jsonl'):
        cards=r['ground_truth'].split()
        if len(cards) not in (3,4,5) or r.get('tile_geometry_status')!='ACCEPTED':continue
        p=Path(r['model_input_png'])
        if not p.is_file() and r.get('source_video')=='smoke_15.mp4':
            p=ROOT/'runs/smoke15_chat_card_sample_5s_20260930/production_trim12_verified'/(r['source_sha256']+'.png')
        data=p.read_bytes();digest=sha(data)
        if digest!=r['model_input_sha256']:raise ValueError('input changed')
        with Image.open(p) as im:
            im=im.convert('RGB');size=im.size;pixelsha=sha(im.tobytes())
        if size[1]!=16 or size[0]>230:raise ValueError('wrongCHATgeometry')
        if digest in seen:continue
        seen.add(digest)
        source=old.get(r['record_id'],{})
        street=source.get('street');kind=source.get('source_user_kind');event=source.get('event_type')
        record={**r,'input_path':str(p.resolve()),'input_sha':digest,'pixel_sha':pixelsha,'size':list(size),
            'additional_to_previous66':digest not in previous,'source_user_kind':kind,'source_event_type':event,'source_street':street}
        if event in ('SHOW','SHOWDOWN') and street not in ('FLOP','TURN','RIVER'):
            record['classification']='CONFLICT_KIND_OR_SHOWDOWN';conflicts.append(record)
        elif kind=='BOARD' or street in ('FLOP','TURN','RIVER') or r.get('gt_status')=='USER_VERIFIED_SMOKE15':
            record['classification']='VERIFIED_CHAT_BOARD';record['board_basis']='user BOARD classification or recorded BOARD street; Smoke15 user-reviewed3..5card sequence'
            accepted.append(record)
        else:
            record['classification']='VERIFIED_CHAT_SEQUENCE_UNKNOWN_EVENT';unknown.append(record)
    for name,rows in [('board_manifest.jsonl',accepted),('unknown_event_manifest.jsonl',unknown),('conflicts_manifest.jsonl',conflicts)]:
        (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    summary={'status':'SAVED_CHAT_ONLY_INVENTORY','verified_board_unique':len(accepted),
        'new_board_vs_previous66':sum(r['additional_to_previous66'] for r in accepted),
        'unknown_event_unique':len(unknown),'conflicts':len(conflicts),
        'board_by_source':dict(collections.Counter(r['source_video'] for r in accepted)),
        'board_by_card_count':dict(collections.Counter(len(r['ground_truth'].split()) for r in accepted)),
        'api_calls':0,'production_changed':False}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
