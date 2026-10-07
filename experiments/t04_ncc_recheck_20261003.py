"""Independent bounded replay of frozen NCC scores; experimental evidence only."""
import collections, hashlib, json, sys, time
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from start_stack_wide_20261003 import ROOT, PARENT, database, table, read_jsonl, dump, jsonl, digest

def run():
    begin=time.perf_counter()
    out=ROOT/'runs/start_stack_frame_ledger_wide_smoke14_20261003'
    target=out/'t04_ncc_acceptance'
    original=out/'ncc_dealing_cards_trigger'
    scores=read_jsonl(original/'ncc_frame_scores.jsonl')
    truth_path=target/'reviewed_positive_matches.json'
    truth=json.loads(truth_path.read_text(encoding='utf-8'))
    truth_map={x['sha']:x for x in truth}
    hands=read_jsonl(out/'selected_hands.jsonl')
    hm={(x['table'],x['hand_id']):x for x in hands}
    profile=json.loads((ROOT/'configs/layout_847x404.json').read_text(encoding='utf-8'))
    template_manifest=json.loads((original/'ncc_template_manifest.json').read_text(encoding='utf-8'))
    ref=Path(template_manifest['source'])
    template_full=cv2.imdecode(np.fromfile(ref,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    template=template_full[:,4:134].copy()
    assert digest(ref)==template_manifest['file_sha256']
    assert hashlib.sha256(template.tobytes()).hexdigest()==template_manifest['template_pixel_sha256']
    threshold=template_manifest['threshold']
    assert threshold==.95
    db=database();results=[];failures=[];matching_seconds=0.;expected=[]
    before_hashes={str(p):digest(p) for p in [original/'summary.json',original/'ncc_frame_scores.jsonl',original/'ncc_template_manifest.json']}
    for s in scores:
        hand=hm[(s['table'],s['hand_id'])]
        im,meta=table(db,hand['session'],s['frame'])
        box=profile['slots'][f"slot_{int(s['table'][1:]):02d}"]['chat']['chat_log']
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        chat=gray[box[1]:,box[0]:box[2]]
        tick=time.perf_counter()
        _,score,_,xy=cv2.minMaxLoc(cv2.matchTemplate(chat,template,cv2.TM_CCOEFF_NORMED))
        matching_seconds+=time.perf_counter()-tick
        x,y=xy;patch=chat[y:y+template.shape[0],x:x+template.shape[1]]
        sha=hashlib.sha256(patch.tobytes()).hexdigest()
        bgr_chat=im[box[1]:,box[0]:box[2]]
        bgr_sha=hashlib.sha256(bgr_chat[y:y+template.shape[0],x:x+template.shape[1]].tobytes()).hexdigest()
        metadata_ok=all(meta[k]==s[k] for k in meta)
        score_ok=abs(score-s['score'])<=1e-7
        xy_ok=list(xy)==s['match_chat_local_xy']
        hit=score>=threshold
        if not(metadata_ok and score_ok and xy_ok and hit==s['ncc_hit']):
            failures.append({'table':s['table'],'hand_id':s['hand_id'],'frame':s['frame'],'metadata_ok':metadata_ok,'score_ok':score_ok,'xy_ok':xy_ok})
        literal=truth_map[bgr_sha]['verified_literal'] if hit and bgr_sha in truth_map else None
        if hit and literal is None:raise ValueError(f'UNREVIEWED_POSITIVE_PATCH {sha}')
        classification=('WRONG_LITERAL' if literal!='Dealer: Dealing cards' else 'MATCH' if s['frame']>=hand['deal_chat_frame'] else 'STALE_TARGET') if hit else 'BELOW_THRESHOLD'
        r=dict(meta,table=s['table'],hand_id=s['hand_id'],score=score,threshold=threshold,
            match_chat_local_xy=list(xy),match_table_xy=[box[0]+x,box[1]+y],
            patch_gray_sha256=sha,patch_bgr_sha256=bgr_sha,ncc_hit=hit,verified_literal=literal,classification=classification,
            expected_deal_chat_frame=hand['deal_chat_frame'],metadata_parity=metadata_ok,score_parity=score_ok,maxloc_parity=xy_ok)
        results.append(r)
        if s['frame']==hand['deal_chat_frame']:
            name=f"recheck_expected_{s['table']}_{s['hand_id']}_{s['frame']}.png"
            cv2.imencode('.png',patch)[1].tofile(target/name)
            expected.append(dict(r,patch_path=str(target/name)))
    db.close()
    hand_results=[]
    for h in hands:
        rs=[r for r in results if (r['table'],r['hand_id'])==(h['table'],h['hand_id'])]
        counts=collections.Counter(r['classification'] for r in rs)
        hand_results.append({'table':h['table'],'hand_id':h['hand_id'],'expected_deal_chat_frame':h['deal_chat_frame'],
            'frames':len(rs),'hit':counts['MATCH']>0,'counts':dict(counts),
            'first_actual_target_candidate_frame':next((r['frame'] for r in rs if r['classification']=='MATCH'),None),
            'maximum_score':max(r['score'] for r in rs)})
    counter=collections.Counter(r['classification'] for r in results)
    summary={'scope':'Original exact 1224 saved TABLE frames only; template and threshold unchanged.',
        'frames':len(results),'hands':len(hands),'threshold':threshold,
        'HITS':sum(x['hit'] for x in hand_results),'MISSES':sum(not x['hit'] for x in hand_results),
        'FALSE_POSITIVES':counter['WRONG_LITERAL'],'false_positive_unit':'source frames; wrong visible lexical literal',
        'false_positive_unique_patch_hashes':len({r['patch_gray_sha256'] for r in results if r['classification']=='WRONG_LITERAL'}),
        'false_positive_hands':sum(x['counts'].get('WRONG_LITERAL',0)>0 for x in hand_results),
        'raw_above_threshold_frames':sum(r['ncc_hit'] for r in results),'raw_above_threshold_hands':sum(any(r['ncc_hit'] for r in results if (r['table'],r['hand_id'])==(x['table'],x['hand_id'])) for x in hands),
        'true_literal_above_threshold_frames':counter['MATCH']+counter['STALE_TARGET'],
        'current_actual_target_candidate_frames':counter['MATCH'],
        'stale_true_literal_frames':counter['STALE_TARGET'],
        'stale_true_literal_hands':sum(x['counts'].get('STALE_TARGET',0)>0 for x in hand_results),
        'unique_reviewed_positive_patch_hashes':len({r['patch_gray_sha256'] for r in results if r['ncc_hit']}),
        'parity_checks_passed':len(results)-len(failures),'parity_checks_failed':len(failures),
        'original_files_unchanged':all(digest(Path(p))==v for p,v in before_hashes.items()),
        'original_hashes':before_hashes,'truth_file_sha256':digest(truth_path),
        'opencv_matching_seconds':matching_seconds,'wall_seconds':time.perf_counter()-begin,
        'live_ocr_latency':'NOT_MEASURED','tracking_state_replay':'NOT_MEASURED',
        'current_candidate_limit':'MATCH means literal card line and frame >= independently selected current DEAL frame; physical row lineage not certified.',
        'negative_frame_limit':'Every below-threshold frame recomputed; its visible literal not automatically classified. Two missed T04 frames require separate visual diagnosis.',
        'hand_results':hand_results,'failures':failures[:2]}
    jsonl(target/'recheck_frames.jsonl',results)
    jsonl(target/'recheck_expected_frames.jsonl',expected)
    dump(target/'recheck_summary.json',summary)
    return summary

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(run(),ensure_ascii=False,indent=2))
