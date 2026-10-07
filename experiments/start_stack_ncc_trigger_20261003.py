"""Offline NCC early capture trigger; no NCC hit grants money/hand authority."""
import argparse,collections,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from start_stack_wide_20261003 import ROOT,PARENT,read_jsonl,jsonl,dump,digest,database,table
THRESHOLD=.95

def run(out):
    import cv2,numpy as np
    target=out/'ncc_dealing_cards_trigger';target.mkdir(exist_ok=True)
    hands=read_jsonl(out/'selected_hands.jsonl');reviews=read_jsonl(out/'visual_reviews.jsonl')
    clean=read_jsonl(out/'clean_forced_window/start_stack_results.jsonl');windows=read_jsonl(out/'clean_forced_window/hand_windows.jsonl')
    rv={(r['table'],r['hand_id']):r for r in reviews};wm={(w['table'],w['hand_id']):w for w in windows}
    profile=json.loads((ROOT/'configs/layout_847x404.json').read_text(encoding='utf-8'))
    reference=PARENT/'crops/rows/table_95984db438dc55f85dee9067/000000681_018.png'
    template_full=cv2.imdecode(np.fromfile(reference,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    template=template_full[:,4:134].copy()
    dump(target/'ncc_template_manifest.json',{'source':str(reference),'file_sha256':digest(reference),
        'source_row_gray_sha256':hashlib.sha256(template_full.tobytes()).hexdigest(),'template_rect':[4,0,134,16],
        'template_pixel_sha256':hashlib.sha256(template.tobytes()).hexdigest(),'verified_literal':'Dealer: Dealing cards',
        'threshold':THRESHOLD,'threshold_selection':'Predeclared0.95; no tuning on sample outcomes.',
        'calibration_hand':'T01 #2875498927','holdout_note':'19otherhands reuse this template; samevideo/session corpus, not generalization acceptance.'})
    db=database();hits=[];captures=[];results=[];opencv_seconds=0.;decode_hash_seconds=0.
    for c in hands:
        key=(c['table'],c['hand_id']);r=rv[key];gate=r['deal_complete_frame'];slot=f"slot_{int(c['table'][1:]):02d}";box=profile['slots'][slot]['chat']['chat_log']
        scanned=[]
        for frame in range(max(0,c['start']-10),gate+4):
            started=time.perf_counter();im,meta=table(db,c['session'],frame);decode_hash_seconds+=time.perf_counter()-started
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);chat=gray[box[1]:,box[0]:box[2]]
            tick=time.perf_counter();match=cv2.matchTemplate(chat,template,cv2.TM_CCOEFF_NORMED);_,maximum,_,location=cv2.minMaxLoc(match);opencv_seconds+=time.perf_counter()-tick
            hit=maximum>=THRESHOLD
            scanned.append(dict(meta,table=c['table'],hand_id=c['hand_id'],score=maximum,ncc_hit=hit,match_chat_local_xy=list(location),
                authority='CAPTURE_TRIGGER_ONLY',actual_hand_start_source_frame=c['start'],deal_complete_confirmation_frame=gate))
        raw=next((p for p in scanned if p['ncc_hit']),None)
        current=next((p for p in scanned if p['frame']>=c['start'] and p['ncc_hit']),None)
        hits.extend(scanned)
        # Capture starts immediately at a raw hit, even when current-hand or
        # row lifecycle is UNKNOWN. Opening acceptance still needs TABLE gate.
        start=raw['frame'] if raw else None
        snapshots=[s for s in clean if (s['table'],s['hand_id'])==key]
        close=wm[key]['window_close_exclusive'];first=wm[key]['first_stack_changing_action']
        confirmed=[s['stable_confirmed_frame'] for s in snapshots if s['status']=='START_STACK_RECOVERED']
        # Retain capture through the latest successful confirmation. Seats
        # without a numeric result stay pending through the clean window.
        capture_end=close if any(s['status']!='START_STACK_RECOVERED' for s in snapshots) else min(close,max(confirmed,default=gate+3)+1)
        if start is not None:
            for frame in range(start,capture_end):
                im,meta=table(db,c['session'],frame)
                for seat in r['occupied_seats']:
                    for kind in ['nick','stack']:
                        roi=profile['slots'][slot]['seats'][seat][kind];pixels=im[roi[1]:roi[3],roi[0]:roi[2]]
                        captures.append(dict(meta,table=c['table'],hand_id=c['hand_id'],seat=seat,kind=kind,roi=roi,
                            crop_bgr_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                            acceptance_window_open=frame>gate,provenance='RAM ROI borrowed, hashed, released; no intermediate image write'))
        result={'table':c['table'],'hand_id':c['hand_id'],'ncc_first_hit_frame':start,
            'ncc_hit_after_hand_start_frame':current['frame'] if current else None,'confirmed_deal_complete_frame':gate,
            'ncc_current_candidate_lead_to_confirmed_deal_frames':gate-current['frame'] if current else None,
            'first_stack_changing_action_frame':first['frame_id'] if first else None,'window_close_exclusive':close,
            'capture_frames_before_close_from_raw_hit':close-start if start is not None else 0,
            'actual_capture_end_exclusive':capture_end,'actual_captured_source_frames':max(0,capture_end-start) if start is not None else 0,
            'confirmed_clean_window_source_frames':max(0,close-gate-1),
            'raw_hit_precedes_current_hand':bool(raw and raw['frame']<c['start']),
            'raw_row_lifecycle':'UNKNOWN_STALE_LINE_POSSIBLE','ncc_hit_is_deal_complete':False,
            'opening_evidence':'Independent reviewed current hand + actual forced TABLE chips/CHAT + all occupied dealt cardbacks at gate.',
            'old_line_not_reopened_rule':'Acceptance opens once per independently confirmed current hand. Persistent/reappearing NCC row never reopens or increments hand/occupancy epoch.',
            'seats':[{k:s.get(k) for k in ['seat','player_id','snapshot_frame','stable_confirmed_frame','status']} for s in snapshots],
            'stable_seats_in_window':sum(s['status']=='START_STACK_RECOVERED' for s in snapshots)}
        results.append(result)
    jsonl(target/'ncc_frame_scores.jsonl',hits);jsonl(target/'trigger_capture_metadata.jsonl',captures);jsonl(target/'hand_trigger_comparison.jsonl',results)
    captured_keys={(p['table'],p['hand_id'],p['seat'],p['kind'],p['frame']) for p in captures}
    baseline_stable=[s for s in clean if s['status']=='START_STACK_RECOVERED']
    captured_stable=sum(all((s['table'],s['hand_id'],s['seat'],'stack',f) in captured_keys for f in s['source_frames']) for s in baseline_stable)
    summary={'ncc_role':'Fast tentative capture trigger only; no downstream OCR required to produce hit.',
        'sample_hands':len(hands),'tables':len({c['table'] for c in hands}),
        'frames_scanned':len(hits),'threshold':THRESHOLD,'hands_with_ncc_hit':sum(r['ncc_first_hit_frame'] is not None for r in results),
        'hands_with_candidate_hit_after_known_header':sum(r['ncc_hit_after_hand_start_frame'] is not None for r in results),
        'early_raw_hits_before_current_hand':sum(r['raw_hit_precedes_current_hand'] for r in results),
        'ncc_hit_can_be_stale':any(r['raw_hit_precedes_current_hand'] for r in results),
        'ncc_lead_frames_to_confirmed_deal':[r['ncc_current_candidate_lead_to_confirmed_deal_frames'] for r in results],
        'capture_roi_observations':len(captures),'intermediate_image_files_written':0,
        'baseline_clean_stable_seats_before_first_money':len(baseline_stable),
        'actual_ncc_captured_stable_seats_before_first_money':captured_stable,
        'baseline_stable_seats_missing_ncc_capture':len(baseline_stable)-captured_stable,
        'hands_without_ncc_hit':[{'table':r['table'],'hand_id':r['hand_id']} for r in results if r['ncc_first_hit_frame'] is None],
        'lead_metric_scope':'Lead to conservative reviewed checkpoint, not earliest visual DEAL_COMPLETE or measured OCR latency.',
        'opencv_matching_seconds':opencv_seconds,'saved_table_decode_hash_seconds':decode_hash_seconds,
        'opencv_match_ms_per_saved_table':1000*opencv_seconds/len(hits) if hits else None,
        'independent_ncc_trigger_benefit_in_seconds':None,'downstream_ocr_wait_time_not_measured':True,
        'verdict':'NCC_CAPTURE_TRIGGER_SUPPORTED_ON_SAVED_SAMPLE; NCC_ONLY_WINDOW_OPENING_NOT_PROVEN',
        'causal_replay_limits':['Saved TABLE/CHAT frames are read offline. New-hand headers and DEAL_COMPLETE confirmation were reviewed offline; live causal confirmation detector is not implemented.',
            'Raw hit can start pending ROI capture without any hand authority; persisted row must not reopen window.',
            'NCC temporal row lifecycle itself unproven; TABLE/current-hand confirmation supplies authoritative acceptance gate.',
            'Referencehand part of sample; independent19hands samevideo, no generalization certificate.']}
    dump(target/'summary.json',summary);db.close();return summary

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.perf_counter();r=run(a.output);r['wall_seconds']=time.perf_counter()-start;print(json.dumps(r,ensure_ascii=False,indent=2))
