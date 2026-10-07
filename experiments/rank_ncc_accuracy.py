"""Bounded offline rank experiment; source-held-out TRAIN, no suit mutations."""
import base64,csv,hashlib,io,json,sys,time,collections,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from parser_core.rank_ncc import RankNCC,rank_feature,RANKS
from parser_core.board_tiles import card_spans,_runs
OUT=ROOT/'runs/rank_ncc_accuracy';PRE=ROOT/'runs/smoke15_unresolved_rootcause';DATA=ROOT/'runs/card_train_smoke16_test_20260930';SPLIT=DATA/'split_v2'
SCORES=(.80,.85,.90,.93,.95,.96,.97,.98,.99,.995,.999,1.001)
MARGINS=(0,.01,.02,.03,.04,.05,.075,.10,.15,.20)
def jl(path):return [json.loads(x) for x in path.read_text('utf8').splitlines()]
def csvrows(path):return list(csv.DictReader(path.open(encoding='utf-8-sig')))
def sha(b):return hashlib.sha256(b).hexdigest()
def write(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
def image(raw):return np.asarray(Image.open(io.BytesIO(raw)).convert('RGB')).copy()
def tileload(row):
 raw=Path(row['tile_png']).read_bytes();assert sha(raw)==row['tile_sha256'];rgb=image(raw);assert sha(rgb.tobytes())==row['tile_pixel_sha256'];return rgb
def metrics(rows,field='pred'):
 n=len(rows);c=sum(x[field]==x['gt_rank'] for x in rows);w=sum(x[field] not in ('UNRESOLVED',None,'') and x[field]!=x['gt_rank'] for x in rows);u=n-c-w
 return dict(total=n,correct=c,wrong=w,unresolved=u,precision=c/(c+w) if c+w else None,coverage=(c+w)/n if n else None,end_to_end_accuracy=c/n if n else None,wrong_accepted_rate=w/n if n else None,unresolved_rate=u/n if n else None)
def decide(row,t,m):return row['rank_candidate'] if row['rank_best_score'] is not None and row['rank_best_score']>=t and row['rank_margin']>=m else 'UNRESOLVED'
def main():
 started=time.perf_counter();frame_provenance=[]
 OUT.mkdir(exist_ok=True);(OUT/'diagnostics').mkdir(exist_ok=True)
 templates=jl(SPLIT/'train_manifest.jsonl');valid=jl(SPLIT/'validation_manifest.jsonl')
 source_records={r['candidate_id']:r for r in jl(ROOT/'runs/paddle_vl15_poker_sft_20260930/verified_chat_card_sequences.jsonl')}
 cases=[];pixels={};provenance=[]
 for split,manifest in [('TRAIN_SOURCE_OUT',templates),('CALIBRATION',valid)]:
  for i,r in enumerate(manifest):
   key=f'{split}_{i:04}';rgb=tileload(r);pixels[key]=rgb
   source=source_records.get(r['record_id'],{})
   cases.append(dict(case_id=key,split=split,source_video=Path(r['source_video']).name.lower(),gt_rank=r['ground_truth'][:-1],gt_card=r['ground_truth'],vl_rank='UNRESOLVED',vl_status='RAW_VL_NOT_IN_THIS_MANIFEST',baseline_rank='UNRESOLVED',baseline_card='UNRESOLVED',suit='UNRESOLVED',crop_path=source.get('artifact_path',''),tile_path=r['tile_png'],frame=source.get('frame_id','UNKNOWN'),table=source.get('table_id','UNKNOWN'),physical_index=r['card_index'],pixel_sha256=r['tile_pixel_sha256'],prequeue=False,old_id=''))
 for i,r in enumerate(templates):
  source=source_records.get(r['record_id'],{});rgb=tileload(r)
  if not source or source.get('gt_status')!='VERIFIED':raise ValueError('Template lacks verified source provenance')
  box=None
  if Path(source['artifact_path']).exists():
   source_raw=Path(source['artifact_path']).read_bytes();assert sha(source_raw)==source['image_sha256']
   gray=np.asarray(Image.open(io.BytesIO(source_raw)).convert('L'));spans,_=card_spans(gray[:,:-12],recover_split=True)
   if len(spans)==len(source['ground_truth'].split()):
    x1,x2=spans[r['card_index']];box=[x1,0,x2+1,16]
  for method in ('fixed_ncc','component_ncc'):
   _,roi,reason=rank_feature(rgb,method)
   provenance.append(dict(template_index=i,rank=r['ground_truth'][:-1],source_video=source['source_video'],source_video_sha256=source['source_video_sha256'],frame=source['frame_id'],table=source['table_id'],row_id=source['row_id'],crop=source['artifact_path'],crop_sha256=source['image_sha256'],card_index=r['card_index'],card_box_in_source_row=json.dumps(box),tile=r['tile_png'],tile_sha256=r['tile_sha256'],tile_pixel_sha256=r['tile_pixel_sha256'],rank_roi=json.dumps(roi),normalization=method,size=16,interpolation='cv2.INTER_LINEAR',centering='mean',norm='L2',status=reason or 'READY'))
 write('template_manifest.csv',provenance)
 records=jl(ROOT/'runs/smoke15_ram_chat_parallel_full_20260930/visual_evidence.jsonl')[:239]
 hybrids={r['record']:r['hybrid'] for r in jl(PRE/'replay_evidence.jsonl')}
 old_tiles={x['id']:x for x in csvrows(ROOT/'runs/smoke15_ncc_accuracy_audit/tile_accuracy.csv')}
 for r in csvrows(PRE/'replay_tile_results.csv'):
  ri=int(r['record']);key=f'SMOKE15_{ri:03}_{int(r["physical_index"])}';h=hybrids[ri]
  production_missing=int(r['physical_index'])>=len(h['tiles'])
  # Structural y2 extraction is diagnostic only. Full-card output stays missing.
  if int(r['physical_index'])>=len(h['tiles']):
   rowrgb=image(base64.b64decode(records[ri-1]['crop_png_base64']))[:,:-12];gray=np.asarray(Image.fromarray(rowrgb).convert('L'))
   spans=[(a,b) for a,b in _runs(gray[2]>150) if a<150 and 18<=b-a+1<=22]
   expected=sum(int(x['record'])==ri for x in csvrows(PRE/'replay_tile_results.csv'))
   valid_geometry=len(spans)==expected and all(2<=spans[j][0]-spans[j-1][1]-1<=5 for j in range(1,len(spans)))
   if valid_geometry:
    x1,x2=spans[int(r['physical_index'])];rgb=rowrgb[:,x1:x2+1].copy()
   else:rgb=np.zeros((16,0,3),dtype=np.uint8)
   tile=dict(vl_rank='UNRESOLVED',final_rank='UNRESOLVED',suit='UNRESOLVED',tile_pixel_sha256=sha(rgb.tobytes()))
  else:
   tile=h['tiles'][int(r['physical_index'])];rgb=image(base64.b64decode(tile['tile_png_base64']));assert sha(rgb.tobytes())==tile['tile_pixel_sha256']
  pixels[key]=rgb
  # Per physical tile VL evidence is unavailable on count mismatch; never shift.
  vl=tile['vl_rank'];vl_status='POSITIONAL_COUNT_ALIGNED' if vl!='UNRESOLVED' else 'COUNT_MISMATCH_OR_NO_RAW_RANK'
  cases.append(dict(case_id=key,split='SMOKE15_DEV',source_video='smoke_15.mp4',gt_rank=r['gt'][:-1],gt_card=r['gt'],vl_rank=vl,vl_status=vl_status,baseline_rank=tile['final_rank'],baseline_card=r['new_card'],suit=tile.get('suit','UNRESOLVED'),crop_path=str(ROOT/f'runs/smoke15_ncc_accuracy_audit/crops/{ri:03}.png'),tile_path='DIAGNOSTIC_Y2_TILE' if production_missing and rgb.shape[1] else 'NO_TILE_PIXEL' if production_missing else old_tiles.get(r['old_id'],{}).get('tile_path','INLINE_REPLAY'),frame=r['frame'],table=r['table'],physical_index=r['physical_index'],pixel_sha256=tile['tile_pixel_sha256'],prequeue=production_missing,old_id=r['old_id']))
 frame_manifest=json.loads((PRE/'frame_context/extraction_manifest.json').read_text('utf8'))
 for r in frame_manifest['adjacent_visible_card_rows']:
  path=PRE/'frame_context'/f't{r["table"]:02}_frame{r["frame_index"]}_span{r["span_index"]:02}_adjacent_card_row.png'
  exact_chat=PRE/'frame_context'/f't{r["table"]:02}_frame{r["frame_index"]}_pyav_exact_chat_roi.png'
  y1,y2=r['padded_crop_y'];rgb=image(exact_chat.read_bytes())[y1:y2].copy();gray=np.asarray(Image.fromarray(rgb).convert('L'));spans,_=card_spans(gray[:,:-12],recover_split=True)
  frame_provenance.append(dict(source=str(exact_chat),source_png_sha256=sha(exact_chat.read_bytes()),row_box=[0,y1,rgb.shape[1],y2],actual_row_rgb_pixel_sha256=sha(rgb.tobytes()),legacy_manifest_pixel_sha256=r['pixel_sha256'],legacy_manifest_sha_matches=sha(rgb.tobytes())==r['pixel_sha256'],authority='EXACT_PYAV_SOURCE_AND_RECORDED_ROW_BOX; LEGACY_MANIFEST_HASH_MISMATCH'))
  labels=r['visible_cards_by_pixel_review'].split();assert len(spans)==len(labels)
  for j,(gt,(x1,x2)) in enumerate(zip(labels,spans)):
   key=f'SOURCE_{r["table"]}_{r["frame_index"]}_{j}';pixels[key]=rgb[:,x1:x2+1]
   cases.append(dict(case_id=key,split='SMOKE15_DEV',source_video='smoke_15.mp4',gt_rank=gt[:-1],gt_card=gt,vl_rank='UNRESOLVED',vl_status='NO_PRODUCTION_EVIDENCE',baseline_rank='UNRESOLVED',baseline_card='UNRESOLVED',suit='UNRESOLVED',crop_path=str(path),tile_path='EXTRACTED_FOR_DIAGNOSTIC_ONLY',frame=r['frame_index'],table=r['table'],physical_index=j,pixel_sha256=sha(pixels[key].tobytes()),prequeue=True,old_id=''))
 assert sum(r['split']=='SMOKE15_DEV' for r in cases)==900
 (OUT/'diagnostic_source_provenance.json').write_text(json.dumps(dict(frame_context=frame_provenance,production_missing_tiles=sum(r['split']=='SMOKE15_DEV' and r['prequeue'] and not r['case_id'].startswith('SOURCE_') for r in cases),diagnostic_y2_tiles=sum(r['tile_path']=='DIAGNOSTIC_Y2_TILE' for r in cases),no_tile_pixels=sum(r['tile_path']=='NO_TILE_PIXEL' for r in cases)),indent=2),'utf8')
 methods=('fixed_ncc','component_ncc');raw_results=[];timings={};models={}
 for method in methods:
  start=time.perf_counter();reader=RankNCC(SPLIT/'train_manifest.jsonl',method=method);models[method]=reader;init=time.perf_counter()-start
  times=[]
  for case in cases:
   tick=time.perf_counter();out=reader.read(pixels[case['case_id']],excluded_source=case['source_video'] if case['split']=='TRAIN_SOURCE_OUT' else None);times.append((time.perf_counter()-tick)*1000)
   raw_results.append({**case,'method':method,**out})
  timings[method]=dict(initialization_seconds=init,total_cards=len(cases),total_wall_seconds=sum(times)/1000,mean_ms_card=float(np.mean(times)),median_ms_card=float(np.median(times)),p95_ms_card=float(np.percentile(times,95)),p99_ms_card=float(np.percentile(times,99)),cards_per_second=1000/np.mean(times))
 sweep=[]
 for method in methods:
  allrows=[r for r in raw_results if r['method']==method]
  for split in ('CALIBRATION','TRAIN_SOURCE_OUT','SMOKE15_DEV','SELECTION_CALIBRATION_AND_SOURCE_OUT'):
   subset=[r for r in allrows if r['split']==split or split=='SELECTION_CALIBRATION_AND_SOURCE_OUT' and r['split'] in ('CALIBRATION','TRAIN_SOURCE_OUT')]
   for threshold,margin in itertools.product(SCORES,MARGINS):
    m=metrics([{**r,'pred':decide(r,threshold,margin)} for r in subset]);sweep.append(dict(method=method,split=split,threshold=threshold,margin=margin,**m))
 write('threshold_sweep.csv',sweep)
 candidates=[r for r in sweep if r['split']=='SELECTION_CALIBRATION_AND_SOURCE_OUT' and r['wrong']==0]
 selected=max(candidates,key=lambda r:(r['correct'],-methods.index(r['method']),r['threshold'],r['margin']))
 # Freeze selected algorithm/threshold before accessing historical Smoke16 labels.
 (OUT/'selection_frozen.json').write_text(json.dumps(dict(selected=selected,train_manifest_sha256=sha((SPLIT/'train_manifest.jsonl').read_bytes()),selection_sources=['smoke_12.mp4 leave-source-out','smoke_13.mp4 leave-source-out','smoke_14.mp4','smoke_15.mp4 historical validation'],smoke16_used_for_selection=False,untouched_validation='NOT_AVAILABLE_THIS_RUN'),indent=2),'utf8')
 method=selected['method'];threshold=selected['threshold'];margin=selected['margin'];reader=models[method]
 for i,r in enumerate(jl(DATA/'smoke16_test_manifest.jsonl')):
  key=f'HISTORICAL16_{i:03}';rgb=tileload(r);pixels[key]=rgb
  case=dict(case_id=key,split='HISTORICAL_SMOKE16',source_video='smoke_16.mp4',gt_rank=r['ground_truth'][:-1],gt_card=r['ground_truth'],vl_rank='UNRESOLVED',vl_status='VL_NOT_LOADED',baseline_rank='UNRESOLVED',baseline_card='UNRESOLVED',suit='UNRESOLVED',crop_path=r['source_row_png'],tile_path=r['tile_png'],frame=r['frame_index_by_pts_512'],table=r['table_id'],physical_index=r['card_index'],pixel_sha256=r['tile_pixel_sha256'],prequeue=False,old_id='')
  cases.append(case);raw_results.append({**case,'method':method,**reader.read(rgb)})
 # Quarantine is a post-selection diagnostic only; never enter threshold sweep.
 for i,r in enumerate(jl(SPLIT/'quarantined_validation_rows.jsonl')):
  key=f'QUARANTINED_{i:03}';rgb=tileload(r);pixels[key]=rgb;source=source_records.get(r['record_id'],{})
  case=dict(case_id=key,split='QUARANTINED_DIAGNOSTIC',source_video=Path(r['source_video']).name.lower(),gt_rank=r['ground_truth'][:-1],gt_card=r['ground_truth'],vl_rank='UNRESOLVED',vl_status='RAW_VL_NOT_IN_THIS_MANIFEST',baseline_rank='UNRESOLVED',baseline_card='UNRESOLVED',suit='UNRESOLVED',crop_path=source.get('artifact_path',''),tile_path=r['tile_png'],frame=source.get('frame_id','UNKNOWN'),table=source.get('table_id','UNKNOWN'),physical_index=r['card_index'],pixel_sha256=r['tile_pixel_sha256'],prequeue=False,old_id='')
  cases.append(case);raw_results.append({**case,'method':method,**reader.read(rgb)})
 # Already disclosed sources: exploratory sweep cannot change frozen selection.
 exploratory=[]
 for scope in ('EXPLORATORY_KNOWN16','EXPLORATORY_QUARANTINED','EXPLORATORY_ALL_KNOWN','EXPLORATORY_NONTRAIN_KNOWN'):
  subset=[r for r in raw_results if r['method']==method and (scope=='EXPLORATORY_ALL_KNOWN' or scope=='EXPLORATORY_NONTRAIN_KNOWN' and r['split']!='TRAIN_SOURCE_OUT' or scope=='EXPLORATORY_KNOWN16' and r['split']=='HISTORICAL_SMOKE16' or scope=='EXPLORATORY_QUARANTINED' and r['split']=='QUARANTINED_DIAGNOSTIC')]
  for t,m in itertools.product(SCORES,MARGINS):
   exploratory.append(dict(method=method,split=scope,threshold=t,margin=m,**metrics([{**r,'pred':decide(r,t,m)} for r in subset])))
 write('threshold_sweep.csv',sweep+exploratory)
 all_known=[r for r in exploratory if r['split']=='EXPLORATORY_ALL_KNOWN']
 best_known=max([r for r in all_known if r['wrong']==0],key=lambda r:(r['correct'],r['threshold'],r['margin']))
 nontrain=[r for r in exploratory if r['split']=='EXPLORATORY_NONTRAIN_KNOWN']
 secondary_split={split:metrics([{**r,'pred':decide(r,best_known['threshold'],best_known['margin'])} for r in raw_results if r['method']==method and r['split']==split]) for split in ('TRAIN_SOURCE_OUT','CALIBRATION','SMOKE15_DEV','HISTORICAL_SMOKE16','QUARANTINED_DIAGNOSTIC')}
 exploratory_summary=dict(scope='POST_FREEZE_ALREADY_DISCLOSED_NOT_INDEPENDENT_NOT_SAFE',best_zero_wrong_all_known=best_known,best_zero_wrong_nontrain=max([r for r in nontrain if r['wrong']==0],key=lambda r:(r['correct'],r['threshold'],r['margin'])),secondary_selected_per_split=secondary_split,max_end_to_end=max(all_known,key=lambda r:r['end_to_end_accuracy']),max_coverage=max(all_known,key=lambda r:r['coverage']),configurations_meeting_999_end_to_end=sum(r['end_to_end_accuracy']>=.999 for r in all_known),configurations_meeting_999_precision=sum(r['precision'] is not None and r['precision']>=.999 for r in all_known))
 # Do not conflate missing CARD segmentation with loss before the queue.
 for row in raw_results:
  row['fullcard_evidence_unavailable']=row['prequeue']
  row['prequeue']=row['case_id'].startswith('SOURCE_')
  row['pixel_evidence_origin']=('NO_NATIVE_TILE' if row['tile_path']=='NO_TILE_PIXEL' else
      'SOURCE_FRAME_DIAGNOSTIC' if row['prequeue'] else 'NATIVE_PRODUCTION_REPLAY' if row['split']=='SMOKE15_DEV' else 'VERIFIED_CORPUS')
 results=[]
 for r in raw_results:
  if r['method']!=method:continue
  result=decide(r,threshold,margin)
  selected_reason=None if result!='UNRESOLVED' else 'RANK_NCC_LOW_CONFIDENCE' if r['rank_candidate'] in RANKS else r['rank_reason']
  results.append({**r,'original_raw_reason':r['rank_reason'],'rank_reason':selected_reason,'rank_accepted':result!='UNRESOLVED','rank_result':result,'correct':result==r['gt_rank'],'wrong':result not in ('UNRESOLVED',r['gt_rank']),'selected_threshold':threshold,'selected_margin':margin})
 for r in raw_results:r['policy_scope']='RAW_DEFAULT_POLICY_NOT_SELECTED'
 write('rank_results.csv',results);write('algorithm_raw_results.csv',raw_results)
 summary={split:metrics([r for r in results if r['split']==split],'rank_result') for split in sorted(set(r['split'] for r in results))}
 perrank=[];matrix=[]
 for split in summary:
  subset=[r for r in results if r['split']==split]
  for rank in RANKS:
   perrank.append(dict(split=split,rank=rank,**metrics([r for r in subset if r['gt_rank']==rank],'rank_result')))
   for pred in (*RANKS,'UNRESOLVED'):matrix.append(dict(split=split,gt_rank=rank,predicted=pred,count=sum(r['gt_rank']==rank and r['rank_result']==pred for r in subset),ungated_count=sum(r['gt_rank']==rank and r['rank_candidate']==pred for r in subset)))
 write('rank_per_class.csv',perrank);write('rank_confusion_matrix.csv',matrix)
 smoke=[r for r in results if r['split']=='SMOKE15_DEV'];vs=[];strategies=[];full=[]
 current84={r['id'] for r in csvrows(PRE/'vl_rank_failures.csv') if r['new_correct']!='True'}
 for r in smoke:
  category=lambda v:'unresolved' if v=='UNRESOLVED' else 'correct' if v==r['gt_rank'] else 'wrong'
  vs.append(dict(case_id=r['case_id'],old_id=r['old_id'],gt=r['gt_rank'],vl=r['vl_rank'],vl_status=r['vl_status'],rankncc=r['rank_result'],comparison='VL_'+category(r['vl_rank'])+'/NCC_'+category(r['rank_result']),in_current84=r['old_id'] in current84))
 write('vl_vs_rankncc.csv',vs)
 for strategy in ('BASELINE','A_VL_PRIMARY_NCC_FALLBACK','B_NCC_PRIMARY_VL_FALLBACK','C_AGREEMENT_ONLY','D_NCC_ONLY','E_BASELINE_PIXEL_FALLBACK'):
  ranked=[];cards=[]
  for r in smoke:
   vl=r['vl_rank'];ncc=r['rank_result'];base=r['baseline_rank']
   pred=(base if strategy=='BASELINE' else vl if strategy=='A_VL_PRIMARY_NCC_FALLBACK' and vl!='UNRESOLVED' else ncc if strategy=='A_VL_PRIMARY_NCC_FALLBACK' else ncc if strategy=='B_NCC_PRIMARY_VL_FALLBACK' and ncc!='UNRESOLVED' else vl if strategy=='B_NCC_PRIMARY_VL_FALLBACK' else vl if strategy=='C_AGREEMENT_ONLY' and vl==ncc else 'UNRESOLVED' if strategy=='C_AGREEMENT_ONLY' else ncc if strategy=='D_NCC_ONLY' else base if base!='UNRESOLVED' else ncc)
   ranked.append({**r,'pred':pred})
   card=r['baseline_card'] if strategy=='BASELINE' else pred+r['suit'] if pred!='UNRESOLVED' and r['suit'] in ('s','h','d','c') and not r['fullcard_evidence_unavailable'] else 'UNRESOLVED'
   cards.append(dict(case_id=r['case_id'],gt_rank=r['gt_card'],pred=card,baseline_card=r['baseline_card']))
  strategies.append(dict(strategy=strategy,unit='RANK',**metrics(ranked)))
  grouped=collections.defaultdict(list)
  for c in cards:grouped[c['case_id'].rsplit('_',1)[0]].append(c['pred'])
  duplicate_rows=sum(len([v for v in vals if v!='UNRESOLVED'])!=len(set(v for v in vals if v!='UNRESOLVED')) for vals in grouped.values())
  full.append(dict(strategy=strategy,unit='FULL_CARD',evidence_scope='TILE_HYPOTHETICAL_NOT_PRODUCTION',duplicate_proposal_rows=duplicate_rows,**metrics(cards),old661_regressions=sum(c['pred']!=c['gt_rank'] for c,r in zip(cards,smoke) if old_tiles.get(r['old_id'],{}).get('category')=='CORRECT_FULL_CARD'),current778_regressions=sum(c['baseline_card']==c['gt_rank'] and c['pred']!=c['gt_rank'] for c in cards)))
 write('hybrid_comparison.csv',strategies);write('full_card_before_after.csv',full)
 problem=[r for r in results if r['rank_result']=='UNRESOLVED' or r['wrong'] or r['vl_rank'] not in ('UNRESOLVED',r['rank_result'])]
 thumbs=[]
 for i,r in enumerate(problem):
  rgb=pixels[r['case_id']];canvas=Image.new('RGB',(520,150),'white');draw=ImageDraw.Draw(canvas);tile=Image.fromarray(rgb)
  if tile.width:canvas.paste(tile.resize((tile.width*5,tile.height*5),Image.Resampling.NEAREST),(5,25))
  else:draw.text((5,35),'NO_TILE_PIXEL',fill='black')
  if r['rank_roi']:
   roi=tile.crop(r['rank_roi']);canvas.paste(roi.resize((roi.width*5,roi.height*5),Image.Resampling.NEAREST),(125,25))
  score=lambda v:'N/A' if v is None else f'{v:.6f}'
  draw.text((5,5),r['case_id']+' '+r['split'],fill='black');draw.text((190,28),f'GT {r["gt_rank"]} VL {r["vl_rank"]} NCC {r["rank_candidate"]}\nFINAL {r["rank_result"]}\nBEST {score(r["rank_best_score"])} SECOND {r["rank_second_candidate"]} {score(r["rank_second_score"])}\nMARGIN {score(r["rank_margin"])}\nROI {r["rank_roi"]}',fill='black')
  canvas.save(OUT/'diagnostics'/f'{i:04}_{r["case_id"]}.png');thumbs.append(canvas)
 for start in range(0,len(thumbs),16):
  page=Image.new('RGB',(1040,1200),'white')
  for j,thumb in enumerate(thumbs[start:start+16]):page.paste(thumb,((j%2)*520,(j//2)*150))
  page.save(OUT/f'diagnostics_{start//16:03}.png')
 bestdev=max([r for r in sweep if r['split']=='SMOKE15_DEV' and r['wrong']==0],key=lambda r:(r['correct'],r['threshold'],r['margin']))
 comparison84=[r for r in smoke if r['old_id'] in current84];assert len(comparison84)==84
 leak={split:sum(r['pixel_sha256'] in {t['tile_pixel_sha256'] for t in templates} for r in results if r['split']==split) for split in summary}
 perf=dict(method=method,**timings[method],all_method_timings=timings,video_overhead='ESTIMATE_ONLY: number_of_card_reads * mean_ms_card; video inference/queue speed not measured',estimated_900_card_seconds=900*timings[method]['mean_ms_card']/1000)
 (OUT/'performance.json').write_text(json.dumps(perf,indent=2),'utf8')
 summary_all=dict(selected=selected,rank_metrics=summary,remaining84=metrics(comparison84,'rank_result'),best_zero_wrong_dev=bestdev,exploratory_known=exploratory_summary,rank_hybrids=strategies,full_card=full,pixel_overlap_train=leak,diagnostic_images=len(problem),template_count=len(templates),template_sources=sorted(set(r['source_video'] for r in templates)),independent_target='NOT_MEASURED',production_enabled=False,performance=perf,runner_elapsed_seconds=time.perf_counter()-started)
 (OUT/'summary.json').write_text(json.dumps(summary_all,indent=2),'utf8');print(json.dumps(summary_all,indent=2))
if __name__=='__main__':main()
