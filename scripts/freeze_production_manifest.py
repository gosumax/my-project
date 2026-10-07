"""INTERNAL ONLY: explicit reviewed golden freeze, never called by runtime/CI."""
import argparse,ast,hashlib,json,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--reviewed-freeze',action='store_true')
 p.add_argument('--version',required=True,help='New unique production version; prior golden is immutable')
 p.add_argument('--benchmark-run',required=True,help='Reviewed completed 500-frame candidate run')
 p.add_argument('--benchmark-video',default='video/smoke_18.mp4',help='Exact reviewed benchmark source, relative to project root')
 p.add_argument('--benchmark-manifest',required=True,help='Exact manifest saved before candidate execution')
 args=p.parse_args()
 if not args.reviewed_freeze:p.error('Explicit reviewed promotion required; hashes are not auto-updated')
 if not args.version.startswith('production-golden-') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.version):
  p.error('Use a safe, explicit production-golden- version name')
 if (ROOT/'provenance'/args.version).exists():p.error('Golden destination already exists; choose a new version')
 seed=['run_production.py','scripts/run_smoke14_ram.py','parser_core/production_integrity.py','scripts/tiny_chat_worker.py','scripts/check_production_integrity.py']
 closure=set(seed);pending=list(seed)
 while pending:
  tree=ast.parse((ROOT/pending.pop()).read_text('utf-8-sig'))
  for n in ast.walk(tree):
   targets=[x.name for x in n.names] if isinstance(n,ast.Import) else [n.module] if isinstance(n,ast.ImportFrom) and n.module else []
   for target in targets:
    for candidate in (target.replace('.','/')+'.py',target.replace('.','/')+'/__init__.py'):
     if (ROOT/candidate).is_file() and candidate not in closure:
      closure.add(candidate);pending.append(candidate)
 modules=sorted(closure)
 for path in list(modules):
  parent=(ROOT/path).parent
  while parent!=ROOT:
   init=parent/'__init__.py'
   if init.is_file():closure.add(init.relative_to(ROOT).as_posix())
   parent=parent.parent
 modules=sorted(closure)
 configs=['configs/ocr_tiny_local.json','configs/card_hybrid_local.json','configs/suit_backup_local.json','configs/rank_ncc_production.json','configs/nine_table_layout.json','configs/layout_847x404.json','configs/chat_log_bottom_rois.json']
 exporter='scripts/build_smoke14_ram_workbook.mjs'
 tests=sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'tests').glob('test_*.py'))
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 golden=args.version
 baseline=json.loads((ROOT/args.benchmark_run/'final_summary.json').read_text('utf8'))
 benchmark_meta=json.loads((ROOT/args.benchmark_run/'run_metadata.json').read_text('utf8'))
 benchmark_manifest=Path(args.benchmark_manifest)
 benchmark_pins=json.loads(benchmark_manifest.read_text('utf8'))
 if sha(benchmark_manifest)!=benchmark_meta.get('production_manifest_sha256'):
  p.error('Saved candidate manifest does not match benchmark fingerprint')
 if set(benchmark_pins['modules'])!=set(modules) or set(benchmark_pins['configs'])!=set(configs):
  p.error('Benchmark dependency closure differs from promoted implementation')
 for name in modules+configs+[exporter]:
  if benchmark_pins['hashes'].get(name)!=sha(ROOT/name):p.error('Benchmark runtime/config changed: '+name)
 if benchmark_meta.get('production_modified_files'):
  p.error('Benchmark was run with modified registered files')
 if baseline.get('frames')!=500 or benchmark_meta.get('frames_requested')!=500:
  p.error('Golden performance requires exactly 500 requested/decoded frames')
 if baseline.get('status') not in ('COMPLETE','PARTIAL_VISUAL') or baseline.get('errors') or baseline.get('error_rows'):
  p.error('Benchmark did not complete successfully')
 if not benchmark_meta.get('cards_enabled') or benchmark_meta.get('vl_enabled'):
  p.error('Benchmark must use production PIXEL_ONLY cards')
 benchmark_video=(ROOT/args.benchmark_video).resolve()
 if Path(benchmark_meta['video']).resolve()!=benchmark_video:
  p.error('Benchmark source path differs from explicitly selected source')
 benchmark_source=benchmark_video.relative_to(ROOT).as_posix()
 benchmark_sha=sha(benchmark_video)
 if benchmark_sha!=benchmark_meta.get('source_video_sha256'):
  p.error('Benchmark source changed since execution')
 code_pins=benchmark_meta.get('production_code_sha256',{})
 if 'parser_core/temporal_overlap.py' not in code_pins or any(sha(ROOT/name)!=digest for name,digest in code_pins.items()):
  p.error('Benchmark runtime implementation differs from promoted code')
 for field,name in [('rank_ncc_config_sha256','configs/rank_ncc_production.json'),
                    ('ncc_config_sha256','configs/suit_backup_local.json')]:
  if benchmark_meta.get(field)!=sha(ROOT/name):p.error('Benchmark config differs: '+name)
 if not (ROOT/args.benchmark_run/'ordered_timeline.xlsx').is_file():p.error('Benchmark Excel export missing')
 stages={
  'orchestration':('scripts/run_smoke14_ram.py','main'),
  'integrity':('parser_core/production_integrity.py','ProductionGuard'),
  'decode_capture_change_detection':('parser_core/ram_first_stage.py','FirstStage.rows'),
  'ROI':('parser_core/layout.py','load_fixed_layout'),
  'row_tracking':('parser_core/chat_rows.py','match_row_blocks'),
  'row_segmentation':('parser_core/chat_rows.py','physical_spans'),
  'temporal_overlap':('parser_core/temporal_overlap.py','TemporalObservations.reduce'),
  'hand_assembly':('parser_core/poker_hand_assembly.py','assemble_hands'),
  'poker_analysis':('parser_core/poker_pipeline.py','analyze_timeline'),
  'poker_logic':('parser_core/poker_logic_engine.py','validate_hand'),
  'poker_math':('parser_core/poker_math_engine.py','audit_observed_ledger'),
  'poker_cards':('parser_core/poker_card_engine.py','best_rank'),
  'poker_diagnostics':('parser_core/poker_diagnostics.py','diagnose_hand'),
  'OCR_queue':('parser_core/parallel_ocr.py','ParallelRowOCR'),
  'Tiny_client':('parser_core/tiny_chat_client.py','TinyChatClient'),
  'Tiny_worker':('scripts/tiny_chat_worker.py','main'),
  'card_queue':('parser_core/visual_tile_queue.py','VisualTileQueue'),
  'card_geometry':('parser_core/board_tiles.py','card_spans'),
  'secondary_geometry':('parser_core/board_tiles.py','secondary_card_spans'),
  'CardHybrid':('parser_core/card_hybrid.py','CardHybrid.read_bytes'),
  'RankNCC':('parser_core/rank_ncc.py','RankNCC.read'),
  'SuitNCC':('parser_core/suit_backup.py','SuitBackup.read'),
  'component_suit':('parser_core/component_suit.py','ComponentSuitBackup.read'),
  'normalization':('parser_core/normalize_chat.py','normalize'),
  'message_grouping':('parser_core/message_assembly.py','group_rows'),
  'event_binding':('parser_core/visual_cards.py','associate_visual_evidence'),
  'event_projection':('parser_core/visual_cards.py','project_chat_card_events'),
  'metrics':('scripts/run_smoke14_ram.py','Telemetry'),
  'CI':('scripts/check_production_integrity.py','main')}
 manifest=dict(schema='poker.production.manifest.v1',production_version=golden,git_commit=commit,golden_tag=None,
  git_state='DIRTY_WORKTREE_SNAPSHOT; no unrelated changes committed',pipeline='RAM_FIRST',card_path='PIXEL_ONLY',ncc_mode=json.loads((ROOT/'configs/suit_backup_local.json').read_text('utf8'))['mode'],
  entrypoint='run_production.py',internal_runner='scripts/run_smoke14_ram.py',
  launch_command=f'python run_production.py --video {benchmark_source} --frames 500 --output runs/production_smoke_fresh --ocr-workers 2 --visual-workers 2 --card-workers 2 --node C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe',
  modules=modules,configs=configs,exporter=exporter,tests=tests,
  config_arguments={'tiny_config':configs[0],'card_hybrid_config':configs[1],'suit_backup_config':configs[2],'rank_ncc_config':configs[3],'layout_profile':configs[5]},
  allowed_dependencies=['experiments/suit_template_experiment.py'],
  dependency_exception_reason='Existing frozen features/predict NCC helper only; hash pinned in suit config. Do not replace or add experimental runners.',
  forbidden_import_roots=['experiments','legacy','archive','archived','runs','outputs','old'],
  forbidden_sources=['experiments/** except exact allowlisted helper','runs/** (code)','outputs/** (code)','legacy/**','archive/**','scripts/capture_video.py','scripts/capture_nine_tables.py','scripts/run_nine_tables_streaming.py','scripts/*benchmark*','tests/fixtures/** (runtime implementation)'],
  png_policy={'transport':'DISABLED','golden_intermediate_count':0,'golden_intermediate_bytes':0,'max_intermediate_count':0,'max_intermediate_bytes':0,'allowed_debug_paths':[],
     'evidence':'RAM PNG encoding and base64 in final JSONL retained; template PNGs read-only. Explicit future debug paths require manifest update.'},
  performance={'early_frames':500,'golden_frames':500,'golden_source_fps':30,'golden_capture_seconds':baseline['first_stage_wall_seconds'],
    'golden_wall_seconds':baseline['wall_seconds'],'golden_fps':baseline['fps'],'golden_rtf':baseline['rtf'],
    'expected_rtf_range':[round(baseline['rtf']*.75,4),round(baseline['rtf']*1.5,4)],'regression_factor':4,
    'policy':'4x capture regression plus changed-path evidence stops; slow clean path warns. RTF is observational range from this workstation, not accuracy/HH gate.'},
  golden_benchmark={'run':args.benchmark_run,'source':benchmark_source,'source_sha256':benchmark_sha,'status':baseline['status']+'; no independent card GT',
     'known_counts':{k:baseline.get(k) for k in ('physical_rows','timeline_rows','dedup_removed_rows','card_tile_count','card_published_rows')},
     'code_freeze_note':'Measured candidate run; inspect its manifest and source fingerprints. Final golden smoke verifies final manifest separately.'},
  quality={'smoke15_dev_tile_correct':900,'smoke15_dev_tile_wrong':0,'smoke15_dev_tile_unresolved':0,
    'smoke15_dev_projected_correct':583,'smoke15_dev_projected_wrong':0,'smoke15_dev_projected_unresolved':317,
    'evidence':'runs/pixel_card_remaining38/replay_WITH_VL_CACHE/summary.json; cached OCR/VL FALLBACK experiment, not current pixel-only acceptance',
    'production_999':'NOT_PROVEN','certified_hand_histories':False},
  VL={'enabled':False,'scope':'No CHAT card model/client/dispatcher launched'},
  stages={k:dict(file=v[0],symbol=v[1]) for k,v in stages.items()},
  outputs=['physical_rows.jsonl','raw_ocr_results.jsonl','visual_evidence.jsonl','removed_duplicates.jsonl','context_decisions.jsonl','ordered_timeline.jsonl','ordered_timeline.xlsx','hand_analysis.jsonl','hand_summary.json','validated_hands.jsonl','run_metadata.json','final_summary.json','module_timings.json','errors_retries.json','production_guard.json'],
  promotion_procedure='PRODUCTION.md#how-to-change-production')
 tracked=modules+configs+[exporter,'models/board_glyphs_v1.json','PRODUCTION.md','AGENTS.md','README.md','requirements.txt','pytest.ini','.github/workflows/production-integrity.yml','scripts/freeze_production_manifest.py']+tests
 manifest['hashes']={path:sha(ROOT/path) for path in sorted(set(tracked))}
 manifest['data_dependencies']=[]
 for cfg in configs[:4]:
  obj=json.loads((ROOT/cfg).read_text('utf8'))
  for key in ('train_manifest','selection_file','validation_results'):
   if key in obj:
    path=Path(obj[key]);manifest['data_dependencies'].append(dict(config=cfg,key=key,path=str(path),sha256=sha(path),read_only=True))
 manifest['external_runtime']={'OCR':json.loads((ROOT/configs[0]).read_text('utf8')),'node':'C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe','export_library':'@oai/artifact-tool (bundled runtime)'}
 (ROOT/'production_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf8')
 snapshot=ROOT/'provenance'/golden;snapshot.mkdir(parents=True,exist_ok=True)
 prior=snapshot/'files/production_manifest.json'
 if prior.is_file():
  revision=snapshot/'revisions'/sha(prior)
  if not revision.exists():
   shutil.copytree(snapshot/'files',revision/'files')
   for name in ('tracked_dirty.patch','git_status.txt','git_head.txt','snapshot_hashes.json'):
    if (snapshot/name).is_file():shutil.copy2(snapshot/name,revision/name)
 for path in sorted(set(tracked+['production_manifest.json'])):
  target=snapshot/'files'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,target)
 subprocess.run(['git','diff','--binary','--output='+str(snapshot/'tracked_dirty.patch')],cwd=ROOT,check=True)
 status=subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT).decode('utf8','replace')
 (snapshot/'git_status.txt').write_text(status,'utf8');(snapshot/'git_head.txt').write_text(commit+'\n','utf8')
 (snapshot/'snapshot_hashes.json').write_text(json.dumps({p:sha(ROOT/p) for p in tracked+['production_manifest.json']},indent=2),'utf8')
 print(json.dumps({'golden':golden,'commit':commit,'modules':len(modules),'tests':len(tests),'snapshot':str(snapshot)},ensure_ascii=False))
if __name__=='__main__':main()
