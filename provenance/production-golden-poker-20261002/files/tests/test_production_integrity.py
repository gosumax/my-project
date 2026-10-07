"""Negative production integrity behaviors in isolated temporary workspaces."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from parser_core.production_integrity import (
    ProductionGuard,ProductionIntegrityError,RegisteredImportGuard,
    load_manifest,sha256,validate_integrity,
)

@pytest.fixture
def workspace(tmp_path):
    files={
        'run_production.py':'def main(): pass\n',
        'scripts/run_smoke14_ram.py':'def main(): pass\n',
        'parser_core/capture.py':'class Capture: pass\n',
        'scripts/export.mjs':'export {};\n',
        'configs/policy.json':'{}',
        'PRODUCTION.md':'THIS FILE DEFINES THE ONLY PRODUCTION PIPELINE.\n',
    }
    for name,text in files.items():
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,'utf8')
    manifest=dict(production_version='test',git_commit='fixture',entrypoint='run_production.py',internal_runner='scripts/run_smoke14_ram.py',pipeline='RAM_FIRST',card_path='PIXEL_ONLY',modules=['scripts/run_smoke14_ram.py','parser_core/capture.py'],configs=['configs/policy.json'],exporter='scripts/export.mjs',tests=['tests/test_integrity.py'],hashes={p:sha256(tmp_path/p) for p in files},png_policy=dict(max_intermediate_count=0,max_intermediate_bytes=0,allowed_debug_paths=['debug/approved.png']),performance=dict(early_frames=100,golden_frames=500,golden_capture_seconds=10,regression_factor=2,expected_rtf_range=[1,2]),stages={'capture':dict(file='parser_core/capture.py',symbol='Capture')},forbidden_import_roots=['experiments','legacy'],config_arguments={'rank_ncc_config':'configs/policy.json'})
    (tmp_path/'production_manifest.json').write_text(json.dumps(manifest),'utf8')
    return tmp_path,manifest

def test_registered_manifest_and_hashes_validate(workspace):
    root,manifest=workspace
    assert load_manifest(root)==manifest
    assert validate_integrity(root,strict_hashes=True)==[]

@pytest.mark.parametrize('field,value',[('entrypoint','scripts/legacy.py'),('pipeline','PNG_FIRST'),('card_path','VL_ONLY')])
def test_manifest_pipeline_entrypoint_substitution_rejected(workspace,field,value):
    root,manifest=workspace;manifest[field]=value
    (root/'production_manifest.json').write_text(json.dumps(manifest),'utf8')
    with pytest.raises(ProductionIntegrityError):load_manifest(root)

def test_foreign_unregistered_and_legacy_static_imports_rejected(workspace):
    root,manifest=workspace
    for target in ('experiments.evil','parser_core.unregistered'):
        p=root/(target.replace('.','/')+'.py');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('VALUE=1\n','utf8')
        (root/'parser_core/capture.py').write_text('import '+target+'\nclass Capture: pass\n','utf8')
        with pytest.raises(ProductionIntegrityError,match='production import'):validate_integrity(root,manifest)

def test_dynamic_import_guard_rejects_external_and_unregistered_modules(workspace,tmp_path,monkeypatch):
    import parser_core.production_integrity as module
    root,manifest=workspace;guard=RegisteredImportGuard(root,manifest)
    foreign=root.parent/'foreign_capture.py';foreign.write_text('','utf8')
    monkeypatch.setattr(module.importlib.machinery.PathFinder,'find_spec',lambda *args:SimpleNamespace(origin=str(foreign)))
    with pytest.raises(ProductionIntegrityError,match='Foreign'):guard.find_spec('parser_core.capture')
    unregistered=root/'parser_core/unknown.py';unregistered.write_text('','utf8')
    monkeypatch.setattr(module.importlib.machinery.PathFinder,'find_spec',lambda *args:SimpleNamespace(origin=str(unregistered)))
    with pytest.raises(ProductionIntegrityError,match='Unregistered'):guard.find_spec('parser_core.unknown')

def test_modified_hashes_are_visible_and_strict_validation_fails(workspace,capsys):
    root,manifest=workspace;(root/'parser_core/capture.py').write_text('class Capture: changed=True\n','utf8')
    assert validate_integrity(root,manifest)==['parser_core/capture.py']
    with pytest.raises(ProductionIntegrityError,match='hashes stale'):validate_integrity(root,manifest,strict_hashes=True)
    guard=ProductionGuard(root,manifest);fp=guard.fingerprint()
    assert fp['modified_files']==['parser_core/capture.py']
    assert 'WORKTREE MODIFIED' in capsys.readouterr().out

def test_configuration_substitution_and_existing_output_rejected(workspace):
    root,manifest=workspace;guard=ProductionGuard(root,manifest)
    with pytest.raises(ProductionIntegrityError,match='config substitution'):guard.configure(SimpleNamespace(rank_ncc_config=root/'wrong.json',output=root/'out'))
    output=root/'out';output.mkdir()
    with pytest.raises(ProductionIntegrityError,match='fresh'):guard.configure(SimpleNamespace(rank_ncc_config=root/'configs/policy.json',output=output))

def test_mass_png_writes_abort_but_explicit_debug_path_allowed(workspace):
    root,manifest=workspace;guard=ProductionGuard(root,manifest);guard.output=root/'out';(guard.output/'debug').mkdir(parents=True)
    (guard.output/'debug/approved.png').write_bytes(b'debug')
    assert guard.check_png()==dict(count=0,bytes=0)
    (guard.output/'transport.PNG').write_bytes(b'pixel transport')
    with pytest.raises(ProductionIntegrityError,match='MASS PNG WRITE'):guard.check_png()

def test_early_performance_modified_path_aborts_once_threshold_is_reached(workspace):
    root,manifest=workspace;(root/'parser_core/capture.py').write_text('class Capture: changed=True\n','utf8')
    guard=ProductionGuard(root,manifest);guard.check_early(99,99)
    assert not guard.early_checked
    with pytest.raises(ProductionIntegrityError,match='PERFORMANCE REGRESSION'):guard.check_early(100,8)
    assert guard.metrics['early_performance']['regression_ratio']==4

def test_slow_clean_path_warns_without_inventing_legacy_evidence(workspace,capsys):
    root,manifest=workspace;guard=ProductionGuard(root,manifest);guard.check_early(100,5)
    assert 'continuing with warning' in capsys.readouterr().out
    assert guard.modified==[] and guard.early_checked

def test_manifest_stage_symbol_must_exist(workspace):
    root,manifest=workspace;manifest['stages']['capture']['symbol']='Absent.method'
    with pytest.raises(ProductionIntegrityError,match='Missing stage symbol'):validate_integrity(root,manifest)

def test_unregistered_stage_file_rejected(workspace):
    root,manifest=workspace;manifest['stages']['capture']['file']='parser_core/rogue.py'
    with pytest.raises(ProductionIntegrityError,match='Unregistered production stage'):validate_integrity(root,manifest)

def test_internal_runner_direct_launch_refuses_before_processing_video(tmp_path):
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'scripts/run_smoke14_ram.py')],cwd=root,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=30)
    assert result.returncode!=0
    assert 'INTERNAL ONLY / LEGACY DIRECT LAUNCH' in result.stderr
    assert not list(tmp_path.iterdir())

@pytest.mark.parametrize('statement', ['from . import rogue','from parser_core import rogue','import replacement'])
def test_relative_from_and_root_namespace_import_substitution_rejected(workspace,statement):
    root,manifest=workspace
    (root/'parser_core/rogue.py').write_text('VALUE=1\n','utf8')
    (root/'replacement.py').write_text('VALUE=1\n','utf8')
    (root/'parser_core/capture.py').write_text(statement+'\nclass Capture: pass\n','utf8')
    with pytest.raises(ProductionIntegrityError,match='Unregistered production import'):validate_integrity(root,manifest)

@pytest.mark.parametrize('foreign',[False,True])
def test_direct_file_loader_execution_audit_rejects_root_or_foreign_source(workspace,foreign):
    root,manifest=workspace
    rogue=(root.parent/'foreign_rogue.py') if foreign else (root/'replacement.py')
    rogue.write_text('MUTATION=True\n','utf8')
    guard_file=Path(__file__).resolve().parents[1]/'parser_core/production_integrity.py'
    code=(f'import importlib.util,json,pathlib\n'
          f's=importlib.util.spec_from_file_location("isolated_guard",{str(guard_file)!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n'
          f'r=pathlib.Path({str(root)!r});g=m.ProductionGuard(r);g.install_import_guard()\n'
          f's=importlib.util.spec_from_file_location("replacement",{str(rogue)!r});x=importlib.util.module_from_spec(s)\n'
          f'try:s.loader.exec_module(x)\n'
          f'except m.ProductionIntegrityError as e:print(e);raise SystemExit(0)\n'
          f'raise SystemExit("UNREGISTERED SOURCE EXECUTED")\n')
    p=subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=30)
    assert p.returncode==0,p.stderr
    assert ('Foreign production source execution' if foreign else 'Unregistered production source execution') in p.stdout

def test_finish_persists_failure_metrics_when_png_guard_aborts(workspace):
    root,manifest=workspace;guard=ProductionGuard(root,manifest);guard.output=root/'out';guard.output.mkdir()
    (guard.output/'transport.png').write_bytes(b'pixels')
    with pytest.raises(ProductionIntegrityError,match='MASS PNG WRITE'):guard.finish(100,2,3,6)
    saved=json.loads((guard.output/'production_guard.json').read_text('utf8'))
    assert 'MASS PNG WRITE' in saved['failure']
    assert saved['intermediate_png']==dict(count=1,bytes=6)
    assert saved['final']['frames']==100 and saved['final']['bytes_written']==6

def test_internal_runner_persists_failed_summary_when_capture_and_final_png_guard_abort(workspace,monkeypatch):
    import importlib.util
    from collections import Counter
    root,manifest=workspace
    actual_root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('isolated_internal_runner',actual_root/'scripts/run_smoke14_ram.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    out=root/'run_failure'
    guard=ProductionGuard(root,manifest)
    def configure(args):guard.output=out
    monkeypatch.setattr(guard,'configure',configure)
    class Capture:
        seen_frames=1;last_video_time=1/30;metrics=Counter();changed=set();board_snapshots={}
        def __init__(self,*args,**kwargs):pass
        def rows(self):
            (out/'transport.png').write_bytes(b'pixels')
            raise RuntimeError('INJECTED_CAPTURE_FAILURE')
            yield
    class OCR:
        _threads=[];results={};spans=[];errors=[];max_queue=0;first_started=None
        jobs=SimpleNamespace(qsize=lambda:0)
        def __init__(self,*args,**kwargs):pass
        def finish(self):pass
    monkeypatch.setattr(runner,'FirstStage',Capture)
    monkeypatch.setattr(runner,'ParallelRowOCR',OCR)
    monkeypatch.setattr(runner,'source_identity',lambda *args:({'source_video_sha256':'synthetic','source_id':'synthetic','layout_id':'synthetic'}, {'1':'s'}))
    monkeypatch.setattr(runner.shutil,'which',lambda *args:None)
    monkeypatch.setattr(sys,'argv',['internal','--video','synthetic.mp4','--frames','1','--output',str(out),'--node',sys.executable,'--skip-cards'])
    with pytest.raises(RuntimeError,match='INJECTED_CAPTURE_FAILURE'):runner.main(production_guard=guard)
    summary=json.loads((out/'final_summary.json').read_text('utf8'))
    errors=json.loads((out/'errors_retries.json').read_text('utf8'))
    saved_guard=json.loads((out/'production_guard.json').read_text('utf8'))
    assert summary['status']=='FAILED'
    assert any('MASS PNG WRITE' in x.get('production_guard','') for x in errors['errors'])
    assert 'MASS PNG WRITE' in saved_guard['failure']
    assert saved_guard['final']['frames']==1
