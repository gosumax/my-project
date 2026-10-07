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
