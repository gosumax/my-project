"""Startup contracts; native arithmetic is independently replayed against Golden."""
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from parser_core import physical_chat_admission as physical


@pytest.fixture
def route(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'parser_core/physical_tail_route.py'
    spec = importlib.util.spec_from_file_location('isolated_proof4_route', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(physical, 'physical_tail_proof', physical.physical_tail_proof)
    config = {'schema': 'poker.physical-tail-proof4.v1', 'enabled': False,
              'native_threads': 4, 'dll': module.DLL, 'dll_sha256': module.DLL_SHA256}
    path = tmp_path / module.CONFIG
    path.parent.mkdir(parents=True)
    manifest = {'speed_state': {'PHYSICAL_TAIL_PROOF4_NATIVE_ACTIVE': 'NO'},
                'hashes': {module.DLL: module.DLL_SHA256}}

    def write(enabled=False):
        config['enabled'] = enabled
        path.write_text(json.dumps(config), 'utf-8')
        manifest['speed_state']['PHYSICAL_TAIL_PROOF4_NATIVE_ACTIVE'] = 'YES' if enabled else 'NO'

    write()
    calls = []
    fake = ModuleType('parser_core.physical_tail_proof4')
    fake.OLD_PROOF = physical.physical_tail_proof

    class Binding:
        def __init__(self, path, *, threads):
            calls.append((path, threads))

        def __call__(self, *args):
            return ('NATIVE', args)

    fake.Proof4Binding = Binding
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    return SimpleNamespace(module=module, root=tmp_path, config=config, manifest=manifest,
                           write=write, calls=calls, fake=fake, path=path)


def test_off_preserves_function_and_does_not_initialize(route):
    original = physical.physical_tail_proof
    assert route.module.install_physical_tail(route.root, route.manifest) is None
    assert route.module.install_physical_tail(route.root, route.manifest) is None
    assert physical.physical_tail_proof is original
    assert not route.calls


def test_on_installs_once_and_preserves_fallback(route):
    original = physical.physical_tail_proof
    route.write(True)
    binding = route.module.install_physical_tail(route.root, route.manifest)
    assert route.module.install_physical_tail(route.root, route.manifest) is binding
    assert len(route.calls) == 1 and route.calls[0] == (route.root / route.module.DLL, 4)
    assert route.fake.OLD_PROOF is original
    assert physical.physical_tail_proof(1, 2, 3, 4) == ('NATIVE', (1, 2, 3, 4))


def test_concurrent_install_publishes_one_binding(route):
    route.write(True)
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(lambda _: route.module.install_physical_tail(route.root, route.manifest), range(16)))
    assert all(result is results[0] for result in results)
    assert len(route.calls) == 1


def test_failed_initialization_does_not_publish(route):
    original = physical.physical_tail_proof
    route.write(True)

    def failed(*args, **kwargs):
        raise RuntimeError('PROOF4_DLL_SEAL_CONFLICT')

    route.fake.Proof4Binding = failed
    with pytest.raises(RuntimeError, match='DLL_SEAL_CONFLICT'):
        route.module.install_physical_tail(route.root, route.manifest)
    assert physical.physical_tail_proof is original and route.module._INSTALLED is None


def test_manifest_config_disagreement_rejects(route):
    route.manifest['speed_state']['PHYSICAL_TAIL_PROOF4_NATIVE_ACTIVE'] = 'YES'
    with pytest.raises(ValueError, match='MANIFEST_CONFIG_CONFLICT'):
        route.module.install_physical_tail(route.root, route.manifest)
    assert not route.calls


def test_manifest_dll_seal_disagreement_rejects(route):
    route.manifest['hashes'][route.module.DLL] = '0' * 64
    with pytest.raises(ValueError, match='MANIFEST_DLL_CONFLICT'):
        route.module.install_physical_tail(route.root, route.manifest)
    assert not route.calls


@pytest.mark.parametrize('field,value', [('enabled', 1), ('native_threads', 2), ('native_threads', 4.0),
    ('dll', 'foreign.dll'), ('dll_sha256', '0' * 64), ('schema', 'foreign'), ('extra', True)])
def test_configuration_unknown_rejects(route, field, value):
    route.config[field] = value
    route.path.write_text(json.dumps(route.config), 'utf-8')
    with pytest.raises(ValueError, match='CONFIGURATION_CONFLICT'):
        route.module.install_physical_tail(route.root, route.manifest)
    assert not route.calls


def test_runtime_route_tamper_rejects(route, monkeypatch):
    route.write(True)
    route.module.install_physical_tail(route.root, route.manifest)
    monkeypatch.setattr(physical, 'physical_tail_proof', lambda *args: None)
    with pytest.raises(RuntimeError, match='RECONFIGURATION_FORBIDDEN'):
        route.module.install_physical_tail(route.root, route.manifest)


def test_off_to_on_requires_a_new_process(route):
    route.module.install_physical_tail(route.root, route.manifest)
    route.write(True)
    with pytest.raises(RuntimeError, match='RECONFIGURATION_FORBIDDEN'):
        route.module.install_physical_tail(route.root, route.manifest)
    assert not route.calls


def test_existing_unexpected_binding_rejects(route, monkeypatch):
    monkeypatch.setattr(physical, 'physical_tail_proof', lambda *args: None)
    with pytest.raises(RuntimeError, match='EXISTING_PROOF_BINDING_CONFLICT'):
        route.module.install_physical_tail(route.root, route.manifest)
    assert not route.calls
