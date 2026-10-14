"""One-time production routing for the sealed Physical Tail PROOF4 binding."""
import json
import threading
from pathlib import Path

from parser_core import physical_chat_admission as physical

_LOCK = threading.Lock()
_INSTALLED = None
_ORIGINAL = physical.physical_tail_proof
CONFIG = 'configs/physical_tail_proof4.json'
DLL = 'native/physical_tail_proof4/wholeproof.dll'
DLL_SHA256 = 'c6ba8e3ebace49deb97cabb7d462341ca6efabbd76d0379a5d92885899ec061b'


def install_physical_tail(root, manifest):
    """Validate before publishing; OFF never imports or initializes native code."""
    global _INSTALLED
    root = Path(root).resolve()
    config = json.loads((root / CONFIG).read_text('utf-8'))
    expected = {'schema': 'poker.physical-tail-proof4.v1',
                'enabled': config.get('enabled'), 'native_threads': 4,
                'dll': DLL, 'dll_sha256': DLL_SHA256}
    if (config != expected or type(config.get('enabled')) is not bool
            or type(config.get('native_threads')) is not int):
        raise ValueError('PROOF4_CONFIGURATION_CONFLICT')
    enabled = config['enabled']
    flag = 'YES' if enabled else 'NO'
    if manifest.get('speed_state', {}).get('PHYSICAL_TAIL_PROOF4_NATIVE_ACTIVE') != flag:
        raise ValueError('PROOF4_MANIFEST_CONFIG_CONFLICT')
    if manifest.get('hashes', {}).get(DLL) != DLL_SHA256:
        raise ValueError('PROOF4_MANIFEST_DLL_CONFLICT')
    identity = (str(root), enabled, DLL_SHA256, 4)
    with _LOCK:
        if _INSTALLED is not None:
            previous, function, binding = _INSTALLED
            if previous != identity or physical.physical_tail_proof is not function:
                raise RuntimeError('PROOF4_ROUTE_RECONFIGURATION_FORBIDDEN')
            return binding
        if physical.physical_tail_proof is not _ORIGINAL:
            raise RuntimeError('PROOF4_EXISTING_PROOF_BINDING_CONFLICT')
        if not enabled:
            _INSTALLED = (identity, _ORIGINAL, None)
            return None
        from parser_core.physical_tail_proof4 import Proof4Binding, OLD_PROOF
        if OLD_PROOF is not _ORIGINAL:
            raise RuntimeError('PROOF4_OLD_FALLBACK_BINDING_CONFLICT')
        binding = Proof4Binding(root / DLL, threads=4)

        def physical_tail_proof4(before, after, old_spans, new_spans):
            return binding(before, after, old_spans, new_spans)

        # No worker may exist yet. A complete, sealed binding is published once.
        physical.physical_tail_proof = physical_tail_proof4
        _INSTALLED = (identity, physical_tail_proof4, binding)
        return binding
