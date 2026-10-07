"""Only official production launcher. The audited runner remains an internal implementation."""
from pathlib import Path
import importlib.util
import sys
from parser_core.production_integrity import ProductionGuard
ROOT=Path(__file__).resolve().parent
def main():
    guard=ProductionGuard(ROOT)
    guard.fingerprint();guard.install_import_guard()
    if '--replay-saved-quality' in sys.argv:
        import argparse
        parser=argparse.ArgumentParser(description='Official saved post-pixel quality replay')
        parser.add_argument('--replay-saved-quality',required=True)
        parser.add_argument('--output',required=True)
        args=parser.parse_args()
        from scripts.replay_saved_quality import main as replay
        replay(args.replay_saved_quality,args.output)
        return
    path=ROOT/guard.manifest['internal_runner']
    spec=importlib.util.spec_from_file_location('production_internal_runner',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.main(production_guard=guard)
if __name__=='__main__':main()
