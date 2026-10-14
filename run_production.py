"""Only official production launcher. The audited runner remains an internal implementation."""
from pathlib import Path
import importlib.util
import sys
from parser_core.production_integrity import ProductionGuard
ROOT=Path(__file__).resolve().parent
def main():
    guard=ProductionGuard(ROOT)
    guard.fingerprint();guard.install_import_guard()
    from parser_core.physical_tail_route import install_physical_tail
    install_physical_tail(ROOT, guard.manifest)
    if '--review-nicknames' in sys.argv:
        from scripts.review_nicknames import main as review
        review()
        return
    if '--validate-saved-physical-tail' in sys.argv:
        import argparse
        parser=argparse.ArgumentParser(description='Official sealed physical producer replay; no acquisition')
        parser.add_argument('--validate-saved-physical-tail',required=True,metavar='BUNDLE_JSON')
        parser.add_argument('--output',required=True)
        args=parser.parse_args()
        from scripts.replay_saved_physical_tail import validate_saved_physical_tail
        validate_saved_physical_tail(ROOT,guard.manifest,args.validate_saved_physical_tail,args.output)
        return
    if '--validate-saved-empty' in sys.argv:
        import argparse
        parser=argparse.ArgumentParser(description='Official bounded native EMPTY lifecycle validation; no acquisition')
        parser.add_argument('--validate-saved-empty',required=True,metavar='CAPTURE_PLAN_JSON')
        parser.add_argument('--capture',required=True)
        parser.add_argument('--output',required=True)
        args=parser.parse_args()
        from scripts.replay_saved_empty import main as validate_empty
        args.plan=args.validate_saved_empty
        validate_empty(args)
        return
    if '--validate-saved-hh' in sys.argv:
        import argparse
        parser=argparse.ArgumentParser(description='Official sealed retained-trace HH validation; no acquisition or solver')
        parser.add_argument('--validate-saved-hh',required=True,metavar='BUNDLE_JSON')
        parser.add_argument('--no-sb-review',required=True)
        parser.add_argument('--output',required=True)
        args=parser.parse_args()
        from scripts.replay_saved_quality import validate_saved_hh
        validate_saved_hh(args.validate_saved_hh,args.output,no_sb_review=args.no_sb_review)
        return
    if '--replay-saved-quality' in sys.argv:
        import argparse
        parser=argparse.ArgumentParser(description='Official saved post-pixel quality replay')
        parser.add_argument('--replay-saved-quality',required=True)
        parser.add_argument('--output',required=True)
        parser.add_argument('--player-database')
        args=parser.parse_args()
        from scripts.replay_saved_quality import main as replay
        replay(args.replay_saved_quality,args.output,player_database=args.player_database)
        return
    path=ROOT/guard.manifest['internal_runner']
    spec=importlib.util.spec_from_file_location('production_internal_runner',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.main(production_guard=guard)
    if '--profile' in sys.argv and sys.argv[sys.argv.index('--profile')+1]=='FULL_HH_PRODUCTION':
        from scripts.review_nicknames import launch_cached_review
        output=Path(sys.argv[sys.argv.index('--output')+1])
        try:
            launch_cached_review(output)
        except Exception as error:
            import json
            (output/'nickname_review_launch.json').write_text(json.dumps(
                dict(status='FAILED',error=repr(error)),ensure_ascii=False,indent=2),'utf-8')
            print('Nickname review could not open:',repr(error),flush=True)
if __name__=='__main__':main()
