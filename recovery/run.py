"""Invoke the unchanged official production entrypoint in a provisioned recovery."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

_spec = importlib.util.spec_from_file_location('recovery_setup', Path(__file__).with_name('restore.py'))
_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup)
verify = _setup.verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--frames', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--ocr-workers', type=int, default=2)
    parser.add_argument('--visual-workers', type=int, default=2)
    parser.add_argument('--card-workers', type=int, default=2)
    args = parser.parse_args()
    destination = args.destination.resolve(strict=True)
    verify(destination)
    if args.frames <= 0 or not args.video.is_file() or args.output.exists():
        raise ValueError('Require existing input video, positive frame count, fresh output directory')
    runtime = destination / 'runtime'
    production = destination / 'production'
    env = os.environ.copy()
    for key in ('PYTHONHOME', 'PYTHONPATH', 'NODE_PATH', 'NODE_OPTIONS'):
        env.pop(key, None)
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PADDLE_PDX_CACHE_HOME'] = str(runtime / 'paddlex-cache')
    env['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    env['PATH'] = str(runtime / 'python') + os.pathsep + env.get('PATH', '')
    command = [str(runtime / 'main/Scripts/python.exe'), str(production / 'run_production.py'),
               '--profile', 'FULL_HH_PRODUCTION', '--video', str(args.video.resolve()),
               '--frames', str(args.frames), '--output', str(args.output.resolve()),
               '--ocr-workers', str(args.ocr_workers), '--visual-workers', str(args.visual_workers),
               '--card-workers', str(args.card_workers), '--node', str(runtime / 'node/bin/node.exe')]
    result = subprocess.run(command, cwd=production, env=env)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
