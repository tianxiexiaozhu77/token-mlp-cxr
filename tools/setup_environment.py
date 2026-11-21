#!/usr/bin/env python3
"""Print/install one model's isolated environment; never replace an existing venv."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

RELEASE = Path(__file__).resolve().parents[1]


def main():
    specs = {s['name']: s for s in json.loads((RELEASE / 'environments/index.json').read_text())}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(specs))
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Install; otherwise print commands only')
    args = parser.parse_args()
    spec = specs[args.model]
    workspace = args.workspace.resolve()
    venv = workspace / spec['path']
    python = venv / 'bin/python'
    if args.execute and not workspace.is_dir():
        parser.error('Prepare the workspace first')
    if args.execute and venv.exists():
        parser.error(f'Refusing to modify existing environment: {venv}')
    commands = [
        ['uv', 'python', 'install', spec['python']],
        ['uv', 'venv', '--seed', '--python', spec['python'], str(venv)],
        [str(python), '-m', 'pip', 'install', 'torch==' + spec['torch'],
         'torchvision==' + spec['torchvision'], '--index-url',
         'https://download.pytorch.org/whl/' + spec['index']],
        [str(python), '-m', 'pip', 'install', '-r', str(RELEASE / spec['requirements'])],
        [str(python), '-m', 'pip', 'check'],
        [str(python), '-c', 'import sys,torch,numpy; print(sys.version); '
         'print(torch.__version__, torch.version.cuda, numpy.__version__)'],
    ]
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=workspace, check=True)


if __name__ == '__main__':
    main()
