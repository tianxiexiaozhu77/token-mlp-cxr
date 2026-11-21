#!/usr/bin/env python3
"""Restore pinned upstream code into a NEW workspace, then overlay this snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

RELEASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Execute; otherwise print the plan only')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if workspace == RELEASE or RELEASE in workspace.parents or workspace in RELEASE.parents:
        parser.error('Use a separate workspace outside the release directory')
    if workspace.exists():
        parser.error('Workspace already exists; choose a new directory')
    manifest = json.loads((RELEASE / 'SOURCE_MANIFEST.json').read_text())
    for item in manifest['files']:
        path = RELEASE / item['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Snapshot hash mismatch: ' + item['path'])
    models = json.loads((RELEASE / 'MODEL_SOURCES.json').read_text())['models']
    for model in models:
        target = workspace / model['checkout_path']
        commands = [
            ['git', 'clone', model['url'], str(target)],
            ['git', '-C', str(target), 'checkout', '--detach', model['revision']],
        ]
        if model.get('local_patch'):
            patch = RELEASE / model['local_patch']
            if hashlib.sha256(patch.read_bytes()).hexdigest() != model['local_patch_sha256']:
                raise ValueError('Patch hash mismatch')
            commands += [
                ['git', '-C', str(target), 'apply', '--check', str(patch)],
                ['git', '-C', str(target), 'apply', str(patch)],
            ]
        for command in commands:
            print(shlex.join(command), flush=True)
            if args.execute:
                target.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(command, check=True)
    # Copy only the registered snapshot and release documentation/tools. Never
    # copy accidentally downloaded weights, data, environments or nested .git.
    paths = {item['path'] for item in manifest['files']}
    paths.update(p.name for p in RELEASE.iterdir() if p.is_file())
    for directory in ('tools', 'environments', 'upstream_patches', 'resources', 'logs'):
        paths.update(str(p.relative_to(RELEASE)) for p in (RELEASE / directory).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts)
    print(f'Overlay {len(paths)} release files into {workspace}', flush=True)
    if args.execute:
        for relative in sorted(paths):
            destination = workspace / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RELEASE / relative, destination)
        print('Prepared workspace. Next: isolated environments, weights and fixed data manifests.')


if __name__ == '__main__':
    main()
