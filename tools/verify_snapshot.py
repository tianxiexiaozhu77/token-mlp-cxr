#!/usr/bin/env python3
"""Validate the source snapshot without importing experiments or starting jobs."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    errors, python_count, shell_count = [], 0, 0
    for row in manifest['files']:
        path = root / row['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            errors.append('Source checksum mismatch: ' + row['path'])
    published_path = root / 'PUBLISHED_LOGS.json'
    published_logs = json.loads(published_path.read_text())['files'] if published_path.is_file() else []
    log_paths = {row['path'] for row in published_logs}
    for row in published_logs:
        path = root / row['path']
        if not path.is_file() or path.stat().st_size != row['size'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            errors.append('Log checksum/size mismatch: ' + row['path'])
        if not row['path'].startswith('logs/') or path.suffix != '.log':
            errors.append('Invalid published log path: ' + row['path'])
    files = [p for p in root.rglob('*') if p.is_file() and '.git' not in p.relative_to(root).parts]
    for path in files:
        rel = str(path.relative_to(root))
        if path.suffix == '.py':
            try:
                tree = ast.parse(path.read_text(), filename=rel)
                python_count += 1
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('experiments.'):
                        module = root.joinpath(*node.module.split('.'))
                        if not module.with_suffix('.py').is_file() and not module.is_dir():
                            errors.append(f'Missing explicit local import: {rel}:{node.lineno}: {node.module}')
            except SyntaxError as error:
                errors.append(str(error))
        elif path.suffix == '.sh':
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            shell_count += 1
            if result.returncode:
                errors.append(rel + ': ' + result.stderr)
        if path.suffix.lower() in {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.npy', '.npz', '.dcm', '.png', '.jpg', '.jpeg'}:
            errors.append('Unexpected data/model payload: ' + rel)
        if any(part in {'runs', '.venv', '.envs', '__pycache__'} for part in path.relative_to(root).parts):
            errors.append('Unexpected runtime directory: ' + rel)
        if path.suffix == '.log' and rel not in log_paths:
            errors.append('Unregistered log: ' + rel)
    report = dict(status='passed' if not errors else 'failed', manifest_files=len(manifest['files']),
                  published_log_files=len(published_logs), log_checksums_verified=True,
                  python_syntax_checked=python_count, shell_syntax_checked=shell_count,
                  explicit_experiments_imports_checked=True, forbidden_payload_check=True,
                  inference_or_training_executed=False, full_runtime_dependency_closure_verified=False,
                  fresh_environment_reproduction_verified=False, errors=errors)
    (root / 'VALIDATION.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
