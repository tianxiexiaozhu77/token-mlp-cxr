#!/usr/bin/env python3
"""Archive text logs from the ten original campaign directories, without editing them."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

CAMPAIGNS = (
    ('direct_anchor_classification', '20261118', '3;9', 'Includes historical bias/direct methods and failed startup attempts.'),
    ('direct_anchor_classification', '20261118_alpha_token_mlp', '3;9', 'AlphaRAD primary Token MLP and associated checks.'),
    ('baseline_token_classification', '20261119', '3;7;9', 'Historical queue also includes MedKLIP, MAVL, and CheXzero, outside the seven-model paper.'),
    ('adapter_placement_classification', '20261119', '4;8', 'Placement and native MaCo pooling controls.'),
    ('both_adapter_classification', '20261119', '10;11;12', 'Both MLP and associated orchestration.'),
    ('gated_asl_classification', '20261120_recovered', '13;14;15', 'Includes recovery, supervision, and diagnostic logs.'),
    ('supervised_baseline_classification', '20261120', '4', 'Includes auxiliary external evaluation and additional comparison tasks.'),
    ('radzero_supervised_classification', '20261120', '4', 'Includes auxiliary external evaluation and bootstrap tasks.'),
    ('large_patient_multilabel', '20261121', '5', 'Includes encoder checks, fitting, selection, evaluation, and statistics.'),
    ('nih_chexpert_transfer', '20261121', '6', 'Encoding, transfer evaluation, tests, and reporting; no new fitting.'),
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    if output == workspace or not (output / 'SOURCE_MANIFEST.json').is_file():
        raise ValueError('Output must be a separate existing source snapshot')
    secret_patterns = [r'AKIA[0-9A-Z]{16}', r'gh[pousr]_[A-Za-z0-9]{30,}',
                       r'hf_[A-Za-z0-9]{30,}', r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
    rows, campaigns = [], []
    for campaign, run_id, tables, note in CAMPAIGNS:
        source_root = workspace / 'experiments' / campaign / 'runs' / run_id
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)
        files = sorted(source_root.rglob('*.log'))
        total = 0
        for source in files:
            relative = source.relative_to(source_root)
            before = source.stat()
            data = source.read_bytes()
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('Log changed during snapshot: ' + str(source))
            if b'\0' in data:
                raise ValueError('Unexpected binary log: ' + str(source))
            text = data.decode('utf-8', errors='replace')
            if any(re.search(pattern, text) for pattern in secret_patterns):
                raise ValueError('Credential-like content requires a redacted copy: ' + str(source))
            target = output / 'logs' / campaign / run_id / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != data:
                raise FileExistsError('Refusing to replace a different archived log: ' + str(target))
            target.write_bytes(data)
            rows.append(dict(path=str(target.relative_to(output)), source_path=str(source.relative_to(workspace)),
                             campaign=campaign, run_id=run_id, campaign_tables=tables,
                             size=len(data), sha256=sha256(data), source_mtime_ns=after.st_mtime_ns))
            total += len(data)
        campaigns.append(dict(campaign=campaign, run_id=run_id, campaign_tables=tables,
                              files=len(files), bytes=total, scope_note=note))
    published = dict(scope='All .log files in the ten listed campaign directories; includes historical auxiliary and failed-attempt logs. Campaign table association is not a per-log attribution or success claim.',
                     snapshot_utc=datetime.now(timezone.utc).isoformat(), original_bytes_preserved=True,
                     files=rows, campaigns=campaigns, file_count=len(rows), total_bytes=sum(r['size'] for r in rows))
    (output / 'PUBLISHED_LOGS.json').write_text(json.dumps(published, indent=2) + '\n')
    with (output / 'LOG_INDEX.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(log_files=len(rows), total_bytes=published['total_bytes'], campaigns=len(campaigns))))


if __name__ == '__main__':
    main()
