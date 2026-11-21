"""Verify the locked RSNA patient mapping and exclude exact source duplicates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.supervised_baseline_classification.common import *


def image_hash(row):
    from PIL import Image
    path=Path(row['image'])
    with Image.open(path) as im:
        rgb=im.convert('RGB');pixels=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
    return dict(id=row['id'],image=str(path),file_sha256=sha256(path),decoded_rgb_sha256=pixels,
        bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN);a=p.parse_args()
    run=a.run_dir;out=run/'external';protocol=json.loads((out/'PROTOCOL.json').read_text())
    assert sha256(protocol['samples_file'])==protocol['samples_sha256']
    assert sha256(protocol['source_manifest'])==protocol['source_manifest_sha256']
    rows=json.loads(Path(protocol['samples_file']).read_text())
    mapping=protocol['official_patient_mapping'];assert sha256(mapping['path'])==mapping['sha256']
    official={r['subset_img_id']:r['img_id'] for r in json.loads(Path(mapping['path']).read_text())}
    for r in rows:
        assert official[r['id']]==r['nih_image_id']
        assert 'NIH:'+Path(r['nih_image_id']).stem.split('_')[0]==r['patient_id']==r['group_id']
    meta,source=read_dataset(DATASETS['chexpert']);start=time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        source_hashes=list(pool.map(image_hash,source));target_hashes=list(pool.map(image_hash,rows))
    save(out/'image_hashes.json',dict(source=source_hashes,target=target_hashes))
    overlap={k:sorted({r[k] for r in source_hashes}&{r[k] for r in target_hashes}) for k in ('file_sha256','decoded_rgb_sha256')}
    # No target exclusion is performed silently. An overlap stops all scoring.
    assert not any(overlap.values()),overlap
    by_hash={}
    for r,h in zip(rows,target_hashes):by_hash.setdefault(h['decoded_rgb_sha256'],set()).add(r['patient_id'])
    cross_patient_duplicates=[v for v in by_hash.values() if len(v)>1]
    assert not cross_patient_duplicates,'Cross-patient exact duplicates require revised grouping'
    result=dict(status='passed',images=len(rows),patients=len({r['patient_id'] for r in rows}),positives=sum(r['labels'][0] for r in rows),
        source_images=len(source),source_patients=len({r['patient_id'] for r in source}),
        mapping_verified=True,source_duplicate_overlap=overlap,cross_patient_exact_duplicate_groups=0,
        protocol_sha256=sha256(out/'PROTOCOL.json'),image_hashes_sha256=sha256(out/'image_hashes.json'),
        code_sha256=sha256(__file__),seconds=time.time()-start,
        independence_scope='NIH-mapped target patients versus Stanford CheXpert adaptation cohort; exact image overlap checked. Cross-institution human identity linkage is unavailable; pretraining overlap is not ruled out.',
        official_split=False,selection='Previously fixed local cohort, locked before this transfer evaluation; no current-target tuning.')
    save(out/'cohort_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
