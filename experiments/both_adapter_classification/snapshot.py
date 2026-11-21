"""Archive the exact runtime sources before the first training worker starts."""
import argparse
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.both_adapter_classification.common import MODELS,save,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True)
    args=p.parse_args();run=args.run_dir.resolve()
    if (run/"source_manifest.json").exists():raise FileExistsError(run/"source_manifest.json")
    here=Path(__file__).parent
    sources=[here/name for name in ("__init__.py","common.py","prepare.py","snapshot.py","run.py","train.py","native_backends.py","pooling_backends.py")]
    sources += [ROOT/"experiments/baseline_token_classification/common.py",ROOT/"experiments/direct_anchor_classification/model.py",
        ROOT/"RadZero/scripts/adapter_model.py",ROOT/"AlphaRAD/local_setup/token_adapter_model.py",ROOT/"AlphaRAD/local_setup/run_pretrained.py"]
    for model in MODELS:
        if model in ("AlphaRAD","RadZero"):continue
        sources += list((ROOT/"baselines"/model/"local_setup").glob("*.py"))
    hashes,archives={},{}
    for path in sorted(set(sources)):
        destination=run/"source"/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        hashes[str(path)]=sha256(path);assert sha256(destination)==hashes[str(path)]
        archives[str(path)]=str(destination)
    save(run/"source_manifest.json",dict(sources_sha256=hashes,archives=archives,protocol_sha256=sha256(run/"PROTOCOL.json")))
    print("Archived",len(hashes),"runtime/dependency files")


if __name__=="__main__":main()
