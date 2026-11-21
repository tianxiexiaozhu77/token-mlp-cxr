"""Archive final runtime sources once, after review and before any new fit."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gated_asl_classification.common import MODELS, save, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if (run / "source_manifest.json").exists() or any((run / "results").glob("*/*/seed*/*/training.json")):
        raise FileExistsError("Source snapshot must precede training and cannot replace an existing manifest")
    protocol = json.loads((run / "PROTOCOL.json").read_text())
    if protocol["new_training_runs"] != 252 or not protocol["all_methods_fresh"]:
        raise ValueError("Expected the frozen all-fresh 252-fit protocol")
    here = Path(__file__).resolve().parent
    sources = [here / name for name in ("__init__.py", "common.py", "components.py", "prepare.py", "snapshot.py", "run.py", "status.py", "train.py")]
    sources += [ROOT / "experiments/both_adapter_classification" / name for name in ("__init__.py", "common.py", "native_backends.py", "pooling_backends.py")]
    sources += [ROOT / "experiments/baseline_token_classification/common.py", ROOT / "experiments/direct_anchor_classification/model.py",
                ROOT / "RadZero/scripts/adapter_model.py", ROOT / "AlphaRAD/local_setup/token_adapter_model.py", ROOT / "AlphaRAD/local_setup/run_pretrained.py"]
    for model in MODELS:
        if model not in ("AlphaRAD", "RadZero"):
            sources += list((ROOT / "baselines" / model / "local_setup").glob("*.py"))
    # Also preserve the declared native backbone/fusion dependencies, which
    # can live outside local_setup (e.g. CARZero's decoder implementation).
    for records in protocol["caches"].values():
        for record in records:
            cache = json.loads(Path(record["path"]).read_text())
            sources += [Path(p) for p in cache.get("backend_metadata", {}).get("source_files", []) if Path(p).suffix == ".py"]
    hashes, archives = {}, {}
    for source in sorted(set(sources)):
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = run / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256(source)
        if destination.exists() and sha256(destination) != digest:
            raise ValueError("An incomplete snapshot contains a different source: " + str(destination))
        shutil.copy2(source, destination)
        if sha256(destination) != digest:
            raise ValueError("Source archive verification failed: " + str(source))
        hashes[str(source)], archives[str(source)] = digest, str(destination)
    save(run / "source_manifest.json", dict(sources_sha256=hashes, archives=archives,
        protocol_sha256=sha256(run / "PROTOCOL.json"), all_methods_fresh=True, new_fits=252))
    print("Archived", len(hashes), "runtime/dependency files; no workers started")


if __name__ == "__main__":
    main()
