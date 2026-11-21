"""Finish independent audits and artifacts after the full training matrix."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from experiment_utils import ROOT, save, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    run = args.run_dir.resolve()
    while True:
        matrix = json.loads((run / "training_matrix.json").read_text())
        if matrix["status"] == "complete":
            break
        if any(j["status"] == "failed" for j in matrix["jobs"]):
            raise RuntimeError("A training worker failed")
        time.sleep(15)
    for script in ("validate_native.py", "report.py"):
        subprocess.run(["bash", str(ROOT / "AlphaRAD/local_setup/python_local.sh"),
                        str(Path(__file__).with_name(script)), "--run-dir", str(run)],
                       cwd=ROOT, check=True)
    logs = run / "runtime_logs"; logs.mkdir(exist_ok=True)
    for path in Path("/tmp").glob("direct_anchor_*.log"):
        shutil.copy2(path, logs / path.name)
    outputs = [run / name for name in ("RESULTS.txt", "RESULTS.csv", "per_class_results.csv",
               "summary.json", "comparison.png", "comparison.pdf", "native_anchor_validation.json", "data_audit.json")]
    save(run / "completion.json", dict(status="complete", evaluations=54, training_seeds=[42,43,44],
        output_sha256={str(p):sha256(p) for p in outputs}, completed=time.time()))


if __name__ == "__main__":
    main()
