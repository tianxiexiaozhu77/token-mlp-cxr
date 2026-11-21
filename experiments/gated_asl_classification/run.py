"""Schedule 21 cache-sharing jobs / 252 fresh fits on idle physical GPUs 4--7."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gated_asl_classification.common import METHODS, MODELS, save, sha256


def completed_job(run, job, protocol, protocol_hash):
    for method in protocol["methods"]:
        for seed in protocol["seeds"]:
            path = run / "results" / job["model"] / job["dataset"] / f"seed{seed}" / method / "result.json"
            if not path.exists():
                return False
            record = json.loads(path.read_text())
            if (record.get("status") != "complete" or record.get("protocol_sha256") != protocol_hash
                    or record.get("model") != job["model"] or record.get("dataset") != job["dataset"]
                    or record.get("method") != method or record.get("seed") != seed):
                raise ValueError("Invalid current-campaign completed fit: " + str(path))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="4,5,6,7")
    args = parser.parse_args()
    run, allowed = args.run_dir.resolve(), set(map(int, args.gpus.split(",")))
    if not allowed or not allowed <= {4, 5, 6, 7}:
        raise ValueError("Only physical GPUs 4--7 are authorized")
    protocol = json.loads((run / "PROTOCOL.json").read_text())
    digest = sha256(run / "PROTOCOL.json")
    manifest = json.loads((run / "source_manifest.json").read_text())
    if (protocol["new_training_runs"] != 252 or protocol["methods"] != list(METHODS)
            or protocol["models"] != list(MODELS) or not protocol["all_methods_fresh"]
            or manifest["protocol_sha256"] != digest):
        raise ValueError("Protocol/source snapshot is not the frozen four-group fresh-fit campaign")
    (run / "logs").mkdir(parents=True, exist_ok=True)
    controller = (run / "controller.lock").open("a+")
    fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = [dict(id=f"{model}_{dataset}", model=model, dataset=dataset, status="pending", expected_fits=12)
            for dataset in protocol["dataset_schedule"] for model in protocol["models"]]
    active = {}
    while True:
        for gpu, (process, job, log) in list(active.items()):
            code = process.poll()
            if code is not None:
                log.close()
                valid = code == 0 and completed_job(run, job, protocol, digest)
                job.update(status="complete" if valid else "failed", exit_code=code, end=time.time())
                del active[gpu]
        failures = any(job["status"] == "failed" for job in jobs)
        if not failures:
            raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True, timeout=20)
            free = [int(line.split(",")[0]) for line in raw.splitlines()
                    if int(line.split(",")[0]) in allowed and int(line.split(",")[1]) < 512]
            free = [gpu for gpu in free if gpu not in active]
            for job in jobs:
                if job["status"] != "pending":
                    continue
                if completed_job(run, job, protocol, digest):
                    # Resume only complete outputs of this new protocol; old
                    # historical groups are never substituted for fresh fits.
                    job["status"] = "complete_current_campaign_resume"
                    continue
                if not free:
                    continue
                gpu = free.pop(0)
                launcher = (ROOT / "AlphaRAD/local_setup/python_local.sh" if job["model"] in ("AlphaRAD", "RadZero")
                            else ROOT / "baselines" / job["model"] / "local_setup/python_local.sh")
                command = ["bash", str(launcher), "-u", str(Path(__file__).with_name("train.py")), "--model", job["model"],
                           "--dataset", job["dataset"], "--run-dir", str(run), "--gpu", str(gpu)]
                log_path = run / "logs" / (job["id"] + ".log")
                log = log_path.open("a")
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                job.update(status="running", pid=process.pid, gpu=gpu, command=command, start=time.time(), log=str(log_path))
                active[gpu] = process, job, log
                print("LAUNCHED", job["id"], "GPU", gpu, "12 fresh fits", flush=True)
        complete = all(job["status"].startswith("complete") for job in jobs)
        save(run / "training_matrix.json", dict(status="complete" if complete else "failed" if failures else "running",
            jobs=jobs, allowed_gpus=sorted(allowed), updated=time.time(), new_fits=252, methods=protocol["methods"],
            all_methods_fresh=True, protocol_sha256=digest))
        if complete:
            break
        if failures and not active:
            raise RuntimeError("A worker failed; inspect its logs and incomplete outputs before retry")
        time.sleep(5)


if __name__ == "__main__":
    main()
