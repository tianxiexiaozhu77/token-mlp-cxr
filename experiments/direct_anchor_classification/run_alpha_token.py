"""Run the nine prespecified AlphaRAD token-MLP fits on physical GPUs 4--7."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

from experiment_utils import ROOT, save, sha256


class Adopted:
    def __init__(self, job): self.job = job
    def poll(self):
        path = Path(f"/proc/{self.job['pid']}/cmdline")
        if path.exists() and self.job["output"].encode() in path.read_bytes(): return None
        return 0 if (Path(self.job["output"])/"token_mlp/result.json").exists() else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--gpus", default="4,5,6,7")
    args = p.parse_args()
    allowed = set(map(int, args.gpus.split(",")))
    if not allowed or not allowed <= {4,5,6,7}:
        raise ValueError("This supplement is restricted to physical GPUs 4--7")
    run, reference = args.run_dir.resolve(), args.reference_run.resolve()
    jobs = []
    for dataset in ("chestxray14", "openi", "chexpert"):
        cache = reference / "cache/alpharad" / dataset
        record = json.loads((cache / "cache_00.json").read_text())
        for seed in (42,43,44):
            jobs.append(dict(id=f"alpharad_{dataset}_{seed}_token_mlp", model="alpharad",
                dataset=dataset, seed=seed, method="token_mlp", status="pending",
                metadata=str(Path(record["dataset"]).resolve()), cache=str(cache),
                output=str(run / "results/alpharad" / dataset / f"seed{seed}")))
    (run / "logs").mkdir(parents=True, exist_ok=True)
    active = {}
    matrix_path = run / "training_matrix.json"
    if matrix_path.exists():
        previous = {j["id"]:j for j in json.loads(matrix_path.read_text())["jobs"]}
        for i, job in enumerate(jobs):
            old = previous.get(job["id"])
            if old and old["status"] == "running":
                if old["gpu"] not in allowed: raise ValueError("Adopted GPU outside allowed set")
                jobs[i] = old; active[old["gpu"]] = Adopted(old), old, None
            elif old and old["status"] == "failed":
                raise RuntimeError("Review failed run before restarting; no automatic seed replacement")
    while True:
        for gpu, (process, job, log) in list(active.items()):
            code = process.poll()
            if code is not None:
                if log: log.close()
                complete = code == 0 and (Path(job["output"])/"token_mlp/result.json").exists()
                job.update(status="complete" if complete else "failed", exit_code=code, end=time.time())
                del active[gpu]
        for job in jobs:
            result = Path(job["output"])/"token_mlp/result.json"
            if job["status"] == "pending" and result.exists():
                data = json.loads(result.read_text())
                if data["status"] != "complete" or data["method"] != "token_mlp" or data["seed"] != job["seed"]:
                    raise ValueError("Existing result does not match scheduled run")
                job["status"] = "complete_reused"
        query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
        free = [int(row.split(",")[0]) for row in query.splitlines() if int(row.split(",")[1]) < 512]
        free = [g for g in free if g in allowed and g not in active]
        lockable = []
        for gpu in free:
            with (ROOT/f"RadZero/runs/gpu_locks/eval_gpu_{gpu}.lock").open("a+") as probe:
                try: fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError: continue
                fcntl.flock(probe, fcntl.LOCK_UN); lockable.append(gpu)
        free = lockable
        for job in jobs:
            if not free or job["status"] != "pending": continue
            records = [json.loads(p.read_text()) for p in Path(job["cache"]).glob("cache_*.json")]
            checks = [run/"cache_checksums"/job["dataset"]/(Path(s["path"]).stem+".json")
                      for r in records for s in r["token_shards"]]
            if not checks or any(not p.exists() or json.loads(p.read_text())["status"] != "complete" for p in checks):
                continue
            gpu = free.pop(0)
            cmd = ["bash", str(ROOT/"AlphaRAD/local_setup/python_local.sh"), "-u",
                   str(ROOT/"AlphaRAD/local_setup/train_direct_anchor.py"),
                   "--dataset", job["metadata"], "--cache", job["cache"], "--output", job["output"],
                   "--gpu", str(gpu), "--seed", str(job["seed"]), "--methods", "token_mlp"]
            log_path = run/"logs"/(job["id"]+".log")
            log = log_path.open("w")
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            job.update(status="running", pid=process.pid, gpu=gpu, start=time.time(), command=cmd, log=str(log_path))
            active[gpu] = process, job, log
            print(f"launched {job['id']} on physical GPU {gpu}", flush=True)
        failed = any(j["status"] == "failed" for j in jobs)
        complete = all(j["status"].startswith("complete") for j in jobs)
        save(matrix_path, dict(status="complete" if complete else ("failed" if failed else "running"),
            jobs=jobs, seeds=[42,43,44], allowed_physical_gpus=sorted(allowed), reference_run=str(reference), updated=time.time()))
        if complete: break
        if failed and not active: raise RuntimeError("Training worker failed; inspect logs")
        time.sleep(10)
    for script in ("validate_alpha_token.py", "report_alpha_token.py"):
        subprocess.run(["bash", str(ROOT/"AlphaRAD/local_setup/python_local.sh"),
            str(Path(__file__).with_name(script)), "--run-dir", str(run), "--reference-run", str(reference)], cwd=ROOT, check=True)
    outputs = [run/name for name in ("RESULTS.txt", "RESULTS.csv", "per_class_results.csv", "summary.json",
                                    "comparison.png", "comparison.pdf", "checkpoint_validation.json")]
    save(run/"completion.json", dict(status="complete", new_evaluations=9, training_seeds=[42,43,44],
        allowed_physical_gpus=sorted(allowed), reference_run=str(reference), completed=time.time(),
        output_sha256={str(p):sha256(p) for p in outputs}))


if __name__ == "__main__":
    main()
