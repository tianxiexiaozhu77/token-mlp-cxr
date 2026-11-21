"""Schedule the predeclared three-seed matrix as frozen caches become ready."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

from experiment_utils import ROOT, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = p.parse_args()
    run = args.run_dir.resolve()
    jobs = []
    for dataset in ("chexpert", "openi", "chestxray14"):
        for model in ("alpharad", "radzero"):
            for seed in (42, 43, 44):
                methods = ["bias", "direct"]
                if seed == 42:
                    methods.insert(0, "baseline")
                if model == "radzero" or seed == 42:
                    methods.append("existing")
                out = run / "results" / model / dataset / f"seed{seed}"
                meta = ROOT / ("RadZero/runs/openi_adapter_20261020_152558/inputs/dataset.json" if dataset == "openi" else f"data/direct_anchor_classification_20261118/{dataset}/dataset.json")
                jobs.append(dict(id=f"{model}_{dataset}_{seed}", model=model, dataset=dataset,
                    seed=seed, methods=methods, output=str(out), metadata=str(meta), status="pending"))
    (run / "logs").mkdir(parents=True, exist_ok=True)
    active = {}
    previous_path = run / "training_matrix.json"
    if previous_path.exists():
        previous = {j["id"]:j for j in json.loads(previous_path.read_text())["jobs"]}
        for index, job in enumerate(jobs):
            old = previous.get(job["id"])
            if old is None:
                continue
            if old["status"] == "running":
                class Adopted:
                    def __init__(self, job): self.job = job
                    def poll(self):
                        cmd = Path(f"/proc/{self.job['pid']}/cmdline")
                        if cmd.exists() and self.job["output"].encode() in cmd.read_bytes():
                            return None
                        return 0 if all((Path(self.job["output"])/m/"result.json").exists() for m in self.job["methods"]) else 1
                jobs[index] = old
                active[old["gpu"]] = Adopted(old), old, None
            elif old["status"] == "failed":
                if Path(old["output"]).exists():
                    raise RuntimeError("Failed job has partial training output; review before retrying")
                archive = run / "failed_attempts" / (old["id"] + "_startup.log")
                archive.parent.mkdir(exist_ok=True)
                if Path(old["log"]).exists():
                    Path(old["log"]).replace(archive)
                job["startup_retry"] = {"previous_gpu":old["gpu"], "previous_log":str(archive), "reason":"GPU lock acquisition failed before training"}
    while True:
        for gpu, (process, job, log) in list(active.items()):
            code = process.poll()
            if code is not None:
                if log is not None:
                    log.close()
                job.update(status="complete" if code == 0 else "failed", exit_code=code, end=time.time())
                del active[gpu]
        for job in jobs:
            if job["status"] == "pending" and all((Path(job["output"])/m/"result.json").exists() for m in job["methods"]):
                job["status"] = "complete_reused"
        query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
        free = [int(row.split(",")[0]) for row in query.splitlines() if int(row.split(",")[1]) < 512]
        free = [g for g in free if str(g) in args.gpus.split(",") and g not in active]
        lockable = []
        for gpu in free:
            with (ROOT / f"RadZero/runs/gpu_locks/eval_gpu_{gpu}.lock").open("a+") as probe:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                fcntl.flock(probe, fcntl.LOCK_UN)
                lockable.append(gpu)
        free = lockable
        for job in jobs:
            if not free or job["status"] != "pending":
                continue
            cache = run / "cache" / job["model"] / job["dataset"]
            states = [json.loads(path.read_text()) for path in cache.glob("cache_*.json")]
            if not states or len(states) != states[0]["shards"] or any(s["status"] != "complete" for s in states):
                continue
            gpu = free.pop(0)
            prefix = "AlphaRAD/local_setup" if job["model"] == "alpharad" else "RadZero/scripts"
            cmd = ["bash", str(ROOT / prefix / "python_local.sh"), "-u", str(ROOT / prefix / "train_direct_anchor.py"),
                   "--dataset", job["metadata"], "--cache", str(cache), "--output", job["output"],
                   "--gpu", str(gpu), "--seed", str(job["seed"]), "--methods", ",".join(job["methods"])]
            log_path = run / "logs" / (job["id"] + ".log")
            log = log_path.open("w")
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            job.update(status="running", pid=process.pid, gpu=gpu, start=time.time(), command=cmd, log=str(log_path))
            active[gpu] = process, job, log
            print(f"launched {job['id']} on GPU {gpu}", flush=True)
        save(run / "training_matrix.json", dict(jobs=jobs, seeds=[42,43,44], updated=time.time(),
            status="complete" if all(j["status"].startswith("complete") for j in jobs) else "running"))
        if all(j["status"].startswith("complete") or j["status"] == "failed" for j in jobs):
            if any(j["status"] == "failed" for j in jobs):
                raise RuntimeError("Training jobs failed; inspect training_matrix.json")
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
