#!/usr/bin/env python3
"""Run released AlphaRAD evaluations on checked, idle physical GPUs 3–7."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
# GPU indices are machine-local. NFS locks shared by several servers must not
# reserve unrelated devices with the same index, or retain stale server leases.
LOCKS = Path(tempfile.gettempdir()) / f"medical-gpu-locks-{os.getuid()}"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def release_gpu(lease):
    try:
        fcntl.flock(lease, fcntl.LOCK_UN)
    finally:
        lease.close()


def reserve_gpu(index):
    if index not in range(3, 8):
        raise ValueError("Only physical GPUs 3–7 are authorized for this run")
    LOCKS.mkdir(parents=True, exist_ok=True, mode=0o700)
    gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", str(index),
        "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=15).strip()
    lease_path = LOCKS / f"{gpu_uuid}.lock"
    lease = lease_path.open("a+")
    os.set_inheritable(lease.fileno(), False)
    try:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            return None, {"reason": "local_gpu_lease_busy", "lock_path": str(lease_path)}
        raw = subprocess.check_output(["nvidia-smi", "-i", str(index),
            "--query-gpu=uuid,name,memory.used", "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
        checked_uuid, name, memory = [v.strip() for v in raw.split(",")]
        if checked_uuid != gpu_uuid:
            raise RuntimeError("GPU index changed while acquiring its UUID lease")
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits"], text=True, timeout=15)
        busy = [row for row in apps.splitlines() if row.split(",")[0].strip() == gpu_uuid]
        info = {"physical_index": index, "uuid": gpu_uuid, "name": name,
                "memory_before_mib": int(memory), "checked_at": now(),
                "lock_path": str(lease_path), "hostname": socket.gethostname()}
        if busy or int(memory) > 512:
            release_gpu(lease)
            return None, {**info, "reason": "gpu_in_use", "processes": busy}
        lease.seek(0)
        lease.truncate()
        lease.write(json.dumps({**info, "controller_pid": os.getpid()}) + "\n")
        lease.flush()
        return lease, info
    except BaseException:
        release_gpu(lease)
        raise


def initial_state(args):
    status_path = args.output / "status.json"
    if not args.resume:
        if status_path.exists():
            raise FileExistsError("Use --resume to continue pending jobs, or a fresh run directory")
        return {"status": "running", "started_at": now(), "controller_pid": os.getpid(),
                "controller_hostname": socket.gethostname(),
                "gpus": args.gpus, "catalog": str(args.catalog.resolve()),
                "jobs": {name: {"status": "pending"} for name in args.datasets}}
    state = json.loads(status_path.read_text())
    if state.get("controller_hostname", socket.gethostname()) != socket.gethostname():
        raise ValueError("Resume on the original host to preserve controller ownership")
    previous_proc = Path("/proc") / str(state["controller_pid"])
    if previous_proc.exists() and state["controller_pid"] != os.getpid():
        proc_state = (previous_proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
        if proc_state != "Z":
            raise RuntimeError(f"Previous controller PID is still alive: {state['controller_pid']}")
    if set(args.datasets) != set(state["jobs"]):
        raise ValueError("Resume dataset list must match the original run")
    if str(args.catalog.resolve()) != state["catalog"]:
        raise ValueError("Resume must use the original catalog")
    for name, job in state["jobs"].items():
        job_dir = args.output / name
        if job["status"] == "complete":
            result = json.loads((job_dir / "result.json").read_text())
            if result["dataset"] != name:
                raise ValueError(f"Completed result dataset mismatch: {name}")
        elif job["status"] == "pending":
            if job_dir.exists():
                raise FileExistsError(f"Pending job already has output; refusing overwrite: {job_dir}")
        else:
            raise ValueError(f"Resume supports complete/pending jobs only: {name}: {job['status']}")
    state.setdefault("resumes", []).append({"resumed_at": now(),
        "previous_controller_pid": state["controller_pid"], "controller_pid": os.getpid()})
    state.update(status="running", controller_pid=os.getpid(),
                 controller_hostname=socket.gethostname(), gpus=args.gpus)
    state.pop("finished_at", None)
    state.pop("last_gpu_checks", None)
    return state


def terminate_group(proc, sig):
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def report(output, state):
    lines = ["AlphaRAD：作者公开预训练模型，无训练/微调/支持集适配。",
             f"状态：{state['status']}；更新：{now()}",
             "只使用运行前经检查空闲的物理 GPU 3–7。",
             "dataset | status | images | AUROC % | AP % | AUPRC(trapezoid) % | other",
             "--- | --- | --- | --- | --- | --- | ---"]
    for name, job in state["jobs"].items():
        path = output / name / "result.json"
        if job["status"] == "complete" and path.exists():
            data = json.loads(path.read_text())
            count = data["rows"]
            c = data.get("classification", {})
            values = ["—" if c.get(k) is None else f"{c[k] * 100:.2f}" for k in
                      ("macro_auroc", "macro_average_precision", "macro_auprc_trapezoidal")]
            other = []
            for section, key in (("zeroshot_pointing_game", "Pointing_Game(Avg)"),
                                 ("zeroshot_seg", "zeroshot_segmentation(Avg)")):
                value = data["author_metrics"].get(section, {}).get(key)
                if value is not None:
                    other.append(f"{key}={100 * value:.2f}%")
            lines.append(f"{name} | complete | {count} | " + " | ".join(values) + " | " + "; ".join(other))
        else:
            lines.append(f"{name} | {job['status']} | — | — | — | — | {job.get('reason', '')}")
    lines += ["", "注意：AP 是作者代码中名为 PR_AUROC 的 Average Precision；梯形AUPRC单独计算。",
              "公开checkpoint是作者重训版，README参考值和论文原始权重结果可能不同。",
              "每项数据来源、完整性与非原论文协议差异见对应 result.json / data_audit.json；样例不是全集复现。"]
    (output / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    write_json(output / "status.json", state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=ROOT / "local_setup/data_20261021_complete/catalog.json")
    p.add_argument("--datasets", nargs="+", default=["chestxdet10", "chestxdet10_gt", "chest14", "siim_pneumothorax", "rsna_pneumonia"])
    p.add_argument("--gpus", type=int, nargs="+", choices=range(3, 8), default=list(range(3, 8)))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true", help="Keep completed results and continue pending jobs")
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    LOCKS.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_key = hashlib.sha256(str(args.output).encode()).hexdigest()
    controller_lock = (LOCKS / f"controller_{run_key}.lock").open("a+")
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = initial_state(args)
    active = {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            for gpu, (proc, name, lease, log) in list(active.items()):
                code = proc.poll()
                if code is not None:
                    complete = code == 0 and (args.output / name / "result.json").exists()
                    state["jobs"][name].update(status="complete" if complete else "failed", exit_code=code, finished_at=now())
                    release_gpu(lease)
                    log.close()
                    del active[gpu]
            catalog = json.loads(args.catalog.read_text())
            ready = [name for name, job in state["jobs"].items()
                     if job["status"] == "pending" and name in catalog["datasets"]]
            for gpu in args.gpus:
                if gpu in active or not ready:
                    continue
                lease, info = reserve_gpu(gpu)
                if lease is None:
                    state.setdefault("last_gpu_checks", {})[str(gpu)] = info
                    continue
                name = ready.pop(0)
                job_dir = args.output / name
                log = None
                try:
                    job_dir.mkdir(exist_ok=False)
                    frozen = {**catalog, "datasets": {name: catalog["datasets"][name]}}
                    write_json(job_dir / "catalog.json", frozen)
                    command = ["bash", str(ROOT / "local_setup/python_local.sh"),
                               str(ROOT / "local_setup/eval_pretrained.py"), "--catalog", str(job_dir / "catalog.json"),
                               "--dataset", name, "--output", str(job_dir), "--batch-size", str(args.batch_size),
                               "--workers", str(args.workers)]
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": info["uuid"],
                           "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4"}
                    log = (job_dir / "run.log").open("w")
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True, close_fds=True)
                except BaseException:
                    if log is not None:
                        log.close()
                    release_gpu(lease)
                    raise
                active[gpu] = (proc, name, lease, log)
                state["jobs"][name] = {"status": "running", "started_at": now(), "pid": proc.pid,
                                       "gpu": info, "command": command}
                print(f"Started {name} on physical GPU {gpu}, PID {proc.pid}", flush=True)
            if not active and all(j["status"] != "pending" for j in state["jobs"].values()):
                state["status"] = "complete" if all(j["status"] == "complete" for j in state["jobs"].values()) else "finished_with_failures"
                break
            if catalog.get("status") == "complete":
                for name, job in state["jobs"].items():
                    if job["status"] == "pending" and name not in catalog["datasets"]:
                        job.update(status="unavailable", reason="No prepared dataset in completed catalog")
            report(args.output, state)
            time.sleep(10)
    finally:
        if active:
            for proc, name, lease, log in active.values():
                if proc.poll() is None:
                    terminate_group(proc, signal.SIGTERM)
            for proc, name, lease, log in active.values():
                try:
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        terminate_group(proc, signal.SIGKILL)
                        proc.wait()
                finally:
                    release_gpu(lease)
                    log.close()
                state["jobs"][name].update(status="interrupted", finished_at=now())
            state["status"] = "interrupted"
        if stopping:
            state["status"] = "interrupted"
        state["finished_at"] = now()
        report(args.output, state)
        release_gpu(controller_lock)
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
