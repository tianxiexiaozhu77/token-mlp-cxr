"""Plan or advise dropping only completed Both jobs' token-file page caches.

Default is plan-only; --apply issues targeted POSIX_FADV_DONTNEED through
O_RDONLY descriptors. It never deletes/writes source files, changes training,
or invokes global drop_caches. This is advisory: in-use/dirty pages may remain.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import save, sha256
from experiments.both_adapter_classification.warm_missing import missing_blocks, SCAN_WINDOW


def check_complete(run, protocol, model, dataset, protocol_hash):
    matrix = json.loads((run / "training_matrix.json").read_text())
    if matrix["protocol_sha256"] != protocol_hash:
        raise ValueError("Training matrix protocol differs")
    jobs = [j for j in matrix["jobs"] if j["model"] == model and j["dataset"] == dataset]
    if len(jobs) != 1 or jobs[0]["status"] != "complete":
        raise ValueError(f"Only exact matrix status complete is allowed: {model}/{dataset}")
    for seed in protocol["seeds"]:
        result_path = run / "results" / model / dataset / f"seed{seed}" / "both_mlp/result.json"
        result = json.loads(result_path.read_text())
        if result["status"] != "complete" or result["protocol_sha256"] != protocol_hash:
            raise ValueError("Incomplete or different-protocol fit: " + str(result_path))


def observable_readers(filenames):
    """Inspect accessible /proc mappings/fds; record limits, do not signal PIDs."""
    matches = {name: [] for name in filenames}
    denied = set()
    checked = 0
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        pid = int(folder.name)
        found = set()
        try:
            for line in (folder / "maps").read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and fields[5].removesuffix(" (deleted)") in matches:
                    found.add(fields[5].removesuffix(" (deleted)"))
            checked += 1
        except PermissionError:
            denied.add(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        try:
            for fd in (folder / "fd").iterdir():
                try:
                    target = os.readlink(fd).removesuffix(" (deleted)")
                    if target in matches:
                        found.add(target)
                except (FileNotFoundError, ProcessLookupError):
                    pass
        except PermissionError:
            denied.add(pid)
        except (FileNotFoundError, ProcessLookupError):
            pass
        for filename in found:
            matches[filename].append(pid)
    return matches, dict(accessible_process_maps_checked=checked, inaccessible_pids=sorted(denied),
                        scope="accessible process mappings and file descriptors; not an exclusive-use guarantee")


def resident_bytes(fd, size):
    missing = 0
    for offset in range(0, size, SCAN_WINDOW):
        _, _, amount = missing_blocks(fd, size, offset, min(SCAN_WINDOW, size - offset))
        missing += amount
    return size - missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--datasets", required=True, help="Comma-separated completed datasets, e.g. openi,chexpert")
    parser.add_argument("--models", help="Comma-separated models; default all protocol models")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    protocol_path = run / "PROTOCOL.json"
    digest = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    datasets = args.datasets.split(",")
    models = args.models.split(",") if args.models else protocol["models"]
    if (not datasets or len(set(datasets)) != len(datasets) or not set(datasets) <= set(protocol["datasets"])
            or not models or len(set(models)) != len(models) or not set(models) <= set(protocol["models"])):
        parser.error("Datasets/models must be unique protocol names")
    targets, seen = [], set()
    for model in models:
        for dataset in datasets:
            check_complete(run, protocol, model, dataset, digest)
            for spec in protocol["caches"][model + "/" + dataset]:
                if sha256(spec["path"]) != spec["sha256"]:
                    raise ValueError("Cache record changed: " + spec["path"])
                cache = json.loads(Path(spec["path"]).read_text())
                paths = ([t["path"] for t in cache["token_shards"]] if "token_shards" in cache
                         else [cache["files"]["tokens.npy"]["path"]])
                for filename in paths:
                    if filename in seen:
                        continue
                    seen.add(filename)
                    expected = spec["files"][filename]
                    stat = Path(filename).stat()
                    if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
                        raise ValueError("Token file metadata changed: " + filename)
                    targets.append(dict(model=model, dataset=dataset, path=filename,
                        bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, device=stat.st_dev, inode=stat.st_ino,
                        cache_record=spec["path"], cache_record_sha256=spec["sha256"], status="pending"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = run / "io_warming" / ("release_completed_" + stamp + ".json")
    started = time.time()
    state = dict(status="running", apply=args.apply, helper_sha256=sha256(__file__),
        protocol_sha256=digest, files=targets, source_files_opened_read_only=True,
        global_drop_caches=False, advice="POSIX_FADV_DONTNEED only when --apply", source_files_modified=False,
        observations_are_snapshots=True, seconds=0.)
    save(output, state)
    try:
        for target in targets:
            check_complete(run, protocol, target["model"], target["dataset"], digest)
            readers, observation = observable_readers([target["path"]])
            target.update(observable_reader_pids=readers[target["path"]], process_observation=observation)
            if readers[target["path"]]:
                target["status"] = "skipped_observable_reader"
                save(output, state)
                continue
            fd = os.open(target["path"], os.O_RDONLY)
            try:
                before = os.fstat(fd)
                identity = target["device"], target["inode"], target["bytes"], target["mtime_ns"]
                if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != identity:
                    raise ValueError("Opened token file identity changed: " + target["path"])
                target["resident_bytes_before"] = resident_bytes(fd, before.st_size)
                if args.apply:
                    # Advisory invalidation of this file only. The kernel may
                    # retain in-use/dirty pages; no fsync or writeback is called.
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    target["resident_bytes_after"] = resident_bytes(fd, before.st_size)
                    target["status"] = "advice_issued"
                else:
                    target["status"] = "planned_only"
                after = os.fstat(fd)
                path_after = Path(target["path"]).stat()
                for stat in (after, path_after):
                    if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != identity:
                        raise ValueError("Token file changed during advisory check: " + target["path"])
                target["metadata_unchanged"] = True
            finally:
                os.close(fd)
            state["seconds"] = time.time() - started
            save(output, state)
            print(target["model"], target["dataset"], Path(target["path"]).name,
                  target["status"], "resident_before_GiB", round(target["resident_bytes_before"] / 1024**3, 3),
                  "resident_after_GiB", round(target.get("resident_bytes_after", 0) / 1024**3, 3), flush=True)
        if sha256(protocol_path) != digest:
            raise ValueError("Frozen protocol changed")
        state.update(status="complete", seconds=time.time() - started,
            advice_issued_files=sum(t["status"] == "advice_issued" for t in targets),
            skipped_files=sum(t["status"].startswith("skipped") for t in targets),
            resident_bytes_before=sum(t.get("resident_bytes_before", 0) for t in targets),
            resident_bytes_after=(sum(t.get("resident_bytes_after", 0) for t in targets) if args.apply else None))
        save(output, state)
        print(json.dumps(dict(status=state["status"], output=str(output), advice_issued_files=state["advice_issued_files"])), flush=True)
    except BaseException as error:
        state.update(status="failed", error=repr(error), seconds=time.time() - started)
        save(output, state)
        raise


if __name__ == "__main__":
    main()
