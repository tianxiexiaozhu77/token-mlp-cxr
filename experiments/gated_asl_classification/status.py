"""Read live four-method progress without changing any experiment artifact."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    raw = (run / "PROTOCOL.json").read_bytes()
    protocol, phash = json.loads(raw), hashlib.sha256(raw).hexdigest()
    matrix = json.loads((run / "training_matrix.json").read_text()) if (run / "training_matrix.json").exists() else dict(status="not_started", jobs=[])
    by_method, by_dataset, completed, invalid = Counter(), Counter(), 0, []
    for path in (run / "results").glob("*/*/seed*/*/result.json"):
        record = json.loads(path.read_text())
        if (record.get("status") != "complete" or record.get("protocol_sha256") != phash
                or record.get("method") not in protocol["methods"] or record.get("model") not in protocol["models"]
                or record.get("dataset") not in protocol["datasets"] or record.get("seed") not in protocol["seeds"]):
            invalid.append(str(path))
            continue
        completed += 1
        by_method[record["method"]] += 1
        by_dataset[record["dataset"]] += 1
    active = []
    for job in matrix["jobs"]:
        if job["status"] not in ("running", "failed"):
            continue
        record = dict(job=job["id"], status=job["status"], gpu=job.get("gpu"))
        candidates = []
        for path in (run / "results" / job["model"] / job["dataset"]).glob("seed*/*/training.json"):
            state = json.loads(path.read_text())
            candidates.append((state.get("status") == "running", path.stat().st_mtime_ns, path, state))
        if candidates:
            _, _, path, state = max(candidates, key=lambda value: (value[0], value[1]))
            record.update(method=state["method"], seed=state["seed"], fit_status=state["status"],
                epochs=len(state.get("history", [])), best_epoch=state.get("best_epoch"))
            progress = path.with_name("progress.json")
            if state["status"] == "running" and progress.exists():
                values = json.loads(progress.read_text())
                record.update({key: values[key] for key in ("epoch", "processed", "total") if key in values})
        else:
            record["stage"] = "cache/model validation"
        if job["status"] == "failed" and Path(job.get("log", "")).is_file():
            record["last_log_lines"] = Path(job["log"]).read_text().splitlines()[-8:]
        active.append(record)
    print(json.dumps(dict(status=matrix["status"], completed=completed, expected=252,
        by_method={method: dict(completed=by_method[method], expected=63) for method in protocol["methods"]},
        by_dataset={dataset: dict(completed=by_dataset[dataset], expected=84) for dataset in protocol["dataset_schedule"]},
        active=active, invalid_results=invalid), ensure_ascii=False))


if __name__ == "__main__":
    main()
