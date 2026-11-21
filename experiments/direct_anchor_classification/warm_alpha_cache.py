"""Sequentially checksum existing frozen token files and warm the OS page cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from experiment_utils import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    jobs = []
    for dataset in ("chexpert", "openi", "chestxray14"):
        for manifest in sorted((args.reference_run / "cache/alpharad" / dataset).glob("cache_*.json")):
            record = json.loads(manifest.read_text())
            for spec in record["token_shards"]:
                jobs.append((dataset, Path(spec["path"])))

    def checksum(job):
        dataset, path = job
        output = args.output / dataset / (path.stem + ".json")
        if output.exists() and json.loads(output.read_text()).get("status") == "complete":
            return json.loads(output.read_text())
        started = time.time(); total = path.stat().st_size; done = 0
        h = hashlib.sha256()
        def progress(status):
            value = dict(status=status, path=str(path), dataset=dataset, bytes=done,
                         total_bytes=total, seconds=time.time()-started)
            if status == "complete": value["sha256"] = h.hexdigest()
            save(output, value)
            return value
        progress("running")
        with path.open("rb") as stream:
            while block := stream.read(16 * 1024**2):
                h.update(block); done += len(block)
                if done % (1024**3) < len(block): progress("running")
        if done != total: raise ValueError(f"Token cache changed while reading: {path}")
        return progress("complete")

    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in as_completed([pool.submit(checksum, job) for job in jobs]):
            print(json.dumps(result.result()), flush=True)


if __name__ == "__main__":
    main()
