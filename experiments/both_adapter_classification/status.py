"""Read the live controller/results without changing running experiments."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,default=Path(__file__).parent/"runs/20261119")
    a=p.parse_args();run=a.run_dir.resolve()
    matrix=json.loads((run/"training_matrix.json").read_text())
    completed=list((run/"results").glob("*/*/seed*/both_mlp/result.json"))
    active=[]
    for job in matrix["jobs"]:
        if job["status"] not in ("running","failed"):continue
        record=dict(job=job["id"],status=job["status"],gpu=job.get("gpu"))
        files=sorted((run/"results"/job["model"]/job["dataset"]).glob("seed*/both_mlp/training.json"))
        if files:
            state=json.loads(files[-1].read_text())
            record.update(seed=state["seed"],fit_status=state["status"],epochs=len(state["history"]),best_epoch=state["best_epoch"])
            progress=files[-1].with_name("progress.json")
            if state["status"]=="running" and progress.exists():
                value=json.loads(progress.read_text());record.update(epoch=value["epoch"],processed=value["processed"],total=value["total"])
        else:record["stage"]="cache/model validation"
        if job["status"]=="failed":record["last_log_lines"]=Path(job["log"]).read_text().splitlines()[-8:]
        active.append(record)
    print(json.dumps(dict(status=matrix["status"],completed=len(completed),expected=63,active=active),ensure_ascii=False))


if __name__=="__main__":main()
