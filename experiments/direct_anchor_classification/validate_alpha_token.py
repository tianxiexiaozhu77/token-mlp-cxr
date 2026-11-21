"""Reload every fitted MLP and spot-check three stored test rows from raw tokens."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from experiment_utils import ROOT, save, sha256
sys.path.insert(0, str(ROOT/"AlphaRAD/local_setup"))
from token_adapter_model import load_token_adapter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--reference-run", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    results = []
    for dataset in ("openi", "chexpert", "chestxray14"):
        arrays, locations = [], {}
        for path in sorted((args.reference_run/"cache/alpharad"/dataset).glob("cache_*.json")):
            record = json.loads(path.read_text())
            for spec in record["token_shards"]:
                array = np.load(spec["path"], mmap_mode="r")
                for local, index in enumerate(spec["indices"]): locations[index] = len(arrays), local
                arrays.append(array)
        for seed in (42,43,44):
            folder = args.run_dir/"results/alpharad"/dataset/f"seed{seed}"/"token_mlp"
            head, adapter, classes = load_token_adapter(folder/"adapter.pt")
            pred = np.load(folder/"test_predictions.npz", allow_pickle=False)
            result = json.loads((folder/"result.json").read_text())
            assert classes == result["class_names"]
            assert result["gpu"]["physical_index"] in (4,5,6,7)
            chosen = [0, len(pred["indices"])//2, len(pred["indices"])-1]
            tensors = []
            for i in pred["indices"][chosen]:
                a, j = locations[int(i)]; tensors.append(np.array(arrays[a][j], copy=True))
            tokens = torch.from_numpy(np.stack(tensors))
            with torch.no_grad(): logits = head(tokens, token_adapter=adapter).numpy()
            np.testing.assert_allclose(logits, pred["logits"][chosen], atol=3e-5, rtol=3e-5)
            results.append(dict(dataset=dataset, seed=seed, checked_test_rows=chosen,
                max_absolute_error=float(np.abs(logits-pred["logits"][chosen]).max()),
                checkpoint_sha256=sha256(folder/"adapter.pt"), physical_gpu=result["gpu"]["physical_index"]))
    save(args.run_dir/"checkpoint_validation.json", dict(status="passed", checks=results))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
