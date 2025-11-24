import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__) + "/..")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import joblib
import numpy as np
import tyro
from tqdm import tqdm

from prompt_hmr.vid_evaluator import Evaluator


def evaluate_cached_results(
    dataset: str = "EMDB",
    split: int = 2,
    results_root: str = "results/emdb_eval",
    device: str = "cuda",
):
    """
    Compute metrics for EMDB sequences using cached inference results.

    Args:
        dataset: Currently only 'EMDB' is supported.
        split: EMDB split id (1 => EMDB-1, 2 => EMDB-2, 3 => custom subset).
        results_root: Directory containing per-sequence results.pkl files.
        device: Torch device used for SMPL forward passes during evaluation.
    """
    evaluator = Evaluator(
        dataset=dataset,
        split=split,
        results_root=results_root,
        device=device,
    )

    accumulator = defaultdict(list)
    per_sequence = {}

    for idx in tqdm(range(len(evaluator.dataset)), desc=f"Evaluating cached {dataset}-split{split}"):
        sample = evaluator.dataset[idx]
        vid = sample["meta"]["vid"]
        cache_path = Path(results_root) / vid / "results.pkl"
        if not cache_path.exists():
            print(f"[WARN] Missing cache for {vid}, skipping.")
            continue

        predictions = joblib.load(cache_path)
        metrics = evaluator._evaluate_sequence(sample, predictions)
        if metrics is None:
            print(f"[WARN] No valid frames for {vid}, skipping.")
            continue

        per_sequence[vid] = {k: float(np.mean(v)) for k, v in metrics.items()}
        for key, value in metrics.items():
            accumulator[key].append(value)

    if not accumulator:
        print("No metrics were computed. Please ensure results exist by running scripts/eval_emdb.py first.")
        return

    aggregated = {}
    for key, value_list in accumulator.items():
        concatenated = np.concatenate(value_list)
        aggregated[key] = concatenated.mean()

    print("=== Aggregate Metrics ===")
    for key, value in aggregated.items():
        print(f"{key}: {value:.4f}")

    print("\n=== Per-sequence Metrics ===")
    for vid, metrics in per_sequence.items():
        items = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"{vid}: {items}")


if __name__ == "__main__":
    tyro.cli(evaluate_cached_results)
