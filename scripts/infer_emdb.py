import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/..")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import tyro

from prompt_hmr.vid_evaluator import Evaluator


def infer_emdb_sequences(
    dataset: str = "EMDB",
    split: int = 2,
    results_root: str = "results/emdb_eval",
    static_camera: bool = False,
    device: str = "cuda",
    force_recompute: bool = False,
):
    """
    Run the PromptHMR video pipeline on EMDB sequences and store only the inference results.

    Args:
        dataset: Currently only 'EMDB' is supported.
        split: EMDB split id (1 => EMDB-1, 2 => EMDB-2, 3 => custom subset).
        results_root: Directory where pipeline outputs/results are stored.
        static_camera: Force the pipeline to assume a static camera.
        device: Torch device for body-model forward passes.
        force_recompute: Re-run the pipeline even if cached results exist.
    """
    evaluator = Evaluator(
        dataset=dataset,
        split=split,
        results_root=results_root,
        static_camera=static_camera,
        device=device,
    )

    iterator = range(len(evaluator.dataset))
    for idx in iterator:
        sample = evaluator.dataset[idx]
        vid = sample["meta"]["vid"]
        print(f"[Infer] Processing {vid} ({idx + 1}/{len(evaluator.dataset)})")
        try:
            evaluator._prepare_predictions(sample, None, force_recompute=force_recompute)
        except FileNotFoundError as err:
            print(f"[WARN] Failed on {vid}: {err}")
            continue

    print("Inference finished. Cached results are stored under:", results_root)


if __name__ == "__main__":
    tyro.cli(infer_emdb_sequences)
