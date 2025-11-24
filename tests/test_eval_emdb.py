import os
from pathlib import Path

import joblib
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')
from prompt_hmr.vid_evaluator import Evaluator
from prompt_hmr.utils.vid_eval_utils import compute_camcoord_metrics, compute_global_metrics


def _get_sample(evaluator, vid_name):
    """
    Helper to fetch the dataset sample for a specific EMDB video id.
    """
    for idx in range(len(evaluator.dataset)):
        sample = evaluator.dataset[idx]
        if sample["meta"]["vid"] == vid_name:
            return sample
    raise ValueError(f"Video {vid_name} not found in dataset.")


def test_eval_emdb_with_cached_results():
    """
    Ensure that the evaluator can consume an existing results.pkl without re-running inference.
    """
    vid_name = "P0_09_outdoor_walk"
    cache_path = Path("results") / "emdb_eval" / vid_name / "results.pkl"
    assert cache_path.exists(), f"Missing cached results at {cache_path}"

    evaluator = Evaluator(split=2, results_root="results/emdb_eval")
    sample = _get_sample(evaluator, vid_name)
    predictions = joblib.load(cache_path)

    metrics = evaluator._evaluate_sequence(sample, predictions)
    assert metrics is not None, "Metrics should not be None when predictions exist."

    # Re-compute metrics manually to ensure evaluator matches utility functions
    seq_len = int(sample["length"])
    eval_mask = sample["mask"].clone().to(torch.bool)
    gt = evaluator._build_ground_truth(sample)
    pred = evaluator._build_predictions(predictions, seq_len, eval_mask)
    assert pred is not None, "Predictions should contain valid tracks."

    combined_mask = (eval_mask & pred["mask"]).cpu()
    cam_batch = {
        "pred_j3d": pred["cam_j3d"],
        "target_j3d": gt["cam_j3d"],
        "pred_verts": pred["cam_verts"],
        "target_verts": gt["cam_verts"],
    }
    world_batch = {
        "pred_j3d_glob": pred["world_j3d"],
        "target_j3d_glob": gt["world_j3d"],
        "pred_verts_glob": pred["world_verts"],
        "target_verts_glob": gt["world_verts"],
    }

    cam_metrics = compute_camcoord_metrics(cam_batch, pelvis_idxs=evaluator.pelvis_idx, fps=evaluator.fps, mask=combined_mask)
    world_metrics = compute_global_metrics(world_batch, mask=combined_mask, chunk_length=evaluator.chunk_length)

    # Compare aggregate means with evaluator outputs
    for key in ["pa_mpjpe", "mpjpe", "pve"]:
        assert np.allclose(cam_metrics[key], metrics[key], atol=1e-6), f"{key} mismatch."
    for key in ["w_mpjpe100", "wa_mpjpe100", "rte"]:
        assert np.allclose(world_metrics[key], metrics[key], atol=1e-6), f"{key} mismatch."

    print(f"\nMetrics for {vid_name}:")
    for key in ["pa_mpjpe", "mpjpe", "pve", "w_mpjpe100", "wa_mpjpe100", "rte"]:
        value = metrics[key]
        if isinstance(value, np.ndarray):
            value = float(value.mean())
        print(f"  {key}: {value:.4f}")


def test_to_rotmat_axis_angle_conversion():
    """
    Validate that Evaluator._to_rotmat converts axis-angle tensors to rotation matrices.
    """
    evaluator = Evaluator(split=2, results_root="results/emdb_eval")
    axis_angle = torch.zeros(5, 23 * 3)
    axis_angle[:, 0] = 1.0

    rotmats = evaluator._to_rotmat(axis_angle, num_joints=23)
    assert rotmats.shape == (5, 23, 3, 3)
    # First rotation matrix should correspond to a unit rotation around x-axis.
    first_rot = rotmats[0, 0]
    assert torch.allclose(first_rot.det(), torch.tensor(1.0), atol=1e-4)


if __name__ == "__main__":
    test_eval_emdb_with_cached_results()
    test_to_rotmat_axis_angle_conversion()
