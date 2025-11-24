import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__) + "/..")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import joblib
import numpy as np
import torch
import tyro
import viser
import viser.transforms as vtf
import time
import cv2
from scipy.spatial.transform import Rotation as R

from prompt_hmr.vid_evaluator import Evaluator
from prompt_hmr.vis.viser import get_color
from prompt_hmr.utils.vid_eval_utils import align_pcl
from pipeline.pipeline import Pipeline


def _get_sample(evaluator, vid_name):
    for idx in range(len(evaluator.dataset)):
        sample = evaluator.dataset[idx]
        if sample["meta"]["vid"] == vid_name:
            return sample
    raise ValueError(f"Video {vid_name} not found in dataset.")


def _compute_alignments(gt_joints, pred_joints, chunk_length):
    """
    Precompute WA-MPJPE-style (global chunk) and W-MPJPE-style (first two frames) transforms.
    Returns two lists with per-chunk (s, R, t) tuples.
    """
    seq_len = gt_joints.shape[0]
    chunk_transforms_glob = []
    chunk_transforms_first = []
    chunk_ranges = []

    for start in range(0, seq_len, chunk_length):
        end = min(seq_len, start + chunk_length)
        chunk_ranges.append((start, end))

        gt_chunk = gt_joints[start:end].reshape(1, -1, 3)
        pred_chunk = pred_joints[start:end].reshape(1, -1, 3)
        s_glob, R_glob, t_glob = align_pcl(gt_chunk, pred_chunk, fixed_scale=False)
        chunk_transforms_glob.append(
            {
                "s": float(s_glob[0].cpu().item()),
                "R": R_glob[0].cpu().numpy(),
                "t": t_glob[0].cpu().numpy(),
            }
        )

        chunk_gt = gt_joints[start:end]
        chunk_pred = pred_joints[start:end]
        if len(chunk_gt) >= 2:
            s_first, R_first, t_first = align_pcl(
                chunk_gt[:2].reshape(1, -1, 3), chunk_pred[:2].reshape(1, -1, 3), fixed_scale=False
            )
            chunk_transforms_first.append(
                {
                    "s": float(s_first[0].cpu().item()),
                    "R": R_first[0].cpu().numpy(),
                    "t": t_first[0].cpu().numpy(),
                }
            )
        else:
            chunk_transforms_first.append(chunk_transforms_glob[-1])

    return chunk_ranges, chunk_transforms_glob, chunk_transforms_first


def _apply_transform(points, transform):
    s_val, R_mat, t_vec = transform
    return s_val * torch.einsum("ij,tnj->tni", R_mat, points) + t_vec


def _collect_vertices(evaluator, sample, predictions, chunk_length=100):
    seq_len = int(sample["length"])
    eval_mask = sample["mask"].clone().to(bool)

    gt = evaluator._build_ground_truth(sample)
    pred = evaluator._build_predictions(predictions, seq_len, eval_mask)
    if pred is None:
        raise RuntimeError("Prediction track not found in cached results.")

    # valid frames = both GT and pred valid
    combined_mask = (eval_mask & pred["mask"]).cpu().numpy()

    # original world coordinates
    gt_verts = gt["world_verts"]
    pred_verts = pred["world_verts"]
    gt_joints = gt["world_j3d"]
    pred_joints = pred["world_j3d"]

    chunk_ranges, chunk_transforms_glob, chunk_transforms_first = _compute_alignments(
        gt_joints, pred_joints, chunk_length
    )

    verts_glob = pred_verts.clone()
    joints_glob = pred_joints.clone()
    verts_first = pred_verts.clone()
    joints_first = pred_joints.clone()

    for (start, end), trans_glob, trans_first in zip(
        chunk_ranges, chunk_transforms_glob, chunk_transforms_first
    ):
        def to_tensors(transform):
            s = torch.tensor(transform["s"], dtype=pred_verts.dtype, device=pred_verts.device)
            R = torch.tensor(transform["R"], dtype=pred_verts.dtype, device=pred_verts.device)
            t = torch.tensor(transform["t"], dtype=pred_verts.dtype, device=pred_verts.device)
            return s, R, t

        s_glob, R_glob, t_glob = to_tensors(trans_glob)
        s_first, R_first, t_first = to_tensors(trans_first)

        verts_glob[start:end] = s_glob * torch.einsum(
            "ij,tnj->tni", R_glob, pred_verts[start:end]
        ) + t_glob
        joints_glob[start:end] = s_glob * torch.einsum(
            "ij,tnj->tni", R_glob, pred_joints[start:end]
        ) + t_glob

        verts_first[start:end] = s_first * torch.einsum(
            "ij,tnj->tni", R_first, pred_verts[start:end]
        ) + t_first
        joints_first[start:end] = s_first * torch.einsum(
            "ij,tnj->tni", R_first, pred_joints[start:end]
        ) + t_first

    return {
        "gt_verts": gt_verts.numpy(),
        "gt_joints": gt_joints.numpy(),
        "pred_glob_verts": verts_glob.numpy(),
        "pred_glob_joints": joints_glob.numpy(),
        "pred_first_verts": verts_first.numpy(),
        "pred_first_joints": joints_first.numpy(),
        "mask": combined_mask,
    }


def _load_camera_data(sample, predictions, vis_data, img_maxsize=320):
    meta_render = sample.get("meta_render", None)
    if meta_render is None or "video_path" not in meta_render:
        return None

    video_path = meta_render["video_path"]
    if not os.path.exists(video_path):
        return None

    T_w2c = sample.get("T_w2c", None)
    if T_w2c is None:
        return None
    T_w2c = T_w2c.cpu().numpy()
    R_w2c = T_w2c[:, :3, :3]
    t_w2c = T_w2c[:, :3, 3]

    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        success, frame = cap.read()
    cap.release()

    if len(frames) == 0:
        return None

    height, width = frames[0].shape[:2]
    aspect = width / height
    focal = sample["K_fullimg"][0, 0, 0].item() if "K_fullimg" in sample else 800.0
    vfov = 2 * np.arctan((height / 2) / focal)

    camera_data = []
    num_frames = min(len(R_w2c), len(frames))
    for i in range(num_frames):
        image = frames[i]
        if max(image.shape[:2]) > img_maxsize:
            scale = img_maxsize / max(image.shape[:2])
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        Rcw = R_w2c[i].T
        Twc = -Rcw @ t_w2c[i]
        quat_xyzw = R.from_matrix(Rcw).as_quat()
        quat_wxyz = np.concatenate([quat_xyzw[3:], quat_xyzw[:3]])

        camera_data.append(
            {
                "quat": quat_wxyz,
                "trans": Twc,
                "image": image,
                "fov": float(vfov),
                "aspect": float(aspect),
            }
        )

    return camera_data



def _vis_compare(
    gt_verts,
    pred_verts_dict,
    gt_joints,
    pred_joints_dict,
    mask,
    faces,
    fps=30,
    num_people=1,
    cameras=None,
):
    faces = faces.cpu().numpy() if hasattr(faces, "cpu") else np.asarray(faces)
    num_frames = gt_verts.shape[0]

    try:
        server.scene.reset()
    except NameError:
        server = viser.ViserServer()

    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+y")

    server.gui.add_text("pred_info", f"Predicted people: {num_people}")
    gui_show_gt = server.gui.add_checkbox("Show GT Mesh", True)
    gui_show_pred = server.gui.add_checkbox("Show Pred Mesh", True)
    gui_show_gt_joints = server.gui.add_checkbox("Show GT Joints", False)
    gui_show_pred_joints = server.gui.add_checkbox("Show Pred Joints", False)
    gui_pin_gt = server.gui.add_checkbox("Pin View to GT Root", False)
    gui_align_mode = server.gui.add_dropdown(
        "Alignment Mode",
        options=("WA-MPJPE (global chunk)", "W-MPJPE (first 2 frames)"),
        initial_value="WA-MPJPE (global chunk)",
    )

    gui_timestep = server.gui.add_slider(
        "Frame",
        min=0,
        max=num_frames - 1,
        step=1,
        initial_value=0,
        disabled=False,
    )
    gui_playing = server.gui.add_checkbox("Playing", True)
    gui_framerate = server.gui.add_slider("FPS", min=1, max=60, step=0.1, initial_value=fps)
    has_camera = cameras is not None and len(cameras) > 0
    gui_show_camera = server.gui.add_checkbox("Show Camera", has_camera)

    server.scene.add_frame(
        "/frames",
        wxyz=vtf.SO3.exp(np.array([0.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )

    gt_handle = server.scene.add_mesh_simple(
        name="/frames/gt",
        vertices=gt_verts[0],
        faces=faces,
        flat_shading=False,
        wireframe=False,
        color=get_color(0),
    )
    pred_handle = server.scene.add_mesh_simple(
        name="/frames/pred",
        vertices=pred_verts_dict["WA-MPJPE"][0],
        faces=faces,
        flat_shading=False,
        wireframe=False,
        color=get_color(1),
    )
    gt_joints_handle = server.scene.add_point_cloud(
        name="/frames/gt_joints",
        points=gt_joints[0],
        colors=np.array([0.0, 0.5, 1.0]),
        point_size=0.01,
    )
    pred_joints_handle = server.scene.add_point_cloud(
        name="/frames/pred_joints",
        points=pred_joints_dict["WA-MPJPE"][0],
        colors=np.array([1.0, 0.5, 0.0]),
        point_size=0.01,
    )

    camera_handle = None
    if has_camera:
        cam = cameras[0]
        camera_handle = server.scene.add_camera_frustum(
            "/frames/camera",
            fov=cam["fov"],
            aspect=cam["aspect"],
            line_width=1.5,
            color=(255, 127, 14),
            scale=0.4,
            wxyz=cam["quat"],
            position=cam["trans"],
            image=cam["image"],
        )

    def apply_frame(frame_idx):
        valid = mask[frame_idx]
        mode_key = (
            "WA-MPJPE" if gui_align_mode.value == "WA-MPJPE (global chunk)" else "W-MPJPE"
        )
        pred_verts = pred_verts_dict[mode_key]
        pred_joints = pred_joints_dict[mode_key]

        offset = np.zeros(3)
        if gui_pin_gt.value and valid:
            offset = gt_joints[frame_idx, 0]

        if valid and gui_show_gt.value:
            gt_handle.vertices = gt_verts[frame_idx] - offset
            gt_handle.visible = True
        else:
            gt_handle.visible = False
        if valid and gui_show_pred.value:
            pred_handle.vertices = pred_verts[frame_idx] - offset
            pred_handle.visible = True
        else:
            pred_handle.visible = False
        if valid and gui_show_gt_joints.value:
            gt_joints_handle.points = gt_joints[frame_idx] - offset
            gt_joints_handle.visible = True
        else:
            gt_joints_handle.visible = False
        if valid and gui_show_pred_joints.value:
            pred_joints_handle.points = pred_joints[frame_idx] - offset
            pred_joints_handle.visible = True
        else:
            pred_joints_handle.visible = False

        if camera_handle is not None and cameras is not None:
            if frame_idx < len(cameras):
                cam = cameras[frame_idx]
                camera_handle.wxyz = cam["quat"]
                camera_handle.position = cam["trans"] - offset
                camera_handle.image = cam["image"]
                camera_handle.visible = gui_show_camera.value and valid
            else:
                camera_handle.visible = False

    apply_frame(0)

    @gui_timestep.on_update
    def _(_) -> None:
        apply_frame(gui_timestep.value)

    @gui_show_gt.on_update
    def _(_) -> None:
        apply_frame(gui_timestep.value)

    @gui_show_pred.on_update
    def _(_) -> None:
        apply_frame(gui_timestep.value)
    @gui_show_gt_joints.on_update
    def _(_) -> None:
        apply_frame(gui_timestep.value)
    @gui_show_pred_joints.on_update
    def _(_) -> None:
        apply_frame(gui_timestep.value)
    @gui_pin_gt.on_update
    def _(_) -> None:
        apply_frame(gui_timestep.value)
    if camera_handle is not None:
        @gui_show_camera.on_update
        def _(_) -> None:
            apply_frame(gui_timestep.value)

    @gui_align_mode.on_update
    def _(_) -> None:
        apply_frame(gui_timestep.value)

    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)


def vis_compare_emdb(
    vid_name: str = "P0_09_outdoor_walk",
    dataset: str = "EMDB",
    split: int = 2,
    results_root: str = "results/emdb_eval",
    device: str = "cuda",
    fps: int = 30,
):
    """
    Visualize GT vs predicted world-coordinate meshes for an EMDB sequence.
    """
    evaluator = Evaluator(
        dataset=dataset,
        split=split,
        results_root=results_root,
        device=device,
    )
    sample = _get_sample(evaluator, vid_name)

    cache_path = Path(results_root) / vid_name / "results.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cached prediction at {cache_path}. Run scripts/eval_emdb.py first.")
    predictions = joblib.load(cache_path)

    vis_data = _collect_vertices(evaluator, sample, predictions)
    gt_verts = vis_data["gt_verts"]
    gt_joints = vis_data["gt_joints"]
    mask = vis_data["mask"]
    pred_verts_dict = {
        "WA-MPJPE": vis_data["pred_glob_verts"],
        "W-MPJPE": vis_data["pred_first_verts"],
    }
    pred_joints_dict = {
        "WA-MPJPE": vis_data["pred_glob_joints"],
        "W-MPJPE": vis_data["pred_first_joints"],
    }

    faces = evaluator.smpl_models["neutral"].faces

    num_people = len(predictions.get("people", {}))

    camera_data = _load_camera_data(sample, predictions, vis_data)

    _vis_compare(
        gt_verts,
        pred_verts_dict,
        gt_joints,
        pred_joints_dict,
        mask,
        faces,
        fps=fps,
        num_people=num_people,
        cameras=camera_data,
    )
    faces = evaluator.smpl_models["neutral"].faces

    num_people = len(predictions.get("people", {}))

    _vis_compare(gt_verts, pred_verts, gt_joints, pred_joints, mask, faces, fps=fps, num_people=num_people)


if __name__ == "__main__":
    tyro.cli(vis_compare_emdb)
