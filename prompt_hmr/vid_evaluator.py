from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import torch
from tqdm import tqdm

from data_config import SMPL_PATH, SMPLX_PATH
from prompt_hmr.smpl_family import SMPL, SMPLX
from prompt_hmr.utils.vid_eval_utils import compute_camcoord_metrics, compute_global_metrics
from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix

from pipeline.pipeline import Pipeline
from pipeline.gvhmr.hmr4d.dataset.emdb.emdb_motion_test import EmdbSmplFullSeqDataset
from pipeline.gvhmr.hmr4d.utils.geo_transform import apply_T_on_points


REPO_ROOT = Path(__file__).resolve().parents[3]
BODY_MODEL_DIR = REPO_ROOT / "pipeline" / "gvhmr" / "hmr4d" / "utils" / "body_model"
SMPL_J_REG_PATH = BODY_MODEL_DIR / "smpl_neutral_J_regressor.pt"
SMPLX2SMPL_PATH = BODY_MODEL_DIR / "smplx2smpl_sparse.pt"


class Evaluator:
    """
    Video evaluator that (optionally) runs the world-coordinate PromptHMR
    pipeline on EMDB videos and reports camera-space (MPJPE/PA-MPJPE/PVE) and
    world-space (W/WA-MPJPE100 + RTE) metrics.
    """

    def __init__(
        self,
        dataset: str = "EMDB",
        split: int = 2,
        results_root: str = "results",
        static_camera: bool = False,
        device: str = "cuda",
        chunk_length: int = 100,
    ):
        if dataset != "EMDB":
            raise ValueError("Video evaluation currently supports the EMDB dataset only.")

        self.ds_name = dataset
        self.split = split
        self.device = torch.device(device)
        self.chunk_length = chunk_length
        self.results_root = Path(results_root)
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.static_camera = static_camera
        self.fps = 30

        # EMDB split-1 -> camera metrics, split-2 -> world metrics.
        self.dataset = EmdbSmplFullSeqDataset(split=split)

        # SMPL/SMPLX models + conversion utilities
        self.smpl_models = {
            "neutral": SMPL(SMPL_PATH, gender="neutral").to(self.device),
            "male": SMPL(SMPL_PATH, gender="male").to(self.device),
            "female": SMPL(SMPL_PATH, gender="female").to(self.device),
        }
        self.smplx = SMPLX(SMPLX_PATH, gender="neutral").to(self.device)
        self.J_regressor = torch.load(SMPL_J_REG_PATH, map_location=self.device).to(self.device)
        self.smplx2smpl = torch.load(SMPLX2SMPL_PATH, map_location=self.device)
        if not torch.is_tensor(self.smplx2smpl):
            raise RuntimeError(f"Unexpected smplx2smpl type: {type(self.smplx2smpl)}")
        self.smplx2smpl = self.smplx2smpl.to(self.device)
        self._smplx2smpl_is_sparse = self.smplx2smpl.layout == torch.sparse_coo
        self.num_smpl_verts = self.smpl_models["neutral"].v_template.shape[0]
        self.pelvis_idx = [1, 2]

        self._default_pipeline: Optional[Pipeline] = None

    def __call__(
        self,
        model: Optional[Pipeline] = None,
        box_prompt: bool = True,
        mask_prompt: bool = False,
        interaction: bool = False,
        force_recompute: bool = False,
        pipeline_runner: Optional[Pipeline] = None,
    ):
        """
        For API compatibility with prompt_hmr.evaluator.Evaluator we keep the same
        signature, but the arguments (box_prompt/mask_prompt/interaction) are
        ignored. Pass a pre-instantiated Pipeline through `model` or
        `pipeline_runner`; if None a default Pipeline will be created.
        """
        del box_prompt, mask_prompt, interaction  # unused but kept for API parity
        runner = pipeline_runner if pipeline_runner is not None else model
        if runner is not None and not isinstance(runner, Pipeline):
            raise ValueError("Video evaluator expects a Pipeline instance or None.")
        return self.eval_dataset(runner, force_recompute=force_recompute)

    # --------------------------------------------------------------------- #
    def eval_dataset(self, pipeline_runner: Optional[Pipeline], force_recompute: bool = False):
        accumulator: Dict[str, list] = defaultdict(list)
        per_sequence = {}

        iterator = tqdm(range(len(self.dataset)), desc=f"Evaluating {self.ds_name}-split{self.split}")
        for idx in iterator:
            sample = self.dataset[idx]
            vid = sample["meta"]["vid"]
            try:
                predictions = self._prepare_predictions(sample, pipeline_runner, force_recompute)
            except FileNotFoundError as err:
                print(f"[WARN] Skip {vid}: {err}")
                continue

            metrics = self._evaluate_sequence(sample, predictions)
            if metrics is None:
                print(f"[WARN] No valid frames for {vid}, skipping.")
                continue

            per_sequence[vid] = {k: float(np.mean(v)) for k, v in metrics.items()}
            for k, v in metrics.items():
                accumulator[k].append(v)

        results = {}
        for k, v in accumulator.items():
            if not v:
                continue
            values = np.concatenate(v)
            results[k] = values.mean()

        accumulator["per_sequence"] = per_sequence
        return results, accumulator

    # ------------------------------------------------------------------ I/O
    def _prepare_predictions(self, sample, pipeline_runner, force_recompute):
        video_path = sample["meta_render"]["video_path"]
        video_name = Path(video_path).stem
        output_folder = self.results_root / video_name
        output_folder.mkdir(parents=True, exist_ok=True)
        cache_file = output_folder / "results.pkl"

        if cache_file.exists() and not force_recompute:
            return joblib.load(cache_file)

        pipeline = self._resolve_pipeline(pipeline_runner)
        results = pipeline(
            video_path,
            str(output_folder),
            static_cam=self.static_camera,
            save_only_essential=True,
        )

        if results is None:
            if not cache_file.exists():
                raise FileNotFoundError(f"Pipeline did not produce {cache_file}.")
            return joblib.load(cache_file)
        return results

    def _resolve_pipeline(self, pipeline_runner: Optional[Pipeline]) -> Pipeline:
        if pipeline_runner is not None:
            return pipeline_runner
        if self._default_pipeline is None:
            self._default_pipeline = Pipeline(static_cam=self.static_camera)
        return self._default_pipeline

    # ------------------------------------------------------------ Evaluation
    def _evaluate_sequence(self, sample, predictions):
        seq_len = int(sample["length"])
        eval_mask = sample["mask"].clone().to(torch.bool)
        gt = self._build_ground_truth(sample)
        pred = self._build_predictions(predictions, seq_len, eval_mask)
        if pred is None:
            return None

        combined_mask = (eval_mask & pred["mask"].cpu()).cpu()
        if combined_mask.sum().item() < 2:
            return None

        cam_batch = {
            "pred_j3d": pred["cam_j3d"],
            "target_j3d": gt["cam_j3d"],
            "pred_verts": pred["cam_verts"],
            "target_verts": gt["cam_verts"],
        }
        cam_metrics = compute_camcoord_metrics(
            cam_batch,
            pelvis_idxs=self.pelvis_idx,
            fps=self.fps,
            mask=combined_mask,
        )

        world_batch = {
            "pred_j3d_glob": pred["world_j3d"],
            "target_j3d_glob": gt["world_j3d"],
            "pred_verts_glob": pred["world_verts"],
            "target_verts_glob": gt["world_verts"],
        }
        world_metrics = compute_global_metrics(
            world_batch,
            mask=combined_mask,
            chunk_length=self.chunk_length,
        )

        metrics = {}
        metrics.update(cam_metrics)
        metrics.update(world_metrics)
        return metrics

    # ------------------------------------------------------------ Groundtruth
    def _build_ground_truth(self, sample):
        gender = sample["gender"]
        smpl_model = self.smpl_models.get(gender, self.smpl_models["neutral"])
        params = {k: v.to(self.device).float() for k, v in sample["smpl_params"].items()}

        global_orient = self._to_rotmat(params["global_orient"], num_joints=1)
        body_pose = self._to_rotmat(params["body_pose"], num_joints=23)

        smpl_out = smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=params["betas"],
            transl=params["transl"],
        )
        verts_world = smpl_out.vertices  # (F, 6890, 3)
        joints_world = torch.einsum("jv,bvi->bji", self.J_regressor, verts_world)

        T_w2c = sample["T_w2c"].to(self.device).float()
        verts_cam = apply_T_on_points(verts_world, T_w2c)
        joints_cam = apply_T_on_points(joints_world, T_w2c)

        return {
            "world_verts": verts_world.cpu(),
            "world_j3d": joints_world.cpu(),
            "cam_verts": verts_cam.cpu(),
            "cam_j3d": joints_cam.cpu(),
        }

    # ------------------------------------------------------------- Prediction
    def _build_predictions(self, predictions, seq_len, eval_mask):
        people = predictions.get("people", {})
        if not people:
            return None

        mask_np = eval_mask.cpu().numpy()
        target_track = self._select_track(people, seq_len, mask_np) # notes: select most regular target
        if target_track is None:
            return None

        frames = np.asarray(target_track["frames"], dtype=np.int64)
        valid = (frames >= 0) & (frames < seq_len)
        if not np.any(valid):
            return None
        frames = frames[valid]

        order = np.argsort(frames)
        frames = frames[order]

        # notes: this part produces SMPLX params under world coordinate and camera coordinate
        cam_params = target_track["smplx_cam"]
        world_params = target_track["smplx_world"]

        pose_cam = torch.from_numpy(cam_params["pose"]).float()[valid][order].to(self.device)
        shape_cam = torch.from_numpy(cam_params["shape"]).float()[valid][order].to(self.device)
        trans_cam = torch.from_numpy(cam_params["trans"]).float()[valid][order].to(self.device)

        pose_world = torch.from_numpy(world_params["pose"]).float()[valid][order].to(self.device)
        shape_world = torch.from_numpy(world_params["shape"]).float()[valid][order].to(self.device)
        trans_world = torch.from_numpy(world_params["trans"]).float()[valid][order].to(self.device)

        cam_out = self._forward_smplx(pose_cam, shape_cam, trans_cam)
        world_out = self._forward_smplx(pose_world, shape_world, trans_world)
        cam_verts = self._convert_to_smpl(cam_out.vertices)
        world_verts = self._convert_to_smpl(world_out.vertices)

        # mesh vertices to joints
        cam_j3d = torch.einsum("jv,bvi->bji", self.J_regressor, cam_verts)
        world_j3d = torch.einsum("jv,bvi->bji", self.J_regressor, world_verts)

        pred_mask = torch.zeros(seq_len, dtype=torch.bool, device=self.device)
        pred_mask[frames] = True

        full_cam_verts = torch.zeros(seq_len, self.num_smpl_verts, 3, device=self.device)
        full_cam_j3d = torch.zeros(seq_len, self.J_regressor.shape[0], 3, device=self.device)
        full_world_verts = torch.zeros_like(full_cam_verts)
        full_world_j3d = torch.zeros_like(full_cam_j3d)

        full_cam_verts[frames] = cam_verts
        full_cam_j3d[frames] = cam_j3d
        full_world_verts[frames] = world_verts
        full_world_j3d[frames] = world_j3d

        return {
            "cam_verts": full_cam_verts.cpu(),
            "cam_j3d": full_cam_j3d.cpu(),
            "world_verts": full_world_verts.cpu(),
            "world_j3d": full_world_j3d.cpu(),
            "mask": pred_mask.cpu(),
        }

    def _select_track(self, people: Dict, seq_len: int, eval_mask: np.ndarray):
        best_track = None
        best_score = -1
        for track in people.values():
            frames = np.asarray(track["frames"], dtype=np.int64)
            valid = (frames >= 0) & (frames < seq_len)
            if not np.any(valid):
                continue
            frames = frames[valid]
            score = eval_mask[frames].sum()
            if score > best_score:
                best_score = score
                best_track = track
        return best_track

    # ----------------------------------------------------------- SMPLX utils
    def _forward_smplx(self, pose_aa, betas, transl):
        num_frames = pose_aa.shape[0]
        components = self._split_smplx_pose(pose_aa)

        global_orient = self._aa_to_rotmat_chunk(components["global_orient"], 1)
        body_pose = self._aa_to_rotmat_chunk(components["body_pose"], 21)
        jaw_pose = self._aa_to_rotmat_chunk(components["jaw_pose"], 1)
        leye_pose = self._aa_to_rotmat_chunk(components["leye_pose"], 1)
        reye_pose = self._aa_to_rotmat_chunk(components["reye_pose"], 1)
        left_hand_pose = self._aa_to_rotmat_chunk(components["left_hand_pose"], 15)
        right_hand_pose = self._aa_to_rotmat_chunk(components["right_hand_pose"], 15)

        zeros_expr = torch.zeros(num_frames, 10, device=self.device)
        return self.smplx(
            global_orient=global_orient,
            body_pose=body_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            betas=betas.to(self.device),
            transl=transl.to(self.device),
            expression=zeros_expr,
        )

    def _split_smplx_pose(self, pose_aa):
        """
        Split flattened axis-angle pose to the SMPLX expected components.
        Missing parts (hands/face) are filled with zeros automatically.
        """
        num_frames, total_dims = pose_aa.shape
        device = pose_aa.device

        def slice_chunk(start, size):
            end = min(start + size, total_dims)
            chunk = pose_aa[:, start:end]
            if chunk.shape[1] < size:
                padding = torch.zeros(num_frames, size - chunk.shape[1], device=device)
                chunk = torch.cat([chunk, padding], dim=1)
            return chunk

        components = {}
        components["global_orient"] = slice_chunk(0, 3)
        components["body_pose"] = slice_chunk(3, 63)
        idx = 66
        components["jaw_pose"] = slice_chunk(idx, 3)
        idx += 3
        components["leye_pose"] = slice_chunk(idx, 3)
        idx += 3
        components["reye_pose"] = slice_chunk(idx, 3)
        idx += 3
        components["left_hand_pose"] = slice_chunk(idx, 45)
        idx += 45
        components["right_hand_pose"] = slice_chunk(idx, 45)
        return components

    def _convert_to_smpl(self, smplx_vertices):
        """
        Multiply SMPLX vertices with the sparse conversion matrix to recover SMPL vertices.
        """
        b, v, _ = smplx_vertices.shape
        verts = []
        for axis in range(3):
            coords = smplx_vertices[:, :, axis]  # (B, V)
            mat_in = coords.T.contiguous()
            if self._smplx2smpl_is_sparse:
                converted = torch.sparse.mm(self.smplx2smpl, mat_in).T
            else:
                converted = torch.matmul(self.smplx2smpl, mat_in).T
            verts.append(converted.unsqueeze(-1))
        return torch.cat(verts, dim=-1)

    def _to_rotmat(self, data, num_joints):
        """
        Convert axis-angle representations to rotation matrices if needed.
        depends on whether:
        1. input tensor is already flattened (batch_size, num_joints, 3)
        2. input tensor is not flattened (batch_size, num_joints*3)
        """
        if data.ndim == 2 and data.shape[-1] == num_joints * 3:
            reshaped = data.view(-1, num_joints, 3)
            rotmats = axis_angle_to_matrix(reshaped.reshape(-1, 3)).view(-1, num_joints, 3, 3)
            return rotmats
        if data.ndim == 3 and data.shape[-1] == 3 and data.shape[-2] == num_joints:
            rotmats = axis_angle_to_matrix(data.reshape(-1, 3)).view(-1, num_joints, 3, 3)
            return rotmats
        return data.view(-1, num_joints, 3, 3)

    def _aa_to_rotmat_chunk(self, chunk, num_joints):
        """
        Convert flattened axis-angle chunks to rotation matrices with shape (B, num_joints, 3, 3).
        """
        if chunk is None:
            return None
        shape = chunk.shape
        if shape[-1] == num_joints * 3:
            chunk = chunk.view(-1, num_joints, 3)
        elif chunk.ndim == 2 and shape[-1] == 3 and num_joints == 1:
            chunk = chunk.view(-1, num_joints, 3)
        elif chunk.ndim == 3 and shape[-1] == 3:
            chunk = chunk.view(-1, num_joints, 3)
        return axis_angle_to_matrix(chunk.reshape(-1, 3)).view(-1, num_joints, 3, 3)
