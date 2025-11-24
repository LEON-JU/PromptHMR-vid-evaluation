下面是 **根据你提供的实际字段** 重新整理后的 **结构化、简洁版 schema**，适合作为 **接口文档 / 给编程 LLM 的上下文**。

---

# 📦 `results.pkl` Schema (PromptHMR)

`results.pkl` 是一个 Python `dict`，字段如下：

---

## 🔹 **1. camera**  — per-frame naive camera parameters

```
camera:
    pred_cam_R: float32 array (N, 3, 3)   # rotation matrices
    pred_cam_T: float32 array (N, 3)      # translations
    img_focal:  float32                    # focal length
    img_center: float32 array (2,)         # principal point
```

---

## 🔹 **2. people** — tracked human instances

```
people:
    <track_id>: dict {
        track_id: int
        bbox_format: str
        frames: int64 array (N,)           # frame indices
        bboxes: float64 array (N,4)        # bounding boxes
        detected: bool array (N,)          # detection mask

        smplx_pose:    float32 (N,22,3,3)  # joint rotations (camera coord)
        smplx_transl:  float32 (N,3)       # translations (camera coord)
        smplx_betas:   float32 (N,10)      # shape parameters

        smplx_cam:
            rotmat: float32 (N,55,3,3)
            pose:   float32 (N,75)
            shape:  float32 (N,10)
            trans:  float32 (N,3)
            contact: bool (N,6)
            static_conf_logits: float16 (N,6)

        smplx_world:
            pose:  float32 (N,165)         # world-coord SMPL-X pose
            shape: float32 (N,10)
            trans: float32 (N,3)
    }
```

> 简述：`people` 中存所有跟踪到的人，包含检测结果、SMPL-X 参数（相机坐标 + 世界坐标）。

---

## 🔹 **3. timings**

```
timings: dict   # processing time logs (optional per stage)
```

---

## 🔹 **4. flags**

```
has_tracks: bool
has_hps_cam: bool
has_hps_world: bool
has_slam: bool
has_hands: bool
has_2d_kpts: bool
has_post_opt: bool
```

> 简述：指示哪些步骤已执行。

---

## 🔹 **5. spec_calib** — camera calibration results

```
spec_calib:
    0..30: dict {
        vfov: float
        f_pix: float32
        pitch: float
        roll: float
    }
    first_frame: dict {...}
    avg_focal_length: float32
    min_focal_length: float32
    max_focal_length: float32
    std_focal_length: float32
    median_focal_length: float32
```

> 简述：逐帧标定 + 全局焦距统计。

---

## 🔹 **6. contact_joint_ids**

```
contact_joint_ids: list[int]
```

---

## 🔹 **7. camera_world** — final world-coordinate camera trajectory

```
camera_world:
    pred_cam_R: float32 (N,3,3)
    pred_cam_T: float32 (N,3)
    Rwc: float32 (N,3,3)     # world ← camera rotation
    Twc: float32 (N,3)
    Rcw: float32 (N,3,3)     # camera ← world rotation
    Tcw: float32 (N,3)
    img_focal: float32 or array
    img_center: float32 (2,)
    viz_scale: float
    viz_center: [float, int, float]
```

> 简述：SLAM 后的相机位姿（用于世界坐标 mesh 渲染）。

