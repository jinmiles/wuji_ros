"""HaMeR per view, boxed from calibrated depth and then from its own 4-view triangulation.

Pass 1 runs HaMeR inside segment.py's depth + RGB boxes. Its 2D joints are
triangulated over the views (DLT; the worst view is dropped while its median
residual exceeds TRIANGULATION_OUTLIER_PX and three views remain). Pass 2 re-boxes
every view around the projection of those 3D joints, so all views frame the same
physical hand, including views where the depth box missed it, and runs HaMeR again;
its result is what is stored and triangulated once more.

`third_party/hamer` is used unmodified: sys.path injection, CACHE_DIR_HAMER set to
assets/hamer, and only `hamer.models.HAMER` and `hamer.configs.get_config` imported. The crop is
reimplemented (same maths as upstream ViTDetDataset, rescale 2.0 as in its demo)
because hamer.datasets needs scikit-image, which the wuji_ros env does not have.

Coordinates: HaMeR predicts in a virtual camera looking straight at the crop
centre. `global_orient_cam` is rotated back into the real camera by the rotation
taking the optical axis onto the ray through the box centre; keypoints_2d are in
full-image pixels.

    output/<session>/keypoints/<camera>/NNNNNN.npz
        keypoints_2d (21,2) px, global_orient_cam (3,3), hand_pose (15,3,3), betas (10,),
        box (4,), box_center (2,), box_size, box_source ("depth" | "triangulated")
    output/<session>/keypoints/triangulated/NNNNNN.npz
        joints_world (21,3), views (str), residual_px (V,) median per used view
"""

from __future__ import annotations

import shutil
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .bagio import Session
from .paths import HAMER_CHECKPOINT, HAMER_DATA_DIR, HAMER_ROOT, frame_name, get_logger, stage_dir
from .rig import rig_for
from . import segment

LOGGER = get_logger("wuji_ros.keypoints")
RESCALE = 2.0
PATCH = 256
BBOX_SHAPE = (192, 256)  # w, h aspect HaMeR's ViT crop expects
BATCH = 16
TRIANGULATION_OUTLIER_PX = 40.0
REBOX_MARGIN = 0.10  # around the projected joints; HaMeR's own RESCALE adds the context


def _load_model(device: str = "cuda"):
    sys.path.insert(0, str(HAMER_ROOT))
    import hamer.configs

    hamer.configs.CACHE_DIR_HAMER = str(HAMER_DATA_DIR)
    from hamer.configs import get_config
    from hamer.models import HAMER

    # Upstream load_hamer's config handling, but with HAMER's own init_renderer=False: its
    # pyrender renderer is for training visualisation only and needs an EGL context, which
    # fails on the 2026-10-22 analysis server (eglInitialize 12289).
    cfg = get_config(str(HAMER_CHECKPOINT.parent.parent / "model_config.yaml"), update_cachedir=True)
    cfg.defrost()
    if cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in cfg.MODEL:
        cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
    cfg.freeze()
    model = HAMER.load_from_checkpoint(str(HAMER_CHECKPOINT), strict=False, cfg=cfg, init_renderer=False)
    return model.to(device).eval(), cfg


def _box_size(box: np.ndarray) -> float:
    w, h = RESCALE * (box[2:] - box[:2])
    target = BBOX_SHAPE[0] / BBOX_SHAPE[1]
    w, h = (h * target, h) if w / h < target else (w, w / target)  # expand_to_aspect_ratio
    return float(max(w, h))


def crop(image: np.ndarray, box: np.ndarray, mean: np.ndarray, std: np.ndarray
         ) -> Tuple[np.ndarray, np.ndarray, float]:
    """(3,256,256) normalised RGB patch, box centre (2,), square box size px."""
    center = (box[:2] + box[2:]) / 2.0
    size = _box_size(box)
    scale = PATCH / size
    if size / PATCH / 2.0 > 1.1:  # upstream blurs before heavy downsampling
        sigma = (size / PATCH / 2.0 - 1) / 2
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    M = np.array([[scale, 0, PATCH / 2 - scale * center[0]], [0, scale, PATCH / 2 - scale * center[1]]])
    patch = cv2.warpAffine(image, M, (PATCH, PATCH), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rgb = patch[:, :, ::-1].astype(np.float32)
    return ((rgb - mean) / std).transpose(2, 0, 1), center, size


def crop_rotation(K: np.ndarray, center: np.ndarray) -> np.ndarray:
    """R_cam_from_crop: the rotation taking the optical axis onto the ray through `center`."""
    ray = np.linalg.solve(K, np.array([center[0], center[1], 1.0]))
    ray /= np.linalg.norm(ray)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, ray)
    s, c = np.linalg.norm(axis), float(z @ ray)
    if s < 1e-12:
        return np.eye(3)
    return cv2.Rodrigues(axis / s * np.arctan2(s, c))[0]


def triangulate(P: Dict[str, np.ndarray], kp: Dict[str, np.ndarray]
                ) -> Optional[Tuple[np.ndarray, List[str], np.ndarray]]:
    """DLT per joint over the views in `kp`, dropping the worst view while it is an outlier."""
    views = sorted(kp)
    while len(views) >= 2:
        X = np.zeros((21, 3))
        for j in range(21):
            A = []
            for c in views:
                u, v = kp[c][j]
                A += [u * P[c][2] - P[c][0], v * P[c][2] - P[c][1]]
            h = np.linalg.svd(np.asarray(A))[2][-1]
            X[j] = h[:3] / h[3]
        res = []
        for c in views:
            q = (P[c] @ np.c_[X, np.ones(21)].T).T
            res.append(np.median(np.linalg.norm(q[:, :2] / q[:, 2:] - kp[c], axis=1)))
        res = np.asarray(res)
        if res.max() <= TRIANGULATION_OUTLIER_PX or len(views) <= 3:
            return X, views, res
        views.pop(int(np.argmax(res)))
    return None


class _Hamer:
    def __init__(self) -> None:
        self.model, cfg = _load_model()
        self.mean = 255.0 * np.asarray(cfg.MODEL.IMAGE_MEAN, dtype=np.float32)
        self.std = 255.0 * np.asarray(cfg.MODEL.IMAGE_STD, dtype=np.float32)

    def __call__(self, s: Session, jobs: List[Tuple[str, int, np.ndarray]]) -> Dict[Tuple[str, int], dict]:
        import torch

        results = {}
        for start in range(0, len(jobs), BATCH):
            chunk = jobs[start:start + BATCH]
            patches, meta = [], []
            for cam, f, box in chunk:
                patch, center, size = crop(s.color(cam, f), box, self.mean, self.std)
                patches.append(patch)
                meta.append((cam, f, box, center, size))
            with torch.no_grad():
                pred = self.model({"img": torch.from_numpy(np.stack(patches)).cuda()})
            kp = pred["pred_keypoints_2d"].float().cpu().numpy()
            params = {k: v.float().cpu().numpy() for k, v in pred["pred_mano_params"].items()}
            for i, (cam, f, box, center, size) in enumerate(meta):
                go_crop = params["global_orient"][i].reshape(3, 3)
                results[cam, f] = dict(
                    keypoints_2d=center + kp[i] * size,  # crop coords are in [-0.5, 0.5] of the box
                    global_orient_crop=go_crop,
                    global_orient_cam=crop_rotation(s.K(cam), center) @ go_crop,
                    hand_pose=params["hand_pose"][i].reshape(15, 3, 3),
                    betas=params["betas"][i].reshape(10),
                    box=box, box_center=center, box_size=np.float64(size))
        return results


def run(session: str, force: bool = False) -> None:
    s = Session(session)
    rig = rig_for(session)
    if rig is None:
        raise SystemExit(f"{session}: no calibration assigned (rig.SESSION_CALIBRATION)")
    out = stage_dir(session, "keypoints")
    tri_dir = out / "triangulated"
    if force and out.is_dir():
        shutil.rmtree(out)  # a frame that fails now must not keep its earlier result
    todo = [f for f in s.frame_ids if force or not (tri_dir / f"{frame_name(f)}.npz").is_file()]
    LOGGER.info("%s: %d frames to run", session, len(todo))
    if not todo:
        return
    P = {c: s.K(c) @ rig.T_cam_from_world[c][:3] for c in s.cameras}
    hamer = _Hamer()

    jobs = []
    for f in todo:
        seg = segment.load(session, f)
        if seg is None:
            continue
        for ci, cam in enumerate(s.cameras):
            if seg["box_valid"][ci]:
                jobs.append((cam, f, seg["boxes"][ci]))
    first = hamer(s, jobs)
    LOGGER.info("pass 1 (depth boxes): %d views", len(first))

    jobs, sources = [], {}
    for f in todo:
        kp = {cam: r["keypoints_2d"] for (cam, ff), r in first.items() if ff == f}
        tri = triangulate(P, kp) if len(kp) >= 2 else None
        for cam in s.cameras:
            if not s.color_path(cam, f).is_file():
                continue
            box = None
            if tri is not None:
                box = segment.project_box(s.K(cam), rig.to_cam(cam, tri[0]), (720, 1280), REBOX_MARGIN)
            if box is None and (cam, f) in first:
                box, sources[cam, f] = first[cam, f]["box"], "depth"
            elif box is not None:
                sources[cam, f] = "triangulated"
            if box is not None:
                jobs.append((cam, f, box))
    second = hamer(s, jobs)
    LOGGER.info("pass 2 (triangulated boxes): %d views", len(second))

    for cam in s.cameras:
        (out / cam).mkdir(parents=True, exist_ok=True)
    tri_dir.mkdir(parents=True, exist_ok=True)
    n_tri = 0
    for f in todo:
        kp = {}
        for cam in s.cameras:
            r = second.get((cam, f))
            if r is None:
                continue
            np.savez(out / cam / f"{frame_name(f)}.npz", box_source=sources[cam, f], **r)
            kp[cam] = r["keypoints_2d"]
        tri = triangulate(P, kp) if len(kp) >= 2 else None
        if tri is not None:
            np.savez(tri_dir / f"{frame_name(f)}.npz", joints_world=tri[0], views=np.array(tri[1]), residual_px=tri[2])
            n_tri += 1
    LOGGER.info("%s: triangulated %d/%d frames", session, n_tri, len(todo))


def load_triangulated(session: str, frame_id: int) -> Optional[Dict[str, np.ndarray]]:
    p = stage_dir(session, "keypoints") / "triangulated" / f"{frame_name(frame_id)}.npz"
    return dict(np.load(p)) if p.is_file() else None


def load(session: str, cam: str, frame_id: int) -> Optional[Dict[str, np.ndarray]]:
    p = stage_dir(session, "keypoints") / cam / f"{frame_name(frame_id)}.npz"
    return dict(np.load(p)) if p.is_file() else None
