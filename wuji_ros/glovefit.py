"""G1: the glove's hand_skeleton -> MANO hand_pose, in the r_wrist frame (no world, no camera).

Per frame: hand_pose theta (15 axis-angle) and T_wrist_from_mano (R, t). One beta
for the whole session. Loss (plan.md M6):
  - palm points (wrist, thumb CMC, four MCPs) position, which are rigid in the palm;
  - finger bones: 1 - cos(direction), so URDF-vs-MANO bone length differences do
    not bend the pose;
  - bone lengths, which are pose-invariant and therefore only move beta;
  - small priors on theta and beta.
Each term is weighted by the glove's per-joint confidence.

The joints do not mean the same thing on both sides (Wuji URDF joint centres vs
MANO rotation centres), so the residuals are reported, not assumed zero.

    output/<session>/glovefit/NNNNNN.npz   hand_pose (15,3,3), betas (10,),
        R_wrist_from_mano (3,3), t_wrist_from_mano (3,), joints_wrist (21,3),
        palm_err_mm (6,), bone_err_deg (15,), tip_err_mm (5,)
    output/<session>/glovefit/summary.json
"""

from __future__ import annotations

import json

import numpy as np

from .bagio import Session
from .paths import frame_name, get_logger, stage_dir

LOGGER = get_logger("wuji_ros.glovefit")
ITERS = 1500
W_PALM, W_BONE, W_LEN, W_THETA, W_BETA = 1.0, 100.0, 1.0, 0.1, 1.0  # palm/len in mm^2, bone unitless


def _kabsch(src: np.ndarray, dst: np.ndarray):
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def run(session: str, force: bool = False) -> None:
    import torch

    from .mano import BONES, FINGER_BONES, PALM_JOINTS, Mano, axis_angle, rotmat

    s = Session(session)
    out = stage_dir(session, "glovefit")
    if (out / "summary.json").is_file() and not force:
        LOGGER.info("%s exists, skipping (use --force)", out)
        return
    frames, skel, conf = [], [], []
    for f in s.frame_ids:
        g = s.glove(f)
        if g is None or "skeleton_wrist" not in g:
            continue
        frames.append(f)
        skel.append(g["skeleton_wrist"])
        conf.append(g["skeleton_confidence"])
    if not frames:
        raise SystemExit(f"{session}: no glove skeleton frames")
    dev = "cuda"
    mano = Mano(dev)
    S = torch.tensor(np.stack(skel), dtype=torch.float32, device=dev)  # (B,21,3) m
    C = torch.tensor(np.stack(conf), dtype=torch.float32, device=dev)
    B = S.shape[0]
    eye = torch.eye(3, device=dev)

    # init: rest MANO palm onto the glove palm, per frame
    _, J0 = mano(torch.zeros(1, 10, device=dev), eye[None], eye.expand(1, 15, 3, 3), torch.zeros(1, 3, device=dev))
    J0 = J0[0].cpu().numpy()
    R_init, t_init = zip(*[_kabsch(J0[list(PALM_JOINTS)], sk[list(PALM_JOINTS)]) for sk in skel])
    rot = torch.tensor(axis_angle(np.stack(R_init)), dtype=torch.float32, device=dev, requires_grad=True)
    trans = torch.tensor(np.stack(t_init), dtype=torch.float32, device=dev, requires_grad=True)
    theta = torch.zeros(B, 15, 3, device=dev, requires_grad=True)
    betas = torch.zeros(1, 10, device=dev, requires_grad=True)
    opt = torch.optim.Adam([rot, trans, theta, betas], lr=0.01)

    palm = list(PALM_JOINTS)
    fp, fc = [p for p, _ in FINGER_BONES], [c for _, c in FINGER_BONES]
    bp, bc = [p for p, _ in BONES], [c for _, c in BONES]

    def joints_wrist():
        _, J = mano(betas.expand(B, 10), eye.expand(B, 3, 3), rotmat(theta), torch.zeros(B, 3, device=dev))
        return torch.einsum("bij,bkj->bki", rotmat(rot), J) + trans[:, None]

    for it in range(ITERS):
        opt.zero_grad()
        J = joints_wrist()
        l_palm = (C[:, palm] * ((J[:, palm] - S[:, palm]) * 1e3).pow(2).sum(-1)).mean()  # mm^2
        dj, ds = J[:, fc] - J[:, fp], S[:, fc] - S[:, fp]
        cos = torch.nn.functional.cosine_similarity(dj, ds, dim=-1)
        l_bone = (C[:, fc] * (1 - cos)).mean()
        lj = (J[:, bc] - J[:, bp]).norm(dim=-1)
        ls = (S[:, bc] - S[:, bp]).norm(dim=-1)
        l_len = (C[:, bc] * ((lj - ls) * 1e3) ** 2).mean()  # mm^2
        loss = W_PALM * l_palm + W_BONE * l_bone + W_LEN * l_len + W_THETA * (theta ** 2).sum(-1).mean() \
            + W_BETA * (betas ** 2).sum()
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == ITERS - 1:
            LOGGER.info("  it %4d  palm %.2f mm  bone %.2f deg  len %.2f mm", it,
                        float(l_palm.sqrt()),
                        float(torch.rad2deg(torch.arccos((1 - l_bone).clamp(-1, 1)))),
                        float(l_len.sqrt()))

    with torch.no_grad():
        J = joints_wrist().cpu().numpy()
        Rm = rotmat(rot).cpu().numpy()
        th = rotmat(theta).cpu().numpy()
    Snp = np.stack(skel)
    beta_np = betas.detach().cpu().numpy()[0]
    out.mkdir(parents=True, exist_ok=True)
    palm_all, bone_all, tip_all = [], [], []
    for i, f in enumerate(frames):
        dj = J[i, fc] - J[i, fp]
        ds = Snp[i, fc] - Snp[i, fp]
        ang = np.degrees(np.arccos(np.clip((dj * ds).sum(-1) / (np.linalg.norm(dj, axis=-1) * np.linalg.norm(ds, axis=-1)), -1, 1)))
        palm_mm = np.linalg.norm(J[i, palm] - Snp[i, palm], axis=-1) * 1e3
        tip_mm = np.linalg.norm(J[i, [4, 8, 12, 16, 20]] - Snp[i, [4, 8, 12, 16, 20]], axis=-1) * 1e3
        np.savez(out / f"{frame_name(f)}.npz", hand_pose=th[i], betas=beta_np, R_wrist_from_mano=Rm[i],
                 t_wrist_from_mano=trans[i].detach().cpu().numpy(), joints_wrist=J[i],
                 palm_err_mm=palm_mm, bone_err_deg=ang, tip_err_mm=tip_mm)
        palm_all.append(palm_mm)
        bone_all.append(ang)
        tip_all.append(tip_mm)
    tip_all = np.stack(tip_all)
    summary = {
        "frames": len(frames), "iters": ITERS, "betas": beta_np.round(4).tolist(),
        "palm_err_mm_median": float(np.median(palm_all)),
        "bone_err_deg_median": float(np.median(bone_all)),
        "tip_err_mm_median": float(np.median(tip_all)),
        "tip_err_mm_median_per_finger": dict(zip(["thumb", "index", "middle", "ring", "pinky"],
                                                 np.median(tip_all, 0).round(2).tolist())),
        "note": "glove skeleton vs fitted MANO joints, r_wrist frame; joint semantics differ (URDF vs MANO)",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("%s: %d frames, palm %.1f mm, bone %.1f deg, tip %.1f mm (medians)", session, len(frames),
                summary["palm_err_mm_median"], summary["bone_err_deg_median"], summary["tip_err_mm_median"])


def load(session: str, frame_id: int):
    p = stage_dir(session, "glovefit") / f"{frame_name(frame_id)}.npz"
    return dict(np.load(p)) if p.is_file() else None
