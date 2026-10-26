"""V1 per camera: the wrist 6D of MANO(beta, theta_glove) from HaMeR 2D + the glove's depth points.

The finger pose is the glove's (G1) and fixed; only the hand's rigid 6D is solved,
in each camera's own frame:

    X_cam = R (X_mano - J_wrist) + c          X_mano = MANO(beta, I, theta_glove, 0)

`run_world` (stage `fit4`) is the main path: one pose for all views in the
calibrated world. `run` (stage `fit`) solves each camera on its own, which is what
the calib stage compares across cameras.

  - reprojection: HaMeR keypoints_2d vs the projected MANO joints, Huber 10 px;
  - surface: the hand's depth points (segment.hand_points: fused depth within 3 cm
    of the triangulated HaMeR skeleton) -> nearest MANO vertex (one direction),
    Huber 10 mm, vertices pushed out along their normals by the glove thickness.
Two starts per frame, HaMeR's global orientation and that flipped 180 deg about the
hand's length axis; the lower final data loss wins (plan.md M7).

Independent check (not a loss): the palm IMU's gravity, carried into the camera by
R_cam_from_wrist, against the table's downward normal (segment.Table.up carried into
that camera). This assumes a level table. Only tilt is compared; IMU yaw drifts.

    output/<session>/fit/<camera>/NNNNNN.npz
        R_cam_from_mano, wrist_cam (3,), joints_cam (21,3), verts_cam (778,3),
        reproj_px (21,), surface_mm_median, n_points, start, gravity_err_deg
    output/<session>/fit/summary.json
"""

from __future__ import annotations

import json
import shutil
from typing import Dict, Optional

import numpy as np

from . import glovefit, keypoints, segment
from .bagio import Session
from .paths import frame_name, get_logger, stage_dir

LOGGER = get_logger("wuji_ros.fit")
ITERS = 300
MAX_POINTS = 1500
HUBER_PX, HUBER_MM = 10.0, 10.0
CHUNK = 128  # fit4 candidates optimised together


def _hand_points(s: Session, session: str, rng: np.random.Generator) -> Dict[int, list]:
    """frame -> per-camera (N,3) world hand points, for frames with a HaMeR triangulation."""
    from .rig import rig_for

    rig, table = rig_for(session), segment.load_table(session)
    out = {}
    for f in s.frame_ids:
        tri = keypoints.load_triangulated(session, f)
        if tri is None:
            continue
        per_cam = segment.hand_points(s, rig, table, f, tri["joints_world"])
        out[f] = [p[rng.choice(len(p), MAX_POINTS, replace=False)] if len(p) > MAX_POINTS else p for p in per_cam]
    return out


def run(session: str, glove_thickness_m: float = 0.0, force: bool = False) -> None:
    import torch

    from .mano import Mano, rotmat

    from .rig import rig_for

    s = Session(session)
    rig = rig_for(session)
    out = stage_dir(session, "fit")
    dev = "cuda"
    mano = Mano(dev)
    faces = torch.tensor(mano.faces, device=dev)
    rng = np.random.default_rng(0)
    summary: Dict[str, Dict[str, object]] = {}
    hand_pts = _hand_points(s, session, rng)
    for ci_cam, cam in enumerate(s.cameras):
        if (out / cam).is_dir() and any((out / cam).glob("*.npz")) and not force:
            LOGGER.info("%s exists, skipping (use --force)", out / cam)
            continue
        if (out / cam).is_dir():
            shutil.rmtree(out / cam)  # a frame that fails now must not keep its earlier result
        K = s.K(cam)
        up = rig.T_cam_from_world[cam][:3, :3] @ segment.load_table(session).up
        frames, kp2d, R0, pts, gfit = [], [], [], [], []
        for f in s.frame_ids:
            g, k = glovefit.load(session, f), keypoints.load(session, cam, f)
            if g is None or k is None or f not in hand_pts:
                continue
            frames.append(f)
            kp2d.append(k["keypoints_2d"])
            R0.append(k["global_orient_cam"])
            pts.append(rig.to_cam(cam, hand_pts[f][ci_cam]))
            gfit.append(g)
        if not frames:
            continue
        B = len(frames)
        eye = torch.eye(3, device=dev)
        betas = torch.tensor(gfit[0]["betas"], dtype=torch.float32, device=dev).expand(B, 10)
        theta = torch.tensor(np.stack([g["hand_pose"] for g in gfit]), dtype=torch.float32, device=dev)
        with torch.no_grad():
            V, J = mano(betas, eye.expand(B, 3, 3), theta, torch.zeros(B, 3, device=dev))
            # vertex normals of the canonical mesh, for the glove-thickness offset
            tri = V[:, faces]
            fn = torch.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0], dim=-1)
            vn = torch.zeros_like(V).index_add_(1, faces.reshape(-1), fn.repeat_interleave(3, dim=1))
            vn = torch.nn.functional.normalize(vn, dim=-1)
            Vs = V + glove_thickness_m * vn
        J0 = J[:, :1]
        Kt = torch.tensor(K, dtype=torch.float32, device=dev)
        target = torch.tensor(np.stack(kp2d), dtype=torch.float32, device=dev)
        npts = [len(p) for p in pts]
        P = torch.zeros(B, max(max(npts), 1), 3, device=dev)
        Pm = torch.zeros(B, P.shape[1], device=dev)
        for i, p in enumerate(pts):
            P[i, :len(p)] = torch.tensor(p, dtype=torch.float32, device=dev)
            Pm[i, :len(p)] = 1.0
        has_pts = Pm.sum(1) > 0

        # hand length axis (wrist -> middle MCP) in the MANO frame, for the flipped start
        axis = torch.nn.functional.normalize(J[:, 9] - J[:, 0], dim=-1)
        flip = rotmat(np.pi * axis)
        R_hamer = torch.tensor(np.stack(R0), dtype=torch.float32, device=dev)

        def solve(R_start):
            centroid = (V.mean(1, keepdim=True) - J0)
            p_med = torch.stack([P[i, :npts[i]].median(0).values if npts[i] else
                                 torch.tensor([0, 0, 0.8], device=dev) for i in range(B)])
            c = (p_med - torch.einsum("bij,bj->bi", R_start, centroid[:, 0])).clone().requires_grad_(True)
            w = torch.zeros(B, 3, device=dev, requires_grad=True)
            opt = torch.optim.Adam([{"params": [w], "lr": 0.01}, {"params": [c], "lr": 0.002}])

            def losses():
                R = torch.einsum("bij,bjk->bik", R_start, rotmat(w))
                Jc = torch.einsum("bij,bkj->bki", R, J - J0) + c[:, None]
                Vc = torch.einsum("bij,bkj->bki", R, Vs - J0) + c[:, None]
                uv = torch.einsum("ij,bkj->bki", Kt, Jc)
                uv = uv[..., :2] / uv[..., 2:].clamp(min=1e-3)
                rep = torch.nn.functional.huber_loss(uv, target, delta=HUBER_PX, reduction="none").sum(-1)
                dmin = torch.cdist(P, Vc).min(-1).values * 1e3  # mm
                srf = torch.nn.functional.huber_loss(dmin, torch.zeros_like(dmin), delta=HUBER_MM, reduction="none")
                srf = (srf * Pm).sum(1) / Pm.sum(1).clamp(min=1)
                return R, Jc, Vc, uv, dmin, rep.mean(1) + srf

            for _ in range(ITERS):
                opt.zero_grad()
                loss = losses()[-1]
                loss.sum().backward()
                opt.step()
            with torch.no_grad():
                return [x.detach() for x in losses()]

        results = [solve(R_hamer), solve(torch.einsum("bij,bjk->bik", R_hamer, flip))]
        pick = torch.stack([r[-1] for r in results]).argmin(0).cpu().numpy()
        (out / cam).mkdir(parents=True, exist_ok=True)
        rep_all, srf_all, grav_all = [], [], []
        for i, f in enumerate(frames):
            R, Jc, Vc, uv, dmin, _ = (x[i] for x in results[pick[i]])
            R = R.cpu().numpy()
            reproj = np.linalg.norm(uv.cpu().numpy() - kp2d[i], axis=-1)
            surf = float(np.median(dmin.cpu().numpy()[:npts[i]])) if npts[i] else float("nan")
            grav_err = float("nan")
            gl = s.glove(f)
            if up is not None and gl is not None and "gravity_wrist" in gl:
                R_cam_from_wrist = R @ gfit[i]["R_wrist_from_mano"].T
                g_cam = R_cam_from_wrist @ gl["gravity_wrist"]
                grav_err = float(np.degrees(np.arccos(np.clip(g_cam @ -up, -1, 1))))
            np.savez(out / cam / f"{frame_name(f)}.npz", R_cam_from_mano=R, wrist_cam=Jc[0].cpu().numpy(),
                     joints_cam=Jc.cpu().numpy(), verts_cam=Vc.cpu().numpy().astype(np.float32),
                     reproj_px=reproj, surface_mm_median=surf, n_points=npts[i], start=int(pick[i]),
                     gravity_err_deg=grav_err)
            rep_all.append(np.median(reproj))
            srf_all.append(surf)
            grav_all.append(grav_err)
        summary[cam] = {
            "frames": B, "flipped_start_chosen": int(pick.sum()),
            "reproj_px_median": float(np.median(rep_all)),
            "surface_mm_median": float(np.nanmedian(srf_all)) if np.isfinite(srf_all).any() else None,
            "gravity_err_deg_median": float(np.nanmedian(grav_all)) if np.isfinite(grav_all).any() else None,
            "gravity_err_deg_p90": float(np.nanpercentile(grav_all, 90)) if np.isfinite(grav_all).any() else None,
            "table_up_cam": None if up is None else up.round(4).tolist(),
        }
        LOGGER.info("%s: %d frames, reproj %.1f px, surface %.1f mm, IMU tilt err %.1f deg (medians)", cam, B,
                    summary[cam]["reproj_px_median"], summary[cam]["surface_mm_median"] or float("nan"),
                    summary[cam]["gravity_err_deg_median"] or float("nan"))
    if summary:
        out.mkdir(parents=True, exist_ok=True)
        prev = json.loads((out / "summary.json").read_text()) if (out / "summary.json").is_file() else {}
        prev.update({"glove_thickness_m": glove_thickness_m, **summary})
        (out / "summary.json").write_text(json.dumps(prev, indent=2) + "\n")


def load(session: str, cam: str, frame_id: int):
    p = stage_dir(session, "fit") / cam / f"{frame_name(frame_id)}.npz"
    return dict(np.load(p)) if p.is_file() else None


IMU_WEIGHT = 100.0  # fit4 IMU term: a palm/back flip costs 2x this, a 4 deg error 0.24; 50 left f73 flipped
# HaMeR's 2D under a palm/back flip: thumb <-> pinky and index <-> ring swap places
# (a right hand palm-up seen from above has the outline of a right hand palm-down
# mirrored). OpenPose-21: thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20.
MIRROR_LABELS = [0, 17, 18, 19, 20, 13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4]


def run_world(session: str, glove_thickness_m: float = 0.0, imu_weight: float = IMU_WEIGHT,
              force: bool = False) -> None:
    """V1 over all views at once, in the world of the session's calibration (plan.md M7).

    Same model and losses as `run`, summed over the views HaMeR ran in: reprojection
    per view, surface over every view's hand points. Starts: the wrist at the
    triangulated HaMeR wrist, the rotation of every view's HaMeR global orientation
    carried into world, and each of those flipped 180 deg about the hand's length
    axis, plus the IMU orientation below. Each start is solved twice: against HaMeR's
    2D as labelled, and against it with the fingers mirrored (MIRROR_LABELS), which is
    what HaMeR's 2D is when it has flipped palm and back. The lowest total loss wins.

    IMU term (plan.md §6 13, 14): imu_weight * (1 - cos) of the angle between the
    fitted hand's R_world_from_wrist and the glove's fused IMU pose carried into world,
    R_world_from_waist R_waist_from_wrist. The SDK's waist frame is z-up, i.e. its
    tilt is gravity: world gravity is minus the measured table normal
    (segment.Table.up; a level table is assumed, world z is not usable, the 20261015
    calibration's board was ~12 deg off the tabletop). Its yaw is one session constant
    (glovefit.waist_in_world over every HaMeR view x the IMU pose); the IMU's slow yaw drift
    within a take is assumed small next to the errors this term is for.
      - 2026-10-16: tilt only. On a black glove HaMeR flips palm and back in every
        view at once (1015-5, t = 31.8-39.4 s), and images cannot outvote that.
      - 2026-10-26: full rotation. While the object is held, HaMeR's views can all be
        wrong about the hand's heading, which tilt cannot see (E008 f320/f328: 84 and
        102 deg of yaw off), and the HaMeR starts never reach the right basin; the
        IMU orientation is also a start.
    A flip costs 2 * imu_weight, a few degrees of IMU-to-wrist misalignment almost
    nothing. imu_weight = 0 is the image-only fit, without the IMU start; only then are
    gravity_err_deg and imu_rot_err_deg independent checks.

        output/<session>/fit4/NNNNNN.npz   R_world_from_mano, wrist_world, T_world_from_wrist (4,4)
            (the glove's r_wrist frame, for its taxels and skeleton), joints_world (21,3),
            verts_world (778,3), views, reproj_px (V,21), surface_mm_median, start_camera
            (the HaMeR view whose orientation started the winner, or "imu"), start_flipped,
            labels_mirrored (the winner used MIRROR_LABELS), gravity_err_deg
            (fused IMU tilt), gravity_err_accel_deg (accelerometer), imu_rot_err_deg
            (fused IMU, whole rotation), image_loss, imu_loss
        output/<session>/fit4/summary.json
    """
    import torch

    from .mano import Mano, rotmat
    from .rig import rig_for

    s = Session(session)
    rig = rig_for(session)
    if rig is None:
        raise SystemExit(f"{session}: no calibration assigned (rig.SESSION_CALIBRATION); use `fit`")
    out = stage_dir(session, "fit4")
    if (out / "summary.json").is_file() and not force:
        LOGGER.info("%s exists, skipping (use --force)", out)
        return
    if out.is_dir():
        shutil.rmtree(out)
    dev = "cuda"
    mano = Mano(dev)
    faces = torch.tensor(mano.faces, device=dev)
    rng = np.random.default_rng(0)
    cams = list(s.cameras)
    Ks = torch.tensor(np.stack([s.K(c) for c in cams]), dtype=torch.float32, device=dev)
    Tcw = torch.tensor(np.stack([rig.T_cam_from_world[c] for c in cams]), dtype=torch.float32, device=dev)
    up = segment.load_table(session).up
    g_world = -up  # assumes a level table

    hand_pts = _hand_points(s, session, rng)
    frames, gfit, targets, vmask, clouds, starts, R_waist = [], [], [], [], [], [], []
    for f in s.frame_ids:
        g, tri = glovefit.load(session, f), keypoints.load_triangulated(session, f)
        if g is None or tri is None:
            continue
        tg = np.zeros((len(cams), 21, 2))
        vm = np.zeros(len(cams))
        pts, st = [], []
        for ci, cam in enumerate(cams):
            k = keypoints.load(session, cam, f)
            if k is None:
                continue
            tg[ci], vm[ci] = k["keypoints_2d"], 1.0
            pts.append(hand_pts[f][ci])
            st.append((ci, rig.T_world_from_cam(cam)[:3, :3] @ k["global_orient_cam"], tri["joints_world"][0]))
        if vm.sum() < 2 or not st:
            continue
        frames.append(f)
        gfit.append(g)
        gl = s.glove(f)
        R_waist.append(gl["R_waist_from_wrist"] if gl is not None and "R_waist_from_wrist" in gl else None)
        targets.append(tg)
        vmask.append(vm)
        cloud = np.concatenate(pts) if pts else np.zeros((0, 3))
        if len(cloud) > MAX_POINTS:
            cloud = cloud[rng.choice(len(cloud), MAX_POINTS, replace=False)]
        clouds.append(cloud)
        starts.append(st)
    if not frames:
        raise SystemExit(f"{session}: no frame with two or more views")
    B, V = len(frames), len(cams)
    LOGGER.info("%s: %d frames with >= 2 views", session, B)
    imu_views = [(R_wm, g["R_wrist_from_mano"], Rw)
                 for st, g, Rw in zip(starts, gfit, R_waist) if Rw is not None for _, R_wm, _ in st]
    R_world_from_waist, yaw_deg, yaw_support = glovefit.waist_in_world(imu_views, up)
    R_imu = [None if Rw is None or R_world_from_waist is None else R_world_from_waist @ Rw @ g["R_wrist_from_mano"]
             for g, Rw in zip(gfit, R_waist)]
    LOGGER.info("IMU waist yaw in world %.0f deg: %.0f%% of %d HaMeR views within 30 deg", yaw_deg,
                100 * yaw_support, len(imu_views))

    eye = torch.eye(3, device=dev)
    betas = torch.tensor(gfit[0]["betas"], dtype=torch.float32, device=dev).expand(B, 10)
    theta = torch.tensor(np.stack([g["hand_pose"] for g in gfit]), dtype=torch.float32, device=dev)
    with torch.no_grad():
        Vm, Jm = mano(betas, eye.expand(B, 3, 3), theta, torch.zeros(B, 3, device=dev))
        tri = Vm[:, faces]
        fn = torch.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0], dim=-1)
        vn = torch.nn.functional.normalize(
            torch.zeros_like(Vm).index_add_(1, faces.reshape(-1), fn.repeat_interleave(3, dim=1)), dim=-1)
        Vs = Vm + glove_thickness_m * vn
    J0 = Jm[:, :1]
    axis = torch.nn.functional.normalize(Jm[:, 9] - Jm[:, 0], dim=-1)
    flip = rotmat(np.pi * axis)

    # candidates: (frame index, start rotation, start wrist, camera, flipped, mirrored labels)
    cand = []
    for i, st in enumerate(starts):
        for ci, R, w in st:
            for fl in (0, 1):
                for mi in (0, 1):
                    cand.append((i, R, w, ci, fl, mi))
        if imu_weight > 0 and R_imu[i] is not None:  # for frames where every HaMeR view is wrong
            for mi in (0, 1):
                cand.append((i, R_imu[i], st[0][2], -1, 0, mi))
    idx = torch.tensor([c[0] for c in cand], device=dev)
    R_start = torch.tensor(np.stack([c[1] for c in cand]), dtype=torch.float32, device=dev)
    fl_t = torch.tensor([c[4] for c in cand], device=dev).bool()
    R_start = torch.where(fl_t[:, None, None], torch.einsum("bij,bjk->bik", R_start, flip[idx]), R_start)
    c0 = torch.tensor(np.stack([c[2] for c in cand]), dtype=torch.float32, device=dev)
    N = len(cand)
    targets_np = np.stack(targets)
    tgt_both = torch.tensor(np.stack([targets_np, targets_np[:, :, MIRROR_LABELS]]), dtype=torch.float32, device=dev)
    mi_t = torch.tensor([c[5] for c in cand], device=dev)
    target = tgt_both[mi_t, idx]  # (N,V,21,2)
    vmk = torch.tensor(np.stack(vmask), dtype=torch.float32, device=dev)[idx]  # (N,V)
    npts = [len(p) for p in clouds]
    P = torch.zeros(B, max(max(npts), 1), 3, device=dev)
    Pm = torch.zeros(B, P.shape[1], device=dev)
    for i, p in enumerate(clouds):
        P[i, :len(p)] = torch.tensor(p, dtype=torch.float32, device=dev)
        Pm[i, :len(p)] = 1.0
    Jl, Vl, Pl, Pml = (Jm - J0)[idx], (Vs - J0)[idx], P[idx], Pm[idx]
    Ri = torch.tensor(np.stack([np.eye(3) if r is None else r for r in R_imu]), dtype=torch.float32, device=dev)[idx]
    Hi = torch.tensor([r is not None for r in R_imu], dtype=torch.float32, device=dev)[idx]

    def solve(sl):
        w = torch.zeros(sl.stop - sl.start, 3, device=dev, requires_grad=True)
        c = c0[sl].clone().requires_grad_(True)
        opt = torch.optim.Adam([{"params": [w], "lr": 0.01}, {"params": [c], "lr": 0.002}])

        def losses():
            R = torch.einsum("bij,bjk->bik", R_start[sl], rotmat(w))
            Jw = torch.einsum("bij,bkj->bki", R, Jl[sl]) + c[:, None]
            Vw = torch.einsum("bij,bkj->bki", R, Vl[sl]) + c[:, None]
            Jc = torch.einsum("vij,bkj->bvki", Tcw[:, :3, :3], Jw) + Tcw[None, :, None, :3, 3]
            uv = torch.einsum("vij,bvkj->bvki", Ks, Jc)
            uv = uv[..., :2] / uv[..., 2:].clamp(min=1e-3)
            rep = torch.nn.functional.huber_loss(uv, target[sl], delta=HUBER_PX, reduction="none").sum(-1).mean(-1)
            rep = (rep * vmk[sl]).sum(1) / vmk[sl].sum(1)
            dmin = torch.cdist(Pl[sl], Vw).min(-1).values * 1e3
            srf = torch.nn.functional.huber_loss(dmin, torch.zeros_like(dmin), delta=HUBER_MM, reduction="none")
            srf = (srf * Pml[sl]).sum(1) / Pml[sl].sum(1).clamp(min=1)
            # 1 - cos of the rotation between fit and IMU, (3 - tr(R^T R_imu)) / 2; 0 where no IMU
            imu = imu_weight * Hi[sl] * (3 - (R * Ri[sl]).sum((-2, -1))) / 2
            return R, Jw, Vw, uv, dmin, rep + srf, imu

        for _ in range(ITERS):
            opt.zero_grad()
            out_ = losses()
            (out_[5] + out_[6]).sum().backward()
            opt.step()
        with torch.no_grad():
            return [x.detach() for x in losses()]

    parts = [solve(slice(a, min(a + CHUNK, N))) for a in range(0, N, CHUNK)]
    R, Jw, Vw, uv, dmin, img_loss, imu_loss = (torch.cat([p[k] for p in parts]) for k in range(7))
    loss = img_loss + imu_loss

    out.mkdir(parents=True, exist_ok=True)
    loss_np = loss.cpu().numpy()
    img_np, imu_np = img_loss.cpu().numpy(), imu_loss.cpu().numpy()
    rep_all, srf_all, grav_all, rot_all, flips, mirrored, imu_starts = [], [], [], [], 0, 0, 0
    for i, f in enumerate(frames):
        mine = [n for n in range(N) if cand[n][0] == i]
        best = min(mine, key=lambda n: loss_np[n])
        Rb = R[best].cpu().numpy()
        views = [cams[v] for v in range(V) if vmask[i][v]]
        tgt = targets[i][:, MIRROR_LABELS] if cand[best][5] else targets[i]
        reproj = np.linalg.norm(uv[best].cpu().numpy() - tgt, axis=-1)[[v for v in range(V) if vmask[i][v]]]
        surf = float(np.median(dmin[best].cpu().numpy()[:npts[i]])) if npts[i] else float("nan")
        grav_err = grav_acc = float("nan")
        gl = s.glove(f)
        if gl is not None and "gravity_wrist_fused" in gl:
            g_w = Rb @ gfit[i]["R_wrist_from_mano"].T @ gl["gravity_wrist_fused"]
            grav_err = float(np.degrees(np.arccos(np.clip(g_w @ g_world, -1, 1))))
        if gl is not None and "gravity_wrist" in gl:
            g_w = Rb @ gfit[i]["R_wrist_from_mano"].T @ gl["gravity_wrist"]
            grav_acc = float(np.degrees(np.arccos(np.clip(g_w @ g_world, -1, 1))))
        rot_err = float("nan") if R_imu[i] is None else \
            float(np.degrees(np.arccos(np.clip((np.trace(Rb.T @ R_imu[i]) - 1) / 2, -1, 1))))
        # glove r_wrist -> world: p_mano = R_g^T (p_wrist - t_g), p_world = Rb (p_mano - J0) + wrist
        R_g, t_g = gfit[i]["R_wrist_from_mano"], gfit[i]["t_wrist_from_mano"]
        T_ww = np.eye(4)
        T_ww[:3, :3] = Rb @ R_g.T
        T_ww[:3, 3] = Jw[best, 0].cpu().numpy() - Rb @ J0[i, 0].cpu().numpy() - Rb @ R_g.T @ t_g
        np.savez(out / f"{frame_name(f)}.npz", R_world_from_mano=Rb, wrist_world=Jw[best, 0].cpu().numpy(),
                 T_world_from_wrist=T_ww,
                 joints_world=Jw[best].cpu().numpy(), verts_world=Vw[best].cpu().numpy().astype(np.float32),
                 views=np.array(views), reproj_px=reproj, surface_mm_median=surf,
                 start_camera=cams[cand[best][3]] if cand[best][3] >= 0 else "imu", start_flipped=int(cand[best][4]),
                 labels_mirrored=int(cand[best][5]), gravity_err_deg=grav_err,
                 gravity_err_accel_deg=grav_acc, imu_rot_err_deg=rot_err,
                 image_loss=float(img_np[best]), imu_loss=float(imu_np[best]))
        rep_all.append(np.median(reproj))
        srf_all.append(surf)
        grav_all.append(grav_err)
        rot_all.append(rot_err)
        flips += cand[best][4]
        mirrored += cand[best][5]
        imu_starts += cand[best][3] < 0
    g, r = np.array(grav_all), np.array(rot_all)
    summary = {
        "calibration": rig.name, "glove_thickness_m": glove_thickness_m, "imu_weight": imu_weight,
        "imu_in_loss": imu_weight > 0, "frames": B, "flipped_start_chosen": flips, "labels_mirrored_frames": mirrored,
        "imu_start_chosen": imu_starts,
        "reproj_px_median": float(np.median(rep_all)), "surface_mm_median": float(np.nanmedian(srf_all)),
        "gravity_err_deg_median": float(np.nanmedian(g)), "gravity_err_deg_p90": float(np.nanpercentile(g, 90)),
        "gravity_err_over_90deg_frames": int((g > 90).sum()),
        "gravity_source": "fused IMU pose (gravity_wrist_fused)",
        "imu_waist_yaw_deg": yaw_deg, "imu_waist_yaw_support": yaw_support,
        "imu_rot_err_deg_median": float(np.nanmedian(r)), "imu_rot_err_deg_p90": float(np.nanpercentile(r, 90)),
        "imu_rot_err_over_30deg_frames": int((r > 30).sum()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("%s (imu_weight %g): %d frames, reproj %.1f px, surface %.1f mm, IMU tilt err %.1f deg median / "
                "%.1f p90, %d > 90 deg; IMU rotation err %.1f deg median, %d > 30 deg", session, imu_weight, B,
                summary["reproj_px_median"], summary["surface_mm_median"], summary["gravity_err_deg_median"],
                summary["gravity_err_deg_p90"], summary["gravity_err_over_90deg_frames"],
                summary["imu_rot_err_deg_median"], summary["imu_rot_err_over_30deg_frames"])


def load_world(session: str, frame_id: int):
    p = stage_dir(session, "fit4") / f"{frame_name(frame_id)}.npz"
    return dict(np.load(p)) if p.is_file() else None
