"""Contact refinement (plan.md §6 18): while the object is held, move the 4-view hand (fit4) onto the
mocap-posed object.

The hand's pose (finger pose, hand size, rotation) is taken as correct (user decision 2026-10-29); only
its world position moves. A frame is held when the object is lifted, its lowest point more than LIFT_MM above
its height in the first frame (resting on the table; along the measured table normal, so the constant offset
between the camera1-posed object and the depth table cancels), and the fit4 hand is within NEAR_MM of it
(signed: penetration counts). Having moved from the first-frame pose is not enough: an object set down
elsewhere, or turned on the table, is not held (20261022_163051 f265, f281). There t solves, in mm,

    min_t |t|^2 + w * ( sum_v relu(-sd(v + t))^2 + relu(min_v sd(v + t))^2 ),   w raised over W_CONTACT

with sd the exact signed distance from a hand vertex to the object mesh (positive outside;
point-triangle distance, sign by the generalised winding number). Both penalty terms vanish exactly
when the hand touches the object without penetrating it, so t is the smallest move that does so.
When the hand goes through opposite faces (fingers on both sides) the penalties have no way out and the
optimiser ends inside; then the smallest exit over EXIT_DIRS directions is taken (exit_search).
Other frames keep the fit4 hand (t = 0). Only hand vertices are tested: a thin object part (the
tracker's marker pins) can still pass between them.

The object pose is a solver input here, so hand-object distances are no longer independent checks.
The independent checks are fit4's image terms, recomputed before and after: HaMeR 2D reprojection,
and the depth surface (the hand's depth points to the nearest MANO vertex). The object enters world
through camera1 only (rig.T_world_from_mocap), so the camera2-4 offsets in docs/TODO.md move with it.

    output/<session>/contact/NNNNNN.npz   every fit4 key, positions shifted by t (reproj_px and
        surface_mm_median after the move), held, t_world (3,) m, T_world_from_obj (4,4), obj_lift_mm,
        obj_move_mm, obj_rot_deg (from the first frame), min_sd_mm_before/after (signed, < 0 = deepest penetration), inside_before/after
        (hand vertices deeper than INSIDE_TOL_MM inside the object), reproj_px_before (V,21), surface_mm_before
    output/<session>/contact/summary.json
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import handfit, keypoints, segment
from .bagio import Session
from .objectpose import ObjectModel
from .paths import frame_name, get_logger, stage_dir

LOGGER = get_logger("wuji_ros.contact")
LIFT_MM = 5.0  # the object's lowest point above its first-frame (resting) height
NEAR_MM = 20.0  # the fit4 hand within this of the object
W_CONTACT = (1e2, 1e3, 1e4, 1e5)  # penalty continuation, each from the previous solution
ITERS = 100
EXIT_DIRS, EXIT_MM, EXIT_CHUNK = 400, 60.0, 16  # exit search when the optimiser ends inside: directions, reach, batch
INSIDE_TOL_MM = 0.1  # a vertex counts as inside the object deeper than this


class SignedDistance:
    """Exact signed distance (m, positive outside) from points to the solid object: the union of the mesh's
    closed bodies. Each body is measured on its own and the minimum is kept, because an assembly STL's bodies
    overlap (the tracker post sits in the prism's socket) and faces buried in another body are not the
    surface. A body wound inward (negative volume) is a sealed cavity (the hollow prism's inner wall); it is
    dropped, so the object is solid where a hand cannot reach anyway."""

    def __init__(self, verts: np.ndarray, faces: np.ndarray, device: str) -> None:
        import trimesh

        tri = torch.tensor(verts[faces], dtype=torch.float64, device=device)
        self.A, self.B, self.C = tri[:, 0], tri[:, 1], tri[:, 2]
        self.n = torch.nn.functional.normalize(torch.linalg.cross(self.B - self.A, self.C - self.A), dim=-1)
        mesh = trimesh.Trimesh(verts, faces, process=False)
        labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(faces))
        bodies = [np.flatnonzero(labels == k) for k in np.unique(labels)]
        self.bodies = [torch.tensor(i, device=device) for i in bodies if mesh.submesh([i], append=True).volume > 0]

    def __call__(self, P: torch.Tensor) -> torch.Tensor:
        A, B, C, n = self.A, self.B, self.C, self.n
        X = P[:, None]  # (N,1,3) against (F,3)
        d_plane = ((X - A) * n).sum(-1)
        Q = X - d_plane[..., None] * n
        in_face = torch.ones_like(d_plane, dtype=torch.bool)
        for U, W in ((A, B), (B, C), (C, A)):
            in_face &= (torch.linalg.cross((W - U).expand_as(Q), Q - U) * n).sum(-1) >= 0

        def edge(U, W):
            E = W - U
            t = (((X - U) * E).sum(-1) / (E * E).sum(-1)).clamp(0, 1)
            return (X - (U + t[..., None] * E)).norm(dim=-1)

        near = torch.minimum(torch.minimum(edge(A, B), edge(B, C)), edge(C, A))
        dist = torch.where(in_face, d_plane.abs(), near)  # (N,F)
        with torch.no_grad():
            # generalised winding number: the faces' solid angles (Van Oosterom-Strackee) summed / 4 pi
            a, b, c = A - X, B - X, C - X
            la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
            num = (a * torch.linalg.cross(b, c)).sum(-1)
            den = la * lb * lc + (a * b).sum(-1) * lc + (a * c).sum(-1) * lb + (b * c).sum(-1) * la
            half_angle = torch.atan2(num, den)
        per_body = []
        for idx in self.bodies:
            d = dist[:, idx].min(1).values
            per_body.append(torch.where(half_angle[:, idx].sum(1) / (2 * np.pi) > 0.5, -d, d))
        return torch.stack(per_body).min(0).values


def touch(sd: SignedDistance, V: torch.Tensor) -> torch.Tensor:
    """The smallest translation (m) after which the vertices V touch the mesh without penetrating it."""
    t = torch.zeros(3, dtype=V.dtype, device=V.device, requires_grad=True)  # mm
    for w in W_CONTACT:
        opt = torch.optim.LBFGS([t], lr=1.0, max_iter=ITERS, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            d = sd(V + t * 1e-3) * 1e3
            loss = (t ** 2).sum() + w * ((-d).clamp(min=0).pow(2).sum() + d.min().clamp(min=0) ** 2)
            loss.backward()
            return loss

        opt.step(closure)
    t = t.detach()
    with torch.no_grad():
        if sd(V + t * 1e-3).min() * 1e3 < -INSIDE_TOL_MM:
            # Penetrating on opposite sides (fingers through two faces) leaves the penalties without a descent
            # direction out; take the smallest exit over directions instead.
            exit_t = exit_search(sd, V)
            if exit_t is not None:
                t = exit_t
    return t * 1e-3


def exit_search(sd: SignedDistance, V: torch.Tensor) -> Optional[torch.Tensor]:
    """The smallest move (mm) over EXIT_DIRS directions (Fibonacci sphere) after which V is outside the mesh, or
    None within EXIT_MM. Along each direction, step by the deepest penetration: a move of d changes no signed
    distance by more than d, so the step never passes the first exit."""
    i = np.arange(EXIT_DIRS) + 0.5
    phi, th = np.arccos(1 - 2 * i / EXIT_DIRS), np.pi * (1 + 5 ** 0.5) * i
    U = torch.tensor(np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1),
                     dtype=V.dtype, device=V.device)
    best = None
    for u in U.split(EXIT_CHUNK):
        s = torch.zeros(len(u), dtype=V.dtype, device=V.device)
        for _ in range(200):
            depth = -(sd((V[None] + (s[:, None] * u)[:, None] * 1e-3).reshape(-1, 3)).reshape(len(u), -1).min(1).values
                      * 1e3)
            step = torch.where((depth > INSIDE_TOL_MM / 10) & (s < EXIT_MM), depth, torch.zeros_like(depth))
            if not (step > 0).any():
                break
            s = s + step
        ok = (depth <= INSIDE_TOL_MM / 10) & (s < EXIT_MM)
        if ok.any():
            k = int(torch.where(ok, s, torch.full_like(s, np.inf)).argmin())
            if best is None or s[k] < best.norm():
                best = s[k] * u[k]
    return best


def _reproj(session: str, rig, Ks: Dict[str, np.ndarray], f: int, joints_world: np.ndarray, views, mirrored: bool
            ) -> np.ndarray:
    """(V,21) px against HaMeR's 2D, mirrored as fit4's winner was."""
    rows = []
    for cam in views:
        k = keypoints.load(session, cam, f)["keypoints_2d"]
        uv = rig.to_cam(cam, joints_world) @ Ks[cam].T
        rows.append(np.linalg.norm(uv[:, :2] / uv[:, 2:] - (k[handfit.MIRROR_LABELS] if mirrored else k), axis=1))
    return np.array(rows)


def run(session: str, object_config: Path, object_mesh: Path, force: bool = False) -> None:
    from .rig import rig_for

    s = Session(session)
    rig = rig_for(session)
    T_world_from_mocap = rig.T_world_from_mocap() if rig is not None else None
    if T_world_from_mocap is None:
        raise SystemExit(f"{session}: the object needs the calibration's delivered mocap2cam.npy; it is never estimated")
    out = stage_dir(session, "contact")
    if (out / "summary.json").is_file() and not force:
        LOGGER.info("%s exists, skipping (use --force)", out)
        return
    if out.is_dir():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    obj = ObjectModel(object_config, object_mesh)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sd = SignedDistance(obj.verts, obj.faces, dev)
    table = segment.load_table(session)
    Ks = {c: s.K(c) for c in s.cameras}
    poses = {}
    for f in s.frame_ids:
        rec = s.object(f)
        pose = obj.T_mocap_from_obj(rec) if rec is not None else None
        if pose is not None:
            poses[f] = pose[0]
    if not poses:
        raise SystemExit(f"{session}: no frame with an object pose (data/{session}/object/, markers named in the config)")
    T_rest = poses[min(poses)]

    def lowest(T_mocap_from_obj: np.ndarray) -> float:
        T = T_world_from_mocap @ T_mocap_from_obj
        return float(table.height(obj.verts @ T[:3, :3].T + T[:3, 3]).min())

    rest = lowest(T_rest)
    LOGGER.info("object rest: frame %d, lowest point %.1f mm above the table; held = lifted > %.0f mm and hand "
                "within %.0f mm", min(poses), rest * 1e3, LIFT_MM, NEAR_MM)

    rows = []
    for f in s.frame_ids:
        fit = handfit.load_world(session, f)
        if fit is None:
            continue
        rec: Dict[str, object] = dict(fit)
        held, t_w, T_wo = False, np.zeros(3), np.full((4, 4), np.nan)
        lift_mm = move_mm = rot_deg = sd0 = sd1 = np.nan
        n_in0 = n_in1 = -1
        if f in poses:
            T_wo = T_world_from_mocap @ poses[f]
            lift_mm = (lowest(poses[f]) - rest) * 1e3
            move_mm = float(np.linalg.norm(poses[f][:3, 3] - T_rest[:3, 3]) * 1e3)
            rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(T_rest[:3, :3].T @ poses[f][:3, :3]) - 1) / 2, -1, 1))))
            # hand vertices in the object frame, where the mesh is fixed; |t| is the same in both frames
            V = torch.tensor((fit["verts_world"] - T_wo[:3, 3]) @ T_wo[:3, :3], dtype=torch.float64, device=dev)
            with torch.no_grad():
                d0 = sd(V)
            sd0, n_in0 = float(d0.min()) * 1e3, int((d0 < -INSIDE_TOL_MM * 1e-3).sum())
            held = lift_mm > LIFT_MM and sd0 <= NEAR_MM
            t_o = touch(sd, V) if held else torch.zeros(3, dtype=torch.float64, device=dev)
            with torch.no_grad():
                d1 = sd(V + t_o)
            sd1, n_in1 = float(d1.min()) * 1e3, int((d1 < -INSIDE_TOL_MM * 1e-3).sum())
            t_w = T_wo[:3, :3] @ t_o.cpu().numpy()
        verts0 = fit["verts_world"].astype(np.float64)
        rec["wrist_world"] = fit["wrist_world"] + t_w
        rec["joints_world"] = fit["joints_world"] + t_w
        rec["verts_world"] = (verts0 + t_w).astype(np.float32)
        T_ww = fit["T_world_from_wrist"].copy()
        T_ww[:3, 3] += t_w
        rec["T_world_from_wrist"] = T_ww

        views, mirrored = list(fit["views"]), bool(fit["labels_mirrored"])
        rec["reproj_px_before"] = _reproj(session, rig, Ks, f, fit["joints_world"], views, mirrored)
        rec["reproj_px"] = _reproj(session, rig, Ks, f, rec["joints_world"], views, mirrored)
        tri = keypoints.load_triangulated(session, f)
        pts = np.concatenate(segment.hand_points(s, rig, table, f, tri["joints_world"]))
        srf = [float(np.median(cKDTree(v).query(pts)[0]) * 1e3) if len(pts) else np.nan for v in (verts0, verts0 + t_w)]
        rec["surface_mm_before"], rec["surface_mm_median"] = srf
        rec.update(held=held, t_world=t_w, T_world_from_obj=T_wo, obj_lift_mm=lift_mm, obj_move_mm=move_mm,
                   obj_rot_deg=rot_deg,
                   min_sd_mm_before=sd0, min_sd_mm_after=sd1, inside_before=n_in0, inside_after=n_in1)
        np.savez(out / f"{frame_name(f)}.npz", **rec)
        rows.append((f, held, float(np.linalg.norm(t_w) * 1e3), sd0, sd1, n_in0, n_in1,
                     float(np.median(rec["reproj_px_before"])), float(np.median(rec["reproj_px"])), *srf))

    a = np.array([r[2:] for r in rows if r[1]], dtype=np.float64).reshape(-1, 9)

    def med(k: int) -> float:
        return float(np.nanmedian(a[:, k])) if len(a) else float("nan")

    summary = {
        "calibration": rig.name, "object_config": str(object_config), "object_mesh": str(object_mesh),
        "lift_mm": LIFT_MM, "near_mm": NEAR_MM, "w_contact": list(W_CONTACT),
        "frames": len(rows), "held_frames": [r[0] for r in rows if r[1]],
        "held_move_mm_median": med(0), "held_move_mm_max": float(a[:, 0].max()) if len(a) else float("nan"),
        "held_min_sd_mm_before_median": med(1), "held_min_sd_mm_after_median": med(2),
        "held_min_sd_mm_after_abs_max": float(np.abs(a[:, 2]).max()) if len(a) else float("nan"),
        "held_inside_verts_before": int(a[:, 3].sum()), "held_inside_verts_after": int(a[:, 4].sum()),
        "held_reproj_px_median_before": med(5), "held_reproj_px_median_after": med(6),
        "held_surface_mm_median_before": med(7), "held_surface_mm_median_after": med(8),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("%s: %d/%d frames held; hand moved %.1f mm median (max %.1f); min distance to the object %.1f -> %.1f mm "
                "(|after| max %.2f), inside vertices %d -> %d; reproj %.1f -> %.1f px, surface %.1f -> %.1f mm",
                session, len(a), len(rows), summary["held_move_mm_median"], summary["held_move_mm_max"],
                summary["held_min_sd_mm_before_median"], summary["held_min_sd_mm_after_median"],
                summary["held_min_sd_mm_after_abs_max"], summary["held_inside_verts_before"],
                summary["held_inside_verts_after"], summary["held_reproj_px_median_before"],
                summary["held_reproj_px_median_after"], summary["held_surface_mm_median_before"],
                summary["held_surface_mm_median_after"])


def load(session: str, frame_id: int) -> Optional[Dict[str, np.ndarray]]:
    p = stage_dir(session, "contact") / f"{frame_name(frame_id)}.npz"
    return dict(np.load(p)) if p.is_file() else None
