"""Calibration check (plan.md M4) for a session's assigned calibration, from channels the fit does not use.

1. Table: each camera's own white tabletop points (the segment stage's footprint,
   within 2 cm of the fused RANSAC plane), fitted as a plane per camera. One rigid
   table seen by four correct extrinsics gives one plane: same normal, same height.
   RealSense depth on the white table is biased outward by ~10 mm, about equally in
   every view, so compare cameras with each other, and confirm with the depth-free
   HaMeR triangulation residual (keypoints stage).
2. Hand: each camera's per-camera V1 wrist and joints, carried into world. The
   cameras were solved independently, so their agreement is evidence about the
   extrinsics (and, where one view is flipped, about that view's fit).

    output/<session>/calib/report.json
"""

from __future__ import annotations

import itertools
import json

import numpy as np

from . import handfit, segment
from .bagio import Session
from .paths import get_logger, stage_dir
from .rig import rig_for

LOGGER = get_logger("wuji_ros.calib")


def _rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1))))


def run(session: str) -> dict:
    s = Session(session)
    rig = rig_for(session)
    if rig is None:
        raise SystemExit(f"{session}: no calibration assigned (rig.SESSION_CALIBRATION)")
    report = {"calibration": rig.name, "table": {}, "hand": {}}

    table = segment.load_table(session)
    first = next(f for f in s.frame_ids if all(s.depth(c, f) is not None for c in s.cameras))
    P, C, idx = segment.fused_cloud(s, rig, first, segment.STRIDE)
    on = table.over(P[:, :2]) & (np.abs(table.height(P)) < 0.02) & (C[:, 2] > 120) & (C[:, 1] < 60)
    centre = P[on].mean(0)
    for ci, cam in enumerate(s.cameras):
        Q = P[on & (idx == ci)]
        if len(Q) < 200:
            continue
        for _ in range(3):
            c = Q.mean(0)
            n = np.linalg.svd(Q - c)[2][2]
            n = -n if n @ table.up < 0 else n
            Q = Q[np.abs((Q - c) @ n) < 0.006]
        h = float((c - centre) @ n / (n @ table.up) + table.height(centre[None])[0])
        report["table"][cam] = {"points": int(len(Q)), "frame": int(first),
                                "tilt_vs_table_deg": float(np.degrees(np.arccos(np.clip(n @ table.up, -1, 1)))),
                                "height_at_centre_mm": h * 1e3,
                                "camera_centre_world": rig.center(cam).round(4).tolist()}
    report["table"]["table_tilt_vs_world_z_deg"] = float(np.degrees(np.arccos(abs(table.up[2]))))
    report["table"]["session_plane_up_world"] = table.up.round(4).tolist()

    pair_wrist, pair_joint, pair_rot = {}, {}, {}
    for f in s.frame_ids:
        res = {}
        for cam in s.cameras:
            fit = handfit.load(session, cam, f)
            if fit is None:
                continue
            Rw = rig.T_world_from_cam(cam)[:3, :3] @ fit["R_cam_from_mano"]
            res[cam] = (rig.to_world(cam, fit["joints_cam"]), Rw)
        for a, b in itertools.combinations(sorted(res), 2):
            key = f"{a}-{b}"
            pair_wrist.setdefault(key, []).append(np.linalg.norm(res[a][0][0] - res[b][0][0]) * 1e3)
            pair_joint.setdefault(key, []).append(np.linalg.norm(res[a][0] - res[b][0], axis=1).mean() * 1e3)
            pair_rot.setdefault(key, []).append(_rot_deg(res[a][1], res[b][1]))
    for key in sorted(pair_wrist):
        report["hand"][key] = {
            "frames": len(pair_wrist[key]),
            "wrist_mm_median": float(np.median(pair_wrist[key])),
            "joints_mm_median": float(np.median(pair_joint[key])),
            "orientation_deg_median": float(np.median(pair_rot[key])),
            "orientation_over_90deg_frames": int((np.array(pair_rot[key]) > 90).sum()),
        }
    out = stage_dir(session, "calib")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for cam in s.cameras:
        t = report["table"].get(cam)
        if t:
            LOGGER.info("table %s: tilt %.2f deg vs the fused plane, height %+.1f mm at its centre (%d points)",
                        cam, t["tilt_vs_table_deg"], t["height_at_centre_mm"], t["points"])
    for key, v in report["hand"].items():
        LOGGER.info("hand %s: wrist %.1f mm, joints %.1f mm, orientation %.1f deg (medians, %d frames, %d > 90 deg)",
                    key, v["wrist_mm_median"], v["joints_mm_median"], v["orientation_deg_median"], v["frames"],
                    v["orientation_over_90deg_frames"])
    return report
