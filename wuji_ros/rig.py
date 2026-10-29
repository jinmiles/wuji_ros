"""Calibration load, camera <-> world transforms, projection.

calibration/<name>/w2c_ext.npz: keys "0".."3" = camera1..camera4, each 3x4
T_cam_from_world (x_cam = R x_world + t). Intrinsics come from the session's own
camera_info (data/<session>/<camera>/intrinsics.json), never from the calibration.

Which calibration a session was recorded under is recorded here, with how it was
established; a session without an entry has no world frame and runs per camera.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .paths import CALIBRATION_DIR, CAMERAS

# session -> calibration name. Measured, not assumed: see docs/ARCHITECTURE.md.
SESSION_CALIBRATION: Dict[str, str] = {
    "1015-5": "20261015",
    "20261022_163051_mocap_mk4_wuji_fixed_rate": "20261015",
}


class Rig:
    def __init__(self, name: str) -> None:
        self.name = name
        data = np.load(CALIBRATION_DIR / name / "w2c_ext.npz", allow_pickle=False)
        self.T_cam_from_world: Dict[str, np.ndarray] = {}
        for i, cam in enumerate(CAMERAS):
            T = np.eye(4)
            T[:3] = np.asarray(data[str(i)], dtype=np.float64).reshape(3, 4)
            self.T_cam_from_world[cam] = T

    def T_world_from_cam(self, cam: str) -> np.ndarray:
        return np.linalg.inv(self.T_cam_from_world[cam])

    def to_world(self, cam: str, X_cam: np.ndarray) -> np.ndarray:
        T = self.T_world_from_cam(cam)
        return X_cam @ T[:3, :3].T + T[:3, 3]

    def to_cam(self, cam: str, X_world: np.ndarray) -> np.ndarray:
        T = self.T_cam_from_world[cam]
        return X_world @ T[:3, :3].T + T[:3, 3]

    def center(self, cam: str) -> np.ndarray:
        return self.T_world_from_cam(cam)[:3, 3]

    def T_world_from_mocap(self) -> Optional[np.ndarray]:
        """T_world_from_cam1 @ T_cam1_from_mocap (calibration/<name>/mocap2cam.npy, m), or None when the
        calibration has no delivered link; it is never estimated."""
        p = CALIBRATION_DIR / self.name / "mocap2cam.npy"
        return self.T_world_from_cam("camera1") @ np.load(p) if p.is_file() else None


def rig_for(session: str) -> Optional[Rig]:
    name = SESSION_CALIBRATION.get(session)
    return Rig(name) if name else None
