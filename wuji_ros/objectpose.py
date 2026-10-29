"""The mocap-tracked rigid object, posed per frame from its recorded markers (plan.md M5, M9).

The object config is the collector's rerun yaml (the `import --camera-config` file). Its `object`
section gives `marker_offsets`: the marker sphere centres in the CAD (OBJ) frame, keyed by the recorded
marker name, in `marker_offsets_unit`. The mesh file is in the same OBJ frame, scaled by `mesh_scale`
(`mesh_T_object` null). The labels are matched by name and trusted, as in the collector's own script.
The yaml's mapping changes between recordings, so check the per-frame RMSE.

    T_mocap_from_obj   Kabsch of the offsets onto the recorded markers (>= min_valid_markers)
    T_world_from_obj = rig.T_world_from_mocap() @ T_mocap_from_obj   (the delivered link, through camera1)

    data/<session>/object/NNNNNN.npz   marker_names (M,), marker_positions (M,3) m, mocap frame
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

UNITS = {"mm": 1e-3, "m": 1.0}


class ObjectModel:
    def __init__(self, config: Path, mesh: Path) -> None:
        import trimesh
        import yaml

        cfg = yaml.safe_load(Path(config).read_text())["object"]
        if cfg.get("mesh_T_object") is not None:
            raise SystemExit(f"{config}: object.mesh_T_object is set; only a mesh in the OBJ frame is supported")
        unit = UNITS[cfg["marker_offsets_unit"]]
        self.offsets = {name: np.asarray(xyz, dtype=np.float64) * unit for name, xyz in cfg["marker_offsets"].items()}
        self.min_markers = int(cfg["min_valid_markers"])
        self.max_rmse_m = float(cfg["max_rmse_warn_m"])
        m = trimesh.load(Path(mesh), force="mesh")
        self.verts = np.asarray(m.vertices, dtype=np.float64) * float(cfg["mesh_scale"])
        self.faces = np.asarray(m.faces, dtype=np.int64)

    def T_mocap_from_obj(self, obj: Dict[str, np.ndarray]) -> Optional[Tuple[np.ndarray, float, int]]:
        """(4x4 T_mocap_from_obj, marker RMSE m, markers used), or None under min_valid_markers."""
        names = [str(n) for n in obj["marker_names"]]
        Y = np.asarray(obj["marker_positions"], dtype=np.float64).reshape(-1, 3)
        keep = [i for i, n in enumerate(names) if n in self.offsets and np.isfinite(Y[i]).all()]
        if len(keep) < self.min_markers:
            return None
        X, Y = np.stack([self.offsets[names[i]] for i in keep]), Y[keep]
        cx, cy = X.mean(0), Y.mean(0)
        U, _, Vt = np.linalg.svd((X - cx).T @ (Y - cy))
        R = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, cy - R @ cx
        rmse = float(np.sqrt((((X @ R.T + T[:3, 3]) - Y) ** 2).sum(1).mean()))
        return T, rmse, len(keep)
