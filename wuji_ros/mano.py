"""Right-hand MANO forward in OpenPose-21 joint order (= MediaPipe-21 = glove hand_skeleton).

Same layer and joint map as HaMeR's mano_wrapper (smplx.MANOLayer, rotation-matrix
pose, no mean-pose offset), so a hand_pose from HaMeR and one fitted here mean the
same thing. The five fingertips are MANO vertices (smplx vertex_ids['mano']).
Units are metres.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from smplx import MANOLayer
from smplx.lbs import batch_rodrigues
from smplx.vertex_ids import vertex_ids

from .paths import MANO_DIR

# MANO's 16 joints + 5 tip vertices -> OpenPose order (HaMeR mano_wrapper.mano_to_openpose).
OPENPOSE_FROM_MANO = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
TIP_VERTICES = list(vertex_ids["mano"].values())  # thumb, index, middle, ring, pinky

# OpenPose-21: 0 wrist; thumb 1-4; index 5-8; middle 9-12; ring 13-16; pinky 17-20.
PALM_JOINTS = (0, 1, 5, 9, 13, 17)  # wrist, thumb CMC, four MCPs: rigid in the palm (plan.md M6)
BONES = [(p, c) for f in range(5) for p, c in zip([0, 1 + 4 * f, 2 + 4 * f, 3 + 4 * f],
                                                     [1 + 4 * f, 2 + 4 * f, 3 + 4 * f, 4 + 4 * f])]
FINGER_BONES = [(p, c) for p, c in BONES if p != 0]  # wrist->base bones are palm, not finger


class Mano(torch.nn.Module):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.layer = MANOLayer(model_path=str(MANO_DIR / "MANO_RIGHT.pkl"), is_rhand=True,
                               use_pca=False).to(device)
        self.register_buffer("tips", torch.tensor(TIP_VERTICES, dtype=torch.long, device=device))
        self.register_buffer("joint_map", torch.tensor(OPENPOSE_FROM_MANO, dtype=torch.long, device=device))
        self.faces = np.asarray(self.layer.faces_tensor.cpu().numpy(), dtype=np.int64)
        self.device = device

    def forward(self, betas: torch.Tensor, global_orient: torch.Tensor, hand_pose: torch.Tensor,
                transl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotation matrices in: global_orient (B,3,3), hand_pose (B,15,3,3). -> (verts (B,778,3), joints (B,21,3))."""
        out = self.layer(betas=betas, global_orient=global_orient[:, None], hand_pose=hand_pose,
                         transl=transl)
        joints = torch.cat([out.joints, out.vertices[:, self.tips]], dim=1)[:, self.joint_map]
        return out.vertices, joints


def rotmat(axis_angle: torch.Tensor) -> torch.Tensor:
    """(...,3) axis-angle -> (...,3,3)."""
    shape = axis_angle.shape[:-1]
    return batch_rodrigues(axis_angle.reshape(-1, 3)).reshape(*shape, 3, 3)


def axis_angle(R: np.ndarray) -> np.ndarray:
    """(...,3,3) -> (...,3), numpy."""
    import cv2

    flat = R.reshape(-1, 3, 3)
    return np.stack([cv2.Rodrigues(np.asarray(r, dtype=np.float64))[0].ravel() for r in flat]).reshape(*R.shape[:-2], 3)
