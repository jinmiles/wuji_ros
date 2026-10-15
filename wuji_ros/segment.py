"""Hand boxes and hand points from calibrated depth + RGB (+ HaMeR, see keypoints.py). plan.md M5.

scene  The table: RANSAC plane of the bright (white) points of the first full
       frame's fused depth near world z = 0, and its footprint as the largest
       connected patch of points on it (1 cm grid). Not a world-z band: the table is
       ~12 deg off world z in the 20261015 calibration.
boxes  Per frame, the 4-view depth fused into world. Candidates are points over
       the footprint, 1.2-35 cm above the table, and dark in RGB (the black glove
       and sleeve; the bare left hand and light clothing drop out), outside the
       static scene: 2 cm voxels that hold candidates in at least STATIC_FRACTION of
       the session's frames (on 1015-5, the checkerboard box at the table's corner,
       which otherwise became a hand while the real one was in the lap, t 19-22 s).
       A largest connected cluster (2 cm voxels) under MIN_HAND_POINTS is no hand.
       Of that cluster, the arm enters over the footprint's edge;
       the hand is the part farthest from that entry (HAND_REACH_M). With no entry
       (the glove lying on the table) the whole cluster is the hand. Its points,
       projected into every view, give that view's first box; keypoints.py then
       re-boxes from the triangulated HaMeR joints, so every view frames the same
       physical hand.
points hand_points(): fused depth within HAND_RADIUS_M of the triangulated HaMeR
       skeleton and above the table, the depth evidence the fits use. The forearm
       lies beyond the wrist joint and is excluded with it.

    output/<session>/scene/table.npz                 up, offset, grid origin, footprint, inner
    output/<session>/segment/NNNNNN.npz              boxes (V,4) px, box_valid (V,), hand_world (N,3)
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .bagio import Session
from .paths import frame_name, get_logger, stage_dir
from .rig import Rig, rig_for

LOGGER = get_logger("wuji_ros.segment")

CELL_M = 0.01
TABLE_BAND_M = 0.01
HEIGHT_RANGE_M = (0.012, 0.35)  # above the table; below 12 mm is table noise and the flat tape
DARK_V = 70  # HSV value: the glove and sleeve are near black, skin and the table are not
VOXEL_M = 0.02
EDGE_M = 0.02  # footprint margin counted as "the arm comes in here"
HAND_REACH_M = 0.13  # hand = cluster points within this of the farthest point from the entry
BOX_MARGIN = 0.15
MIN_BOX_POINTS = 30
MIN_HAND_POINTS = 200  # the glove gives 1000+ at STRIDE 3; static clutter gave 56-82
STATIC_FRACTION = 0.8  # 0.5 also took the hand: it hovers over the same spot for half of 1015-5
HAND_RADIUS_M = 0.03
STRIDE = 3  # px stride for the box stage cloud
POINT_STRIDE = 2  # px stride for hand_points


@dataclass
class Table:
    up: np.ndarray  # (3,) unit plane normal pointing up (world z-down: up_z < 0)
    offset: float  # height(P) = P . up - offset
    origin: np.ndarray  # (2,) world xy of grid cell (0, 0)
    footprint: np.ndarray  # (H,W) uint8, grid over world xy
    inner: np.ndarray  # footprint eroded by EDGE_M

    def over(self, xy: np.ndarray, inner: bool = False) -> np.ndarray:
        grid = self.inner if inner else self.footprint
        ij = np.floor((xy - self.origin) / CELL_M).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < grid.shape[0]) & (ij[:, 1] < grid.shape[1])
        out = np.zeros(len(xy), bool)
        out[ok] = grid[ij[ok, 0], ij[ok, 1]] > 0
        return out

    def height(self, P: np.ndarray) -> np.ndarray:
        return P @ self.up - self.offset


def fused_cloud(s: Session, rig: Rig, frame_id: int, stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N,3) world points, (N,3) HSV, (N,) camera index of the 4-view depth."""
    P, C, idx = [], [], []
    for ci, cam in enumerate(s.cameras):
        d, im = s.depth(cam, frame_id), s.color(cam, frame_id)
        if d is None or im is None:
            continue
        K = s.K(cam)
        v, u = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
        z = d[v, u].astype(np.float64) * 1e-3
        ok = (z > 0.2) & (z < 2.5)
        X = np.stack([(u[ok] - K[0, 2]) * z[ok] / K[0, 0], (v[ok] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], 1)
        P.append(rig.to_world(cam, X))
        C.append(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[v[ok], u[ok]])
        idx.append(np.full(int(ok.sum()), ci))
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros(0, int)
    return np.concatenate(P), np.concatenate(C), np.concatenate(idx)


def build_table(s: Session, rig: Rig) -> Table:
    for f in s.frame_ids:
        P, C, idx = fused_cloud(s, rig, f, STRIDE)
        if len(np.unique(idx)) == len(s.cameras):
            break
    # The tabletop is white and is the largest bright plane near world z = 0. RANSAC,
    # not a z band: the table is ~12 deg off world z in the 20261015 calibration,
    # so a horizontal slice of it is a strip, not the table.
    hsv = C
    cand = P[(np.abs(P[:, 2]) < 0.3) & (hsv[:, 2] > 120) & (hsv[:, 1] < 60)]
    rng = np.random.default_rng(0)
    best = None
    for _ in range(2000):
        a3 = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(a3[1] - a3[0], a3[2] - a3[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[2]) < 0.8:  # a table, not a wall
            continue
        inl = np.abs((cand - a3[0]) @ n) < TABLE_BAND_M / 2
        if best is None or inl.sum() > best.sum():
            best = inl
    Q = cand[best]
    for _ in range(3):
        c = Q.mean(0)
        up = np.linalg.svd(Q - c)[2][2]
        up = -up if up[2] > 0 else up
        Q = cand[np.abs((cand - c) @ up) < TABLE_BAND_M / 2]
    offset = float(c @ up)
    sel = np.abs(P @ up - offset) < TABLE_BAND_M
    T = P[sel, :2]
    origin = T.min(0)
    ij = np.floor((T - origin) / CELL_M).astype(int)
    occ = np.zeros(ij.max(0) + 1, np.uint8)
    occ[ij[:, 0], ij[:, 1]] = 1
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    biggest = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    contour = max(cv2.findContours(biggest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    footprint = np.zeros_like(biggest)
    cv2.fillPoly(footprint, [cv2.convexHull(contour)], 1)  # the hand occludes part of the table
    k = 2 * int(round(EDGE_M / CELL_M)) + 1
    inner = cv2.erode(footprint, np.ones((k, k), np.uint8))
    tilt = float(np.degrees(np.arccos(abs(up[2]))))
    LOGGER.info("table: plane tilt %.2f deg from world z, through world %s, footprint %.2f m^2 (frame %d)",
                tilt, np.round(c, 3), footprint.sum() * CELL_M ** 2, f)
    return Table(up, offset, origin, footprint, inner)


def load_table(session: str) -> Table:
    d = np.load(stage_dir(session, "scene") / "table.npz")
    return Table(d["up"], float(d["offset"]), d["origin"], d["footprint"], d["inner"])


def _largest_cluster(Q: np.ndarray) -> np.ndarray:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    vox = np.floor(Q / VOXEL_M).astype(int)
    keys, inv = np.unique(vox, axis=0, return_inverse=True)
    pairs = cKDTree(keys).query_pairs(1.8, output_type="ndarray")  # 26-neighbourhood
    A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(keys),) * 2)
    _, lab = connected_components(A, directed=False)
    big = np.argmax(np.bincount(lab, weights=np.bincount(inv)))
    return Q[lab[inv] == big]


def dark_points(s: Session, rig: Rig, table: Table, frame_id: int) -> np.ndarray:
    """Dark points over the table footprint within the hand's height range."""
    P, C, _ = fused_cloud(s, rig, frame_id, STRIDE)
    h = table.height(P)
    sel = (h > HEIGHT_RANGE_M[0]) & (h < HEIGHT_RANGE_M[1]) & (C[:, 2] < DARK_V) & table.over(P[:, :2])
    return P[sel]


def _voxels(Q: np.ndarray) -> np.ndarray:
    return np.unique(np.floor(Q / VOXEL_M).astype(np.int64), axis=0)


def static_voxels(per_frame: List[np.ndarray]) -> set:
    """2 cm voxels holding dark candidates in at least STATIC_FRACTION of the frames."""
    counts: Dict[tuple, int] = {}
    for Q in per_frame:
        for v in map(tuple, _voxels(Q)):
            counts[v] = counts.get(v, 0) + 1
    return {v for v, n in counts.items() if n >= STATIC_FRACTION * len(per_frame)}


def hand_candidate(Q: np.ndarray, table: Table, static: set) -> np.ndarray:
    """World points of the hand by depth + RGB alone (box stage), from dark_points()."""
    if static and len(Q):
        keep = np.array([tuple(v) not in static for v in np.floor(Q / VOXEL_M).astype(np.int64)])
        Q = Q[keep]
    if len(Q) < MIN_HAND_POINTS:
        return np.zeros((0, 3))
    Q = _largest_cluster(Q)
    if len(Q) < MIN_HAND_POINTS:
        return np.zeros((0, 3))
    edge = ~table.over(Q[:, :2], inner=True)
    if edge.sum() < 20:
        return Q
    entry = Q[edge].mean(0)
    d = np.linalg.norm(Q[:, :2] - entry[:2], axis=1)
    return Q[d > d.max() - HAND_REACH_M]


def project_box(K: np.ndarray, X_cam: np.ndarray, shape: Tuple[int, int], margin: float = BOX_MARGIN
                ) -> Optional[np.ndarray]:
    X_cam = X_cam[X_cam[:, 2] > 0.1]
    if len(X_cam) == 0:
        return None
    uv = (K @ X_cam.T).T
    uv = uv[:, :2] / uv[:, 2:]
    lo, hi = uv.min(0), uv.max(0)
    pad = margin * (hi - lo).max()
    lo, hi = np.maximum(lo - pad, 0), np.minimum(hi + pad, [shape[1] - 1, shape[0] - 1])
    if np.any(hi - lo < 16):
        return None
    return np.array([lo[0], lo[1], hi[0], hi[1]])


def run(session: str, force: bool = False) -> None:
    s = Session(session)
    rig = rig_for(session)
    if rig is None:
        raise SystemExit(f"{session}: no calibration assigned (rig.SESSION_CALIBRATION)")
    scene = stage_dir(session, "scene")
    if not (scene / "table.npz").is_file() or force:
        t = build_table(s, rig)
        scene.mkdir(parents=True, exist_ok=True)
        np.savez(scene / "table.npz", up=t.up, offset=t.offset, origin=t.origin, footprint=t.footprint, inner=t.inner)
    table = load_table(session)
    out = stage_dir(session, "segment")
    if (out / "summary.json").is_file() and not force:
        LOGGER.info("%s exists, skipping (use --force)", out)
        return
    if out.is_dir():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    dark = {f: dark_points(s, rig, table, f) for f in s.frame_ids}
    static = static_voxels(list(dark.values()))
    LOGGER.info("static scene: %d voxels of 2 cm left out", len(static))
    found = 0
    for f in s.frame_ids:
        target = out / f"{frame_name(f)}.npz"
        hand = hand_candidate(dark[f], table, static)
        boxes = np.zeros((len(s.cameras), 4))
        valid = np.zeros(len(s.cameras), bool)
        if len(hand) >= MIN_BOX_POINTS:
            for ci, cam in enumerate(s.cameras):
                if not s.color_path(cam, f).is_file():
                    continue
                b = project_box(s.K(cam), rig.to_cam(cam, hand), (720, 1280))
                if b is not None:
                    boxes[ci], valid[ci] = b, True
            found += 1
        np.savez(target, boxes=boxes, box_valid=valid, hand_world=hand.astype(np.float32))
    (out / "summary.json").write_text(json.dumps({
        "frames": len(s.frame_ids), "with_hand": found, "static_voxels": len(static),
        "static_fraction": STATIC_FRACTION, "min_hand_points": MIN_HAND_POINTS}, indent=2) + "\n")
    LOGGER.info("%s: hand candidate in %d/%d frames", session, found, len(s.frame_ids))


def load(session: str, frame_id: int) -> Optional[Dict[str, np.ndarray]]:
    p = stage_dir(session, "segment") / f"{frame_name(frame_id)}.npz"
    return dict(np.load(p)) if p.is_file() else None


def _segment_distance(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(N,) distance from each point to the nearest of the segments A[k]-B[k]."""
    AB = B - A
    t = np.clip(np.einsum("nkd,kd->nk", P[:, None] - A[None], AB) / np.maximum((AB ** 2).sum(1), 1e-12), 0, 1)
    closest = A[None] + t[..., None] * AB[None]
    return np.linalg.norm(P[:, None] - closest, axis=-1).min(1)


def hand_points(s: Session, rig: Rig, table: Table, frame_id: int, joints_world: np.ndarray
                ) -> List[np.ndarray]:
    """Per camera, the world depth points of the hand: within HAND_RADIUS_M of the skeleton, above the table."""
    from .mano import BONES

    P, _, idx = fused_cloud(s, rig, frame_id, POINT_STRIDE)
    lo, hi = joints_world.min(0) - HAND_RADIUS_M, joints_world.max(0) + HAND_RADIUS_M
    pre = np.all((P >= lo) & (P <= hi), axis=1) & (table.height(P) > 0.005)
    A = joints_world[[p for p, _ in BONES]]
    B = joints_world[[c for _, c in BONES]]
    keep = np.zeros(len(P), bool)
    keep[pre] = _segment_distance(P[pre], A, B) < HAND_RADIUS_M
    return [P[keep & (idx == ci)] for ci in range(len(s.cameras))]
