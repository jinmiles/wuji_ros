"""rosbag2 recording -> data/<session>/ (master-clock synchronised dataset).

    camera{1..4}/color/NNNNNN.png      rgb8 -> BGR png
    camera{1..4}/depth/NNNNNN.png      aligned_depth_to_color, uint16 mm
    camera{1..4}/intrinsics.json       first color/camera_info
    object/NNNNNN.npz                  /RigidBody/pose (+ markers when published)
    glove/NNNNNN.npz                   wuji glove, see GLOVE_KEYS
    metadata.csv                       per frame: every stream's header.stamp and dt
    manifest.json                      bag, settings, tf_static, glove info, counts

Every time is header.stamp (capture / EMF sampling time on the recording host
clock), never the bag write time. Master clock is camera1 color, one frame per
`every_s`. Every other stream is matched nearest-neighbour; a stream outside its
tolerance is left out of that frame and marked missing in metadata.csv, the
frame itself is kept (plan.md M3).

The bag is read as SQLite directly, so a bag whose recorder died (empty
metadata.yaml) works once `sqlite3 broken.db3 .recover | sqlite3 fixed.db3` has
made its database readable.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from rosbags.typesys import Stores, get_typestore

from .paths import CAMERAS, MASTER_CAMERA, frame_name, get_logger, session_dir

LOGGER = get_logger("wuji_ros.bagio")
TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)

CAMERA_TOL_S = 0.017  # half a 30 fps period: the nearest frame is the same exposure
STREAM_TOL_S = 0.010  # glove and mocap run at 120 Hz

GLOVE_NS = "/wuji_glove/right"
OBJECT_POSE_TOPIC = "/RigidBody/pose"
# glove npz keys: skeleton_wrist (21,3) m and skeleton_confidence (21,) in r_wrist;
# emf_poses / tip_poses (5,8) x,y,z,qx,qy,qz,qw,confidence (frames r_hand_emf_tx /
# r_wrist); joint_angles (21,) rad; accel_wrist (3,) raw palm accelerometer rotated
# into r_wrist; gravity_wrist (3,) unit, = -accel_wrist / |accel_wrist|.
# Tactile (32FC1, -1 = invalid taxel): tactile / tactile_binary / tactile_residual
# (24,31) row-major; tactile_zone_<zone> (n,); tactile_points (526,3) m in r_wrist
# with tactile_pressure (526,), the SDK's taxel cloud as recorded.
TACTILE_ZONES = ("palm", "thumb", "index", "middle", "ring", "pinky")
GLOVE_STREAMS = {
    "skeleton": f"{GLOVE_NS}/hand_skeleton",
    "emf": f"{GLOVE_NS}/emf_poses",
    "tips": f"{GLOVE_NS}/tip_poses",
    "angles": f"{GLOVE_NS}/hand_joint_angles",
    "imu": f"{GLOVE_NS}/imu_raw/palm",
    "tactile": f"{GLOVE_NS}/tactile",
    "tactile_binary": f"{GLOVE_NS}/tactile_binary",
    "tactile_residual": f"{GLOVE_NS}/tactile_residual",
    "tactile_cloud": f"{GLOVE_NS}/tactile_point_cloud",
    **{f"tactile_zone_{z}": f"{GLOVE_NS}/tactile_zones/{z}" for z in TACTILE_ZONES},
}


def color_topic(cam: str) -> str:
    return f"/{cam}/{cam}/color/image_raw"


def depth_topic(cam: str) -> str:
    return f"/{cam}/{cam}/aligned_depth_to_color/image_raw"


def info_topic(cam: str) -> str:
    return f"/{cam}/{cam}/color/camera_info"


def _stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _header_stamp(raw: bytes) -> float:
    # CDR: 4-byte encapsulation, then std_msgs/Header.stamp (int32 sec, uint32 nanosec).
    return int.from_bytes(raw[4:8], "little", signed=True) + int.from_bytes(raw[8:12], "little") * 1e-9


def _nearest(stamps: np.ndarray, t: float) -> Optional[int]:
    if stamps.size == 0:
        return None
    i = int(np.searchsorted(stamps, t))
    return min((j for j in (i - 1, i) if 0 <= j < stamps.size), key=lambda j: abs(stamps[j] - t))


def _cloud(msg) -> np.ndarray:
    """PointCloud2 of float32 fields only -> (width, fields)."""
    return np.frombuffer(bytes(msg.data), dtype="<f4").reshape(msg.width * msg.height, msg.point_step // 4).copy()


def _quat_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _image(msg) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc == "rgb8":
        return cv2.cvtColor(np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)
    if enc == "bgr8":
        return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
    if enc in ("16uc1", "mono16"):
        return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width)
    raise ValueError(f"unsupported image encoding '{msg.encoding}'")


def db3_path(bag: Path) -> Path:
    if bag.is_file():
        return bag
    found = sorted(bag.glob("*.db3"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one .db3 in {bag}, found {len(found)}")
    return found[0]


class _Bag:
    def __init__(self, path: Path) -> None:
        self.path = db3_path(path)
        self.con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.topics = {n: (i, t) for i, n, t in self.con.execute("select id, name, type from topics")}
        self.names = {i: n for n, (i, _) in self.topics.items()}

    def has(self, topic: str) -> bool:
        return topic in self.topics

    def stamps(self, topics: Sequence[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Header stamps and row ids, read from the first bytes of each message only."""
        out: Dict[str, Tuple[List[float], List[int]]] = {t: ([], []) for t in topics}
        ids = ",".join(str(self.topics[t][0]) for t in topics)
        for rowid, tid, head in self.con.execute(
                f"select id, topic_id, substr(data, 1, 16) from messages where topic_id in ({ids})"):
            s, r = out[self.names[tid]]
            s.append(_header_stamp(head))
            r.append(rowid)
        result = {}
        for t, (s, r) in out.items():
            order = np.argsort(s, kind="stable")
            result[t] = (np.asarray(s)[order], np.asarray(r, dtype=np.int64)[order])
        return result

    def read(self, rowid: int):
        raw, tid = self.con.execute("select data, topic_id from messages where id = ?", (int(rowid),)).fetchone()
        return TYPESTORE.deserialize_cdr(raw, self.topics[self.names[tid]][1])

    def first(self, topic: str):
        row = self.con.execute("select data from messages where topic_id = ? order by timestamp limit 1",
                               (self.topics[topic][0],)).fetchone()
        return None if row is None else TYPESTORE.deserialize_cdr(row[0], self.topics[topic][1])

    def all(self, topic: str) -> list:
        return [TYPESTORE.deserialize_cdr(raw, self.topics[topic][1]) for (raw,) in self.con.execute(
            "select data from messages where topic_id = ? order by timestamp", (self.topics[topic][0],))]


def _tf_static(bag: _Bag) -> Dict[str, Dict[str, object]]:
    out = {}
    for msg in bag.all("/tf_static") if bag.has("/tf_static") else []:
        for tr in msg.transforms:
            t, q = tr.transform.translation, tr.transform.rotation
            out[f"{tr.header.frame_id}->{tr.child_frame_id}"] = {
                "parent": tr.header.frame_id, "child": tr.child_frame_id,
                "translation": [t.x, t.y, t.z], "rotation_xyzw": [q.x, q.y, q.z, q.w]}
    return out


def default_session_name(bag: Path) -> str:
    name = bag.name if bag.is_dir() else bag.stem
    return name.removesuffix("_0").removesuffix("_recovered")


def extract(bag_path: Path, session: Optional[str] = None, every_s: float = 0.5,
            start_s: float = 0.0, end_s: Optional[float] = None, force: bool = False) -> Path:
    """Write data/<session>/ from `bag_path`; returns the session directory."""
    bag = _Bag(Path(bag_path))
    session = session or default_session_name(Path(bag_path))
    out = session_dir(session)
    if (out / "metadata.csv").is_file() and not force:
        LOGGER.info("%s exists, skipping (use --force to re-extract)", out)
        return out
    cams = [c for c in CAMERAS if bag.has(color_topic(c)) and bag.has(depth_topic(c))]
    if MASTER_CAMERA not in cams:
        raise SystemExit(f"{bag.path}: no {MASTER_CAMERA} color + aligned depth")
    glove = {k: t for k, t in GLOVE_STREAMS.items() if bag.has(t)}
    has_object = bag.has(OBJECT_POSE_TOPIC)
    marker_topics = sorted(t for t in bag.topics if t.startswith("/RigidBody/marker") and t.endswith("/pose"))

    LOGGER.info("%s: scanning header stamps", bag.path)
    image_topics = [f(c) for c in cams for f in (color_topic, depth_topic)]
    small_topics = list(glove.values()) + ([OBJECT_POSE_TOPIC] + marker_topics if has_object else [])
    stamps = bag.stamps(image_topics + small_topics)
    empty = [t for t in small_topics if stamps[t][0].size == 0]
    if empty:
        LOGGER.warning("topics with no messages (left out): %s", empty)

    master = stamps[color_topic(MASTER_CAMERA)][0]
    t0 = float(master[0])
    t_end = t0 + end_s if end_s is not None else float(master[-1])
    picks, next_t = [], t0 + start_s
    for i, t in enumerate(master):
        if next_t <= t <= t_end:
            picks.append(i)
            next_t = t + every_s

    tf_static = _tf_static(bag)
    imu_tf = tf_static.get("r_wrist->r_palm_imu_link")
    R_wrist_from_imu = _quat_matrix(imu_tf["rotation_xyzw"]) if imu_tf else None
    if "imu" in glove and R_wrist_from_imu is None:
        LOGGER.warning("no r_wrist->r_palm_imu_link in /tf_static: gravity_wrist not written")

    for cam in cams:
        for kind in ("color", "depth"):
            (out / cam / kind).mkdir(parents=True, exist_ok=True)
        info = bag.first(info_topic(cam)) if bag.has(info_topic(cam)) else None
        if info is not None:
            (out / cam / "intrinsics.json").write_text(json.dumps({
                "width": int(info.width), "height": int(info.height),
                "K": np.asarray(info.k, dtype=float).reshape(3, 3).tolist(),
                "distortion_model": info.distortion_model, "D": [float(x) for x in info.d],
                "frame_id": info.header.frame_id, "source": info_topic(cam)}, indent=2) + "\n")
    if glove:
        (out / "glove").mkdir(exist_ok=True)
    if has_object:
        (out / "object").mkdir(exist_ok=True)

    def match(topic: str, t: float, tol: float) -> Tuple[Optional[int], Optional[float]]:
        st, rows = stamps[topic]
        j = _nearest(st, t)
        if j is None or abs(st[j] - t) > tol:
            return None, None
        return int(rows[j]), float(st[j])

    rows_out, missing = [], {}
    for frame_id, i in enumerate(picks, start=1):
        t = float(master[i])
        name = frame_name(frame_id)
        row: Dict[str, object] = {"frame_id": frame_id, "master_stamp": f"{t:.6f}"}
        for cam in cams:
            ref = t
            for kind, topic in (("color", color_topic(cam)), ("depth", depth_topic(cam))):
                # depth is matched to its own camera's colour frame, colour to the master
                rowid, st = match(topic, ref, CAMERA_TOL_S)
                key = f"{cam}_{kind}"
                if rowid is None:
                    row[f"{key}_stamp"], row[f"{key}_dt_ms"] = "", ""
                    missing[key] = missing.get(key, 0) + 1
                    continue
                cv2.imwrite(str(out / cam / kind / f"{name}.png"), _image(bag.read(rowid)))
                row[f"{key}_stamp"], row[f"{key}_dt_ms"] = f"{st:.6f}", f"{(st - t) * 1e3:.2f}"
                if kind == "color":
                    ref = st

        if glove:
            g: Dict[str, np.ndarray] = {}
            for key, topic in glove.items():
                rowid, st = match(topic, t, STREAM_TOL_S)
                row[f"glove_{key}_stamp"] = "" if st is None else f"{st:.6f}"
                row[f"glove_{key}_dt_ms"] = "" if st is None else f"{(st - t) * 1e3:.2f}"
                if rowid is None:
                    missing[f"glove_{key}"] = missing.get(f"glove_{key}", 0) + 1
                    continue
                msg = bag.read(rowid)
                if key == "skeleton":
                    c = _cloud(msg)
                    g["skeleton_wrist"], g["skeleton_confidence"] = c[:, :3], c[:, 3]
                elif key in ("emf", "tips"):
                    g[f"{key}_poses" if key == "emf" else "tip_poses"] = _cloud(msg)
                elif key == "angles":
                    g["joint_angles"] = np.asarray(msg.position, dtype=np.float64)
                    g["joint_names"] = np.array(list(msg.name))
                elif key in ("tactile", "tactile_binary", "tactile_residual") or key.startswith("tactile_zone_"):
                    img = np.frombuffer(bytes(msg.data), dtype="<f4").reshape(msg.height, msg.width).copy()
                    g[key] = img.ravel() if key.startswith("tactile_zone_") else img
                elif key == "tactile_cloud":
                    c = _cloud(msg)
                    g["tactile_points"], g["tactile_pressure"] = c[:, :3], c[:, 3]
                elif key == "imu" and R_wrist_from_imu is not None:
                    a = msg.linear_acceleration
                    acc = R_wrist_from_imu @ np.array([a.x, a.y, a.z])
                    g["accel_wrist"], g["gravity_wrist"] = acc, -acc / max(np.linalg.norm(acc), 1e-9)
                g[f"{key}_stamp"] = np.float64(st)
            if g:
                np.savez(out / "glove" / f"{name}.npz", **g)

        if has_object:
            rowid, st = match(OBJECT_POSE_TOPIC, t, STREAM_TOL_S)
            row["object_stamp"] = "" if st is None else f"{st:.6f}"
            row["object_dt_ms"] = "" if st is None else f"{(st - t) * 1e3:.2f}"
            if rowid is not None:
                pose = bag.read(rowid).pose
                markers = []
                for mt in marker_topics:
                    mrow, _ = match(mt, st, STREAM_TOL_S)
                    p = bag.read(mrow).point if mrow is not None else None
                    markers.append([np.nan] * 3 if p is None else [p.x, p.y, p.z])
                np.savez(out / "object" / f"{name}.npz",
                         position=np.array([pose.position.x, pose.position.y, pose.position.z]),
                         orientation_xyzw=np.array([pose.orientation.x, pose.orientation.y,
                                                    pose.orientation.z, pose.orientation.w]),
                         marker_names=np.array([m.split("/")[2] for m in marker_topics]),
                         marker_positions=np.array(markers, dtype=np.float64).reshape(-1, 3))
            else:
                missing["object"] = missing.get("object", 0) + 1
        rows_out.append(row)
        if frame_id % 25 == 0:
            LOGGER.info("  %d/%d frames", frame_id, len(picks))

    fields = list(dict.fromkeys(k for r in rows_out for k in r))
    with (out / "metadata.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    info_msg = bag.first(f"{GLOVE_NS}/info") if bag.has(f"{GLOVE_NS}/info") else None
    color_dt = [abs(float(r[f"{c}_color_dt_ms"])) for r in rows_out for c in cams
                if c != MASTER_CAMERA and r.get(f"{c}_color_dt_ms") not in ("", None)]
    manifest = {
        "bag": str(bag.path), "session": session, "master": f"{MASTER_CAMERA} color header.stamp",
        "every_s": every_s, "start_s": start_s, "end_s": end_s,
        "camera_tol_ms": CAMERA_TOL_S * 1e3, "stream_tol_ms": STREAM_TOL_S * 1e3,
        "first_master_stamp": t0, "master_span_s": float(master[-1] - t0),
        "frames": len(rows_out), "missing": missing, "cameras": cams,
        "camera_color_dt_ms": {"median": float(np.median(color_dt)), "max": float(np.max(color_dt))} if color_dt else None,
        "topics_without_messages": empty, "tf_static": tf_static,
        "glove_info": json.loads(info_msg.data) if info_msg is not None else None,
        "calibration": None,  # set by the calib stage once a calibration is assigned
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    LOGGER.info("%s: %d frames, missing %s, other cameras vs master median %.1f ms",
                out, len(rows_out), missing or "none",
                manifest["camera_color_dt_ms"]["median"] if color_dt else float("nan"))
    return out


class Session:
    """Reader for data/<session>/."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.root = session_dir(name)
        if not (self.root / "metadata.csv").is_file():
            raise FileNotFoundError(f"{self.root} is not an extracted session (run `extract`)")
        with (self.root / "metadata.csv").open(newline="") as fh:
            self.rows = {int(r["frame_id"]): r for r in csv.DictReader(fh)}
        self.frame_ids = sorted(self.rows)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.cameras = list(self.manifest["cameras"])

    def color_path(self, cam: str, frame_id: int) -> Path:
        return self.root / cam / "color" / f"{frame_name(frame_id)}.png"

    def color(self, cam: str, frame_id: int) -> Optional[np.ndarray]:
        p = self.color_path(cam, frame_id)
        return cv2.imread(str(p), cv2.IMREAD_COLOR) if p.is_file() else None

    def depth(self, cam: str, frame_id: int) -> Optional[np.ndarray]:
        p = self.root / cam / "depth" / f"{frame_name(frame_id)}.png"
        return cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.is_file() else None

    def K(self, cam: str) -> np.ndarray:
        return np.asarray(json.loads((self.root / cam / "intrinsics.json").read_text())["K"], dtype=np.float64)

    def glove(self, frame_id: int) -> Optional[Dict[str, np.ndarray]]:
        p = self.root / "glove" / f"{frame_name(frame_id)}.npz"
        return dict(np.load(p)) if p.is_file() else None
