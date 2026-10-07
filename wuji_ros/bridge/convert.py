"""Wuji SDK frames -> plain values for ROS 2 messages.

Nothing here imports ROS or the SDK: the functions only read the SDK objects'
attributes (see third_party/wuji-sdk and the wuji_sdk type stubs), so the
conversion can be tested on a machine that has neither.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

SKELETON_JOINTS = 21  # MediaPipe landmark order, same index order as OpenPose-21
EMF_FINGERS = 5  # thumb, index, middle, ring, pinky
SKELETON_FIELDS: Tuple[str, ...] = ("x", "y", "z", "confidence")
# emf_poses and tip_poses: one row per finger, thumb..pinky.
FINGER_POSE_FIELDS: Tuple[str, ...] = ("x", "y", "z", "qx", "qy", "qz", "qw", "confidence")
IMU_LINKS: Tuple[str, ...] = ("palm", "thumb", "index", "middle", "ring", "pinky")
# tactile, tactile_binary and tactile_residual: 744 taxels, row-major 24x31, -1.0 = invalid taxel.
TACTILE_SHAPE: Tuple[int, int] = (24, 31)
TACTILE_ZONES: Tuple[str, ...] = ("palm", "thumb", "index", "middle", "ring", "pinky")
TACTILE_CLOUD_POINTS = 526  # active taxels of the 24x31 grid (SDK changelog, tactile_point_cloud contract)
# Byte size per sensor_msgs/PointField datatype (INT8=1 .. FLOAT64=8).
POINT_FIELD_SIZES: Dict[int, int] = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8}

FINGERS: Tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
# Used slots of HandJointAngles.fingers[i].angles: the thumb uses all five, the
# other fingers four, and their fifth slot is zero padding. The names below follow
# this SDK slot order; the SDK does not name the slots anatomically.
ANGLE_SLOTS: Tuple[int, ...] = (5, 4, 4, 4, 4)


def stamp_from_us(timestamp_us: int) -> Tuple[int, int]:
    """Device timestamp (UTC microseconds) -> (sec, nanosec) of builtin_interfaces/Time."""
    sec, rem_us = divmod(int(timestamp_us), 1_000_000)
    return sec, rem_us * 1000


def clock_skew_s(timestamp_us: int, host_now_ns: int) -> float:
    """Device stamp minus host wall clock in seconds.

    Before the SDK's time sync the device stamps its uptime instead of UTC, which
    shows up here as a skew of decades.
    """
    return (int(timestamp_us) * 1000 - int(host_now_ns)) * 1e-9


def skeleton_points(skeleton: Any) -> np.ndarray:
    """HandSkeleton -> (21, 4) float32 [x, y, z, confidence] in metres, in header.frame_id."""
    joints = skeleton.joints
    if len(joints) != SKELETON_JOINTS:
        raise ValueError(f"hand_skeleton has {len(joints)} joints, expected {SKELETON_JOINTS}")
    return np.array([[*j.pose.position, j.confidence] for j in joints], dtype=np.float32)


def skeleton_joint_names(skeleton: Any) -> List[str]:
    return [joint.name for joint in skeleton.joints]


def finger_pose_points(frame: Any, stream: str) -> np.ndarray:
    """EmfPoseArray / FingertipPoses -> (5, 8) float32 [x, y, z, qx, qy, qz, qw, confidence], thumb..pinky."""
    poses = frame.poses
    if len(poses) != EMF_FINGERS:
        raise ValueError(f"{stream} has {len(poses)} poses, expected {EMF_FINGERS}")
    rows = []
    for finger_pose in poses:
        q = finger_pose.pose.orientation
        rows.append([*finger_pose.pose.position, q.x, q.y, q.z, q.w, finger_pose.confidence])
    return np.array(rows, dtype=np.float32)


def joint_angle_names() -> List[str]:
    return [f"{finger}_{slot}" for finger, used in zip(FINGERS, ANGLE_SLOTS) for slot in range(used)]


def joint_angle_positions(angles: Any) -> List[float]:
    """HandJointAngles -> 21 angles in radians, in joint_angle_names() order."""
    fingers = angles.fingers
    if len(fingers) != len(FINGERS):
        raise ValueError(f"hand_joint_angles has {len(fingers)} fingers, expected {len(FINGERS)}")
    return [float(value) for finger, used in zip(fingers, ANGLE_SLOTS) for value in finger.angles[:used]]


def imu_fields(imu: Any) -> Dict[str, Any]:
    """ImuData -> sensor_msgs/Imu field values, passed through unchanged."""
    q = imu.orientation
    w = imu.angular_velocity
    a = imu.linear_acceleration
    fields = {
        "orientation": (q.x, q.y, q.z, q.w),
        "orientation_covariance": [float(v) for v in imu.orientation_covariance],
        "angular_velocity": (w.x, w.y, w.z),
        "angular_velocity_covariance": [float(v) for v in imu.angular_velocity_covariance],
        "linear_acceleration": (a.x, a.y, a.z),
        "linear_acceleration_covariance": [float(v) for v in imu.linear_acceleration_covariance],
    }
    for name, value in fields.items():
        if name.endswith("covariance") and len(value) != 9:
            raise ValueError(f"imu {name} has {len(value)} entries, sensor_msgs/Imu needs 9")
    return fields


def tactile_grid(frame: Any, stream: str) -> np.ndarray:
    """TactileFrame / TactileBinary / TactileResidual -> (24, 31) float32, values unchanged."""
    data = frame.data
    rows, cols = TACTILE_SHAPE
    if len(data) != rows * cols:
        raise ValueError(f"{stream} has {len(data)} taxels, expected {rows}x{cols}")
    return np.asarray(data, dtype=np.float32).reshape(rows, cols)


def tactile_zone_values(zones: Any) -> Dict[str, np.ndarray]:
    """TactileZones -> {zone: (n,) float32} for TACTILE_ZONES, values unchanged."""
    return {zone: np.asarray(getattr(zones, zone), dtype=np.float32) for zone in TACTILE_ZONES}


def cloud_layout(cloud: Any) -> Dict[str, Any]:
    """SDK PointCloud -> PointCloud2 layout over the unchanged byte payload.

    The SDK PointField.type is taken as the sensor_msgs/PointField datatype code; the
    layout check below rejects codes or offsets that cannot be that encoding.
    """
    stride = int(cloud.point_stride)
    data = bytes(cloud.data)
    fields = [(field.name, int(field.offset), int(field.type)) for field in cloud.fields]
    for name, offset, datatype in fields:
        size = POINT_FIELD_SIZES.get(datatype)
        if size is None or offset + size > stride:
            raise ValueError(f"tactile_point_cloud field {name!r} (offset {offset}, type {datatype}) "
                             f"does not fit a {stride}-byte point")
    if stride <= 0 or len(data) % stride:
        raise ValueError(f"tactile_point_cloud has {len(data)} bytes, not a multiple of point_stride {stride}")
    width = len(data) // stride
    if width != TACTILE_CLOUD_POINTS:
        raise ValueError(f"tactile_point_cloud has {width} points, expected {TACTILE_CLOUD_POINTS}")
    return {"fields": fields, "point_step": stride, "width": width, "data": data}


def side_transforms(transforms: Any, frame_prefix: str) -> List[Dict[str, Any]]:
    """FrameTransforms -> this glove's transforms, sorted by child frame.

    tf and tf_static are global SDK topics merged over every connected device, so only
    the transforms with this glove's side prefix ("r_") on either frame are kept
    (tf_static: r_wrist -> r_*, tf: waist -> r_wrist).
    """
    records = []
    for transform in transforms.transforms:
        if not (transform.parent_frame_id.startswith(frame_prefix)
                or transform.child_frame_id.startswith(frame_prefix)):
            continue
        q = transform.rotation
        records.append({
            "parent": transform.parent_frame_id,
            "child": transform.child_frame_id,
            "translation": [float(v) for v in transform.translation],
            "rotation_xyzw": [q.x, q.y, q.z, q.w],
            "timestamp_us": int(transform.timestamp_us),
        })
    return sorted(records, key=lambda record: record["child"])
