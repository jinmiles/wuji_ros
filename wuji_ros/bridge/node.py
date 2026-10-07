"""ROS 2 bridge: the right-hand Wuji Glove -> /wuji_glove/right/* for rosbag recording.

Every message carries the device's timestamp_us (end of EMF sampling, UTC once the
SDK has synced the device clock) as header.stamp, never the host receive time, so
the glove lines up with the RealSense header.stamp (capture time) on the same host
clock. Logging goes through the node's ROS logger so it lands in /rosout and in the
bag next to the data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import threading
import time
from array import array
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Image, Imu, JointState, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from tf2_msgs.msg import TFMessage
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from wuji_sdk import DeviceType, SdkManager, WujiException, WujiGlove

from . import convert

HAND_SIDE = "right"  # plan.md: right hand only
FRAME_PREFIX = "r_"
TOPIC_NS = f"/wuji_glove/{HAND_SIDE}"
NODE_NAME = "wuji_glove_bridge"
# Reliable with a deep queue: the recorder must not lose 120 Hz frames to a slow tick.
STREAM_QOS = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE)
LATCHED_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
IMU_STREAMS = tuple(f"imu_{kind}/{link}" for kind in ("raw", "data") for link in convert.IMU_LINKS)
TACTILE_GRID_STREAMS = ("tactile", "tactile_binary", "tactile_residual")
TACTILE_ZONE_STREAMS = tuple(f"tactile_zones/{zone}" for zone in convert.TACTILE_ZONES)
# Streams that should produce frames once connected; tf_static is checked through info.
EXPECTED_STREAMS = ("emf_poses", "tip_poses", "hand_skeleton", "hand_joint_angles", *IMU_STREAMS,
                    *TACTILE_GRID_STREAMS, "tactile_zones", "tactile_point_cloud", "tf")
FIRST_FRAME_WAIT_S = 5.0


class BridgeError(RuntimeError):
    """A condition under which recording must stop instead of continuing."""


def _point_fields(names: Sequence[str]) -> List[PointField]:
    return [PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1)
            for i, name in enumerate(names)]


def _float_image(header: Header, values: np.ndarray) -> Image:
    """2D float32 array -> 32FC1 image, little-endian, row-major."""
    rows, cols = values.shape
    return Image(header=header, height=rows, width=cols, encoding="32FC1", is_bigendian=0, step=4 * cols,
                 data=array("B", values.astype("<f4").tobytes()))


def _transform_message(record: Dict[str, Any]) -> TransformStamped:
    message = TransformStamped()
    message.header.stamp = _stamp(record["timestamp_us"])
    message.header.frame_id = record["parent"]
    message.child_frame_id = record["child"]
    (message.transform.translation.x, message.transform.translation.y,
     message.transform.translation.z) = record["translation"]
    (message.transform.rotation.x, message.transform.rotation.y,
     message.transform.rotation.z, message.transform.rotation.w) = record["rotation_xyzw"]
    return message


def _stamp(timestamp_us: int) -> Time:
    sec, nanosec = convert.stamp_from_us(timestamp_us)
    return Time(sec=sec, nanosec=nanosec)


def _static_key(records: List[Dict[str, Any]]) -> List[Tuple]:
    """tf_static content without its timestamps, which change on every 1 Hz republish."""
    return [(r["parent"], r["child"], tuple(r["translation"]), tuple(r["rotation_xyzw"])) for r in records]


class GloveBridge(Node):
    def __init__(self, glove: Any, glove_info: Dict[str, Any], max_clock_skew_s: float,
                 stall_timeout_s: float) -> None:
        super().__init__(NODE_NAME)
        self._glove = glove
        self._glove_info = glove_info
        self._max_clock_skew_s = max_clock_skew_s
        self._stall_timeout_s = stall_timeout_s
        self._cloud_pubs = {
            name: self.create_publisher(PointCloud2, f"{TOPIC_NS}/{name}", STREAM_QOS)
            for name in ("emf_poses", "tip_poses", "hand_skeleton", "tactile_point_cloud")
        }
        self._angles_pub = self.create_publisher(JointState, f"{TOPIC_NS}/hand_joint_angles", STREAM_QOS)
        self._imu_pubs = {
            name: self.create_publisher(Imu, f"{TOPIC_NS}/{name}", STREAM_QOS) for name in IMU_STREAMS
        }
        self._image_pubs = {
            name: self.create_publisher(Image, f"{TOPIC_NS}/{name}", STREAM_QOS)
            for name in (*TACTILE_GRID_STREAMS, *TACTILE_ZONE_STREAMS)
        }
        # The IMU-driven waist -> r_wrist transform drifts in yaw (plan.md), so it is
        # recorded under the glove namespace instead of joining the global /tf tree.
        self._tf_pub = self.create_publisher(TFMessage, f"{TOPIC_NS}/tf", STREAM_QOS)
        self._info_pub = self.create_publisher(String, f"{TOPIC_NS}/info", LATCHED_QOS)
        self._static_broadcaster = StaticTransformBroadcaster(self)
        self._static_records: Optional[List[Dict[str, Any]]] = None
        self._logged_streams: set = set()
        self._info_lock = threading.Lock()
        self._failure: Optional[str] = None
        self._last_emf_ns: Optional[int] = None
        self._sdk_subs: List[Any] = []
        self._started_ns: Optional[int] = None
        self._silence_checked = False
        self.create_timer(0.2, self._check_health)
        self._publish_info()

    # SDK callbacks run on the SDK's background threads; rclpy publishers accept that.
    def start(self) -> None:
        glove = self._glove
        on_error = self._on_sdk_error
        manager = SdkManager.instance()
        self._started_ns = time.monotonic_ns()
        self._sdk_subs = [
            glove.emf_poses().subscribe_with_callback(self._on_emf_poses, on_error),
            glove.tip_poses().subscribe_with_callback(self._on_tip_poses, on_error),
            glove.hand_skeleton().subscribe_with_callback(self._on_hand_skeleton, on_error),
            glove.hand_joint_angles().subscribe_with_callback(self._on_hand_joint_angles, on_error),
            *(getattr(glove, f"imu_{link}")().subscribe_with_callback(
                partial(self._on_imu, f"imu_raw/{link}"), on_error) for link in convert.IMU_LINKS),
            *(getattr(glove, f"imu_data_{link}")().subscribe_with_callback(
                partial(self._on_imu, f"imu_data/{link}"), on_error) for link in convert.IMU_LINKS),
            *(getattr(glove, stream)().subscribe_with_callback(
                partial(self._on_tactile_grid, stream), on_error) for stream in TACTILE_GRID_STREAMS),
            glove.tactile_zones().subscribe_with_callback(self._on_tactile_zones, on_error),
            glove.tactile_point_cloud().subscribe_with_callback(self._on_tactile_point_cloud, on_error),
            manager.tf().subscribe_with_callback(self._on_tf, on_error),
            manager.tf_static().subscribe_with_callback(self._on_tf_static, on_error),
        ]
        self.get_logger().info(f"streaming to {TOPIC_NS}/*")

    def stop(self) -> None:
        for subscription in self._sdk_subs:
            subscription.close()
        self._sdk_subs = []

    def _fail(self, reason: str) -> None:
        if self._failure is None:
            self._failure = reason

    def _check_health(self) -> None:
        last = self._last_emf_ns
        if last is not None and (time.monotonic_ns() - last) * 1e-9 > self._stall_timeout_s:
            self._fail(f"emf_poses stalled for more than {self._stall_timeout_s} s")
        started = self._started_ns
        if (not self._silence_checked and started is not None
                and (time.monotonic_ns() - started) * 1e-9 > FIRST_FRAME_WAIT_S):
            self._silence_checked = True
            silent = [stream for stream in EXPECTED_STREAMS if stream not in self._logged_streams]
            # Not fatal: tactile_binary / tactile_residual need a tactile calibration.
            if silent:
                self.get_logger().warning(f"no frame after {FIRST_FRAME_WAIT_S} s from: {silent}")
        if self._failure is not None:
            raise BridgeError(self._failure)

    def _stamp_ok(self, stream: str, timestamp_us: int, frame_id: str) -> bool:
        """False once recording has to stop; logs the first frame of each stream."""
        if self._failure is not None:
            return False
        skew = convert.clock_skew_s(timestamp_us, time.time_ns())
        if abs(skew) > self._max_clock_skew_s:
            self._fail(f"{stream}: device stamp is {skew:+.3f} s from the host clock "
                       f"(limit {self._max_clock_skew_s} s); the SDK time sync has not taken effect")
            return False
        if stream not in self._logged_streams:
            self._logged_streams.add(stream)
            self.get_logger().info(
                f"{stream}: first frame, frame_id={frame_id!r}, clock skew {skew * 1e3:+.1f} ms")
        return True

    def _header(self, stream: str, header: Any) -> Optional[Header]:
        """ROS header from the device header, or None once recording has to stop."""
        if not self._stamp_ok(stream, header.timestamp_us, header.frame_id):
            return None
        return Header(stamp=_stamp(header.timestamp_us), frame_id=header.frame_id)

    def _on_emf_poses(self, frame: Any) -> None:
        self._last_emf_ns = time.monotonic_ns()
        self._on_finger_poses("emf_poses", frame)

    def _on_tip_poses(self, frame: Any) -> None:
        self._on_finger_poses("tip_poses", frame)

    def _on_finger_poses(self, stream: str, frame: Any) -> None:
        header = self._header(stream, frame.header)
        if header is None:
            return
        try:
            points = convert.finger_pose_points(frame, stream)
        except ValueError as exc:
            self._fail(str(exc))
            return
        self._cloud_pubs[stream].publish(
            point_cloud2.create_cloud(header, _point_fields(convert.FINGER_POSE_FIELDS), points))

    def _on_hand_skeleton(self, frame: Any) -> None:
        header = self._header("hand_skeleton", frame.header)
        if header is None:
            return
        try:
            points = convert.skeleton_points(frame)
        except ValueError as exc:
            self._fail(str(exc))
            return
        if "skeleton_joint_names" not in self._glove_info:
            self._update_info(skeleton_joint_names=convert.skeleton_joint_names(frame))
        self._cloud_pubs["hand_skeleton"].publish(
            point_cloud2.create_cloud(header, _point_fields(convert.SKELETON_FIELDS), points))

    def _on_hand_joint_angles(self, frame: Any) -> None:
        header = self._header("hand_joint_angles", frame.header)
        if header is None:
            return
        try:
            positions = convert.joint_angle_positions(frame)
        except ValueError as exc:
            self._fail(str(exc))
            return
        self._angles_pub.publish(JointState(header=header, name=convert.joint_angle_names(), position=positions))

    def _on_imu(self, stream: str, frame: Any) -> None:
        header = self._header(stream, frame.header)
        if header is None:
            return
        try:
            fields = convert.imu_fields(frame)
        except ValueError as exc:
            self._fail(str(exc))
            return
        message = Imu(header=header)
        (message.orientation.x, message.orientation.y,
         message.orientation.z, message.orientation.w) = fields["orientation"]
        message.orientation_covariance = fields["orientation_covariance"]
        (message.angular_velocity.x, message.angular_velocity.y,
         message.angular_velocity.z) = fields["angular_velocity"]
        message.angular_velocity_covariance = fields["angular_velocity_covariance"]
        (message.linear_acceleration.x, message.linear_acceleration.y,
         message.linear_acceleration.z) = fields["linear_acceleration"]
        message.linear_acceleration_covariance = fields["linear_acceleration_covariance"]
        self._imu_pubs[stream].publish(message)

    def _on_tactile_grid(self, stream: str, frame: Any) -> None:
        header = self._header(stream, frame.header)
        if header is None:
            return
        try:
            grid = convert.tactile_grid(frame, stream)
        except ValueError as exc:
            self._fail(str(exc))
            return
        self._image_pubs[stream].publish(_float_image(header, grid))

    def _on_tactile_zones(self, frame: Any) -> None:
        header = self._header("tactile_zones", frame.header)
        if header is None:
            return
        zones = convert.tactile_zone_values(frame)
        if "tactile_zone_sizes" not in self._glove_info:
            self._update_info(tactile_zone_sizes={zone: len(values) for zone, values in zones.items()})
        for zone, values in zones.items():
            self._image_pubs[f"tactile_zones/{zone}"].publish(_float_image(header, values.reshape(1, -1)))

    def _on_tactile_point_cloud(self, frame: Any) -> None:
        header = self._header("tactile_point_cloud", frame.header)
        if header is None:
            return
        try:
            layout = convert.cloud_layout(frame)
        except ValueError as exc:
            self._fail(str(exc))
            return
        # The cloud carries its own frame_id next to the header's; the points are in the former.
        header.frame_id = frame.frame_id or header.frame_id
        if "tactile_point_cloud_layout" not in self._glove_info:
            self._update_info(tactile_point_cloud_layout={
                "header_frame_id": frame.header.frame_id, "cloud_frame_id": frame.frame_id,
                "point_step": layout["point_step"],
                "width": layout["width"], "fields": layout["fields"]})
        self._cloud_pubs["tactile_point_cloud"].publish(PointCloud2(
            header=header, height=1, width=layout["width"],
            fields=[PointField(name=name, offset=offset, datatype=datatype, count=1)
                    for name, offset, datatype in layout["fields"]],
            is_bigendian=False, point_step=layout["point_step"],
            row_step=layout["point_step"] * layout["width"], data=array("B", layout["data"]), is_dense=False))

    def _on_tf(self, transforms: Any) -> None:
        records = convert.side_transforms(transforms, FRAME_PREFIX)
        if not records or not self._stamp_ok("tf", records[0]["timestamp_us"], records[0]["parent"]):
            return
        self._tf_pub.publish(TFMessage(transforms=[_transform_message(record) for record in records]))

    def _on_tf_static(self, transforms: Any) -> None:
        records = convert.side_transforms(transforms, FRAME_PREFIX)
        if not records or (self._static_records is not None
                            and _static_key(records) == _static_key(self._static_records)):
            return
        self._static_records = records
        self._static_broadcaster.sendTransform([_transform_message(record) for record in records])
        self._update_info(tf_static=records)
        self.get_logger().info(f"tf_static: {[(r['parent'], r['child'], r['translation']) for r in records]}")

    def _on_sdk_error(self, message: str) -> None:
        # Not fatal by itself: a lost device shows up as a stall of emf_poses.
        self.get_logger().warning(f"SDK stream error: {message}")

    def _update_info(self, **entries: Any) -> None:
        with self._info_lock:
            self._glove_info.update(entries)
            self._publish_info()

    def _publish_info(self) -> None:
        self._info_pub.publish(String(data=json.dumps(self._glove_info, sort_keys=True)))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def connect_glove(sn: Optional[str], address: Optional[str]) -> Tuple[Any, Any]:
    """Connect the one Wuji Glove on the network (or the one with `sn`), or the glove at `address`.

    `address` skips the SDK scan, whose discovery multicast leaves through the default-route
    NIC and so misses a glove on a second, glove-only NIC.
    """
    manager = SdkManager.instance()
    if address is not None:
        try:
            return manager, manager.connect(address=address, device_name=f"glove_{HAND_SIDE}")
        except WujiException as exc:
            raise BridgeError(f"cannot connect to the glove at {address}: {exc}") from exc
    gloves = [device for device in manager.scan() if device.device_type == DeviceType.WujiGlove]
    if sn is not None:
        gloves = [device for device in gloves if device.sn == sn]
    if len(gloves) != 1:
        wanted = f" with SN {sn}" if sn is not None else ""
        raise BridgeError(f"expected exactly one Wuji Glove{wanted}, found {[d.sn for d in gloves]}")
    return manager, manager.connect(sn=gloves[0].sn, device_name=f"glove_{HAND_SIDE}")


def describe_glove(manager: Any, glove: Any) -> Dict[str, Any]:
    """Device, SDK user and hand-model provenance, published once on the latched info topic."""
    side = glove.hand_side().get()
    if side != HAND_SIDE:
        raise BridgeError(f"glove {glove.serial_number} is a {side} glove; this bridge records the {HAND_SIDE} hand")
    try:
        override = glove.hand_model_path().get()
    except WujiException:  # ParamNotSet: no custom URDF; the SDK user's model is used
        override = None
    # The offline pipeline resolves the hand URDF for this SN and SDK user the same way
    # online IK does; its answer is what an offline recomputation from emf_poses needs.
    pipeline = WujiGlove.offline_pipeline(sn=glove.serial_number, hand_side=side)
    urdf_path = pipeline.urdf_source_path
    user = manager.current_user()
    sync = glove.sync_time()
    return {
        "sn": glove.serial_number,
        "hand_side": side,
        "firmware": glove.version().get(),
        "wuji_sdk": importlib.metadata.version("wuji-sdk"),
        # user_id only: display names can be a subject's name.
        "sdk_user_id": user["user_id"],
        "sdk_user_is_default": user["is_default"],
        "hand_model_path_override": override,
        "urdf_source": pipeline.urdf_source,
        "urdf_path": urdf_path,
        "urdf_sha256": _sha256(Path(urdf_path)) if urdf_path else None,
        "time_sync": {"offset_us": sync.offset_us, "round_trip_us": sync.round_trip_us,
                      "synced_at_us": sync.synced_at_us},
        "joint_angle_names": convert.joint_angle_names(),
        "stream_fields": {"emf_poses": list(convert.FINGER_POSE_FIELDS),
                          "tip_poses": list(convert.FINGER_POSE_FIELDS),
                          "hand_skeleton": list(convert.SKELETON_FIELDS)},
        "tactile_shape": list(convert.TACTILE_SHAPE),
        # Every topic the device offers, to check the bridge against what the SDK exposes.
        "sdk_topics": sorted(topic.path for topic in glove.topics() if topic.can_sub),
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Publish the {HAND_SIDE}-hand Wuji Glove to {TOPIC_NS}/* for rosbag recording.")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--address", default=None,
                        help="glove ip:port, e.g. 192.168.1.101:50001; connects without the SDK scan")
    target.add_argument("--sn", default=None,
                        help="glove serial number; needed only when several gloves are on the network")
    parser.add_argument("--max-clock-skew", type=float, default=1.0,
                        help="stop when a device stamp is further than this from the host clock, s "
                             "(catches stamps taken before the SDK time sync)")
    parser.add_argument("--stall-timeout", type=float, default=2.0,
                        help="stop when emf_poses has not arrived for this long, s")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    args = parse_args(remove_ros_args(args=argv)[1:])
    rclpy.init(args=argv)
    logger = rclpy.logging.get_logger(NODE_NAME)
    manager, node = None, None
    try:
        manager, glove = connect_glove(args.sn, args.address)
        info = describe_glove(manager, glove)
        logger.info(f"glove: {json.dumps(info, sort_keys=True)}")
        node = GloveBridge(glove, info, args.max_clock_skew, args.stall_timeout)
        node.start()
        rclpy.spin(node)
    except BridgeError as exc:
        logger.error(str(exc))
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        if manager is not None:
            manager.disconnect_all()
        rclpy.try_shutdown()
    return 0
