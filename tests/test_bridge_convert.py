"""Contract of wuji_ros.bridge.convert: shapes, field order, units and stamps of the
glove streams as they will sit in the bag. Uses stand-ins with the attributes of the
wuji_sdk types, so it runs without ROS or the SDK."""

import unittest
from types import SimpleNamespace as NS

import numpy as np

from wuji_ros.bridge import convert


def quat(x, y, z, w):
    return NS(x=x, y=y, z=z, w=w)


def pose(position, orientation=None):
    return NS(position=list(position), orientation=orientation or quat(0.0, 0.0, 0.0, 1.0))


def header(timestamp_us, frame_id):
    return NS(seq=1, timestamp_us=timestamp_us, frame_id=frame_id)


class StampTest(unittest.TestCase):
    def test_stamp_splits_microseconds(self):
        self.assertEqual(convert.stamp_from_us(1_788_155_500_514_286), (1_788_155_500, 514_286_000))

    def test_clock_skew_detects_uptime_stamps(self):
        host_now_ns = 1_788_155_500 * 10**9
        self.assertAlmostEqual(convert.clock_skew_s(1_788_155_500_004_000, host_now_ns), 0.004, places=9)
        uptime_us = 3_600 * 10**6  # one hour after power-on, before time sync
        self.assertLess(convert.clock_skew_s(uptime_us, host_now_ns), -1e9)


class SkeletonTest(unittest.TestCase):
    def test_points_keep_joint_order_and_confidence(self):
        joints = [NS(name=f"j{i}", pose=pose((i * 0.01, -i * 0.02, i * 0.03)), confidence=0.5 + i * 0.01)
                  for i in range(21)]
        skeleton = NS(header=header(0, "r_wrist"), joints=joints)
        points = convert.skeleton_points(skeleton)
        self.assertEqual(points.shape, (21, 4))
        self.assertEqual(points.dtype, np.float32)
        np.testing.assert_allclose(points[7], [0.07, -0.14, 0.21, 0.57], rtol=1e-6)
        self.assertEqual(convert.skeleton_joint_names(skeleton)[20], "j20")

    def test_wrong_joint_count_is_rejected(self):
        skeleton = NS(joints=[NS(name="j", pose=pose((0, 0, 0)), confidence=1.0)] * 20)
        with self.assertRaises(ValueError):
            convert.skeleton_points(skeleton)


class EmfTest(unittest.TestCase):
    def test_points_are_position_quaternion_xyzw_confidence(self):
        poses = [NS(pose=pose((f, 0.1, -0.05), quat(0.1 * f, 0.2, 0.3, 0.9)), confidence=0.95)
                 for f in range(5)]
        points = convert.emf_points(NS(header=header(0, "r_hand_emf_tx"), poses=poses))
        self.assertEqual(points.shape, (5, 8))
        self.assertEqual(convert.EMF_FIELDS, ("x", "y", "z", "qx", "qy", "qz", "qw", "confidence"))
        np.testing.assert_allclose(points[2], [2, 0.1, -0.05, 0.2, 0.2, 0.3, 0.9, 0.95], rtol=1e-6)


class JointAngleTest(unittest.TestCase):
    def test_padding_slot_of_long_fingers_is_dropped(self):
        fingers = [NS(angles=[10 * f + k for k in range(5)], confidence=1.0) for f in range(5)]
        positions = convert.joint_angle_positions(NS(fingers=fingers))
        names = convert.joint_angle_names()
        self.assertEqual(len(positions), 21)
        self.assertEqual(len(names), 21)
        self.assertEqual(positions[:6], [0, 1, 2, 3, 4, 10])
        self.assertEqual(names[:6], ["thumb_0", "thumb_1", "thumb_2", "thumb_3", "thumb_4", "index_0"])
        self.assertEqual(names[-1], "pinky_3")
        self.assertEqual(positions[-1], 43)


class ImuTest(unittest.TestCase):
    def test_fields_pass_through(self):
        imu = NS(orientation=quat(0.0, 0.0, 0.7071, 0.7071), orientation_covariance=[0.0] * 9,
                 angular_velocity=NS(x=0.1, y=0.2, z=0.3), angular_velocity_covariance=[0.0] * 9,
                 linear_acceleration=NS(x=0.0, y=0.0, z=9.8), linear_acceleration_covariance=[0.0] * 9)
        fields = convert.imu_fields(imu)
        self.assertEqual(fields["orientation"], (0.0, 0.0, 0.7071, 0.7071))
        self.assertEqual(fields["linear_acceleration"], (0.0, 0.0, 9.8))
        self.assertEqual(len(fields["angular_velocity_covariance"]), 9)

    def test_short_covariance_is_rejected(self):
        imu = NS(orientation=quat(0.0, 0.0, 0.0, 1.0), orientation_covariance=[-1.0],
                 angular_velocity=NS(x=0.0, y=0.0, z=0.0), angular_velocity_covariance=[0.0] * 9,
                 linear_acceleration=NS(x=0.0, y=0.0, z=9.8), linear_acceleration_covariance=[0.0] * 9)
        with self.assertRaises(ValueError):
            convert.imu_fields(imu)


class TfStaticTest(unittest.TestCase):
    def test_keeps_only_this_side(self):
        def transform(parent, child, translation):
            return NS(timestamp_us=5, parent_frame_id=parent, child_frame_id=child,
                      translation=translation, rotation=quat(0.0, 0.0, 0.0, 1.0))

        merged = NS(transforms=[
            transform("r_wrist", "r_palm_imu_link", [0.0, -0.01, -0.05]),
            transform("l_wrist", "l_hand_emf_tx", [0.0, 0.02, -0.06]),
            transform("r_wrist", "r_hand_emf_tx", [0.0, 0.02, -0.06]),
        ])
        records = convert.side_transforms(merged, "r_")
        self.assertEqual([r["child"] for r in records], ["r_hand_emf_tx", "r_palm_imu_link"])
        self.assertEqual(records[0]["rotation_xyzw"], [0.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
