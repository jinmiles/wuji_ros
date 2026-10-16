"""Glove timing check of a recorded bag (recording PC, ROS 2 Humble).

Per glove topic: delay = bag receive time - header.stamp (device sampling time, UTC),
stamp monotonicity and the largest stamp gap. Also the hand_skeleton scale in metres.
Only glove topics are read, so camera images in the same bag are skipped.

    python3.10 -m wuji_ros.bridge.bagcheck data/bags/<name>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2

TOPIC_PREFIX = "/wuji_glove/right/"
NO_HEADER = {"/wuji_glove/right/info"}
WRIST, MIDDLE_TIP = 0, 12  # MediaPipe landmark indices in hand_skeleton


def _stamp_ns(message) -> int:
    header = message.transforms[0].header if hasattr(message, "transforms") else message.header
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def check(bag: Path) -> int:
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()
             if t.name.startswith(TOPIC_PREFIX) and t.name not in NO_HEADER}
    reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(types)))
    classes = {name: get_message(kind) for name, kind in types.items()}
    stamps: Dict[str, List[int]] = {name: [] for name in types}
    recv: Dict[str, List[int]] = {name: [] for name in types}
    skeleton_span_m: Optional[float] = None
    while reader.has_next():
        topic, data, recv_ns = reader.read_next()
        message = deserialize_message(data, classes[topic])
        if hasattr(message, "transforms") and not message.transforms:
            continue
        stamps[topic].append(_stamp_ns(message))
        recv[topic].append(recv_ns)
        if topic.endswith("/hand_skeleton") and skeleton_span_m is None:
            points = point_cloud2.read_points_numpy(message, field_names=("x", "y", "z"))
            skeleton_span_m = float(np.linalg.norm(points[MIDDLE_TIP] - points[WRIST]))

    print(f"{'topic':45s} {'n':>6s} {'delay ms: median':>17s} {'p95':>7s} {'max':>7s} {'min':>7s}"
          f" {'non-incr':>8s} {'max gap ms':>10s}")
    failures = 0
    for topic in sorted(types):
        if not stamps[topic]:
            print(f"{topic[len(TOPIC_PREFIX):]:45s} {0:6d}  (no messages)")
            continue
        stamp = np.asarray(stamps[topic], dtype=np.int64)
        delay_ms = (np.asarray(recv[topic], dtype=np.int64) - stamp) * 1e-6
        steps = np.diff(stamp)
        non_increasing = int(np.sum(steps <= 0))
        max_gap_ms = float(steps.max() * 1e-6) if steps.size else 0.0
        failures += non_increasing
        print(f"{topic[len(TOPIC_PREFIX):]:45s} {stamp.size:6d} {np.median(delay_ms):17.1f}"
              f" {np.percentile(delay_ms, 95):7.1f} {delay_ms.max():7.1f} {delay_ms.min():7.1f}"
              f" {non_increasing:8d} {max_gap_ms:10.1f}")
    if skeleton_span_m is not None:
        print(f"\nhand_skeleton wrist -> middle tip, first frame: {skeleton_span_m:.3f} "
              f"(metres if about 0.15-0.21)")
    print("\nstamps monotonic on every topic" if failures == 0 else
          f"\n{failures} non-increasing stamps (see non-incr column)")
    return 0 if failures == 0 else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Glove delay / stamp / unit check of a bag.")
    parser.add_argument("bag", type=Path, help="bag directory, e.g. data/bags/subj01_take01")
    return check(parser.parse_args(argv).bag)


if __name__ == "__main__":
    sys.exit(main())
