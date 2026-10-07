#!/usr/bin/env bash
# Recording PC: install wuji-sdk for ROS 2 Humble's python3.10 (user site, no venv). Run once.
# rclpy is built for python3.10; plain python3 fails with "No module named
# 'rclpy._rclpy_pybind11'" (same fix as mocap_ros_py e94e059).
set -eo pipefail
cd "$(dirname "$0")/../.."
source /opt/ros/humble/setup.bash
python3.10 -m pip install --user -r requirements-bridge.txt
python3.10 -c "import wuji_sdk, rclpy, sensor_msgs_py, tf2_ros, tf2_msgs.msg; print('bridge env ok')"
