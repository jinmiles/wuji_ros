#!/usr/bin/env bash
# Recording PC: create the bridge venv (.venv-bridge) on ROS 2 Humble's python3.10. Run once.
set -euo pipefail
cd "$(dirname "$0")/../.."
source /opt/ros/humble/setup.bash
python3.10 -m venv --system-site-packages .venv-bridge
.venv-bridge/bin/python -m pip install -r requirements-bridge.txt
.venv-bridge/bin/python -c "import wuji_sdk, rclpy, sensor_msgs_py, tf2_ros, tf2_msgs.msg; print('bridge env ok')"
