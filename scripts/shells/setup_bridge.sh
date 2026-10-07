#!/usr/bin/env bash
# Recording PC: create the bridge venv (.venv-bridge) on the ROS 2 python3. Run once.
set -eo pipefail
cd "$(dirname "$0")/../.."
source scripts/shells/_ros_env.sh
python3 -m venv --system-site-packages .venv-bridge
.venv-bridge/bin/python -m pip install -r requirements-bridge.txt
.venv-bridge/bin/python -c "import wuji_sdk, rclpy, sensor_msgs_py, tf2_ros, tf2_msgs.msg; print('bridge env ok')"
