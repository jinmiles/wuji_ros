#!/usr/bin/env bash
# Recording PC: create the bridge venv (.venv-bridge) on the ROS 2 system python. Run once.
set -eo pipefail
cd "$(dirname "$0")/../.."
source scripts/shells/_ros_env.sh
"$ROS_PYTHON" -m venv --clear --system-site-packages .venv-bridge
.venv-bridge/bin/python -m pip install -r requirements-bridge.txt
.venv-bridge/bin/python -c "import wuji_sdk, rclpy, sensor_msgs_py, tf2_ros, tf2_msgs.msg; print('bridge env ok')"
