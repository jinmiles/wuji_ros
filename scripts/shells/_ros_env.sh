# Sourced by the recording PC scripts: use the ROS 2 already sourced in this shell,
# otherwise the single install under /opt/ros. ROS setup.bash breaks under `set -u`.
if [[ -z "${ROS_DISTRO:-}" ]]; then
  ros_setups=(/opt/ros/*/setup.bash)
  if [[ ${#ros_setups[@]} -ne 1 || ! -f "${ros_setups[0]}" ]]; then
    echo "ROS 2 not found or ambiguous (${ros_setups[*]}); source your ROS 2 setup.bash first" >&2
    exit 1
  fi
  set +u
  source "${ros_setups[0]}"
fi
# rclpy's C extension is built for the system python that ROS was built with; a conda or
# pyenv python3 earlier on PATH cannot load it (No module named 'rclpy._rclpy_pybind11').
ROS_PYTHON="${ROS_PYTHON:-/usr/bin/python3}"
if ! "$ROS_PYTHON" -c "import rclpy" 2>/dev/null; then
  echo "rclpy does not import with ${ROS_PYTHON} ($("$ROS_PYTHON" --version 2>&1)); set ROS_PYTHON to the python ROS 2 ${ROS_DISTRO} was built for" >&2
  exit 1
fi
echo "ROS 2 ${ROS_DISTRO}, ${ROS_PYTHON} ($("$ROS_PYTHON" --version 2>&1))" >&2
