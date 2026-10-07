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
# rclpy's C extension is built for one python version (Humble: 3.10). Any other python3
# on PATH (conda, pyenv, a newer system python) cannot load it:
# "No module named 'rclpy._rclpy_pybind11'" (same fix as mocap_ros_py e94e059).
# The version is read from where rclpy is installed, e.g. /opt/ros/humble/lib/python3.10/.
if [[ -z "${ROS_PYTHON:-}" ]]; then
  rclpy_dirs=(/opt/ros/"${ROS_DISTRO}"/lib/python3.*/site-packages/rclpy)
  if [[ ${#rclpy_dirs[@]} -eq 1 && -d "${rclpy_dirs[0]}" ]]; then
    ros_py_version="${rclpy_dirs[0]#/opt/ros/${ROS_DISTRO}/lib/}"
    ROS_PYTHON="/usr/bin/${ros_py_version%%/*}"
  else
    ROS_PYTHON=/usr/bin/python3
  fi
fi
if ! "$ROS_PYTHON" -c "import rclpy" 2>/dev/null; then
  echo "rclpy does not import with ${ROS_PYTHON} ($("$ROS_PYTHON" --version 2>&1)); set ROS_PYTHON to the python ROS 2 ${ROS_DISTRO} was built for" >&2
  exit 1
fi
echo "ROS 2 ${ROS_DISTRO}, ${ROS_PYTHON} ($("$ROS_PYTHON" --version 2>&1))" >&2
