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
echo "ROS 2 ${ROS_DISTRO}, $(python3 --version)" >&2
