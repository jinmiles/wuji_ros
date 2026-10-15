#!/usr/bin/env bash
# Recording PC: record the glove into data/bags/<name>.
# Usage:
#   record_glove.sh <name> [extra topics...]   glove topics + /tf_static (+ the named topics)
#   record_glove.sh <name> --all               every topic on the ROS graph, including topics
#                                              that appear while recording, except the
#                                              image_transport duplicates (compressed, theora)
# Humble's `ros2 bag record` applies -e only to the topics listed by name when both are
# given, so the named topics go into one regex instead.
set -eo pipefail
cd "$(dirname "$0")/../.."
name="${1:?usage: record_glove.sh <name> [--all | extra topics...]}"
shift
source /opt/ros/humble/setup.bash
mkdir -p data/bags
if [[ "${1:-}" == --all ]]; then
  exec ros2 bag record -o "data/bags/${name}" -a -x '.*/(compressed|compressedDepth|theora)$'
fi
regex='^/wuji_glove/right/.*|^/tf_static$'
for topic in "$@"; do regex+="|^${topic}\$"; done
exec ros2 bag record -o "data/bags/${name}" -e "$regex"
