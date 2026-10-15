#!/usr/bin/env bash
# Recording PC: record every glove topic and /tf_static into data/bags/<name>.
# Usage: record_glove.sh <name> [extra topics...]   (e.g. camera and mocap topics)
# Humble's `ros2 bag record` applies -e only to the topics listed by name when both are
# given, so every topic goes into one regex instead.
set -eo pipefail
cd "$(dirname "$0")/../.."
name="${1:?usage: record_glove.sh <name> [extra topics...]}"
shift
regex='^/wuji_glove/right/.*|^/tf_static$'
for topic in "$@"; do regex+="|^${topic}\$"; done
source /opt/ros/humble/setup.bash
mkdir -p data/bags
exec ros2 bag record -o "data/bags/${name}" -e "$regex"
