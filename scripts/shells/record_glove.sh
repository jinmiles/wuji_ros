#!/usr/bin/env bash
# Recording PC: record every glove topic and /tf_static into data/bags/<name>.
# Usage: record_glove.sh <name> [extra topics...]   (e.g. camera and mocap topics)
set -eo pipefail
cd "$(dirname "$0")/../.."
name="${1:?usage: record_glove.sh <name> [extra topics...]}"
shift
source /opt/ros/humble/setup.bash
mkdir -p data/bags
exec ros2 bag record -o "data/bags/${name}" -e '^/wuji_glove/right/.*' /tf_static "$@"
