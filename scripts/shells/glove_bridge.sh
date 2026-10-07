#!/usr/bin/env bash
# Recording PC: run the glove bridge. Arguments go to scripts/glove_bridge.py (e.g. --sn <SN>).
set -eo pipefail
cd "$(dirname "$0")/../.."
source /opt/ros/humble/setup.bash
exec python3.10 scripts/glove_bridge.py "$@"
