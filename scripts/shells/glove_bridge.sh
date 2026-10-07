#!/usr/bin/env bash
# Recording PC: run the glove bridge. Arguments go to scripts/glove_bridge.py (e.g. --sn <SN>).
set -eo pipefail
cd "$(dirname "$0")/../.."
source scripts/shells/_ros_env.sh
exec .venv-bridge/bin/python scripts/glove_bridge.py "$@"
