#!/usr/bin/env bash
# Analysis server or robot room PC: one recording folder -> the contact hand and every overlay (ANNOTATION.md).
# Usage:
#   ./annotate.sh <recording folder> [--force] [--calibration NAME] [--camera-config YAML] [--object-mesh STL]
# The yaml and the STL default to the only *.yaml / *.stl next to the folder. Outputs: output/<folder name>/.
set -eo pipefail
: "${1:?usage: annotate.sh <recording folder> [--force] [--calibration NAME] [--camera-config YAML] [--object-mesh STL]}"
# The conda env's packages only: python3.10 would also load the recording PC's user-site bridge packages
# (~/.local/lib/python3.10, ahead of the env) and a sourced ROS 2 PYTHONPATH.
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 conda run --no-capture-output -n wuji_ros python "$(dirname "$0")/scripts/run.py" annotate "$@"
