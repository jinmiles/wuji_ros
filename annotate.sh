#!/usr/bin/env bash
# Analysis server: one recording folder -> the contact hand and every overlay (ANNOTATION.md).
# Usage:
#   ./annotate.sh <recording folder> [--force] [--calibration NAME] [--camera-config YAML] [--object-mesh STL]
# The yaml and the STL default to the only *.yaml / *.stl next to the folder. Outputs: output/<folder name>/.
set -eo pipefail
: "${1:?usage: annotate.sh <recording folder> [--force] [--calibration NAME] [--camera-config YAML] [--object-mesh STL]}"
exec conda run --no-capture-output -n wuji_ros python "$(dirname "$0")/scripts/run.py" annotate "$@"
