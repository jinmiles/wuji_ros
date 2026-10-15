#!/usr/bin/env bash
# Recording PC: per-topic message count and rate of data/bags/<name>, empty topics flagged.
# Usage: check_bag.sh <name>
set -eo pipefail
cd "$(dirname "$0")/../.."
name="${1:?usage: check_bag.sh <name>}"
source /opt/ros/humble/setup.bash
info="$(ros2 bag info "data/bags/${name}")"
echo "$info" | grep -E '^(Bag size|Duration|Messages):'
duration="$(echo "$info" | sed -n 's/^Duration: *\([0-9.]*\)s.*/\1/p')"
echo
printf '%-55s %9s %9s  %s\n' topic count Hz type
echo "$info" | sed -n 's/.*Topic: \([^ ]*\) | Type: \([^ ]*\) | Count: \([0-9]*\).*/\1 \3 \2/p' | sort |
  awk -v d="$duration" '{hz = d > 0 ? $2 / d : 0; flag = $2 == 0 ? "  <- EMPTY" : "";
                         printf "%-55s %9d %9.1f  %s%s\n", $1, $2, hz, $3, flag}'
echo
du -sh "data/bags/${name}"
