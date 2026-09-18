#!/usr/bin/env python3
"""Publish the right-hand Wuji Glove to ROS 2 (recording PC). See README "녹화 PC"."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wuji_ros.bridge.node import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
