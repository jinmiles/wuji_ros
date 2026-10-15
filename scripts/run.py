#!/usr/bin/env python
"""Analysis entrypoint (README "분석 파이프라인").

    conda run --no-capture-output -n wuji_ros python scripts/run.py extract <bag_dir> --session <name>
    conda run --no-capture-output -n wuji_ros python scripts/run.py segment --session <name>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wuji_ros.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
