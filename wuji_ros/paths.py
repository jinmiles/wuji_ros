"""Single source of repo paths, the data/output layout and external asset locations."""

from __future__ import annotations

import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
CALIBRATION_DIR = REPO_ROOT / "calibration"
ASSETS_DIR = REPO_ROOT / "assets"

HAMER_ROOT = REPO_ROOT / "third_party" / "hamer"
# HaMeR's _DATA layout (README "자산"); hamer.configs.CACHE_DIR_HAMER points here.
HAMER_DATA_DIR = ASSETS_DIR / "hamer"
HAMER_CHECKPOINT = HAMER_DATA_DIR / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
MANO_DIR = ASSETS_DIR / "mano"

CAMERAS = ("camera1", "camera2", "camera3", "camera4")
MASTER_CAMERA = "camera1"


def session_dir(session: str) -> Path:
    """data/<session>/: the extracted, synchronised recording."""
    return DATA_DIR / session


def stage_dir(session: str, stage: str) -> Path:
    """output/<session>/<stage>/: one stage's results, mirroring data/<session>/."""
    return OUTPUT_DIR / session / stage


def frame_name(frame_id: int) -> str:
    return f"{frame_id:06d}"


_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def get_logger(name: str = "wuji_ros") -> logging.Logger:
    """Shared analysis logger (the bridge logs through its ROS node instead)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
        logger.setLevel(logging.INFO)
    return logger
