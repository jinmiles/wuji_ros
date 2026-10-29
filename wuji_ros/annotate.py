"""One recording folder -> every stage up to the contact hand, and every overlay (ANNOTATION.md).

    import -> segment -> glovefit -> keypoints -> fit4 -> contact
           -> overlay (fit4 hand), overlay_object (+ object), overlay_contact_object (contact hand + object), tactile

The session is the folder's name. Unless given, the collector's rerun yaml (camera intrinsics and the
object's marker offsets) and the object mesh are the only *.yaml and *.stl next to the folder (its parent).
Each stage keeps its outputs unless force.

The calibration is never estimated. It is, first that applies:
  - `calibration`;
  - the session's rig.SESSION_CALIBRATION entry;
  - the calibration whose w2c_ext.npz is byte-identical (sha256) to a w2c_ext.npz delivered with the
    recording (in the folder or its parent).
It is written to data/<session>/manifest.json ("calibration", "calibration_source"), which rig.rig_for reads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

from .paths import CALIBRATION_DIR, get_logger, session_dir

LOGGER = get_logger("wuji_ros.annotate")


def _only(folder: Path, pattern: str, option: str) -> Path:
    found = sorted(folder.glob(pattern))
    if len(found) != 1:
        raise SystemExit(f"{folder}: {len(found)} files match {pattern} ({[p.name for p in found]}); pass {option}")
    return found[0]


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def resolve_calibration(session: str, src: Path, name: Optional[str]) -> Tuple[str, str]:
    """(calibration name, how it was established), in the module docstring's order."""
    from .rig import SESSION_CALIBRATION

    fixed = SESSION_CALIBRATION.get(session)
    if name is not None:
        if not (CALIBRATION_DIR / name / "w2c_ext.npz").is_file():
            raise SystemExit(f"calibration/{name}/w2c_ext.npz does not exist")
        if fixed is not None and fixed != name:
            raise SystemExit(f"{session} is assigned {fixed} in rig.SESSION_CALIBRATION, not {name}")
        return name, "--calibration"
    if fixed is not None:
        return fixed, "rig.SESSION_CALIBRATION"
    known = {_sha256(p): p.parent.name for p in sorted(CALIBRATION_DIR.glob("*/w2c_ext.npz"))}
    for p in (src / "w2c_ext.npz", src.parent / "w2c_ext.npz"):
        if p.is_file() and _sha256(p) in known:
            return known[_sha256(p)], f"{p} is byte-identical to calibration/{known[_sha256(p)]}/w2c_ext.npz"
    raise SystemExit(f"{session}: no calibration. No w2c_ext.npz delivered with the recording matches calibration/*; "
                     "pass --calibration <name> or add the session to rig.SESSION_CALIBRATION (never estimated)")


def run(src: Path, camera_config: Optional[Path] = None, object_mesh: Optional[Path] = None,
        calibration: Optional[str] = None, every_s: float = 0.5, force: bool = False) -> None:
    from . import contactfit, glovefit, handfit, keypoints, segment, viz
    from .bagio import import_dir

    src = Path(src)
    if not (src / "image_metadata.csv").is_file():
        raise SystemExit(f"{src} is not a recording folder (no image_metadata.csv)")
    camera_config = camera_config or _only(src.resolve().parent, "*.yaml", "--camera-config")
    object_mesh = object_mesh or _only(src.resolve().parent, "*.stl", "--object-mesh")
    session = src.resolve().name
    name, how = resolve_calibration(session, src.resolve(), calibration)
    LOGGER.info("%s: camera/object config %s, object mesh %s, calibration %s (%s)", session, camera_config,
                object_mesh, name, how)

    import_dir(src, camera_config, session=session, every_s=every_s, force=force)
    manifest_path = session_dir(session) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(calibration=name, calibration_source=how)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    segment.run(session, force=force)
    glovefit.run(session, force=force)
    keypoints.run(session, force=force)
    handfit.run_world(session, force=force)
    contactfit.run(session, object_config=camera_config, object_mesh=object_mesh, force=force)
    viz.run(session, force=force)
    viz.run(session, object_config=camera_config, object_mesh=object_mesh, force=force)
    viz.run(session, object_config=camera_config, object_mesh=object_mesh, hand="contact", force=force)
    viz.run_tactile(session, force=force)
