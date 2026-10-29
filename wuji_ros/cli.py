"""Analysis entrypoint: one subcommand per stage, thin handlers over the package modules.

Stages and their environments (README "환경"):
    extract    bag -> data/<session>/                    wuji_ros
    import     pre-sampled folder -> data/<session>/     wuji_ros
    segment   table + hand boxes from depth and RGB     wuji_ros
    keypoints  HaMeR per view, re-boxed by triangulation wuji_ros
    glovefit   G1: glove skeleton -> MANO (r_wrist)      wuji_ros
    fit        V1 per camera: wrist 6D from HaMeR+depth  wuji_ros
    calib      calibration check (table, hand agreement) wuji_ros
    fit4       V1 over all views in world                wuji_ros
    contact    held frames: fit4 hand moved onto the object  wuji_ros
    overlay    MANO mesh + skeleton in every view        wuji_ros
               (+ the mocap-posed object: overlay_object)
    tactile    glove tactile map per frame               wuji_ros
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional


def _session_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--session", required=True, help="session name under data/")
    p.add_argument("--force", action="store_true", help="regenerate existing outputs")


DEFAULT_EVERY_S = 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run.py", description="wuji_ros analysis stages")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="rosbag2 -> data/<session>/ on the camera1 color clock")
    p.add_argument("bag", type=Path, help="rosbag2 directory or its .db3")
    p.add_argument("--session", default=None, help="default: the bag directory name")
    p.add_argument("--every", type=float, default=DEFAULT_EVERY_S,
                   help="seconds between extracted master frames")
    p.add_argument("--start", type=float, default=0.0, help="seconds after the first master frame")
    p.add_argument("--end", type=float, default=None, help="seconds after the first master frame")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("import", help="pre-sampled recording folder (image_metadata.csv, camera*/, wuji/, mocap/) "
                                      "-> data/<session>/ on the camera1 color clock")
    p.add_argument("src", type=Path, help="recording folder")
    p.add_argument("--camera-config", type=Path, required=True,
                   help="yaml whose cameras[] give name, width, height and intrinsics fx, fy, cx, cy")
    p.add_argument("--session", default=None, help="default: the folder name")
    p.add_argument("--every", type=float, default=DEFAULT_EVERY_S,
                   help="seconds between imported master frames (0 = every frame)")
    p.add_argument("--start", type=float, default=0.0, help="seconds after the first master frame")
    p.add_argument("--end", type=float, default=None, help="seconds after the first master frame")
    p.add_argument("--align", choices=("index", "stamp"), default="index",
                   help="index: keep every stream of the collector's row (offsets still in metadata.csv); "
                        "stamp: leave out streams beyond 17 ms (cameras) / 10 ms (glove, mocap) of camera1 color")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("segment", help="table footprint + per-view hand boxes from calibrated depth and RGB")
    _session_args(p)

    p = sub.add_parser("keypoints", help="HaMeR per view: depth boxes, then boxes from its own triangulation")
    _session_args(p)

    p = sub.add_parser("glovefit", help="G1: fit MANO pose + shape to the glove skeleton (r_wrist frame)")
    _session_args(p)

    p = sub.add_parser("fit", help="V1 per camera: MANO(glove pose) wrist 6D from HaMeR 2D + depth")
    _session_args(p)
    p.add_argument("--glove-thickness-mm", type=float, default=0.0,
                   help="MANO surface offset along normals for the depth term (unmeasured, docs/TODO.md)")

    p = sub.add_parser("fit4", help="V1 over all views in world (needs the session's calibration)")
    _session_args(p)
    p.add_argument("--glove-thickness-mm", type=float, default=0.0)
    p.add_argument("--imu-weight", type=float, default=100.0,
                   help="weight of the glove IMU orientation term, tilt and session-yaw (0 = image-only fit; "
                        "a palm/back flip costs 2x this)")
    p.add_argument("--shape", choices=("glove", "image"), default="glove",
                   help="hand size: glove = glovefit's beta (glove bone lengths); image = one beta for the "
                        "session fitted with every frame's pose (2D + depth)")

    p = sub.add_parser("contact", help="while the object is held, move the fit4 hand (position only) by the least "
                                       "that makes it touch the mocap-posed object without penetrating it")
    _session_args(p)
    p.add_argument("--object-config", type=Path, required=True,
                   help="collector's rerun yaml (object.marker_offsets by marker name, units, mesh_scale)")
    p.add_argument("--object-mesh", type=Path, required=True, help="object mesh (STL) in the OBJ frame")

    p = sub.add_parser("calib", help="check the assigned calibration: table planes + per-camera hand agreement")
    p.add_argument("--session", required=True)

    p = sub.add_parser("tactile", help="the glove's tactile map per frame, mp4 + gif")
    _session_args(p)
    p.add_argument("--fps", type=float, default=2.0, help="playback rate in sequence order")

    p = sub.add_parser("overlay", help="draw the fitted hand (mesh + skeleton) into every full camera frame")
    _session_args(p)
    p.add_argument("--fps", type=float, default=2.0,
                   help="playback rate in sequence order (extracted frames are every --every s)")
    p.add_argument("--object-config", type=Path, default=None,
                   help="collector's rerun yaml whose object section gives marker_offsets (OBJ frame, by marker name), "
                        "marker_offsets_unit, mesh_scale, min_valid_markers; with --object-mesh the object is drawn too, "
                        "into output/<session>/overlay_object/ (needs the calibration's mocap2cam.npy)")
    p.add_argument("--object-mesh", type=Path, default=None, help="object mesh (STL) in the OBJ frame")
    p.add_argument("--hand", choices=("fit4", "contact"), default="fit4",
                   help="hand source: fit4, or the contact stage's (into overlay_contact[_object]/)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "extract":
        from .bagio import extract
        extract(args.bag, session=args.session, every_s=args.every, start_s=args.start,
                end_s=args.end, force=args.force)
    elif args.command == "import":
        from .bagio import import_dir
        import_dir(args.src, args.camera_config, session=args.session, every_s=args.every, start_s=args.start,
                   end_s=args.end, align=args.align, force=args.force)
    elif args.command == "segment":
        from .segment import run
        run(args.session, force=args.force)
    elif args.command == "keypoints":
        from .keypoints import run
        run(args.session, force=args.force)
    elif args.command == "glovefit":
        from .glovefit import run
        run(args.session, force=args.force)
    elif args.command == "fit":
        from .handfit import run
        run(args.session, glove_thickness_m=args.glove_thickness_mm / 1000.0, force=args.force)
    elif args.command == "fit4":
        from .handfit import run_world
        run_world(args.session, glove_thickness_m=args.glove_thickness_mm / 1000.0, imu_weight=args.imu_weight,
                  shape=args.shape, force=args.force)
    elif args.command == "contact":
        from .contactfit import run
        run(args.session, object_config=args.object_config, object_mesh=args.object_mesh, force=args.force)
    elif args.command == "calib":
        from .calibcheck import run
        run(args.session)
    elif args.command == "tactile":
        from .viz import run_tactile
        run_tactile(args.session, fps=args.fps, force=args.force)
    elif args.command == "overlay":
        if (args.object_config is None) != (args.object_mesh is None):
            parser.error("--object-config and --object-mesh go together")
        from .viz import run
        run(args.session, fps=args.fps, object_config=args.object_config, object_mesh=args.object_mesh,
            hand=args.hand, force=args.force)
    return 0
