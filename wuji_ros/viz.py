"""Overlay and tactile views, each a per-frame image sequence plus H.264 mp4 and looping gif.

overlay  2x2 montage of the four whole camera frames (camera1 camera2 / camera3
         camera4), never cropped. On each view only the fitted hand: the MANO mesh
         (shaded, translucent) and its 21-joint skeleton, from the 4-view world fit
         (fit4) projected into every view; a session without a calibration shows
         each camera's own fit instead.
tactile  The glove's 24x31 tactile map per frame (pressure colour, grey = invalid
         taxel, white box = binary contact) with each zone's maximum, on its own,
         so it never covers the camera images.
The source filename and the numbers go in a strip under the image, not on it.

    output/<session>/overlay/NNNNNN.jpg, overlay.mp4, overlay.gif
    output/<session>/tactile/NNNNNN.jpg, tactile.mp4, tactile.gif
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from . import handfit
from .bagio import TACTILE_ZONES, Session
from .mano import BONES
from .paths import OUTPUT_DIR, frame_name, get_logger, stage_dir

LOGGER = get_logger("wuji_ros.viz")
JPEG_QUALITY = 95
GIF_WIDTH = 960  # downscale only
CRF = 18
FINGER_COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 128, 0)]  # BGR, thumb..pinky


def draw_hand(img: np.ndarray, K: np.ndarray, verts: np.ndarray, joints: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Painter's-algorithm mesh + skeleton, camera-frame points in metres."""
    def proj(X):
        uv = (K @ X.T).T
        return uv[:, :2] / uv[:, 2:]

    uv = proj(verts)
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    view = -tri.mean(1) / np.linalg.norm(tri.mean(1), axis=1, keepdims=True)
    shade = np.abs((n * view).sum(1))
    layer = img.copy()
    for fi in np.argsort(-tri[:, :, 2].mean(1)):  # far to near
        c = 80 + 175 * shade[fi]
        cv2.fillConvexPoly(layer, np.round(uv[faces[fi]] * 4).astype(np.int32), (c * 0.9, c * 0.75, c * 0.55),
                           lineType=cv2.LINE_AA, shift=2)
    out = cv2.addWeighted(layer, 0.55, img, 0.45, 0)
    j2 = proj(joints)
    for k, (p, c) in enumerate(BONES):
        cv2.line(out, tuple(np.round(j2[p]).astype(int)), tuple(np.round(j2[c]).astype(int)),
                 FINGER_COLORS[k // 4], 2, cv2.LINE_AA)
    for q in j2:
        cv2.circle(out, tuple(np.round(q).astype(int)), 3, (255, 255, 255), -1, cv2.LINE_AA)
    return out


TACTILE_SCALE = 24  # px per taxel in the tactile view


def _pressure_bgr(p: np.ndarray) -> np.ndarray:
    """(N,) pressure in [0,1] -> (N,3) BGR, OpenCV's inferno colormap."""
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_INFERNO)[:, 0]
    return lut[np.clip(np.round(p * 255), 0, 255).astype(int)]


def tactile_panel(glove: Optional[dict]) -> Optional[np.ndarray]:
    """The 24x31 tactile map (pressure colour, grey = invalid taxel, white box = binary contact)."""
    if glove is None or "tactile" not in glove:
        return None
    t = glove["tactile"]
    rgb = _pressure_bgr(np.clip(t, 0, 1).ravel()).reshape(*t.shape, 3)
    rgb[t < 0] = (90, 90, 90)
    panel = cv2.resize(rgb, (t.shape[1] * TACTILE_SCALE, t.shape[0] * TACTILE_SCALE), interpolation=cv2.INTER_NEAREST)
    if "tactile_binary" in glove:
        for r, c in zip(*np.nonzero(glove["tactile_binary"] > 0.5)):
            cv2.rectangle(panel, (c * TACTILE_SCALE, r * TACTILE_SCALE),
                          ((c + 1) * TACTILE_SCALE - 1, (r + 1) * TACTILE_SCALE - 1), (255, 255, 255), 2)
    return panel


def _strip(width: int, lines: Sequence[str]) -> np.ndarray:
    h = 34 * len(lines) + 12
    strip = np.full((h, width, 3), 255, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(strip, text, (14, 34 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return strip


def write_media(frames: List[Path], mp4: Path, gif: Path, fps: float) -> None:
    """H.264 (yuv444p, lossless PNG in) and a palette GIF, regenerated from the stills on disk."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        LOGGER.warning("ffmpeg not found: skipping %s / %s", mp4.name, gif.name)
        return
    h, w = cv2.imread(str(frames[0])).shape[:2]
    w2, h2 = w - w % 2, h - h % 2

    def feed(cmd):
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for p in frames:
            img = cv2.imread(str(p))[:h2, :w2]
            proc.stdin.write(cv2.imencode(".png", img)[1].tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}")

    src = [ffmpeg, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "png", "-i", "-"]
    feed(src + ["-c:v", "libx264", "-pix_fmt", "yuv444p", "-crf", str(CRF), "-preset", "slow", str(mp4)])
    gw = min(GIF_WIDTH, w2)
    feed(src + ["-vf", f"scale={gw}:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                "-loop", "0", str(gif)])
    LOGGER.info("%d frames -> %s, %s", len(frames), mp4.name, gif.name)


def run(session: str, fps: float = 2.0, force: bool = False) -> None:
    from .mano import Mano

    s = Session(session)
    out = stage_dir(session, "overlay")
    out.mkdir(parents=True, exist_ok=True)
    from .rig import rig_for

    faces = Mano("cpu").faces
    Ks = {c: s.K(c) for c in s.cameras}
    rig = rig_for(session)
    written = []
    for f in s.frame_ids:
        target = out / f"{frame_name(f)}.jpg"
        written.append(target)
        if target.is_file() and not force:
            continue
        tiles, notes = [], []
        world = handfit.load_world(session, f) if rig is not None else None
        for cam in s.cameras:
            img = s.color(cam, f)
            if img is None:
                tiles.append(np.zeros((720, 1280, 3), np.uint8))
                notes.append(f"{cam}: no frame")
                continue
            if world is not None:
                tiles.append(draw_hand(img, Ks[cam], rig.to_cam(cam, world["verts_world"]),
                                       rig.to_cam(cam, world["joints_world"]), faces))
                views = list(world["views"])
                notes.append(f"{cam}: reproj {np.median(world['reproj_px'][views.index(cam)]):.1f}px"
                             if cam in views else f"{cam}: no HaMeR")
                continue
            fit = handfit.load(session, cam, f) if rig is None else None  # calibrated: only the 4-view fit
            if fit is None:
                tiles.append(img)
                notes.append(f"{cam}: no fit")
                continue
            tiles.append(draw_hand(img, Ks[cam], fit["verts_cam"], fit["joints_cam"], faces))
            notes.append(f"{cam}: reproj {np.median(fit['reproj_px']):.1f}px  depth {float(fit['surface_mm_median']):.1f}mm"
                         f"  tilt {float(fit['gravity_err_deg']):.0f}deg")
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
        t = float(s.rows[f]["master_stamp"]) - float(s.rows[s.frame_ids[0]]["master_stamp"])
        source = (f"4-view world fit, calibration {rig.name}" if world is not None
                  else "no 4-view fit for this frame" if rig is not None else "per-camera fit")
        lines = [f"{session}  frame {frame_name(f)}.png   t={t:.1f}s   MANO mesh + skeleton: glove fingers, posed by {source}",
                 "   |   ".join(notes)]
        if world is not None:
            tilt = float(world["gravity_err_deg"])
            lines.append(f"depth {float(world['surface_mm_median']):.1f}mm   IMU tilt vs fit {tilt:.0f}deg"
                         + ("   ** IMU disagrees > 90deg: palm/back flip suspected **" if tilt > 90 else ""))
        cv2.imwrite(str(target), np.vstack([grid, _strip(grid.shape[1], lines)]), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    write_media(written, OUTPUT_DIR / session / "overlay.mp4", OUTPUT_DIR / session / "overlay.gif", fps)


def _zone_bars(glove: dict, width: int, height: int) -> np.ndarray:
    """Max pressure per zone as horizontal bars, 0-1."""
    img = np.full((height, width, 3), 255, np.uint8)
    row = height // (len(TACTILE_ZONES) + 1)
    x0, x1 = 150, width - 90
    for i, z in enumerate(TACTILE_ZONES):
        v = glove.get(f"tactile_zone_{z}")
        m = float(v[v >= 0].max()) if v is not None and (v >= 0).any() else float("nan")
        y = row * (i + 1)
        cv2.putText(img, z, (12, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y - 14), (x1, y + 14), (210, 210, 210), 1)
        if np.isfinite(m):
            color = tuple(int(c) for c in _pressure_bgr(np.array([m]))[0])
            cv2.rectangle(img, (x0, y - 14), (x0 + int((x1 - x0) * m), y + 14), color, -1)
            cv2.putText(img, f"{m:.2f}", (x1 + 10, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def run_tactile(session: str, fps: float = 2.0, force: bool = False) -> None:
    s = Session(session)
    out = stage_dir(session, "tactile")
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for f in s.frame_ids:
        glove = s.glove(f)
        panel = tactile_panel(glove)
        if panel is None:
            continue
        target = out / f"{frame_name(f)}.jpg"
        written.append(target)
        if target.is_file() and not force:
            continue
        bars = _zone_bars(glove, 640, panel.shape[0])
        img = np.hstack([panel, np.full((panel.shape[0], 20, 3), 255, np.uint8), bars])
        t = float(s.rows[f]["master_stamp"]) - float(s.rows[s.frame_ids[0]]["master_stamp"])
        n_contact = int((glove.get("tactile_binary", np.zeros(1)) > 0.5).sum())
        lines = [f"{session}  frame {frame_name(f)}.png   t={t:.1f}s",
                 f"tactile 24x31: colour = pressure 0-1, grey = invalid taxel, white box = binary contact ({n_contact})"]
        cv2.imwrite(str(target), np.vstack([img, _strip(img.shape[1], lines)]), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if written:
        write_media(written, OUTPUT_DIR / session / "tactile.mp4", OUTPUT_DIR / session / "tactile.gif", fps)
