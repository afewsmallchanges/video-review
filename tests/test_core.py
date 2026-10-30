"""Checks for the video core (no Qt needed).

Builds test clips whose frame number is drawn as 16 black/white blocks, encodes
them like normal camera/broadcast footage (long GOP, B-frames), imports them,
then confirms every random seek and backward step returns the right frame.

Run:  python -m tests.test_core
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from replaydesk import ingest  # noqa: E402
from replaydesk.frames import FrameSource  # noqa: E402

BITS, BLOCK = 16, 32


def make_clip(path: Path, w: int, h: int, fps: str, n_frames: int):
    """Encode a long-GOP H.264 clip with the frame index drawn in the corner."""
    exe = ingest.ffmpeg_exe()
    proc = subprocess.Popen(
        [exe, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", fps,
         "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-g", "250",
         "-bf", "3", "-pix_fmt", "yuv420p" if w % 2 == 0 and h % 2 == 0 else "yuv444p",
         str(path)],
        stdin=subprocess.PIPE,
    )
    yy, xx = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        frame = np.empty((h, w, 3), np.uint8)
        frame[..., 0] = (xx + i * 7) % 256
        frame[..., 1] = (yy + i * 3) % 256
        frame[..., 2] = 128
        frame = np.clip(frame + rng.integers(-12, 12, frame.shape), 0, 255).astype(np.uint8)
        for b in range(BITS):
            v = 255 if (i >> b) & 1 else 0
            frame[0:BLOCK, b * BLOCK:(b + 1) * BLOCK] = v
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0


def read_index(rgb: np.ndarray) -> int:
    n = 0
    for b in range(BITS):
        c = rgb[BLOCK // 4: 3 * BLOCK // 4, b * BLOCK + BLOCK // 4: (b + 1) * BLOCK - BLOCK // 4]
        if c.mean() > 128:
            n |= 1 << b
    return n


def opencv_step_back_ms(path: Path, n_frames: int, k: int = 60) -> tuple[float, int]:
    """Baseline: step back k frames through the ORIGINAL file with OpenCV."""
    import cv2
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    wrong = 0
    t0 = time.perf_counter()
    for i in range(n_frames - 1, n_frames - 1 - k, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        wrong += (not ok) or read_index(bgr[..., ::-1]) != i
    return (time.perf_counter() - t0) / k * 1000, wrong


def check_clip(label, w, h, fps, n_frames, tmp: Path):
    src = tmp / f"{label}.mp4"
    make_clip(src, w, h, fps, n_frames)

    progress_seen = []
    t0 = time.perf_counter()
    info = ingest.import_clip(src, progress=progress_seen.append)
    t_import = time.perf_counter() - t0

    t0 = time.perf_counter()
    again = ingest.import_clip(src)
    t_cached = time.perf_counter() - t0
    assert again.folder == info.folder, "second import should reuse the cache"

    fs = FrameSource(info, cache_mb=0)  # cache off: measure real decodes
    errors = []
    for i in range(fs.frame_count):                       # every frame, in order
        if read_index(fs.frame(i)) != i:
            errors.append(("seq", i))
    rnd = random.Random(1)
    for i in [rnd.randrange(fs.frame_count) for _ in range(300)]:  # random seeks
        if read_index(fs.frame(i)) != i:
            errors.append(("rand", i))
    t0 = time.perf_counter()
    for i in range(fs.frame_count - 1, -1, -1):           # scroll back from end
        if read_index(fs.frame(i)) != i:
            errors.append(("back", i))
    back_ms = (time.perf_counter() - t0) / fs.frame_count * 1000
    assert read_index(fs.frame(-5)) == 0                  # clamping
    assert read_index(fs.frame(10**6)) == fs.frame_count - 1

    raw_ms, raw_wrong = opencv_step_back_ms(src, n_frames)
    disk_mb = sum(f.stat().st_size for f in Path(info.folder).glob("*.jpg")) / 1e6

    print(f"\n[{label}] {w}x{h} @ {fps} fps, {n_frames} frames")
    print(f"  imported            {fs.width}x{fs.height}, fps {fs.fps}, frames {fs.frame_count}")
    print(f"  import time         {t_import:.2f}s (re-import {t_cached*1000:.0f} ms)")
    print(f"  disk                {disk_mb:.1f} MB ({disk_mb/n_frames*1000:.0f} KB/frame)")
    print(f"  progress callbacks  {len(progress_seen)}, last={progress_seen[-1]:.2f}")
    print(f"  wrong frames        {len(errors)} of {fs.frame_count * 2 + 300} checks")
    print(f"  step back, imported {back_ms:.1f} ms/frame")
    print(f"  step back, original {raw_ms:.1f} ms/frame via OpenCV ({raw_wrong} wrong of 60)")

    assert fs.frame_count == n_frames, "frame count mismatch"
    assert (fs.width, fs.height) == (w, h)
    assert not errors, errors[:10]


def check_cancel(tmp: Path):
    import threading
    src = tmp / "cancel.mp4"
    make_clip(src, 1920, 1080, "50", 400)
    ev = threading.Event()
    def on_progress(p):
        if p > 0.05:
            ev.set()
    try:
        ingest.import_clip(src, progress=on_progress, cancel=ev)
    except ingest.ImportCancelled:
        pass
    else:
        raise AssertionError("expected ImportCancelled")
    leftovers = [p.name for p in ingest.cache_dir().iterdir() if p.name.startswith("cancel")]
    assert not leftovers, leftovers
    print("\n[cancel] import cancelled cleanly, no partial folder left")


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        os.environ["XDG_CACHE_HOME"] = str(tmp / "cache")
        check_clip("hd50", 1280, 720, "50", 400, tmp)
        check_clip("ntsc_odd", 641, 361, "30000/1001", 240, tmp)
        check_clip("fullhd50", 1920, 1080, "50", 300, tmp)
        check_cancel(tmp)
    print("\nAll core checks passed.")


if __name__ == "__main__":
    main()
