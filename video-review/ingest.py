# ingest.py

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

# Bump when the import settings change, so old cached copies are redone.
CACHE_VERSION = 2
FRAME_PATTERN = "%06d.jpg"
JPEG_QUALITY = 3  # FFmpeg -q:v scale, 2 (best) .. 31 (worst)

ProgressCallback = Callable[[float], None]


class ImportCancelled(Exception):
    pass


class ImportFailed(Exception):
    pass


@dataclass
class ClipInfo:
    source: str
    folder: str
    fps: float
    frame_count: int
    width: int
    height: int
    version: int = CACHE_VERSION

    def frame_path(self, n: int) -> Path:
        return Path(self.folder) / (FRAME_PATTERN % n)


# --------------------------------------------------------------------------
# Locating FFmpeg
# --------------------------------------------------------------------------

def ffmpeg_exe() -> str:
    """Return the FFmpeg binary to use.

    Prefers the copy bundled with the ``imageio-ffmpeg`` pip package, so the
    app behaves the same on every distro. Set IMAGEIO_FFMPEG_EXE to force a
    specific binary.
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise ImportFailed(
                "FFmpeg not found. Install it with `pip install imageio-ffmpeg` "
                "or your distro's ffmpeg package."
            )
        return exe


_version_cache: dict[str, int] = {}


def _major_version(exe: str) -> int:
    if exe not in _version_cache:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True).stdout
        m = re.search(r"ffmpeg version n?(\d+)", out)
        _version_cache[exe] = int(m.group(1)) if m else 7
    return _version_cache[exe]


# --------------------------------------------------------------------------
# Cache location
# --------------------------------------------------------------------------

def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache")
    d = Path(base) / "replaydesk" / "clips"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_folder(src: Path) -> Path:
    st = src.stat()
    key = f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}|v{CACHE_VERSION}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", src.stem)[:40]
    return cache_dir() / f"{safe}-{digest}"


def load_clip_info(folder: Path) -> Optional[ClipInfo]:
    meta = Path(folder) / "clip.json"
    try:
        data = json.loads(meta.read_text())
        info = ClipInfo(**data)
    except (OSError, ValueError, TypeError):
        return None
    if info.version != CACHE_VERSION or info.frame_count < 1:
        return None
    info.folder = str(Path(folder))  # survive the cache dir being moved
    return info


def cached_clip(src: Path) -> Optional[ClipInfo]:
    """Return the imported copy of ``src`` if it already exists."""
    return load_clip_info(_cache_folder(Path(src)))


# --------------------------------------------------------------------------
# Probing and conversion
# --------------------------------------------------------------------------

def probe(src: Path, exe: Optional[str] = None) -> tuple[Optional[float], Optional[float]]:
    """(duration_seconds, fps) from FFmpeg's banner; either may be None."""
    exe = exe or ffmpeg_exe()
    err = subprocess.run(
        [exe, "-hide_banner", "-nostdin", "-i", str(src)],
        capture_output=True,
        text=True,
    ).stderr
    duration = fps = None
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if m:
        h, mnt, s = m.groups()
        duration = int(h) * 3600 + int(mnt) * 60 + float(s)
    video_line = next((l for l in err.splitlines() if "Video:" in l), "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", video_line) or re.search(
        r"(\d+(?:\.\d+)?)\s*tbr", video_line
    )
    if m:
        fps = float(m.group(1))
    return duration, fps


def build_command(src: Path, out_folder: Path, exe: str) -> list[str]:
    cmd = [exe, "-hide_banner", "-nostdin", "-y", "-i", str(src), "-map", "0:v:0"]
    # Constant frame rate, so "frame N" always means the same moment in time.
    cmd += ["-fps_mode", "cfr"] if _major_version(exe) >= 5 else ["-vsync", "cfr"]
    cmd += [
        "-c:v", "mjpeg", "-q:v", str(JPEG_QUALITY), "-pix_fmt", "yuvj420p",
        "-start_number", "0",
        "-progress", "pipe:1", "-nostats",
        str(out_folder / FRAME_PATTERN),
    ]
    return cmd


def import_clip(
    src: Path | str,
    progress: Optional[ProgressCallback] = None,
    cancel: Optional[threading.Event] = None,
) -> ClipInfo:
    """Import ``src`` (or reuse an earlier import) and return its ClipInfo.

    ``progress`` is called with values from 0.0 to 1.0, from this thread.
    Setting ``cancel`` stops the import and raises ImportCancelled.
    """
    src = Path(src)
    if not src.exists():
        raise ImportFailed(f"File not found: {src}")

    existing = cached_clip(src)
    if existing:
        if progress:
            progress(1.0)
        return existing

    exe = ffmpeg_exe()
    final = _cache_folder(src)
    work = final.with_name(final.name + ".part")
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(final, ignore_errors=True)
    work.mkdir(parents=True)

    duration, fps = probe(src, exe)
    proc = subprocess.Popen(
        build_command(src, work, exe),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    # Drain stderr on a side thread so FFmpeg never blocks on a full pipe.
    err_lines: list[str] = []
    drain = threading.Thread(target=lambda: err_lines.extend(proc.stderr), daemon=True)
    drain.start()

    try:
        for line in proc.stdout:
            if cancel is not None and cancel.is_set():
                proc.terminate()
                proc.wait()
                raise ImportCancelled()
            key, _, value = line.strip().partition("=")
            if key == "out_time_us" and duration and progress:
                try:
                    progress(min(int(value) / 1e6 / duration, 0.99))
                except ValueError:
                    pass
        proc.wait()
        drain.join(timeout=2)
        if proc.returncode != 0:
            tail = "".join(err_lines[-15:]).strip()
            raise ImportFailed(f"FFmpeg could not import {src.name}:\n{tail}")

        frames = sorted(work.glob("*.jpg"))
        if not frames:
            raise ImportFailed(f"No video frames found in {src.name}")
        width, height = _jpeg_size(frames[0])
        info = ClipInfo(
            source=str(src.resolve()),
            folder=str(final),
            fps=fps or 25.0,
            frame_count=len(frames),
            width=width,
            height=height,
        )
        (work / "clip.json").write_text(json.dumps(asdict(info), indent=2))
        work.rename(final)
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        shutil.rmtree(work, ignore_errors=True)
        raise

    if progress:
        progress(1.0)
    return info


def _jpeg_size(path: Path) -> tuple[int, int]:
    """Read width/height from a JPEG header without decoding it."""
    with open(path, "rb") as f:
        data = f.read(65536)
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        length = int.from_bytes(data[i + 2:i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            h = int.from_bytes(data[i + 5:i + 7], "big")
            w = int.from_bytes(data[i + 7:i + 9], "big")
            return w, h
        i += 2 + length
    raise ImportFailed(f"Could not read frame size from {path.name}")
