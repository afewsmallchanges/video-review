# frames.py

from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np

from .ingest import ClipInfo


class FrameSource:
    """Decode any frame of an imported clip by index, with an LRU cache.

    Each frame is its own JPEG file, so any frame costs exactly one decode,
    forwards or backwards.
    """

    def __init__(self, info: ClipInfo, cache_mb: int = 256):
        self.info = info
        self.fps = info.fps
        self.width = info.width
        self.height = info.height
        self.frame_count = info.frame_count
        frame_bytes = max(1, info.width * info.height * 3)
        self._cache_frames = max(0, (cache_mb * 1024 * 1024) // frame_bytes)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def frame(self, n: int) -> np.ndarray:
        """Return frame ``n`` (clamped to the clip) as an HxWx3 RGB array."""
        n = self.clamp(n)
        hit = self._cache.get(n)
        if hit is not None:
            self._cache.move_to_end(n)
            return hit

        buf = np.fromfile(self.info.frame_path(n), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Could not decode frame {n} of {self.info.source}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self._cache_frames:
            self._cache[n] = img
            if len(self._cache) > self._cache_frames:
                self._cache.popitem(last=False)
        return img

    def clamp(self, n: int) -> int:
        return max(0, min(int(n), self.frame_count - 1))

    def timestamp(self, n: int) -> float:
        """Seconds from the start of the clip to frame ``n``."""
        return n / self.fps

    def timecode(self, n: int) -> str:
        """mm:ss.ff label, where ff is the frame number within the second."""
        fps_int = max(1, round(self.fps))
        total_s, ff = divmod(n, fps_int)
        m, s = divmod(total_s, 60)
        return f"{m:02d}:{s:02d}.{ff:02d}"
