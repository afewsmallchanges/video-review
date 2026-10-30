# player.py
# Qt widgets for viewing one clip: the picture, the timeline and transport.

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QWheelEvent
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .frames import FrameSource

# --- Scroll wheel settings ---------------------------------------------------
WHEEL_UP_IS_FORWARD = True   # flip if you prefer scroll-down = forward
NOTCH = 120                  # Qt units per mouse-wheel click
TOUCHPAD_PX_PER_FRAME = 12   # touchpad pixels per frame step
SHIFT_MULTIPLIER = 10        # Shift + scroll moves this many frames per step


class WheelStepper:
    """Turns wheel events into whole frame steps.

    Mouse wheels send 120 units per click; touchpads send many small pixel
    deltas. Leftovers are accumulated so slow touchpad swipes still step.
    """

    def __init__(self):
        self._acc = 0.0

    def steps(self, event: QWheelEvent) -> int:
        pixel = event.pixelDelta()
        angle = event.angleDelta()
        if not pixel.isNull():
            delta = (pixel.y() or pixel.x()) / TOUCHPAD_PX_PER_FRAME
        else:
            # Some systems turn Shift+wheel into horizontal scrolling.
            delta = (angle.y() or angle.x()) / NOTCH
        if not WHEEL_UP_IS_FORWARD:
            delta = -delta
        if self._acc and (delta > 0) != (self._acc > 0):
            self._acc = 0.0  # direction changed: drop the leftover
        self._acc += delta
        whole = int(self._acc)
        self._acc -= whole
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            whole *= SHIFT_MULTIPLIER
        return whole


class VideoView(QWidget):
    """Draws the current frame, letterboxed, and turns scrolling into steps."""

    stepRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: Optional[QImage] = None
        self._array: Optional[np.ndarray] = None  # keeps QImage's memory alive
        self._wheel = WheelStepper()
        self.placeholder = "Drop a clip here, or press Ctrl+O"
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 180)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    def set_frame(self, rgb: Optional[np.ndarray]):
        if rgb is None:
            self._image = self._array = None
        else:
            rgb = np.ascontiguousarray(rgb)
            h, w, _ = rgb.shape
            self._array = rgb
            self._image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.update()

    def image_rect(self) -> QRectF:
        """Where the picture sits inside the widget (used later for drawing)."""
        if self._image is None:
            return QRectF(self.rect())
        iw, ih = self._image.width(), self._image.height()
        scale = min(self.width() / iw, self.height() / ih)
        w, h = iw * scale, ih * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._image is None:
            p.setPen(QColor(140, 140, 140))
            f = QFont(self.font())
            f.setPointSize(f.pointSize() + 4)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(self.image_rect(), self._image)

    def wheelEvent(self, event: QWheelEvent):
        n = self._wheel.steps(event)
        if n:
            self.stepRequested.emit(n)
        event.accept()


class TimelineSlider(QSlider):
    """Horizontal scrubber. Its scroll wheel steps frames like the picture."""

    def __init__(self, on_steps: Callable[[int], None], parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._on_steps = on_steps
        self._wheel = WheelStepper()
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setTracking(True)

    def wheelEvent(self, event: QWheelEvent):
        n = self._wheel.steps(event)
        if n:
            self._on_steps(n)
        event.accept()

    def mousePressEvent(self, event):
        # Jump straight to the clicked point instead of paging.
        if event.button() == Qt.MouseButton.LeftButton and self.maximum() > 0:
            x = event.position().x()
            value = round(self.minimum() + (self.maximum() - self.minimum()) * x / max(1, self.width()))
            self.setValue(value)
        super().mousePressEvent(event)


SPEEDS = [("1×", 1.0), ("½×", 0.5), ("¼×", 0.25), ("⅒×", 0.1)]


class PlayerWidget(QWidget):
    """One clip: picture, timeline, transport buttons and frame readout."""

    frameChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source: Optional[FrameSource] = None
        self.current = 0
        self.speed = 1.0

        self.view = VideoView()
        self.view.stepRequested.connect(self.step)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._tick)

        self.slider = TimelineSlider(self.step)
        self.slider.valueChanged.connect(self._slider_moved)

        def button(text, tip, slot):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setAutoRaise(True)
            b.clicked.connect(slot)
            return b

        self.btn_start = button("⏮", "First frame (Home)", lambda: self.seek(0))
        self.btn_back = button("◀|", "Back one frame (←, scroll)", lambda: self.step(-1))
        self.btn_play = button("▶", "Play / pause (Space)", self.toggle_play)
        self.btn_fwd = button("|▶", "Forward one frame (→, scroll)", lambda: self.step(1))
        self.btn_end = button("⏭", "Last frame (End)", lambda: self.seek(10**9))
        for b in (self.btn_start, self.btn_back, self.btn_play, self.btn_fwd, self.btn_end):
            f = b.font()
            f.setPointSize(f.pointSize() + 3)
            b.setFont(f)

        self.speed_box = QComboBox()
        self.speed_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.speed_box.setToolTip("Playback speed")
        for label, value in SPEEDS:
            self.speed_box.addItem(label, value)
        self.speed_box.currentIndexChanged.connect(self._speed_changed)

        self.readout = QLabel("—")
        mono = QFont("monospace")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.readout.setFont(mono)
        self.readout.setMinimumWidth(260)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        bar = QHBoxLayout()
        bar.setContentsMargins(8, 4, 8, 6)
        for w in (self.btn_start, self.btn_back, self.btn_play, self.btn_fwd, self.btn_end):
            bar.addWidget(w)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.speed_box)
        bar.addWidget(self.readout)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.view, 1)
        layout.addLayout(bar)
        self._set_enabled(False)

    # ------------------------------------------------------------------
    def load(self, source: FrameSource):
        self.pause()
        self.source = source
        self.slider.blockSignals(True)
        self.slider.setRange(0, source.frame_count - 1)
        self.slider.setPageStep(max(1, round(source.fps)))
        self.slider.blockSignals(False)
        self._set_enabled(True)
        self.current = -1
        self.seek(0)

    def seek(self, n: int):
        if not self.source:
            return
        n = self.source.clamp(n)
        if n == self.current:
            return
        self.current = n
        self.view.set_frame(self.source.frame(n))
        self.slider.blockSignals(True)
        self.slider.setValue(n)
        self.slider.blockSignals(False)
        self._update_readout()
        self.frameChanged.emit(n)

    def step(self, delta: int):
        """Manual stepping always pauses playback first."""
        self.pause()
        self.seek(self.current + delta)

    # --- playback -----------------------------------------------------
    def toggle_play(self):
        if self.timer.isActive():
            self.pause()
        else:
            self.play()

    def play(self):
        if not self.source:
            return
        if self.current >= self.source.frame_count - 1:
            self.seek(0)
        self.timer.start(max(1, round(1000 / (self.source.fps * self.speed))))
        self.btn_play.setText("⏸")

    def pause(self):
        self.timer.stop()
        self.btn_play.setText("▶")

    def _tick(self):
        if self.current >= self.source.frame_count - 1:
            self.pause()
        else:
            self.seek(self.current + 1)

    def _speed_changed(self, _index):
        self.speed = float(self.speed_box.currentData())
        if self.timer.isActive():
            self.play()  # restart with the new interval
        self._update_readout()

    # ------------------------------------------------------------------
    def _slider_moved(self, value: int):
        self.pause()
        self.seek(value)

    def _update_readout(self):
        s = self.source
        if not s:
            self.readout.setText("—")
            return
        width = len(str(s.frame_count))
        self.readout.setText(
            f"{s.timecode(self.current)}   frame {self.current + 1:>{width}}/{s.frame_count}"
        )

    def _set_enabled(self, on: bool):
        for w in (self.btn_start, self.btn_back, self.btn_play, self.btn_fwd,
                  self.btn_end, self.slider, self.speed_box):
            w.setEnabled(on)
