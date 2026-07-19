"""RTSP/файловый источник через ffmpeg-subprocess c NVDEC (спека §3).

Декод сразу в разрешение инференса (scale на стороне ffmpeg) — без
промежуточного full-HD в памяти. NVDEC (-hwaccel cuda) разгружает CPU и
почти не конкурирует с CUDA-ядрами детектора. Встроенный watchdog
перезапускает процесс, если кадров нет дольше stale-порога.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time

import numpy as np

from ..config import CameraConfig
from ..types import Frame
from .base import LatestSlot

log = logging.getLogger(__name__)

_CUVID = {"h264": "h264_cuvid", "hevc": "hevc_cuvid"}


def ffmpeg_available() -> str | None:
    return shutil.which("ffmpeg")


class FFmpegSource:
    def __init__(self, cam: CameraConfig, width: int, height: int) -> None:
        self.camera_id = cam.id
        self.cam = cam
        self.w, self.h = width, height
        self.slot = LatestSlot()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._seq = 0
        self._restarts = 0
        self._last_frame_mono = 0.0
        self._fps_win: list[float] = []

    # --- команда ffmpeg ---
    def _cmd(self) -> list[str]:
        cam = self.cam
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error"]
        if cam.use_nvdec:
            cmd += ["-hwaccel", "cuda", "-c:v", _CUVID[cam.codec]]
        if cam.source == "rtsp":
            cmd += ["-rtsp_transport", "tcp"]
        elif cam.source == "file":
            cmd += ["-re", "-stream_loop", "-1"]      # файл — как живой поток (отладка)
        cmd += ["-i", cam.url]
        vf = []
        if cam.roi:
            x, y, w, h = cam.roi
            vf.append(f"crop={w}:{h}:{x}:{y}")        # ROI входной группы из full-HD (§10.3)
        vf.append(f"fps={cam.fps}")
        vf.append(f"scale={self.w}:{self.h}")
        cmd += ["-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1"]
        return cmd

    # --- жизненный цикл ---
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"cap-{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._kill()
        if self._thread:
            self._thread.join(timeout=5)

    def _kill(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    def _run(self) -> None:
        frame_bytes = self.w * self.h * 3
        while not self._stop.is_set():
            try:
                self._proc = subprocess.Popen(
                    self._cmd(), stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, bufsize=frame_bytes * 2,
                )
            except FileNotFoundError:
                log.error("[%s] ffmpeg not found in PATH — source stopped", self.camera_id)
                return
            log.info("[%s] ffmpeg started (nvdec=%s)", self.camera_id, self.cam.use_nvdec)
            assert self._proc.stdout is not None
            buf = bytearray()
            while not self._stop.is_set():
                chunk = self._proc.stdout.read(frame_bytes - len(buf))
                if not chunk:
                    break                              # EOF — камера отвалилась
                buf.extend(chunk)
                if len(buf) < frame_bytes:
                    continue
                img = np.frombuffer(bytes(buf), np.uint8).reshape(self.h, self.w, 3)
                buf.clear()
                now = time.monotonic()
                self._seq += 1
                self._fps_win.append(now)
                self._fps_win = [t for t in self._fps_win if now - t < 5.0]
                self._last_frame_mono = now
                self.slot.put(Frame(self.camera_id, self._seq, now, time.time(), img))
            self._kill()
            if not self._stop.is_set():
                self._restarts += 1
                log.warning("[%s] stream interrupted, restart #%d in 2s",
                            self.camera_id, self._restarts)
                self._stop.wait(2.0)

    def restart(self) -> None:
        """Принудительный перезапуск (внешний watchdog)."""
        log.warning("[%s] watchdog: restarting ffmpeg", self.camera_id)
        self._kill()

    # --- статистика ---
    def is_stale(self, max_age_s: float) -> bool:
        return self._last_frame_mono > 0 and (time.monotonic() - self._last_frame_mono) > max_age_s

    def stats(self) -> dict:
        now = time.monotonic()
        return {
            "decode_fps": round(len(self._fps_win) / 5.0, 2),
            "frames": self._seq,
            "restarts": self._restarts,
            "stale_s": round(now - self._last_frame_mono, 1) if self._last_frame_mono else -1.0,
        }
