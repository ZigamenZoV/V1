"""Финализация по завершении трека (спека §5.2).

Событие фиксируется, когда трек потерян окончательно (lost > track_buffer):
классифицируется вся траектория — по началу/концу (сторона A vs B) и по
факту полной последовательности состояний. Один трек → максимум одно
событие (этим же покрыт cooldown §5.1: внутри трека вторая
последовательность события не породит). Короткие/малоподвижные треки
отбрасываются как шум.

Классификация устойчива к фрагментации трека: обломок трека человека,
уже прошедшего дверь, начинается в зоне B и полную последовательность
A→B не проходит — двойного счёта не возникает. Поэтому временнОй дедуп
не применяется: в плотном потоке люди идут друг за другом чаще, чем раз
в cooldown, и подавление по времени занижало бы счёт.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..config import CameraConfig, ReidConfig
from ..types import Event, Frame, Track
from .fsm import TrackFSM
from .zones import ZONE_A, ZONE_B, ZonePair

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _TrackCtx:
    fsm: TrackFSM
    best_crop: np.ndarray | None = None
    best_crop_score: float = -1.0


class TrajectoryCounter:
    """Состояние счёта одной gate-камеры: FSM по каждому треку + финализация."""

    def __init__(self, cam: CameraConfig, zones: ZonePair, fps: float,
                 reid: ReidConfig | None = None) -> None:
        self.cam = cam
        self.zones = zones
        self.fps = fps
        self.reid = reid if (reid and reid.enabled) else None
        self._tracks: dict[int, _TrackCtx] = {}

    # --- обновление на такте детекции -----------------------------------
    def update(self, tracks: list[Track], frame: Frame | None) -> None:
        for t in tracks:
            ctx = self._tracks.get(t.track_id)
            if ctx is None:
                ctx = _TrackCtx(TrackFSM(self.cam.fsm))
                self._tracks[t.track_id] = ctx
            zone = self.zones.locate(t.center)
            ctx.fsm.update(zone, t.center, t.score)
            if self.reid is not None and frame is not None and zone is not None:
                self._update_crop(ctx, t, frame)

    def _update_crop(self, ctx: _TrackCtx, t: Track, frame: Frame) -> None:
        """Кроп тела: бокс головы, расширенный вширь и вниз (спека §6)."""
        quality = t.score * float((t.box[2] - t.box[0]) * (t.box[3] - t.box[1]))
        if quality <= ctx.best_crop_score:
            return
        h, w = frame.image.shape[:2]
        bw = t.box[2] - t.box[0]
        bh = t.box[3] - t.box[1]
        cx = (t.box[0] + t.box[2]) / 2
        x1 = int(max(0, cx - bw * self.reid.crop_expand_w / 2))
        x2 = int(min(w, cx + bw * self.reid.crop_expand_w / 2))
        y1 = int(max(0, t.box[1]))
        y2 = int(min(h, t.box[1] + bh * self.reid.crop_expand_h))
        if x2 - x1 < 4 or y2 - y1 < 8:
            return
        ctx.best_crop = frame.image[y1:y2, x1:x2].copy()
        ctx.best_crop_score = quality

    # --- финализация потерянных треков -----------------------------------
    def finalize(self, removed_ids: list[int]) -> list[Event]:
        events: list[Event] = []
        for tid in removed_ids:
            ctx = self._tracks.pop(tid, None)
            if ctx is None:
                continue
            ev = self._classify(tid, ctx)
            if ev is not None:
                # итоговое событие логирует main-процесс (с контекстом occupancy)
                log.debug("[%s] track %d -> %s (frames=%d, conf=%.2f)",
                          self.cam.id, tid, ev.type, ev.trajectory_len, ev.confidence)
                events.append(ev)
        return events

    def _classify(self, tid: int, ctx: _TrackCtx) -> Event | None:
        fsm = ctx.fsm
        cfg = self.cam.fsm
        if fsm.frames < cfg.min_track_frames or fsm.path_px < cfg.min_path_px:
            return None
        zones_seq = fsm.zone_sequence()
        if len(zones_seq) < 2:
            return None
        first, last = zones_seq[0].state, zones_seq[-1].state

        etype: str | None = None
        # полный проход A→B (вход): начало на стороне A, конец на стороне B
        if first == ZONE_A and last == ZONE_B and self._has_pair(fsm, ZONE_A, ZONE_B):
            etype = "enter"
        # полный проход B→A (выход)
        elif first == ZONE_B and last == ZONE_A and self._has_pair(fsm, ZONE_B, ZONE_A):
            etype = "exit"
        if etype is None:
            return None

        return Event.make(self.cam.id, tid, etype,
                          trajectory_len=fsm.frames,
                          confidence=round(fsm.mean_score, 3),
                          crop=ctx.best_crop)

    def _has_pair(self, fsm: TrackFSM, z_from: str, z_to: str) -> bool:
        """Есть ли в стабильной последовательности переход z_from→z_to
        с OUT-разрывом не больше max_gap кадров."""
        st = fsm.stable
        idx_zones = [(i, s.state) for i, s in enumerate(st) if s.state is not None]
        for k in range(len(idx_zones) - 1):
            (i, zi), (j, zj) = idx_zones[k], idx_zones[k + 1]
            if zi == z_from and zj == z_to and fsm.gap_between(i, j) <= fsm.max_gap:
                return True
        return False

    def drain(self) -> list[Event]:
        """Финализирует все живые треки (останов сервиса)."""
        return self.finalize(list(self._tracks.keys()))

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)
