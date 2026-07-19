"""FSM устойчивости по треку (спека §5.1).

Сырое положение (A/B/OUT) на каждом такте дебаунсится: состояние
фиксируется только после K последовательных кадров. Топтание на пороге
и дрожание бокса стабильных состояний не порождают — последовательность
не завершается. Итог трека — сжатая последовательность стабильных
состояний, которую классифицирует финализатор.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import FsmConfig


@dataclass(slots=True)
class StableState:
    state: str | None      # 'A' | 'B' | None(OUT)
    start_frame: int       # индекс кадра начала сырой серии


class TrackFSM:
    def __init__(self, cfg: FsmConfig) -> None:
        self.k = max(1, cfg.k_frames)
        self.max_gap = cfg.max_gap_frames
        self.frames = 0
        self.path_px = 0.0
        self.score_sum = 0.0
        self.stable: list[StableState] = []
        self._run_state: str | None = None
        self._run_len = 0
        self._run_start = 0
        self._prev_center: tuple[float, float] | None = None

    def update(self, zone: str | None, center: tuple[float, float], score: float) -> None:
        self.frames += 1
        self.score_sum += score
        if self._prev_center is not None:
            dx = center[0] - self._prev_center[0]
            dy = center[1] - self._prev_center[1]
            self.path_px += (dx * dx + dy * dy) ** 0.5
        self._prev_center = center

        if zone == self._run_state:
            self._run_len += 1
        else:
            self._run_state = zone
            self._run_len = 1
            self._run_start = self.frames
        if self._run_len == self.k:
            if not self.stable or self.stable[-1].state != zone:
                self.stable.append(StableState(zone, self._run_start))

    # --- свёртки для классификации ---
    def zone_sequence(self) -> list[StableState]:
        """Стабильные состояния без OUT."""
        return [s for s in self.stable if s.state is not None]

    def gap_between(self, i_from: int, i_to: int) -> int:
        """Кадровый разрыв между стабильными зонами stable[i_from] и stable[i_to]
        (суммарная длительность OUT между ними)."""
        gap = 0
        for j in range(i_from + 1, i_to):
            if self.stable[j].state is None:
                nxt = self.stable[j + 1].start_frame if j + 1 < len(self.stable) else self.stable[j].start_frame
                gap += max(0, nxt - self.stable[j].start_frame)
        return gap

    @property
    def mean_score(self) -> float:
        return self.score_sum / self.frames if self.frames else 0.0
