"""ReID-леджер «кто внутри здания» (спека §6).

enter → эмбеддинг добавляется в леджер; exit → cosine-матчинг против
леджера с временным приором (недавно вошедшие ранжируются выше), матч
гасит запись. Доля unmatched-выходов — метрика деградации детекта/трекинга.

Корректирующий слой, не первичный счётчик: ReID по одежде ломается на
форме/униформе — зафиксировано в доке проекта.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import ReidConfig
from ..types import new_embedding_id

log = logging.getLogger(__name__)


@dataclass(slots=True)
class LedgerEntry:
    entry_id: str
    ts: float                 # wall-clock входа
    camera_id: str
    emb: np.ndarray           # L2-нормированный


@dataclass(slots=True)
class MatchResult:
    entry_id: str | None      # None → unmatched
    similarity: float


class Ledger:
    def __init__(self, cfg: ReidConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._entries: dict[str, LedgerEntry] = {}
        self._exit_window: list[bool] = []    # скользящее окно: True = unmatched
        self._last_dump = 0.0
        self._restore()

    # --- события ---------------------------------------------------------
    def on_enter(self, camera_id: str, emb: np.ndarray | None) -> str:
        entry_id = new_embedding_id()
        if emb is not None:
            with self._lock:
                self._entries[entry_id] = LedgerEntry(entry_id, time.time(), camera_id, emb)
        return entry_id

    def on_exit(self, camera_id: str, emb: np.ndarray | None) -> MatchResult:
        if emb is None:
            self._note_exit(unmatched=True)
            return MatchResult(None, 0.0)
        now = time.time()
        with self._lock:
            self._purge(now)
            best_id, best_score, best_cos = None, -1.0, 0.0
            for e in self._entries.values():
                cos = float(np.dot(e.emb, emb))
                # временной приор: недавно вошедшие ранжируются выше
                age = max(0.0, now - e.ts)
                bonus = self.cfg.time_bonus * float(np.exp(-age / max(self.cfg.time_half_life_s, 1.0)))
                score = cos + bonus
                if score > best_score:
                    best_id, best_score, best_cos = e.entry_id, score, cos
            if best_id is not None and best_cos >= self.cfg.threshold:
                del self._entries[best_id]
                self._note_exit(unmatched=False)
                return MatchResult(best_id, round(best_cos, 3))
        self._note_exit(unmatched=True)
        return MatchResult(None, round(best_cos, 3) if best_id else 0.0)

    # --- обслуживание -----------------------------------------------------
    def _purge(self, now: float) -> None:
        dead = [k for k, e in self._entries.items() if now - e.ts > self.cfg.ttl_s]
        for k in dead:
            del self._entries[k]
        if dead:
            log.info("ledger: %d entries expired by TTL", len(dead))

    def _note_exit(self, unmatched: bool) -> None:
        self._exit_window.append(unmatched)
        if len(self._exit_window) > 200:
            self._exit_window = self._exit_window[-200:]

    @property
    def unmatched_ratio(self) -> float:
        """Доля exit без матча (окно 200): рост = детект/трекинг деградировал."""
        if not self._exit_window:
            return 0.0
        return sum(self._exit_window) / len(self._exit_window)

    @property
    def size(self) -> int:
        return len(self._entries)

    # --- дамп/восстановление (леджер в памяти + периодический дамп) -------
    def dump_now(self) -> None:
        self._last_dump = 0.0
        self.maybe_dump()

    def maybe_dump(self) -> None:
        now = time.time()
        if now - self._last_dump < self.cfg.dump_interval_s:
            return
        self._last_dump = now
        try:
            with self._lock:
                if not self._entries:
                    Path(self.cfg.dump_path).unlink(missing_ok=True)
                    return
                ids = list(self._entries.keys())
                Path(self.cfg.dump_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    self.cfg.dump_path,
                    ids=np.array(ids),
                    ts=np.array([self._entries[i].ts for i in ids]),
                    cams=np.array([self._entries[i].camera_id for i in ids]),
                    embs=np.stack([self._entries[i].emb for i in ids]),
                )
        except OSError as e:
            log.warning("ledger dump failed: %s", e)

    def _restore(self) -> None:
        p = Path(self.cfg.dump_path)
        if not p.is_file():
            return
        try:
            data = np.load(p, allow_pickle=False)
            now = time.time()
            n = 0
            for i in range(len(data["ids"])):
                ts = float(data["ts"][i])
                if now - ts > self.cfg.ttl_s:
                    continue
                eid = str(data["ids"][i])
                self._entries[eid] = LedgerEntry(eid, ts, str(data["cams"][i]),
                                                 data["embs"][i].astype(np.float32))
                n += 1
            log.info("ledger restored: %d entries", n)
        except Exception as e:  # повреждённый дамп не должен валить сервис
            log.warning("ledger restore failed (%s) — starting empty", e)
