"""Occupancy-интегратор с аудит-коррекцией (спека §5.3, §7).

Occupancy = материализованный интеграл Σenter − Σexit. Интеграл копит
дрейф (каждая пропущенная ошибка живёт вечно), поэтому редкие абсолютные
замеры аудита втягивают значение: occupancy += α·(audit_ema − occupancy).
По сути одномерный фильтр: непрерывный сигнал от гейтов + редкие
абсолютные измерения.
"""
from __future__ import annotations

import logging
import threading
import time

from .config import AuditConfig

log = logging.getLogger(__name__)


class OccupancyIntegrator:
    def __init__(self, audit_cfg: AuditConfig, initial: float = 0.0) -> None:
        self.cfg = audit_cfg
        self._lock = threading.Lock()
        self._value = max(0.0, initial)
        self._enters_total = 0
        self._exits_total = 0
        self._audit_ema: float | None = None
        self._pending_outlier: float | None = None
        self._last_audit_delta = 0.0
        self._last_audit_ts: float | None = None
        self.updated_at = time.time()

    # --- события гейтов ---------------------------------------------------
    def apply_event(self, etype: str) -> float:
        with self._lock:
            if etype == "enter":
                self._value += 1
                self._enters_total += 1
            elif etype == "exit":
                self._value = max(0.0, self._value - 1)
                self._exits_total += 1
            self.updated_at = time.time()
            return self._value

    # --- аудит ------------------------------------------------------------
    def apply_audit(self, audit_count: float) -> tuple[float, float, bool]:
        """Замер аудита → (применённая дельта, новое значение, алерт).

        Выброс (|audit − occ| > outlier_abs) применяется только после
        второго подряд подтверждения — одиночный сбой счётчика толпы
        не должен дёргать occupancy.
        """
        with self._lock:
            delta_raw = audit_count - self._value
            if abs(delta_raw) > self.cfg.outlier_abs:
                if self._pending_outlier is None or \
                        abs(audit_count - self._pending_outlier) > self.cfg.outlier_abs:
                    self._pending_outlier = audit_count
                    log.warning("audit outlier %.1f (occupancy=%.1f) — waiting for confirmation",
                                audit_count, self._value)
                    return 0.0, self._value, False
            self._pending_outlier = None

            if self._audit_ema is None:
                self._audit_ema = audit_count
            else:
                e = self.cfg.ema
                self._audit_ema = e * audit_count + (1 - e) * self._audit_ema

            applied = self.cfg.alpha * (self._audit_ema - self._value)
            self._value = max(0.0, self._value + applied)
            self._last_audit_delta = self._audit_ema - self._value
            self._last_audit_ts = time.time()
            self.updated_at = self._last_audit_ts
            alert = abs(delta_raw) > self.cfg.alert_delta
            if alert:
                log.warning("occupancy drift: audit=%.1f occupancy=%.1f (delta %.1f) — "
                            "check cameras/zones/detector", audit_count, self._value, delta_raw)
            return applied, self._value, alert

    # --- чтение -----------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "occupancy": int(round(self._value)),
                "occupancy_raw": round(self._value, 2),
                "enters_total": self._enters_total,
                "exits_total": self._exits_total,
                "audit_ema": round(self._audit_ema, 2) if self._audit_ema is not None else None,
                "last_audit_delta": round(self._last_audit_delta, 2),
                "last_audit_ts": self._last_audit_ts,
                "updated_at": self.updated_at,
            }

    @property
    def value(self) -> float:
        with self._lock:
            return self._value
