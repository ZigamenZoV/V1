"""Event Store: SQLite (один узел, спека §5.3). WAL, единственный писатель —
main-процесс. Для нескольких узлов интерфейс переносится на PostgreSQL.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

from ..types import Event

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('enter','exit')),
    trajectory_len INTEGER NOT NULL,
    confidence REAL NOT NULL,
    embedding_id TEXT,
    matched_entry_id TEXT,
    unmatched INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS occupancy_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    value REAL NOT NULL,
    source TEXT NOT NULL,            -- events | audit | restore | snapshot
    delta REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS audits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    camera_id TEXT NOT NULL,
    raw_count REAL NOT NULL,
    applied_delta REAL NOT NULL,
    occupancy_after REAL NOT NULL
);
"""


class EventStore:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # --- запись -----------------------------------------------------------
    def add_event(self, ev: Event) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO events (ts, camera_id, track_id, type, trajectory_len,"
                " confidence, embedding_id, matched_entry_id, unmatched)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (ev.ts, ev.camera_id, ev.track_id, ev.type, ev.trajectory_len,
                 ev.confidence, ev.embedding_id, ev.matched_entry_id, int(ev.unmatched)))
            self._db.commit()
            return int(cur.lastrowid)

    def log_occupancy(self, value: float, source: str, delta: float = 0.0) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO occupancy_log (ts, value, source, delta) VALUES (?,?,?,?)",
                (time.time(), value, source, delta))
            self._db.commit()

    def add_audit(self, camera_id: str, raw: float, applied: float, occ_after: float) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO audits (ts, camera_id, raw_count, applied_delta, occupancy_after)"
                " VALUES (?,?,?,?,?)", (time.time(), camera_id, raw, applied, occ_after))
            self._db.commit()

    # --- чтение (API) -----------------------------------------------------
    def last_occupancy(self) -> float | None:
        with self._lock:
            row = self._db.execute(
                "SELECT value FROM occupancy_log ORDER BY id DESC LIMIT 1").fetchone()
        return float(row[0]) if row else None

    def events(self, limit: int = 100, since: str | None = None,
               camera_id: str | None = None) -> list[dict]:
        q = ("SELECT id, ts, camera_id, track_id, type, trajectory_len, confidence,"
             " embedding_id, matched_entry_id, unmatched FROM events")
        cond, args = [], []
        if since:
            cond.append("ts >= ?")
            args.append(since)
        if camera_id:
            cond.append("camera_id = ?")
            args.append(camera_id)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        cols = ["id", "ts", "camera_id", "track_id", "type", "trajectory_len",
                "confidence", "embedding_id", "matched_entry_id", "unmatched"]
        return [dict(zip(cols, r)) for r in rows]

    def counts_today(self) -> dict:
        day = time.strftime("%Y-%m-%d")
        with self._lock:
            rows = self._db.execute(
                "SELECT type, COUNT(*) FROM events WHERE ts >= ? GROUP BY type",
                (day,)).fetchall()
        out = {"enter": 0, "exit": 0}
        out.update({t: int(n) for t, n in rows})
        return out

    def occupancy_history(self, hours: float = 24.0) -> list[dict]:
        t0 = time.time() - hours * 3600
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, value, source FROM occupancy_log WHERE ts >= ? ORDER BY id",
                (t0,)).fetchall()
        return [{"ts": r[0], "value": r[1], "source": r[2]} for r in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()
