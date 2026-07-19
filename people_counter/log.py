"""Логирование: консоль + ротируемый файл на процесс."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_FMT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_configured = False


def setup(process_name: str, log_dir: str | Path = "logs", level: str = "INFO") -> logging.Logger:
    """Настраивает root-логгер процесса. Вызывается один раз на процесс
    (main, gpu, worker-<cam>), файлы не пересекаются между процессами."""
    global _configured
    if _configured:
        return logging.getLogger(process_name)
    _configured = True

    root = logging.getLogger()
    root.setLevel(level.upper())

    con = logging.StreamHandler(sys.stderr)
    con.setFormatter(logging.Formatter(_FMT))
    root.addHandler(con)

    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            Path(log_dir) / f"{process_name}.log",
            maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter(_FMT))
        root.addHandler(fh)
    except OSError:
        root.warning("cannot open log file in %s, console only", log_dir)

    return logging.getLogger(process_name)
