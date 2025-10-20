# src/RetinoNet/__init__.py
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

PKG_NAME = "RetinoNet"

def _build_formatter(json_mode: bool) -> logging.Formatter:
    if json_mode:
        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload = {
                    "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "module": record.module,
                    "msg": record.getMessage(),
                }
                if record.exc_info:
                    payload["exc"] = self.formatException(record.exc_info)
                return json.dumps(payload, ensure_ascii=False)
        return JSONFormatter()
    return logging.Formatter(
        fmt="[%(asctime)s] | %(levelname)s | %(name)s - %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S%z",
    )

def _configure() -> logging.Logger:
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    json_mode = os.getenv("LOG_JSON", "0").lower() in {"1", "true", "yes"}
    log_dir = Path(os.getenv("LOG_DIR", "logs")); log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "RetinoNet.log"

    lg = logging.getLogger(PKG_NAME)
    if lg.handlers:
        return lg

    lg.setLevel(level)
    fmt = _build_formatter(json_mode)

    fh = TimedRotatingFileHandler(logfile, when="midnight", backupCount=7, encoding="utf-8")
    fh.setLevel(level); fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(level); ch.setFormatter(fmt)

    lg.addHandler(fh); lg.addHandler(ch)
    lg.propagate = False
    return lg

def get_logger(name: str | None = None) -> logging.Logger:
    base = logging.getLogger(PKG_NAME) if logging.getLogger(PKG_NAME).handlers else _configure()
    return base if not name else base.getChild(name)

# default logger accessible via: from RetinoNet import logger
logger = _configure()

__all__ = ["logger", "get_logger"]