import logging
import os
import sys

_loggers: dict[str, logging.Logger] = {}
_initialized: bool = False
_file_handler: logging.FileHandler | None = None


def _ensure_initialized():
    global _initialized
    if _initialized:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)-5s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger("refusal_direction")
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    _initialized = True


def enable_file_logging(log_dir: str):
    global _file_handler
    _ensure_initialized()
    if _file_handler is not None:
        return
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "pipeline.log")
    _file_handler = logging.FileHandler(log_path, encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)-5s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _file_handler.setLevel(logging.DEBUG)
    logging.getLogger("refusal_direction").addHandler(_file_handler)


def get_logger(name: str) -> logging.Logger:
    _ensure_initialized()
    if name not in _loggers:
        _loggers[name] = logging.getLogger(f"refusal_direction.{name}")
    return _loggers[name]
