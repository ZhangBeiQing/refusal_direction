import logging
import sys

_loggers: dict[str, logging.Logger] = {}
_initialized: bool = False


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


def get_logger(name: str) -> logging.Logger:
    _ensure_initialized()
    if name not in _loggers:
        _loggers[name] = logging.getLogger(f"refusal_direction.{name}")
    return _loggers[name]
