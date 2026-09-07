"""Tiny dependency-free colored console logger for gate decisions.

Kept separate from ``main`` so the crawler can narrate its per-gate decisions
without importing the demo runner. Colors auto-disable when stdout is not a TTY
or when ``NO_COLOR`` is set.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime


class _C:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GREY = "\033[90m"


class ConsoleLog:
    """Minimal leveled + gate-aware logger."""

    def __init__(self, enabled: bool = True) -> None:
        self.color = (
            enabled
            and sys.stdout.isatty()
            and os.getenv("NO_COLOR") is None
        )

    def _paint(self, text: str, color: str) -> str:
        return f"{color}{text}{_C.RESET}" if self.color else text

    def _stamp(self) -> str:
        return self._paint(datetime.now().strftime("%H:%M:%S"), _C.GREY)

    def _emit(self, tag: str, tag_color: str, msg: str) -> None:
        print(f"{self._stamp()} {self._paint(tag, tag_color)} {msg}")

    # --- levels ----------------------------------------------------------- #
    def info(self, msg: str) -> None:
        self._emit("INFO ", _C.BLUE, msg)

    def success(self, msg: str) -> None:
        self._emit("OK   ", _C.GREEN, msg)

    def warn(self, msg: str) -> None:
        self._emit("WARN ", _C.YELLOW, msg)

    def error(self, msg: str) -> None:
        self._emit("ERROR", _C.RED, msg)

    def debug(self, msg: str) -> None:
        self._emit("DEBUG", _C.GREY, self._paint(msg, _C.DIM))

    # --- gate narration --------------------------------------------------- #
    def gate(self, name: str, msg: str, ok: bool = True) -> None:
        arrow = self._paint("▶", _C.CYAN)
        gate_label = self._paint(f"[{name}]", _C.MAGENTA if ok else _C.RED)
        print(f"{self._stamp()} {arrow} {gate_label} {msg}")

    def header(self, msg: str) -> None:
        line = "═" * min(len(msg) + 4, 78)
        print("\n" + self._paint(line, _C.CYAN))
        print(self._paint(f"  {msg}", _C.BOLD + _C.CYAN))
        print(self._paint(line, _C.CYAN))

    def rule(self, char: str = "─", width: int = 78) -> None:
        print(self._paint(char * width, _C.GREY))


# Shared module-level instance.
log = ConsoleLog()
