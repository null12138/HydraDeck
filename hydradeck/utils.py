from __future__ import annotations

import datetime
import sys
import threading
import time


def log(enabled: bool, msg: str) -> None:
    if not enabled:
        return
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] {msg}")


JSON = dict[str, object]


class Heartbeat:
    def __init__(self, enabled: bool, label: str, interval_s: float = 5.0) -> None:
        self._enabled = enabled
        self._label = label
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self) -> Heartbeat:
        if not self._enabled:
            return self

        def run() -> None:
            start = time.time()
            while not self._stop.wait(self._interval_s):
                elapsed = int(time.time() - start)
                sys.stderr.write(f"[heartbeat] {self._label} ({elapsed}s)\n")
                sys.stderr.flush()

        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = (exc_type, exc, tb)
        if not self._enabled:
            return
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=1.0)


class Progress:
    def __init__(
        self,
        enabled: bool,
        total: int,
        label: str = "",
        stream=None,
    ) -> None:
        self._enabled = enabled
        self._total = max(int(total), 1)
        self._label = label
        self._stream = stream or sys.stderr
        self._current = 0
        self._last_len = 0

    def update(self, step: str, inc: int = 1) -> None:
        if not self._enabled:
            return
        self._current = min(self._total, self._current + max(int(inc), 0))
        pct = int((self._current / self._total) * 100)
        bar_len = 24
        filled = int(bar_len * self._current / self._total)
        bar = "#" * filled + "-" * (bar_len - filled)
        msg = f"[progress] {self._label} [{bar}] {pct:3d}%  {step}"
        pad = " " * max(0, self._last_len - len(msg))
        self._stream.write("\r" + msg + pad)
        self._stream.flush()
        self._last_len = len(msg)

    def done(self, step: str = "done") -> None:
        if not self._enabled:
            return
        self._current = self._total
        self.update(step, inc=0)
        self._stream.write("\n")
        self._stream.flush()
