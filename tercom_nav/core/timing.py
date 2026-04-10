"""Lightweight execution-time and call-rate tracker for ROS 2 callbacks.

Usage
-----
    from tercom_nav.core.timing import ComponentTimer
    import time

    _t = ComponentTimer(window=200)

    def some_callback(self, msg):
        t0 = _t.start()
        ... do work ...
        _t.stop(t0)

    # Later, read metrics:
    exec_ms = _t.avg_exec_ms()   # average execution time in milliseconds
    rate_hz = _t.avg_hz()        # average call frequency in Hz
"""
import time
import collections


class ComponentTimer:
    """Rolling-window execution-time and call-rate tracker.

    All measurements are kept in fixed-size deques so memory is bounded
    and averages reflect recent behavior (not a lifetime average).

    Parameters
    ----------
    window : int
        Number of most-recent samples to keep (default 200).
    """

    def __init__(self, window: int = 200):
        self._exec_times_ms: collections.deque = collections.deque(maxlen=window)
        self._intervals_s: collections.deque = collections.deque(maxlen=window)
        self._last_call_perf: float | None = None

    # ------------------------------------------------------------------ public

    def start(self) -> float:
        """Record the call timestamp and return it for use with stop()."""
        now = time.perf_counter()
        if self._last_call_perf is not None:
            self._intervals_s.append(now - self._last_call_perf)
        self._last_call_perf = now
        return now

    def stop(self, t0: float) -> None:
        """Record elapsed time since t0 (returned by start())."""
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._exec_times_ms.append(elapsed_ms)

    def avg_exec_ms(self) -> float:
        """Average execution time in milliseconds (0.0 if no data yet)."""
        if not self._exec_times_ms:
            return 0.0
        return sum(self._exec_times_ms) / len(self._exec_times_ms)

    def avg_hz(self) -> float:
        """Average call frequency in Hz (0.0 if fewer than 2 calls recorded)."""
        if len(self._intervals_s) < 2:
            return 0.0
        avg_interval = sum(self._intervals_s) / len(self._intervals_s)
        return 1.0 / avg_interval if avg_interval > 0.0 else 0.0

    def reset(self) -> None:
        """Clear all stored samples."""
        self._exec_times_ms.clear()
        self._intervals_s.clear()
        self._last_call_perf = None
