"""Latency measurement sinks for the Tip 4 inference-speed benchmark.

``SegmentTimerSink`` implements the ``LatencySink`` protocol consumed by
``Policy.predict_action``: the policy reports segment boundaries, the sink owns
the clock and the device synchronization. Keeping synchronization here means
the policy's inference path carries no timing logic, and an unset sink leaves
it untouched.

Every boundary synchronizes the CUDA device before reading the clock;
without that, the asynchronous queue would silently shift generation cost
into the first downstream CPU access (the ``.cpu()`` inside detokenization).
"""

import logging
import time
from dataclasses import dataclass, field

import torch

# Segment names in the order ``Policy.predict_action`` reports them.
SEGMENTS = ("pre", "encode", "generate", "detok", "unnorm")


@dataclass
class LatencyRecord:
    """Wall-clock milliseconds for one ``predict_action`` call."""

    segments_ms: dict[str, float]
    total_ms: float
    # Emitted action-token count (EOS included) for tokenized heads; None for
    # continuous heads, whose sequential depth is assigned per method later.
    generated_tokens: int | None


@dataclass
class SegmentTimerSink:
    """Records per-segment durations of every ``predict_action`` call.

    Attributes:
        device: Device the policy runs on; a CUDA device is synchronized
            before every clock read so each segment is charged the kernels it
            actually launched.
        records: One ``LatencyRecord`` per completed call, in call order.
    """

    device: torch.device
    records: list[LatencyRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._start_time = 0.0
        self._last_time = 0.0
        self._segments_ms: dict[str, float] = {}

    def _now(self) -> float:
        """Synchronize the device (CUDA only) and read the monotonic clock."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def start(self) -> None:
        """Open one call's measurement window."""
        self._segments_ms = {}
        self._start_time = self._now()
        self._last_time = self._start_time

    def mark(self, segment: str) -> None:
        """Charge the time since the previous boundary to ``segment``."""
        now = self._now()
        self._segments_ms[segment] = (now - self._last_time) * 1000.0
        self._last_time = now

    def finish(self, generated_tokens: int | None) -> None:
        """Close the window and append the completed record."""
        total_ms = (self._now() - self._start_time) * 1000.0
        self.records.append(
            LatencyRecord(
                segments_ms=dict(self._segments_ms),
                total_ms=total_ms,
                generated_tokens=generated_tokens,
            )
        )


class DetokWarningCounter(logging.Handler):
    """Counts detokenizer warnings without letting them reach the log output.

    FAST's decoder warns once per chunk on a coefficient-count mismatch or an
    out-of-range token ID (``action_discretizer.py``). On a degenerate
    checkpoint that is one warning per timed call, and the logging I/O would
    dominate the detokenization segment. This handler counts the warnings for
    the results row and, attached alongside a raised level on the root logger,
    keeps them out of the timed path.
    """

    MODULE_SUBSTRING = "action_discretizer"

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if self.MODULE_SUBSTRING in record.pathname:
            self.count += 1

    def take(self) -> int:
        """Return the warnings counted since the last call and reset."""
        counted = self.count
        self.count = 0
        return counted


class _DropDiscretizerRecords(logging.Filter):
    """Filter that hides discretizer warnings from an output handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        return DetokWarningCounter.MODULE_SUBSTRING not in record.pathname


class SilencedDetokWarnings:
    """Context manager: count discretizer warnings, keep them off the console.

    Attaches a ``DetokWarningCounter`` to the root logger and adds a filter to
    every pre-existing root handler, so the warnings are counted for the
    results row but their logging I/O stays out of the timed path. The root
    logger's level is left alone: lowering it would also drop the records
    before the counter sees them.
    """

    def __init__(self) -> None:
        self.counter = DetokWarningCounter()
        self._filter = _DropDiscretizerRecords()
        self._filtered_handlers: list[logging.Handler] = []

    def __enter__(self) -> DetokWarningCounter:
        root = logging.getLogger()
        self._filtered_handlers = list(root.handlers)
        for handler in self._filtered_handlers:
            handler.addFilter(self._filter)
        root.addHandler(self.counter)
        return self.counter

    def __exit__(self, *exc_info: object) -> None:
        root = logging.getLogger()
        root.removeHandler(self.counter)
        for handler in self._filtered_handlers:
            handler.removeFilter(self._filter)
        self._filtered_handlers = []
