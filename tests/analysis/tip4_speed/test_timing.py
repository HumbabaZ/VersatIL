"""Tests for the Tip 4 latency sink and the detok warning counter."""

import logging

import torch

from versatil.analysis.tip4_speed.timing import (
    SEGMENTS,
    DetokWarningCounter,
    SegmentTimerSink,
    SilencedDetokWarnings,
)


class TestSegmentTimerSink:
    def test_records_all_segments_and_generated_tokens(self) -> None:
        sink = SegmentTimerSink(device=torch.device("cpu"))

        sink.start()
        for segment in SEGMENTS:
            sink.mark(segment)
        sink.finish(generated_tokens=17)

        assert len(sink.records) == 1
        record = sink.records[0]
        assert set(record.segments_ms) == set(SEGMENTS)
        assert record.generated_tokens == 17
        assert all(value >= 0.0 for value in record.segments_ms.values())

    def test_segments_sum_close_to_total(self) -> None:
        sink = SegmentTimerSink(device=torch.device("cpu"))

        sink.start()
        for segment in SEGMENTS:
            sink.mark(segment)
        sink.finish(generated_tokens=None)

        record = sink.records[0]
        # finish() reads the clock once more after the last mark, so the total
        # can only exceed the segment sum, and only by clock overhead.
        assert record.total_ms >= sum(record.segments_ms.values())
        assert record.total_ms - sum(record.segments_ms.values()) < 5.0

    def test_consecutive_calls_append_independent_records(self) -> None:
        sink = SegmentTimerSink(device=torch.device("cpu"))

        for tokens in (3, None):
            sink.start()
            for segment in SEGMENTS:
                sink.mark(segment)
            sink.finish(generated_tokens=tokens)

        assert [record.generated_tokens for record in sink.records] == [3, None]


def _discretizer_record() -> logging.LogRecord:
    return logging.LogRecord(
        name="root",
        level=logging.WARNING,
        pathname="/repo/src/versatil/data/tokenization/action_discretizer.py",
        lineno=265,
        msg="FAST token sequence decodes to 3 DCT coefficients",
        args=(),
        exc_info=None,
    )


def _unrelated_record() -> logging.LogRecord:
    return logging.LogRecord(
        name="root",
        level=logging.WARNING,
        pathname="/repo/src/versatil/other_module.py",
        lineno=1,
        msg="unrelated",
        args=(),
        exc_info=None,
    )


class TestDetokWarningCounter:
    def test_counts_only_discretizer_records_and_take_resets(self) -> None:
        counter = DetokWarningCounter()

        counter.emit(_discretizer_record())
        counter.emit(_unrelated_record())
        counter.emit(_discretizer_record())

        assert counter.take() == 2
        assert counter.take() == 0

    def test_silencer_counts_while_hiding_from_other_handlers(self) -> None:
        root = logging.getLogger()
        captured: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record)

        capture = _Capture(level=logging.WARNING)
        root.addHandler(capture)
        try:
            with SilencedDetokWarnings() as counter:
                root.handle(_discretizer_record())
                root.handle(_unrelated_record())
                assert counter.take() == 1
            assert [record.msg for record in captured] == ["unrelated"]
            # The filter is removed on exit: discretizer records flow again.
            root.handle(_discretizer_record())
            assert len(captured) == 2
        finally:
            root.removeHandler(capture)
