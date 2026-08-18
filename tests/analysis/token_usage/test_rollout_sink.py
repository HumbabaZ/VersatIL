"""Tests for versatil.analysis.token_usage.rollout_sink module."""

import json

import numpy as np
import pytest
import torch

from versatil.analysis.token_usage.rollout_sink import (
    RolloutTokenField,
    RolloutTokenSink,
)


def _read_rows(jsonl_path):
    with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
        return [json.loads(line) for line in jsonl_file]


class TestRolloutTokenSinkRecord:
    def test_records_one_row_per_environment_in_batch(self, tmp_path):
        sink = RolloutTokenSink(output_path=tmp_path / "rollout.jsonl")
        sink.record(action_tokens=torch.tensor([[1, 2, 3], [4, 5, 6]]))
        flushed = sink.flush()

        rows = _read_rows(flushed)
        assert [row[RolloutTokenField.TOKENS.value] for row in rows] == [
            [1, 2, 3],
            [4, 5, 6],
        ]
        assert [row[RolloutTokenField.ENV_INDEX.value] for row in rows] == [0, 1]

    def test_promotes_single_environment_to_one_row(self, tmp_path):
        sink = RolloutTokenSink(output_path=tmp_path / "rollout.jsonl")
        sink.record(action_tokens=torch.tensor([7, 8, 9]))
        rows = _read_rows(sink.flush())
        assert len(rows) == 1
        assert rows[0][RolloutTokenField.TOKENS.value] == [7, 8, 9]

    def test_call_index_increments_per_record(self, tmp_path):
        sink = RolloutTokenSink(output_path=tmp_path / "rollout.jsonl")
        sink.record(action_tokens=torch.tensor([[1]]))
        sink.record(action_tokens=torch.tensor([[2]]))
        rows = _read_rows(sink.flush())
        assert [row[RolloutTokenField.CALL_INDEX.value] for row in rows] == [0, 1]

    def test_context_is_attached_to_rows(self, tmp_path):
        sink = RolloutTokenSink(output_path=tmp_path / "rollout.jsonl")
        sink.set_context(context={"rollout_index": 3, "task_name": "sequential"})
        sink.record(action_tokens=torch.tensor([[1, 2]]))
        rows = _read_rows(sink.flush())
        assert rows[0][RolloutTokenField.CONTEXT.value] == {
            "rollout_index": 3,
            "task_name": "sequential",
        }

    def test_context_change_only_affects_later_rows(self, tmp_path):
        sink = RolloutTokenSink(output_path=tmp_path / "rollout.jsonl")
        sink.set_context(context={"rollout_index": 0})
        sink.record(action_tokens=torch.tensor([[1]]))
        sink.set_context(context={"rollout_index": 1})
        sink.record(action_tokens=torch.tensor([[2]]))
        rows = _read_rows(sink.flush())
        assert rows[0][RolloutTokenField.CONTEXT.value] == {"rollout_index": 0}
        assert rows[1][RolloutTokenField.CONTEXT.value] == {"rollout_index": 1}

    def test_rejects_three_dimensional_tokens(self, tmp_path):
        sink = RolloutTokenSink(output_path=tmp_path / "rollout.jsonl")
        with pytest.raises(ValueError, match="Expected 1D or 2D action tokens"):
            sink.record(action_tokens=torch.zeros((2, 2, 2), dtype=torch.long))


class TestRolloutTokenSinkFlush:
    def test_flush_clears_buffer(self, tmp_path):
        sink = RolloutTokenSink(output_path=tmp_path / "rollout.jsonl")
        sink.record(action_tokens=torch.tensor([[1]]))
        sink.flush()
        sink.flush()
        assert _read_rows(tmp_path / "rollout.jsonl") == []

    def test_flush_accepts_numpy_tokens(self, tmp_path):
        sink = RolloutTokenSink(output_path=tmp_path / "rollout.jsonl")
        sink.record(action_tokens=np.array([[3, 4]], dtype=np.int64))
        rows = _read_rows(sink.flush())
        assert rows[0][RolloutTokenField.TOKENS.value] == [3, 4]
