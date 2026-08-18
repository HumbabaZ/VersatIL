"""Rollout-time capture of predicted action tokens for usage analysis."""

import enum
import json
from pathlib import Path

import numpy as np
import torch

RolloutRow = dict[str, int | list[int] | dict[str, int | str]]


class RolloutTokenField(enum.StrEnum):
    """Field names for one captured rollout token row."""

    CALL_INDEX = "call_index"
    ENV_INDEX = "env_index"
    TOKENS = "tokens"
    CONTEXT = "context"


class RolloutTokenSink:
    """Buffers predicted action token IDs during rollout and writes JSONL.

    Records the raw model-vocab token IDs emitted at each
    ``Policy.predict_action`` call, one JSON line per (call, environment). Token
    IDs are stored unchanged; mapping to local or coefficient space happens
    later in the report so the rollout hook stays minimal. Structurally
    satisfies the ``TokenUsageSink`` protocol consumed by ``Policy``.
    """

    def __init__(self, output_path: str | Path):
        """Initialize an empty sink.

        Args:
            output_path: JSONL destination written by ``flush``.
        """
        self._output_path = Path(output_path)
        self._rows: list[RolloutRow] = []
        self._call_index = 0
        self._context: dict[str, int | str] = {}

    def set_context(self, context: dict[str, int | str] | None) -> None:
        """Set tags attached to rows recorded next, e.g. rollout index or task."""
        self._context = dict(context) if context is not None else {}

    def record(self, action_tokens: torch.Tensor | np.ndarray) -> None:
        """Record one prediction's model-vocab action token IDs.

        Args:
            action_tokens: Predicted token IDs with shape (batch, sequence_len)
                or (sequence_len,) for a single environment.

        Raises:
            ValueError: If the token tensor is neither 1D nor 2D.
        """
        tokens_array = self._to_numpy(action_tokens=action_tokens)
        if tokens_array.ndim == 1:
            tokens_array = tokens_array[None, :]
        if tokens_array.ndim != 2:
            raise ValueError(
                f"Expected 1D or 2D action tokens, got shape {tokens_array.shape}"
            )
        for env_index, row_tokens in enumerate(tokens_array):
            self._rows.append(
                {
                    RolloutTokenField.CALL_INDEX.value: self._call_index,
                    RolloutTokenField.ENV_INDEX.value: env_index,
                    RolloutTokenField.TOKENS.value: row_tokens.tolist(),
                    RolloutTokenField.CONTEXT.value: dict(self._context),
                }
            )
        self._call_index += 1

    def flush(self) -> Path:
        """Write buffered rows to the JSONL output path and clear the buffer."""
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("w", encoding="utf-8") as output_file:
            for row in self._rows:
                output_file.write(json.dumps(row) + "\n")
        self._rows.clear()
        return self._output_path

    @staticmethod
    def _to_numpy(action_tokens: torch.Tensor | np.ndarray) -> np.ndarray:
        """Return a CPU int64 NumPy view of predicted token IDs."""
        if isinstance(action_tokens, torch.Tensor):
            return action_tokens.detach().cpu().numpy().astype(np.int64)
        return np.asarray(action_tokens).astype(np.int64)
