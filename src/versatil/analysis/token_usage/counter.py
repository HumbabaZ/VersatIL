"""Per-token frequency counting for tokenizer usage analysis."""

import enum
import json
from pathlib import Path

import numpy as np


class TokenUsageField(enum.StrEnum):
    """Field names for serialized token-usage counts."""

    TOKENS = "tokens"
    COUNTS = "counts"
    LABEL = "label"
    TOTAL = "total"


class TokenUsageCounter:
    """Accumulates per-token frequency over an integer token stream.

    Counts an arbitrary integer token domain, including the negative DCT
    coefficient tokens produced by FAST reverse-BPE, so bin IDs, FAST BPE IDs,
    and FAST coefficient tokens all share one counter type.
    """

    def __init__(self, label: str = ""):
        """Initialize an empty counter.

        Args:
            label: Human-readable name for the stream, e.g. "train" or "rollout".
        """
        self.label = label
        self._counts: dict[int, int] = {}

    def update(self, token_ids: np.ndarray | list[int]) -> None:
        """Add token IDs to the running frequency counts.

        Args:
            token_ids: Integer token IDs of any shape; flattened before counting.

        Raises:
            ValueError: If the token IDs are not an integer dtype.
        """
        tokens_array = np.asarray(token_ids).reshape(-1)
        if tokens_array.size == 0:
            return
        if not np.issubdtype(tokens_array.dtype, np.integer):
            raise ValueError(
                f"Token IDs must be integers, got dtype {tokens_array.dtype}"
            )
        values, counts = np.unique(tokens_array, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist(), strict=True):
            self._counts[value] = self._counts.get(value, 0) + count

    @property
    def support(self) -> set[int]:
        """Token IDs observed at least once."""
        return {token for token, count in self._counts.items() if count > 0}

    @property
    def total(self) -> int:
        """Total number of counted tokens."""
        return int(sum(self._counts.values()))

    def counts_as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return sorted ``(token_ids, counts)`` arrays."""
        if not self._counts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        tokens = np.array(sorted(self._counts.keys()), dtype=np.int64)
        counts = np.array(
            [self._counts[token] for token in tokens.tolist()], dtype=np.int64
        )
        return tokens, counts

    def probability(self, token_id: int) -> float:
        """Empirical probability of one token ID."""
        total = self.total
        if total == 0:
            return 0.0
        return self._counts.get(token_id, 0) / total

    def save(self, path: str | Path) -> Path:
        """Save counts to a ``.npz`` file plus a ``.json`` sidecar.

        Args:
            path: Destination path; a ``.npz`` suffix is enforced.

        Returns:
            The resolved ``.npz`` path written.
        """
        npz_path = Path(path)
        if npz_path.suffix != ".npz":
            npz_path = npz_path.with_suffix(".npz")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        tokens, counts = self.counts_as_arrays()
        np.savez(
            npz_path,
            **{
                TokenUsageField.TOKENS.value: tokens,
                TokenUsageField.COUNTS.value: counts,
            },
        )
        metadata_path = npz_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    TokenUsageField.LABEL.value: self.label,
                    TokenUsageField.TOTAL.value: self.total,
                }
            )
        )
        return npz_path

    @classmethod
    def load(cls, path: str | Path) -> "TokenUsageCounter":
        """Load counts from a ``.npz`` file written by ``save``."""
        npz_path = Path(path)
        if npz_path.suffix != ".npz":
            npz_path = npz_path.with_suffix(".npz")
        archive = np.load(npz_path)
        tokens = archive[TokenUsageField.TOKENS.value]
        counts = archive[TokenUsageField.COUNTS.value]
        label = ""
        metadata_path = npz_path.with_suffix(".json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            label = metadata.get(TokenUsageField.LABEL.value, "")
        counter = cls(label=label)
        counter._counts = {
            int(token): int(count)
            for token, count in zip(tokens.tolist(), counts.tolist(), strict=True)
        }
        return counter
