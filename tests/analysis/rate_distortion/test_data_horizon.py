"""Tests for per-horizon chunk windowing in versatil.analysis.rate_distortion.data."""

import numpy as np

from versatil.analysis.rate_distortion.data import (
    ActionComponentLayout,
    PerStepContext,
    chunk_data_at_horizon,
)


class _IdentityField:
    def unnormalize(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)


class _IdentityNormalizer:
    def __getitem__(self, key: str) -> _IdentityField:
        return _IdentityField()


def _context(episode_selection_mask: np.ndarray) -> PerStepContext:
    per_step_actions = (
        np.random.default_rng(0).standard_normal((100, 7)).astype(np.float32)
    )
    return PerStepContext(
        per_step_actions=per_step_actions,
        normalizer=_IdentityNormalizer(),
        layout=[ActionComponentLayout("action", 0, 7, True)],
        episode_ends=np.array([50, 100]),
        episode_selection_mask=episode_selection_mask,
    )


def test_chunk_count_and_shape_track_horizon() -> None:
    context = _context(np.array([True, True]))
    chunks_10 = chunk_data_at_horizon(context=context, horizon=10)
    assert chunks_10.chunks_normalized.shape == ((50 - 10 + 1) * 2, 10, 7)
    chunks_20 = chunk_data_at_horizon(context=context, horizon=20)
    assert chunks_20.chunks_normalized.shape == ((50 - 20 + 1) * 2, 20, 7)


def test_unselected_episode_is_excluded() -> None:
    context = _context(np.array([True, False]))
    chunks = chunk_data_at_horizon(context=context, horizon=10)
    assert chunks.chunks_normalized.shape[0] == (50 - 10 + 1)
