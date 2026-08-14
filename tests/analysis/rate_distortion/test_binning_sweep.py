"""Tests for versatil.analysis.rate_distortion.binning_sweep and rate metrics."""

import numpy as np
import pytest

from versatil.analysis.rate_distortion.binning_sweep import run_binning_cell
from versatil.analysis.rate_distortion.data import (
    ActionChunkData,
    ActionComponentLayout,
)
from versatil.analysis.rate_distortion.metrics import binning_rate, bits_per_step


class _IdentityField:
    def unnormalize(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)


class _IdentityNormalizer:
    def __getitem__(self, key: str) -> _IdentityField:
        return _IdentityField()


def _layout() -> list[ActionComponentLayout]:
    return [
        ActionComponentLayout("ee_pos_action", 0, 3, True),
        ActionComponentLayout("ee_ori_action", 3, 6, True),
        ActionComponentLayout("gripper_state_action", 6, 7, False),
    ]


@pytest.fixture
def chunk_data(rng) -> ActionChunkData:
    chunks = (rng.standard_normal((64, 10, 7)) * 0.3).clip(-1, 1).astype(np.float32)
    chunks[:, :, 6] = np.sign(rng.standard_normal((64, 10)))
    return ActionChunkData(
        chunks_normalized=chunks, normalizer=_IdentityNormalizer(), layout=_layout()
    )


def test_binning_rate_is_exact() -> None:
    rate = binning_rate(num_bins=256, time_horizon=10, action_dim=7)
    assert rate["mean_token_len"] == 70.0
    assert rate["bits_per_chunk"] == pytest.approx(70 * 8.0)


def test_bits_per_step_divides_by_horizon() -> None:
    assert bits_per_step(bits_per_chunk=560.0, time_horizon=10) == pytest.approx(56.0)


def test_binning_distortion_decreases_with_more_bins(chunk_data) -> None:
    coarse = run_binning_cell(
        chunk_data=chunk_data, num_bins=16, binning_strategy="quantile"
    )
    fine = run_binning_cell(
        chunk_data=chunk_data, num_bins=256, binning_strategy="quantile"
    )
    assert fine["rmse_continuous"] <= coarse["rmse_continuous"] + 1e-9
    assert coarse["bits_per_chunk"] < fine["bits_per_chunk"]
