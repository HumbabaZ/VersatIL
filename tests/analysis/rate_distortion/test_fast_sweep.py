"""Tests for versatil.analysis.rate_distortion.fast_sweep module."""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from versatil.analysis.rate_distortion.data import (
    ActionChunkData,
    ActionComponentLayout,
)
from versatil.analysis.rate_distortion.fast_sweep import load_fast_class, run_fast_cell


class _IdentityField:
    def unnormalize(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)


class _IdentityNormalizer:
    def __getitem__(self, key: str) -> _IdentityField:
        return _IdentityField()


def _chunk_data(chunks: np.ndarray) -> ActionChunkData:
    layout = [
        ActionComponentLayout(
            key="ee_pos_action", start=0, end=chunks.shape[2], needs_normalization=True
        )
    ]
    return ActionChunkData(
        chunks_normalized=chunks, normalizer=_IdentityNormalizer(), layout=layout
    )


def test_run_fast_cell_skips_cell_below_alphabet_floor() -> None:
    chunk_data = _chunk_data(np.zeros((2, 4, 3), dtype=np.float32))
    with (
        patch(
            "versatil.analysis.rate_distortion.fast_sweep.dct_alphabet_size",
            return_value=5000,
        ),
        patch(
            "versatil.analysis.rate_distortion.fast_sweep.fit_fast_processor"
        ) as mock_fit,
    ):
        row = run_fast_cell(
            fast_class=MagicMock(),
            chunk_data=chunk_data,
            scale=50.0,
            vocab_size=256,
            sweep="vocab",
            is_operating_point=False,
        )
    assert row["feasible"] is False
    mock_fit.assert_not_called()
    assert "bits_per_chunk" not in row


def test_run_fast_cell_fits_encodes_and_scores_feasible_cell() -> None:
    chunks = np.zeros((2, 4, 3), dtype=np.float32)
    chunk_data = _chunk_data(chunks)
    token_lists = [[1, 2], [1, 2, 3]]
    processor = MagicMock(return_value=token_lists)
    processor.decode.return_value = np.zeros((2, 4, 3), dtype=np.float32)
    with (
        patch(
            "versatil.analysis.rate_distortion.fast_sweep.dct_alphabet_size",
            return_value=4,
        ),
        patch(
            "versatil.analysis.rate_distortion.fast_sweep.fit_fast_processor",
            return_value=processor,
        ) as mock_fit,
    ):
        row = run_fast_cell(
            fast_class=MagicMock(),
            chunk_data=chunk_data,
            scale=10.0,
            vocab_size=8,
            sweep="scale",
            is_operating_point=True,
        )
    mock_fit.assert_called_once()
    assert mock_fit.call_args.kwargs["scale"] == 10.0
    assert mock_fit.call_args.kwargs["vocab_size"] == 8
    assert mock_fit.call_args.kwargs["time_horizon"] == 4
    assert mock_fit.call_args.kwargs["action_dim"] == 3
    assert row["feasible"] is True
    assert row["bits_per_chunk"] == pytest.approx(2.5 * math.log2(8))
    assert row["rmse_continuous"] == pytest.approx(0.0)


@pytest.fixture
def fast_class():
    return load_fast_class("physical-intelligence/fast")


@pytest.fixture
def continuous_chunk_data(rng) -> ActionChunkData:
    chunks = (rng.standard_normal((64, 8, 3)) * 0.3).astype(np.float32)
    return _chunk_data(chunks)


@pytest.mark.integration
def test_vocab_sweep_leaves_distortion_exactly_flat(
    fast_class, continuous_chunk_data
) -> None:
    small_vocab = run_fast_cell(
        fast_class=fast_class,
        chunk_data=continuous_chunk_data,
        scale=10.0,
        vocab_size=256,
        sweep="vocab",
        is_operating_point=False,
    )
    large_vocab = run_fast_cell(
        fast_class=fast_class,
        chunk_data=continuous_chunk_data,
        scale=10.0,
        vocab_size=1024,
        sweep="vocab",
        is_operating_point=True,
    )
    assert small_vocab["feasible"] and large_vocab["feasible"]
    assert large_vocab["rmse_continuous"] == pytest.approx(
        small_vocab["rmse_continuous"], abs=1e-9
    )


@pytest.mark.integration
def test_finer_scale_does_not_increase_distortion(
    fast_class, continuous_chunk_data
) -> None:
    coarse = run_fast_cell(
        fast_class=fast_class,
        chunk_data=continuous_chunk_data,
        scale=2.0,
        vocab_size=1024,
        sweep="scale",
        is_operating_point=False,
    )
    fine = run_fast_cell(
        fast_class=fast_class,
        chunk_data=continuous_chunk_data,
        scale=20.0,
        vocab_size=1024,
        sweep="scale",
        is_operating_point=False,
    )
    assert fine["rmse_continuous"] <= coarse["rmse_continuous"] + 1e-6
