"""Tests for versatil.analysis.rate_distortion.metrics module."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from versatil.analysis.rate_distortion.data import ActionComponentLayout
from versatil.analysis.rate_distortion.metrics import (
    check_near_lossless,
    dct_alphabet_size,
    rate_metrics,
    reconstruction_distortion,
)


class _IdentityField:
    def unnormalize(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)


class _IdentityNormalizer:
    def __getitem__(self, key: str) -> _IdentityField:
        return _IdentityField()


@pytest.fixture
def continuous_gripper_layout() -> list[ActionComponentLayout]:
    return [
        ActionComponentLayout(
            key="ee_pos_action", start=0, end=2, needs_normalization=True
        ),
        ActionComponentLayout(
            key="gripper_state_action", start=2, end=3, needs_normalization=False
        ),
    ]


def test_rate_metrics_uses_mean_length_and_log2_vocab_bits() -> None:
    metrics = rate_metrics(token_lists=[[1, 2, 3], [1, 2, 3, 4, 5]], vocab_size=1024)
    assert metrics["mean_token_len"] == 4.0
    assert metrics["bits_per_chunk"] == 40.0


def test_dct_alphabet_size_grows_with_scale(rng) -> None:
    chunks = rng.standard_normal((8, 10, 4)).astype(np.float32)
    small = dct_alphabet_size(chunks_normalized=chunks, scale=2.0)
    large = dct_alphabet_size(chunks_normalized=chunks, scale=20.0)
    assert large >= small


def test_reconstruction_distortion_reports_original_space_continuous_rmse(
    continuous_gripper_layout,
) -> None:
    ground_truth = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
    reconstruction = np.array([[[0.3, 0.4, -0.2]]], dtype=np.float32)
    metrics = reconstruction_distortion(
        ground_truth_normalized=ground_truth,
        reconstruction_normalized=reconstruction,
        normalizer=_IdentityNormalizer(),
        layout=continuous_gripper_layout,
    )
    expected_rmse = float(np.sqrt((0.3**2 + 0.4**2) / 2))
    assert metrics["rmse_ee_pos_action"] == pytest.approx(expected_rmse, abs=1e-6)
    assert metrics["rmse_continuous"] == pytest.approx(expected_rmse, abs=1e-6)


def test_reconstruction_distortion_scores_gripper_as_classification_mismatch(
    continuous_gripper_layout,
) -> None:
    ground_truth = np.array(
        [[[0.0, 0.0, 1.0]], [[0.0, 0.0, -1.0]], [[0.0, 0.0, 1.0]]], dtype=np.float32
    )
    reconstruction = np.array(
        [[[0.0, 0.0, 0.5]], [[0.0, 0.0, 0.4]], [[0.0, 0.0, -0.2]]], dtype=np.float32
    )
    metrics = reconstruction_distortion(
        ground_truth_normalized=ground_truth,
        reconstruction_normalized=reconstruction,
        normalizer=_IdentityNormalizer(),
        layout=continuous_gripper_layout,
    )
    assert metrics["gripper_mismatch_rate"] == pytest.approx(2.0 / 3.0, abs=1e-6)


@pytest.mark.parametrize(
    "distortion_rmse, expectation",
    [
        (0.01, does_not_raise()),
        (
            0.5,
            pytest.raises(
                ValueError,
                match=(
                    "Near-lossless FAST round trip did not reproduce the input "
                    r"\(continuous RMSE 0.5000 > tolerance 0.0500\)"
                ),
            ),
        ),
    ],
)
def test_check_near_lossless_rejects_above_tolerance(
    distortion_rmse, expectation
) -> None:
    with expectation:
        check_near_lossless(distortion_rmse=distortion_rmse, tolerance=0.05)
