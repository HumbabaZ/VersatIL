"""Tests for versatil.analysis.tip1_noise.measure_token_length module."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.analysis.tip1_noise.measure_token_length import (
    MEASUREMENT_CAP,
    measure_cell,
    measurement_overrides,
    stage_fast_cells,
    suggested_cap,
)
from versatil.analysis.tip1_noise.sweep import (
    ACTION_TOKENIZER_MAX_TOKEN_LEN_KEY,
    CONDITIONAL_METHOD_CONFIG,
    CONDITIONAL_TASK,
    NOISY_ZARR_DIR_ENV,
    POSITION,
    DataCell,
    TrainCell,
)
from versatil.data.constants import SampleKey


@pytest.fixture
def noisy_store_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv(NOISY_ZARR_DIR_ENV, str(tmp_path))
    return tmp_path


@pytest.fixture
def fast_cell_factory() -> Callable[..., TrainCell]:
    def factory(trajectory_length: int = 240) -> TrainCell:
        return TrainCell(
            data=DataCell(
                task=CONDITIONAL_TASK,
                injection=POSITION,
                smoothing_sigma=0.0,
                sigma_multiplier=1.0,
                data_seed=42,
                trajectory_length=trajectory_length,
            ),
            method="fast",
            seed=0,
        )

    return factory


@pytest.fixture
def padded_dataset_factory() -> Callable[..., MagicMock]:
    def factory(kept_tokens_per_chunk: list[int], padded_length: int = 16):
        samples = []
        for kept in kept_tokens_per_chunk:
            is_pad = torch.ones(padded_length, dtype=torch.bool)
            is_pad[:kept] = False
            samples.append(
                {SampleKey.ACTION.value: {SampleKey.IS_PAD_ACTION.value: is_pad}}
            )
        dataset = MagicMock()
        dataset.__len__.return_value = len(samples)
        dataset.__getitem__.side_effect = lambda index: samples[index]
        return dataset

    return factory


@pytest.mark.unit
def test_measurement_overrides_carry_the_training_horizon_and_the_raised_cap(
    noisy_store_root: Path, fast_cell_factory: Callable[..., TrainCell]
):
    overrides = measurement_overrides(fast_cell_factory(trajectory_length=240))

    assert "task.prediction_horizon=239" in overrides
    assert "task.dataset_schema.trajectory_length=240" in overrides
    assert f"{ACTION_TOKENIZER_MAX_TOKEN_LEN_KEY}={MEASUREMENT_CAP}" in overrides
    assert "experiment.device=cpu" in overrides
    assert sum(ACTION_TOKENIZER_MAX_TOKEN_LEN_KEY in item for item in overrides) == 1


@pytest.mark.unit
def test_measure_cell_reports_the_largest_action_token_count_across_splits(
    noisy_store_root: Path,
    fast_cell_factory: Callable[..., TrainCell],
    padded_dataset_factory: Callable[..., MagicMock],
):
    cell = fast_cell_factory(trajectory_length=240)
    train_loader = MagicMock()
    train_loader.dataset = padded_dataset_factory(kept_tokens_per_chunk=[5, 9])
    val_loader = MagicMock()
    val_loader.dataset = padded_dataset_factory(kept_tokens_per_chunk=[12])

    with (
        patch(
            "versatil.analysis.tip1_noise.measure_token_length.initialize_config_dir"
        ),
        patch(
            "versatil.analysis.tip1_noise.measure_token_length.compose"
        ) as mock_compose,
        patch("versatil.analysis.tip1_noise.measure_token_length.instantiate"),
        patch(
            "versatil.analysis.tip1_noise.measure_token_length.get_dataloaders",
            return_value=(train_loader, val_loader, None, None, None),
        ),
    ):
        row = measure_cell(cell)

    assert (
        mock_compose.call_args.kwargs["config_name"]
        == (CONDITIONAL_METHOD_CONFIG["fast"])
    )
    assert "task.prediction_horizon=239" in mock_compose.call_args.kwargs["overrides"]
    # Each kept count includes the EOS, so the largest action count is 12 - 1.
    assert row["max_action_tokens"] == 11
    assert row["chunks"] == 3
    assert row["trajectory_length"] == 240


@pytest.mark.unit
def test_stage_fast_cells_keeps_one_fast_cell_per_store(noisy_store_root: Path):
    cells = stage_fast_cells("rate_conditional_s0")

    assert [cell.data.trajectory_length for cell in cells] == [60, 120, 240]
    assert {cell.method for cell in cells} == {"fast"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "max_action_tokens, margin, expected",
    [
        (94, 8, 103),
        (400, 0, 401),
    ],
)
def test_suggested_cap_clears_the_maximum_and_the_eos(
    max_action_tokens: int, margin: int, expected: int
):
    assert suggested_cap(max_action_tokens, margin) == expected
