"""Tests for versatil.analysis.tip1_noise.sweep module."""

import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from versatil.analysis.tip1_noise.sweep import (
    ACTION_TOKENIZER_MAX_TOKEN_LEN_KEY,
    CONDITIONAL_METHOD_CONFIG,
    CONDITIONAL_TASK,
    FAST_MAX_TOKEN_LEN_BY_LENGTH,
    FAST_OVERRIDES,
    FINAL_METHODS,
    GPT_MAX_SEQ_LEN_KEY,
    METHOD_CONFIG,
    NOISY_ZARR_DIR_ENV,
    POSITION,
    DataCell,
    TrainCell,
    add_effective_bins,
    check_paths_unique,
    data_cells,
    fast_max_token_len,
    measured_snr,
    method_config,
    noisy_zarr_root,
    reference_cells,
    stage_cells,
)


@pytest.fixture
def noisy_store_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv(NOISY_ZARR_DIR_ENV, str(tmp_path))
    return tmp_path


@pytest.fixture
def data_cell_factory() -> Callable[..., DataCell]:
    def factory(
        task: str = CONDITIONAL_TASK,
        sigma_multiplier: float = 2.0,
        data_seed: int = 42,
        num_episodes: int | None = None,
        trajectory_length: int = 60,
    ) -> DataCell:
        return DataCell(
            task=task,
            injection=POSITION,
            smoothing_sigma=0.0,
            sigma_multiplier=sigma_multiplier,
            data_seed=data_seed,
            num_episodes=num_episodes,
            trajectory_length=trajectory_length,
        )

    return factory


@pytest.fixture
def train_cell_factory(
    data_cell_factory: Callable[..., DataCell],
) -> Callable[..., TrainCell]:
    def factory(
        method: str = "fast",
        trajectory_length: int = 60,
        sigma_multiplier: float = 1.0,
    ) -> TrainCell:
        return TrainCell(
            data=data_cell_factory(
                sigma_multiplier=sigma_multiplier,
                trajectory_length=trajectory_length,
            ),
            method=method,
            seed=0,
        )

    return factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "task, method, expected",
    [
        (CONDITIONAL_TASK, "fast", CONDITIONAL_METHOD_CONFIG["fast"]),
        (CONDITIONAL_TASK, "qfat", CONDITIONAL_METHOD_CONFIG["qfat"]),
        (CONDITIONAL_TASK, "bcat", CONDITIONAL_METHOD_CONFIG["bcat"]),
        ("sequential", "fast", METHOD_CONFIG["fast"]),
        ("corridor", "bcat", METHOD_CONFIG["bcat"]),
    ],
)
def test_method_config_uses_context_configs_only_for_conditional_task(
    task: str, method: str, expected: str
):
    assert method_config(task=task, method=method) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "method, expectation",
    [
        ("fast", does_not_raise()),
        ("unknown", pytest.raises(KeyError, match=re.escape("'unknown'"))),
    ],
)
def test_method_config_rejects_unknown_method(
    method: str, expectation: AbstractContextManager
):
    with expectation:
        method_config(task=CONDITIONAL_TASK, method=method)


@pytest.mark.unit
@pytest.mark.parametrize(
    "task, sigma_multiplier, expected_noise_std",
    [
        (CONDITIONAL_TASK, 2.0, 0.016),
        (CONDITIONAL_TASK, 1.0, 0.008),
        ("sequential", 3.0, 0.036),
    ],
)
def test_data_cell_noise_std_scales_the_task_default(
    data_cell_factory,
    task: str,
    sigma_multiplier: float,
    expected_noise_std: float,
):
    cell = data_cell_factory(task=task, sigma_multiplier=sigma_multiplier)
    assert cell.noise_std == pytest.approx(expected_noise_std)


@pytest.mark.unit
def test_data_cell_name_and_path_carry_every_generation_parameter(
    noisy_store_root: Path, data_cell_factory
):
    cell = data_cell_factory(task=CONDITIONAL_TASK, sigma_multiplier=2.0, data_seed=43)

    assert cell.name == "conditional__inj-position__band-high__sig-2__dseed-43"
    assert cell.zarr_path == str(
        noisy_store_root / "tip1_noisy_synthetic" / f"{cell.name}.zarr"
    )


@pytest.mark.unit
def test_noisy_zarr_root_requires_the_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv(NOISY_ZARR_DIR_ENV, raising=False)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"{NOISY_ZARR_DIR_ENV} is not set. The Tip 1 sweep writes "
            "noise-corrupted datasets and keeps them out of the shared clean "
            f"store on purpose, so point {NOISY_ZARR_DIR_ENV} at a directory "
            "reserved for them, for example "
            "/data/horse/ws/qizh093f-versatil/noisy_zarr."
        ),
    ):
        noisy_zarr_root()


@pytest.mark.unit
def test_final_conditional_stage_runs_every_arm_on_shared_stores(
    noisy_store_root: Path,
):
    cells = stage_cells("final_conditional_s0")
    stores = data_cells(cells)

    assert len(cells) == 16
    assert len(stores) == 4
    check_paths_unique(stores)
    assert [cell.method for cell in cells[: len(FINAL_METHODS)]] == list(FINAL_METHODS)
    assert sorted({cell.data.sigma_multiplier for cell in cells}) == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    assert {cell.seed for cell in cells} == {0}
    assert {cell.data.task for cell in cells} == {CONDITIONAL_TASK}


@pytest.mark.unit
@pytest.mark.parametrize(
    "method, expects_fast_cap",
    [
        ("fast", True),
        ("qfat", False),
    ],
)
def test_train_cell_command_selects_task_config_and_fast_cap(
    noisy_store_root: Path,
    data_cell_factory,
    method: str,
    expects_fast_cap: bool,
):
    cell = TrainCell(data=data_cell_factory(), method=method, seed=0)

    command = cell.command(matched=True)

    config_index = command.index("--config-name") + 1
    assert command[config_index] == CONDITIONAL_METHOD_CONFIG[method]
    assert f"experiment.name={cell.name}" in command
    assert "task/dataset_schema=synthetic/conditional_circle" in command
    assert all(override in command for override in FAST_OVERRIDES) is expects_fast_cap


@pytest.mark.unit
@pytest.mark.parametrize(
    "trajectory_length, num_episodes, expected_suffix",
    [
        (60, None, ""),
        (240, None, "__T-240"),
        (240, 50, "__ep-50__T-240"),
        (60, 50, "__ep-50"),
    ],
)
def test_data_cell_name_carries_trajectory_length_only_when_not_default(
    noisy_store_root: Path,
    data_cell_factory: Callable[..., DataCell],
    trajectory_length: int,
    num_episodes: int | None,
    expected_suffix: str,
):
    cell = data_cell_factory(
        trajectory_length=trajectory_length, num_episodes=num_episodes
    )

    expected = f"conditional__inj-position__band-high__sig-2__dseed-42{expected_suffix}"
    assert cell.name == expected
    assert cell.zarr_path.endswith(f"/{expected}.zarr")


@pytest.mark.unit
@pytest.mark.parametrize(
    "trajectory_length, sigma_multiplier, expected_noise_std",
    [
        (60, 1.0, 0.008),
        (120, 1.0, 0.004),
        (240, 1.0, 0.002),
        (240, 2.0, 0.004),
    ],
)
def test_data_cell_noise_std_scales_inversely_with_trajectory_length(
    data_cell_factory: Callable[..., DataCell],
    trajectory_length: int,
    sigma_multiplier: float,
    expected_noise_std: float,
):
    cell = data_cell_factory(
        trajectory_length=trajectory_length, sigma_multiplier=sigma_multiplier
    )
    assert cell.noise_std == pytest.approx(expected_noise_std)


@pytest.mark.unit
@pytest.mark.parametrize(
    "trajectory_length, expected_present",
    [
        (60, False),
        (240, True),
    ],
)
def test_schema_overrides_set_trajectory_length_only_when_not_default(
    noisy_store_root: Path,
    data_cell_factory: Callable[..., DataCell],
    trajectory_length: int,
    expected_present: bool,
):
    overrides = data_cell_factory(
        sigma_multiplier=1.0, trajectory_length=trajectory_length
    ).schema_overrides()

    length_override = f"task.dataset_schema.trajectory_length={trajectory_length}"
    assert (length_override in overrides) is expected_present
    if expected_present:
        assert "task.dataset_schema.noise_std=0.002" in overrides


@pytest.mark.unit
@pytest.mark.parametrize(
    "method, trajectory_length, expected",
    [
        ("qfat", 60, []),
        ("fast", 60, []),
        ("qfat", 240, ["task.prediction_horizon=240"]),
        ("bcat", 240, ["task.prediction_horizon=240"]),
        (
            "fast",
            240,
            ["task.prediction_horizon=239", f"{GPT_MAX_SEQ_LEN_KEY}=1024"],
        ),
        (
            "binned",
            120,
            [
                "task.prediction_horizon=119",
                f"{ACTION_TOKENIZER_MAX_TOKEN_LEN_KEY}=240",
                f"{GPT_MAX_SEQ_LEN_KEY}=1024",
            ],
        ),
        (
            "binned",
            240,
            [
                "task.prediction_horizon=239",
                f"{ACTION_TOKENIZER_MAX_TOKEN_LEN_KEY}=480",
                f"{GPT_MAX_SEQ_LEN_KEY}=1024",
            ],
        ),
    ],
)
def test_length_overrides_follow_the_method_family(
    noisy_store_root: Path,
    train_cell_factory: Callable[..., TrainCell],
    method: str,
    trajectory_length: int,
    expected: list[str],
):
    cell = train_cell_factory(method=method, trajectory_length=trajectory_length)
    assert cell.length_overrides() == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "trajectory_length, expectation",
    [
        (60, does_not_raise()),
        (
            240,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "FAST max_token_len is not measured for trajectory_length=240; "
                    "run measure_token_length.py --stage <stage> and add the value "
                    "to FAST_MAX_TOKEN_LEN_BY_LENGTH."
                ),
            ),
        ),
    ],
)
def test_fast_max_token_len_requires_a_measurement_for_the_length(
    monkeypatch: pytest.MonkeyPatch,
    trajectory_length: int,
    expectation: AbstractContextManager,
):
    monkeypatch.delitem(FAST_MAX_TOKEN_LEN_BY_LENGTH, 240, raising=False)
    with expectation:
        fast_max_token_len(trajectory_length)


@pytest.mark.unit
def test_train_cell_overrides_use_the_measured_fast_cap_for_the_length(
    noisy_store_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_cell_factory: Callable[..., TrainCell],
):
    monkeypatch.setitem(FAST_MAX_TOKEN_LEN_BY_LENGTH, 240, 400)
    cell = train_cell_factory(method="fast", trajectory_length=240)

    overrides = cell.overrides(matched=True)

    assert f"{ACTION_TOKENIZER_MAX_TOKEN_LEN_KEY}=400" in overrides
    assert overrides.count("task.prediction_horizon=239") == 1
    assert overrides[-1] == f"{ACTION_TOKENIZER_MAX_TOKEN_LEN_KEY}=400"


@pytest.mark.unit
def test_rate_stage_orders_cells_by_length_then_method(noisy_store_root: Path):
    cells = stage_cells("rate_conditional_s0")
    anchor_names = {
        cell.name
        for cell in stage_cells("final_conditional_s0")
        if cell.data.sigma_multiplier == 1.0
    }

    assert len(cells) == 12
    assert len(data_cells(cells)) == 3
    assert [cell.data.trajectory_length for cell in cells] == [60] * 4 + [120] * 4 + [
        240
    ] * 4
    assert [cell.method for cell in cells[:4]] == list(FINAL_METHODS)
    assert {cell.name for cell in cells[:4]} == anchor_names


@pytest.mark.unit
def test_existing_stages_keep_the_default_length(noisy_store_root: Path):
    cells = stage_cells("final_conditional_s0")
    assert {cell.data.trajectory_length for cell in cells} == {60}
    assert all("__T-" not in cell.name for cell in cells)


@pytest.mark.unit
def test_reference_cells_keep_the_trajectory_length(
    noisy_store_root: Path, data_cell_factory: Callable[..., DataCell]
):
    cell = data_cell_factory(sigma_multiplier=1.0, trajectory_length=240)

    references = reference_cells([cell])

    assert [reference.name for reference in references] == [
        "conditional__inj-position__band-high__sig-0__dseed-42__T-240"
    ]


@pytest.mark.unit
def test_measured_snr_reads_the_clean_store_of_the_same_length(
    noisy_store_root: Path, data_cell_factory: Callable[..., DataCell]
):
    cell = data_cell_factory(sigma_multiplier=1.0, trajectory_length=240)
    clean_actions = np.full((4, 2), 0.02, dtype=np.float32)
    noisy_actions = clean_actions + np.full((4, 2), 0.01, dtype=np.float32)
    fake_buffer = MagicMock()
    fake_buffer.__getitem__.return_value.__getitem__.return_value = clean_actions

    with patch(
        "versatil.analysis.tip1_noise.sweep.ReplayBuffer.create_from_path",
        return_value=fake_buffer,
    ) as mock_create:
        snr = measured_snr(cell=cell, actions=noisy_actions)

    assert mock_create.call_args.args[0].endswith("__sig-0__dseed-42__T-240.zarr")
    assert snr == pytest.approx(2.0)


@pytest.mark.unit
def test_add_effective_bins_caches_the_reference_per_trajectory_length(
    noisy_store_root: Path, data_cell_factory: Callable[..., DataCell]
):
    short_cell = data_cell_factory(sigma_multiplier=1.0, trajectory_length=60)
    long_cell = data_cell_factory(sigma_multiplier=1.0, trajectory_length=240)
    rows = [
        {"cell": short_cell.name, "action_range": 0.4},
        {"cell": long_cell.name, "action_range": 0.2},
    ]
    references = {60: 0.2, 240: 0.05}

    with patch(
        "versatil.analysis.tip1_noise.sweep.clean_reference_range",
        side_effect=lambda cell: references[cell.trajectory_length],
    ) as mock_reference:
        add_effective_bins(rows, [short_cell, long_cell])

    assert mock_reference.call_count == 2
    assert rows[0]["range_inflation"] == pytest.approx(2.0)
    assert rows[1]["range_inflation"] == pytest.approx(4.0)
