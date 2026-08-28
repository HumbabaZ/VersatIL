"""Tests for versatil.analysis.tip1_noise.sweep module."""

import re
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest

from versatil.analysis.tip1_noise.sweep import (
    CONDITIONAL_METHOD_CONFIG,
    CONDITIONAL_TASK,
    FAST_OVERRIDES,
    FINAL_METHODS,
    METHOD_CONFIG,
    NOISY_ZARR_DIR_ENV,
    POSITION,
    DataCell,
    TrainCell,
    check_paths_unique,
    data_cells,
    method_config,
    noisy_zarr_root,
    stage_cells,
)


@pytest.fixture
def noisy_store_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv(NOISY_ZARR_DIR_ENV, str(tmp_path))
    return tmp_path


@pytest.fixture
def data_cell_factory():
    def factory(
        task: str = CONDITIONAL_TASK,
        sigma_multiplier: float = 2.0,
        data_seed: int = 42,
    ) -> DataCell:
        return DataCell(
            task=task,
            injection=POSITION,
            smoothing_sigma=0.0,
            sigma_multiplier=sigma_multiplier,
            data_seed=data_seed,
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
