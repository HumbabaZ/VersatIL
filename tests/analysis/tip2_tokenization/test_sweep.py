"""Tests for versatil.analysis.tip2_tokenization.sweep module."""

from contextlib import nullcontext as does_not_raise

import pytest

from versatil.analysis.tip2_tokenization.stores import STORES
from versatil.analysis.tip2_tokenization.sweep import (
    BINNING,
    CONDITIONAL_METHOD_CONFIG,
    CONDITIONAL_TASK,
    FAST,
    FAST_MAX_TOKEN_LEN,
    FAST_SCALES,
    METHOD_CONFIG,
    NUM_BINS_GRID,
    TrainCell,
    check_paths_unique,
    method_config,
    stage_cells,
)


@pytest.fixture
def train_cell_factory():
    def factory(
        task: str = "sequential",
        method: str = FAST,
        param: float = 8.843660238726597,
        train_seed: int = 0,
    ) -> TrainCell:
        return TrainCell(
            store=STORES[task],
            method=method,
            param=param,
            train_seed=train_seed,
        )

    return factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "task, method, expected",
    [
        (CONDITIONAL_TASK, FAST, CONDITIONAL_METHOD_CONFIG[FAST]),
        (CONDITIONAL_TASK, BINNING, CONDITIONAL_METHOD_CONFIG[BINNING]),
        ("sequential", FAST, METHOD_CONFIG[FAST]),
        ("sequential", BINNING, METHOD_CONFIG[BINNING]),
    ],
)
def test_method_config_uses_context_configs_only_for_conditional_task(
    task: str, method: str, expected: str
):
    assert method_config(task=task, method=method) == expected


class TestStageCells:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "stage, expected_count",
        [("pilot", 14), ("main", 42)],
    )
    def test_cell_count_is_families_times_grid_times_seeds(
        self, stage: str, expected_count: int
    ):
        assert len(stage_cells(stage=stage, task="sequential")) == expected_count

    @pytest.mark.unit
    def test_fast_cells_precede_binning_cells(self):
        cells = stage_cells(stage="main", task="sequential")
        methods = [cell.method for cell in cells]
        assert methods[:21] == [FAST] * 21
        assert methods[21:] == [BINNING] * 21

    @pytest.mark.unit
    @pytest.mark.parametrize("task", ["sequential", "conditional"])
    def test_every_grid_point_and_seed_is_present(self, task: str):
        cells = stage_cells(stage="main", task=task)
        fast_params = {cell.param for cell in cells if cell.method == FAST}
        binning_params = {int(cell.param) for cell in cells if cell.method == BINNING}
        assert fast_params == set(FAST_SCALES[task])
        assert binning_params == set(NUM_BINS_GRID)
        assert {cell.train_seed for cell in cells} == {0, 1, 2}


class TestCheckPathsUnique:
    @pytest.mark.unit
    def test_stage_cells_are_all_unique(self):
        check_paths_unique(stage_cells(stage="main", task="sequential"))

    @pytest.mark.unit
    def test_duplicate_names_raise(self, train_cell_factory):
        cell = train_cell_factory()
        with pytest.raises(ValueError, match="Duplicate"):
            check_paths_unique([cell, cell])


class TestGranularityOverrides:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "task, scale, expected_scale_suffix",
        [
            ("sequential", 187.86120017758375, ".scale=187.861"),
            ("conditional", 267.01065377512424, ".scale=267.011"),
        ],
    )
    def test_fast_sets_scale_and_the_per_task_token_cap(
        self, train_cell_factory, task: str, scale: float, expected_scale_suffix: str
    ):
        overrides = train_cell_factory(
            task=task, method=FAST, param=scale
        ).granularity_overrides()
        assert any(o.endswith(expected_scale_suffix) for o in overrides)
        assert any(
            o.endswith(f".max_token_len={FAST_MAX_TOKEN_LEN[task]}") for o in overrides
        )

    @pytest.mark.unit
    def test_binning_sets_num_bins_and_never_overrides_token_cap(
        self, train_cell_factory
    ):
        overrides = train_cell_factory(
            method=BINNING, param=64.0
        ).granularity_overrides()
        assert overrides == [
            "task.dataloader.tokenization.action_tokenizer."
            "action_discretizer.num_bins=64"
        ]


class TestTrainCellNaming:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "method, param, expected_tag",
        [
            (FAST, 0.41631974215059914, "scale-0p41632"),
            (BINNING, 1024.0, "bins-1024"),
        ],
    )
    def test_param_tag_is_filename_safe(
        self, train_cell_factory, method: str, param: float, expected_tag: str
    ):
        assert train_cell_factory(method=method, param=param).param_tag == expected_tag

    @pytest.mark.unit
    def test_name_carries_task_family_grid_and_seed(self, train_cell_factory):
        cell = train_cell_factory(method=BINNING, param=8.0, train_seed=2)
        assert cell.name == "sequential__binning__bins-8__seed-2"


class TestCommand:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "override",
        [
            "task.dataset_schema.eval_reference_noise_std=0.0",
            "experiment.data_seed=42",
            "experiment.seed=1",
            "experiment.name=sequential__fast__scale-8p84366__seed-1",
        ],
    )
    def test_command_pins_data_and_seed(self, train_cell_factory, override: str):
        cell = train_cell_factory(method=FAST, param=8.843660238726597, train_seed=1)
        assert override in cell.command()

    @pytest.mark.unit
    def test_command_points_at_the_fixed_store(self, train_cell_factory):
        cell = train_cell_factory()
        zarr_override = (
            f"task.dataset_schema.zarr_path={STORES['sequential'].zarr_path}"
        )
        assert zarr_override in cell.command()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "index, expectation",
        [(0, does_not_raise()), (41, does_not_raise())],
    )
    def test_stage_cells_are_indexable_for_the_slurm_array(self, index, expectation):
        cells = stage_cells(stage="main", task="sequential")
        with expectation:
            _ = cells[index]
