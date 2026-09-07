"""Tests for versatil.analysis.tip2_tokenization.plot_decomposition module."""

import math

import numpy as np
import pytest

from versatil.analysis.tip2_tokenization.plot_decomposition import (
    arithmetic_mean,
    band_edges,
    failing_runs,
    format_tick,
    geometric_mean,
    group_by_param,
)


class TestGeometricMean:
    @pytest.mark.unit
    def test_matches_log_space_arithmetic_mean(self, rng: np.random.Generator):
        values = list(rng.uniform(0.1, 10.0, size=5))

        assert geometric_mean(values=values) == pytest.approx(
            math.exp(np.mean(np.log(values)))
        )

    @pytest.mark.unit
    def test_single_value_is_itself(self):
        assert geometric_mean(values=[3.5]) == pytest.approx(3.5)

    @pytest.mark.unit
    def test_is_not_dragged_by_one_diverged_seed(self):
        # An arithmetic mean of (1e-4, 1e-4, 1e2) sits at ~33, hugging the
        # outlier; the geometric mean stays with the bulk.
        center = geometric_mean(values=[1e-4, 1e-4, 1e2])

        assert center == pytest.approx(1e-2, rel=1e-6)

    @pytest.mark.unit
    @pytest.mark.parametrize("values", [[0.0, 1.0], [-1.0, 2.0], []])
    def test_non_positive_or_empty_returns_none(self, values: list[float]):
        assert geometric_mean(values=values) is None


class TestArithmeticMean:
    @pytest.mark.unit
    def test_plain_mean_including_zero(self):
        assert arithmetic_mean(values=[0.0, 0.5, 1.0]) == pytest.approx(0.5)


class TestGroupByParam:
    @pytest.mark.unit
    def test_groups_seeds_and_sorts_coarse_to_fine(self):
        rows = [
            {"param": "10", "train_seed": "0"},
            {"param": "1", "train_seed": "0"},
            {"param": "10", "train_seed": "1"},
            {"param": "1", "train_seed": "1"},
        ]

        groups = group_by_param(rows=rows)

        assert [param for param, _ in groups] == [1.0, 10.0]
        assert [row["train_seed"] for row in groups[1][1]] == ["0", "1"]


class TestBandEdges:
    @pytest.mark.unit
    def test_interior_edges_are_geometric_midpoints(self):
        params = [1.0, 4.0, 16.0, 64.0]

        left, right = band_edges(params=params, start=1, end=2)

        assert left == pytest.approx(math.sqrt(1.0 * 4.0))
        assert right == pytest.approx(math.sqrt(16.0 * 64.0))

    @pytest.mark.unit
    def test_outermost_edges_extend_to_the_panel_margin(self):
        params = [1.0, 4.0, 16.0]

        left, right = band_edges(params=params, start=0, end=2)

        assert left == pytest.approx(1.0 / 1.6)
        assert right == pytest.approx(16.0 * 1.6)


class TestFailingRuns:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "failed, expected",
        [
            ([True, True, False, True], [(0, 1), (3, 3)]),
            ([False, False], []),
            ([True, True, True], [(0, 2)]),
        ],
    )
    def test_returns_runs_of_consecutive_failures(
        self, failed: list[bool], expected: list[tuple[int, int]]
    ):
        assert failing_runs(failed=failed) == expected


class TestFormatTick:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "value, expected",
        [(0.1, "0.1"), (1.0, "1"), (3.0, "3"), (250.0, "250"), (4096.0, "4096")],
    )
    def test_compact_grid_labels(self, value: float, expected: str):
        assert format_tick(value=value) == expected
