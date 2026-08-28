"""Tests for versatil.analysis.tip2_tokenization.plot_decomposition module."""

import csv
from pathlib import Path

import pytest

from versatil.analysis.tip2_tokenization.plot_decomposition import (
    DEGENERATE_UNIQUE_COUNT,
    is_degenerate,
    load_family,
)


@pytest.fixture
def manifest_factory(tmp_path: Path):
    def factory(rows: list[dict[str, str]]) -> Path:
        path = tmp_path / "tip2_eval_task_pilot.csv"
        with open(path, "w", newline="") as manifest:
            writer = csv.DictWriter(manifest, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    return factory


class TestIsDegenerate:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "unique_count, expected",
        [
            (1, True),
            (DEGENERATE_UNIQUE_COUNT, True),
            (DEGENERATE_UNIQUE_COUNT + 1, False),
            (3800, False),
        ],
    )
    def test_collapsed_sequence_count_marks_the_point(
        self, unique_count: int, expected: bool
    ):
        row = {"unique_gt_sequence_count": str(unique_count)}
        assert is_degenerate(row) is expected


class TestLoadFamily:
    @pytest.mark.unit
    def test_filters_one_method_and_sorts_coarse_to_fine(self, manifest_factory):
        rows = [
            {"method": "fast", "param": "8.0", "unique_gt_sequence_count": "100"},
            {"method": "binning", "param": "4.0", "unique_gt_sequence_count": "100"},
            {"method": "fast", "param": "0.4", "unique_gt_sequence_count": "1"},
        ]

        loaded = load_family(csv_path=manifest_factory(rows), method="fast")

        assert [float(row["param"]) for row in loaded] == [0.4, 8.0]
        assert all(row["method"] == "fast" for row in loaded)

    @pytest.mark.unit
    def test_missing_method_yields_empty(self, manifest_factory):
        rows = [{"method": "fast", "param": "1.0", "unique_gt_sequence_count": "5"}]

        assert load_family(csv_path=manifest_factory(rows), method="binning") == []
