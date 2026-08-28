"""Tests for versatil.analysis.tip2_tokenization.gate_report module."""

import csv
from pathlib import Path

import pytest

from versatil.analysis.tip2_tokenization.gate_report import (
    REQUIRED_COLUMNS,
    STAND_STILL_TOLERANCE,
    adjacent_resolution,
    gate_report,
    is_at_stand_still,
    missing_columns,
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


@pytest.fixture
def row_factory():
    def factory(
        method: str,
        param: float,
        total: float,
        se: float = 1e-5,
        unique: int = 100,
        mean_square: float = 5e-4,
        mode_match: float = 1.0,
        catastrophic: float = 0.0,
        rollout: float = 1.0,
    ) -> dict[str, str]:
        return {
            "method": method,
            "param": str(param),
            "total_mse": str(total),
            "total_mse_se": str(se),
            "unique_gt_sequence_count": str(unique),
            "expert_mean_square": str(mean_square),
            "mode_match_rate": str(mode_match),
            "catastrophic_fraction": str(catastrophic),
            "rollout_success": str(rollout),
            "identity_gap": "1e-12",
            "no_eos_rate": "0.0",
        }

    return factory


class TestIsAtStandStill:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "total, expected",
        [
            (5e-4, True),
            (5e-4 * (1 + 0.9 * STAND_STILL_TOLERANCE), True),
            (5e-4 * (1 - 0.9 * STAND_STILL_TOLERANCE), True),
            (5e-4 * (1 + 1.1 * STAND_STILL_TOLERANCE), False),
            (2e-4, False),
            (7e-4, False),
        ],
    )
    def test_band_around_mean_square(self, total: float, expected: bool):
        assert is_at_stand_still(total_mse=total, expert_mean_square=5e-4) is expected


class TestAdjacentResolution:
    @pytest.mark.unit
    def test_separated_pairs_resolve_and_noisy_pairs_do_not(self):
        flags = adjacent_resolution(
            totals=[1.0, 2.0, 2.01],
            standard_errors=[0.1, 0.1, 0.1],
        )

        assert flags == [True, False]

    @pytest.mark.unit
    def test_single_point_has_no_pairs(self):
        assert adjacent_resolution(totals=[1.0], standard_errors=[0.1]) == []


class TestMissingColumns:
    @pytest.mark.unit
    def test_reports_absent_required_columns(self, row_factory):
        row = row_factory(method="fast", param=1.0, total=1e-3)
        del row["mode_match_rate"]

        assert missing_columns(row) == ["mode_match_rate"]

    @pytest.mark.unit
    def test_complete_row_reports_nothing(self, row_factory):
        assert missing_columns(row_factory(method="fast", param=1.0, total=1e-3)) == []
        assert set(REQUIRED_COLUMNS) <= set(
            row_factory(method="fast", param=1.0, total=1.0)
        )


class TestGateReport:
    @pytest.mark.unit
    def test_degenerate_point_is_excluded_from_resolution(
        self, manifest_factory, row_factory
    ):
        rows = [
            row_factory(method="fast", param=0.3, total=4e-4, unique=1),
            row_factory(method="fast", param=3.0, total=1e-3),
            row_factory(method="fast", param=30.0, total=3e-3),
        ]

        report = gate_report(csv_path=manifest_factory(rows))

        assert "resolved adjacent pairs: 1/1 (healthy points: 2)" in report
        assert "3.000 -> 30.000: resolved" in report
        assert "healthy-point spread (max/min - 1): 200.0%" in report

    @pytest.mark.unit
    def test_task_failing_point_is_excluded_but_stand_still_is_not(
        self, manifest_factory, row_factory
    ):
        rows = [
            row_factory(method="fast", param=1.0, total=2e-4, rollout=0.0),
            row_factory(method="fast", param=3.0, total=5e-4),
            row_factory(method="fast", param=30.0, total=1.5e-3),
        ]

        report = gate_report(csv_path=manifest_factory(rows))

        assert "resolved adjacent pairs: 1/1 (healthy points: 2)" in report
        assert "3.000 -> 30.000: resolved" in report

    @pytest.mark.unit
    def test_exploded_point_is_left_out_of_the_calm_spread(
        self, manifest_factory, row_factory
    ):
        rows = [
            row_factory(method="fast", param=3.0, total=1e-3),
            row_factory(method="fast", param=30.0, total=3e-3),
            row_factory(method="fast", param=300.0, total=10.0, catastrophic=0.7),
        ]

        report = gate_report(csv_path=manifest_factory(rows))

        assert "healthy-point spread (max/min - 1): 999900.0%" in report
        assert "catastrophic_fraction < 0.05: 200.0% (2 points)" in report

    @pytest.mark.unit
    def test_missing_column_raises(self, manifest_factory, row_factory):
        row = row_factory(method="binning", param=4.0, total=1e-3)
        del row["rollout_success"]

        with pytest.raises(ValueError, match="rollout_success"):
            gate_report(csv_path=manifest_factory([row]))
