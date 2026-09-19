"""Tests for versatil.analysis.tip2_tokenization.trajectory_shape module."""

import math

import numpy as np
import pytest

from versatil.analysis.tip2_tokenization.trajectory_shape import (
    aggregate_rows,
    cell_metrics,
    mean_step_speed,
    mean_turning_angle,
    parse_cell_name,
    rms_jerk,
)


class TestRmsJerk:
    @pytest.mark.unit
    def test_constant_velocity_has_zero_jerk(self):
        # A straight, constant-speed path has zero second difference.
        actions = np.ones((4, 10, 2), dtype=np.float64) * 0.02

        assert rms_jerk(actions=actions) == pytest.approx(0.0)

    @pytest.mark.unit
    def test_matches_hand_computed_second_difference(self):
        # One chunk, one dim: displacements 0,1,4,9,16 -> second diff constant 2.
        actions = np.array([[[0.0], [1.0], [4.0], [9.0], [16.0]]])

        assert rms_jerk(actions=actions) == pytest.approx(2.0)


class TestMeanTurningAngle:
    @pytest.mark.unit
    def test_straight_line_has_zero_turning(self):
        actions = np.tile(np.array([0.03, 0.0]), (2, 6, 1))

        assert mean_turning_angle(actions=actions) == pytest.approx(0.0)

    @pytest.mark.unit
    def test_right_angle_turn(self):
        # x-step then y-step then x-step: two 90-degree turns.
        actions = np.array([[[0.1, 0.0], [0.0, 0.1], [0.1, 0.0]]])

        assert mean_turning_angle(actions=actions) == pytest.approx(math.pi / 2)

    @pytest.mark.unit
    def test_stand_still_returns_zero_when_no_valid_step(self):
        actions = np.zeros((3, 5, 2), dtype=np.float64)

        assert mean_turning_angle(actions=actions) == 0.0


class TestMeanStepSpeed:
    @pytest.mark.unit
    def test_mean_displacement_magnitude(self):
        # Every step is a 3-4-5 vector -> magnitude 5.
        actions = np.tile(np.array([3.0, 4.0]), (2, 4, 1))

        assert mean_step_speed(actions=actions) == pytest.approx(5.0)


class TestCellMetrics:
    @pytest.mark.unit
    def test_emits_all_metrics_for_present_references(self, rng: np.random.Generator):
        arrays = {
            "argmax": rng.normal(size=(5, 8, 2)),
            "expert": rng.normal(size=(5, 8, 2)),
            "stochastic": rng.normal(size=(2, 5, 8, 2)),
        }

        metrics = cell_metrics(arrays=arrays)

        # round_trip absent -> no round_trip_* keys; stochastic ignored.
        assert set(metrics) == {
            "argmax_jerk",
            "argmax_turning_angle",
            "argmax_step_speed",
            "expert_jerk",
            "expert_turning_angle",
            "expert_step_speed",
        }


class TestParseCellName:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("conditional__fast__scale-10__seed-0.npz", ("fast", 10.0, 0)),
            ("conditional__fast__scale-0p1__seed-2.npz", ("fast", 0.1, 2)),
            ("conditional__binning__bins-256__seed-1.npz", ("binning", 256.0, 1)),
            ("summary.csv", None),
            ("conditional__qfat__scale-1__seed-0.npz", None),
        ],
    )
    def test_parses_family_param_seed(self, filename, expected):
        assert parse_cell_name(filename=filename) == expected


class TestAggregateRows:
    @pytest.mark.unit
    def test_averages_metrics_over_seeds(self):
        rows = [
            {"method": "fast", "param": 10.0, "seed": 0, "argmax_jerk": 0.02},
            {"method": "fast", "param": 10.0, "seed": 1, "argmax_jerk": 0.04},
            {"method": "fast", "param": 3.0, "seed": 0, "argmax_jerk": 0.10},
        ]

        aggregated = aggregate_rows(rows=rows)

        assert aggregated[("fast", 10.0)]["argmax_jerk"] == pytest.approx(0.03)
        assert aggregated[("fast", 3.0)]["argmax_jerk"] == pytest.approx(0.10)
