"""Tests for versatil.analysis.tip2_tokenization.eval_hook module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.analysis.tip2_tokenization.eval_hook import (
    DECOMPOSITION_KEYS,
    catastrophic_fraction,
    compare_generation_modes,
    count_unique_sequences,
    decompose_in_position_space,
    decompose_prediction_error,
    fraction_without_eos,
    generate_actions_with_argmax,
    mask_after_ground_truth_eos,
    mean_sequence_length,
    mode_match_rate,
    per_chunk_mse,
    prefixed_metrics,
    robust_error_summary,
    save_eval_arrays,
    standard_error,
)


@pytest.fixture
def chunk_arrays_factory(rng: np.random.Generator):
    """Factory for (expert, round_trip, generation) chunks of shape (N, H, D)."""

    def factory(
        num_chunks: int = 5, horizon: int = 7, action_dim: int = 2
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = (num_chunks, horizon, action_dim)
        expert = rng.normal(size=shape)
        round_trip = expert + rng.normal(scale=0.1, size=shape)
        generation = round_trip + rng.normal(scale=0.2, size=shape)
        return expert, round_trip, generation

    return factory


class TestDecomposeInPositionSpace:
    @pytest.mark.unit
    def test_identity_holds_on_integrated_paths(self, chunk_arrays_factory):
        expert, round_trip, generation = chunk_arrays_factory()

        metrics = decompose_in_position_space(
            expert_chunks=expert,
            round_trip_chunks=round_trip,
            generated_chunks=generation,
        )

        assert metrics["identity_gap"] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.unit
    def test_total_equals_action_decomposition_of_cumsum_inputs(
        self, chunk_arrays_factory
    ):
        expert, round_trip, generation = chunk_arrays_factory()
        action_dim = expert.shape[-1]

        metrics = decompose_in_position_space(
            expert_chunks=expert,
            round_trip_chunks=round_trip,
            generated_chunks=generation,
        )
        expected = decompose_prediction_error(
            action_true=np.cumsum(expert, axis=1).reshape(-1, action_dim),
            reconstruction_from_gt_tokens=np.cumsum(round_trip, axis=1).reshape(
                -1, action_dim
            ),
            reconstruction_from_generation=np.cumsum(generation, axis=1).reshape(
                -1, action_dim
            ),
        )

        for key in DECOMPOSITION_KEYS:
            assert metrics[key] == pytest.approx(expected[key])

    @pytest.mark.unit
    def test_stand_still_level_is_mean_square_of_expert_path(
        self, chunk_arrays_factory
    ):
        expert, round_trip, generation = chunk_arrays_factory()

        metrics = decompose_in_position_space(
            expert_chunks=expert,
            round_trip_chunks=round_trip,
            generated_chunks=generation,
        )

        assert metrics["expert_mean_square"] == pytest.approx(
            float(np.mean(np.cumsum(expert, axis=1) ** 2))
        )

    @pytest.mark.unit
    def test_constant_per_step_bias_grows_with_horizon(self):
        expert = np.zeros((1, 4, 1))
        biased = np.full((1, 4, 1), 0.5)

        metrics = decompose_in_position_space(
            expert_chunks=expert, round_trip_chunks=expert, generated_chunks=biased
        )

        assert metrics["total_mse"] == pytest.approx(
            np.mean(np.array([0.5, 1.0, 1.5, 2.0]) ** 2)
        )


class TestPrefixedMetrics:
    @pytest.mark.unit
    def test_selects_and_prefixes_only_the_requested_keys(self):
        metrics = {"total_mse": 1.0, "cross": 2.0, "term1_rmse": 3.0}

        result = prefixed_metrics(
            metrics=metrics, prefix="position_", keys=("total_mse", "cross")
        )

        assert result == {"position_total_mse": 1.0, "position_cross": 2.0}


class TestSaveEvalArrays:
    @pytest.mark.unit
    def test_writes_every_array_under_its_key(self, tmp_path, chunk_arrays_factory):
        expert, round_trip, argmax = chunk_arrays_factory(num_chunks=3, horizon=4)
        stochastic = np.stack([argmax, argmax + 1.0])
        mode_ids = np.array([0, 1, 0])
        array_path = tmp_path / "nested" / "cell.npz"

        save_eval_arrays(
            array_path=array_path,
            expert_chunks=expert,
            round_trip_chunks=round_trip,
            argmax_chunks=argmax,
            stochastic_chunks=stochastic,
            mode_ids=mode_ids,
        )

        with np.load(array_path) as saved:
            assert saved["expert"].shape == (3, 4, 2)
            assert saved["stochastic"].shape == (2, 3, 4, 2)
            np.testing.assert_array_equal(saved["round_trip"], round_trip)
            np.testing.assert_array_equal(saved["argmax"], argmax)
            np.testing.assert_array_equal(saved["mode_ids"], mode_ids)


@pytest.fixture
def action_arrays_factory():
    """Factory for (expert, round_trip, generation) denormalized action arrays."""

    def factory(
        num_elements: int = 64,
        action_dim: int = 2,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        expert = rng.normal(size=(num_elements, action_dim))
        round_trip = expert + rng.normal(scale=0.1, size=(num_elements, action_dim))
        generation = round_trip + rng.normal(scale=0.2, size=(num_elements, action_dim))
        return expert, round_trip, generation

    return factory


class TestMaskAfterGroundTruthEos:
    @pytest.mark.unit
    def test_replaces_predictions_from_the_gt_eos_onward(self):
        eos, pad = 9, 0
        ground_truth = torch.tensor([[1, 2, 3, eos, pad, pad]])
        predicted = torch.tensor([[5, 6, 7, 8, 4, 4]])

        masked = mask_after_ground_truth_eos(
            predicted_tokens=predicted,
            ground_truth_tokens=ground_truth,
            eos_token_id=eos,
        )

        assert masked.tolist() == [[5, 6, 7, eos, pad, pad]]

    @pytest.mark.unit
    def test_rows_without_eos_are_unchanged(self):
        ground_truth = torch.tensor([[1, 2, 3, 4]])
        predicted = torch.tensor([[5, 6, 7, 8]])

        masked = mask_after_ground_truth_eos(
            predicted_tokens=predicted,
            ground_truth_tokens=ground_truth,
            eos_token_id=9,
        )

        assert masked.tolist() == predicted.tolist()

    @pytest.mark.unit
    def test_no_eos_id_returns_the_input(self):
        predicted = torch.tensor([[5, 6]])

        masked = mask_after_ground_truth_eos(
            predicted_tokens=predicted,
            ground_truth_tokens=torch.tensor([[1, 2]]),
            eos_token_id=None,
        )

        assert masked is predicted

    @pytest.mark.unit
    def test_mixed_batch_masks_each_row_at_its_own_eos(self):
        eos, pad = 9, 0
        ground_truth = torch.tensor([[1, eos, pad], [1, 2, 3]])
        predicted = torch.tensor([[7, 7, 7], [8, 8, 8]])

        masked = mask_after_ground_truth_eos(
            predicted_tokens=predicted,
            ground_truth_tokens=ground_truth,
            eos_token_id=eos,
        )

        assert masked.tolist() == [[7, eos, pad], [8, 8, 8]]


class TestCatastrophicFraction:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "bad_chunk_value, expected",
        [(20.0, 0.5), (5.0, 0.0)],
    )
    def test_counts_chunks_beyond_factor_times_bound(
        self, bad_chunk_value: float, expected: float
    ):
        good = np.full((1, 3, 2), 0.5)
        bad = np.full((1, 3, 2), bad_chunk_value)

        fraction = catastrophic_fraction(
            generated_chunks=np.concatenate([good, bad]),
            action_bound=1.0,
            factor=10.0,
        )

        assert fraction == pytest.approx(expected)

    @pytest.mark.unit
    def test_empty_input_is_zero(self):
        fraction = catastrophic_fraction(
            generated_chunks=np.zeros((0, 3, 2)), action_bound=1.0, factor=10.0
        )

        assert fraction == 0.0


class TestRobustErrorSummary:
    @pytest.mark.unit
    def test_trimmed_mean_drops_the_largest_errors(self):
        errors = np.array([1.0] * 9 + [1000.0])

        summary = robust_error_summary(squared_errors=errors, trim_fraction=0.1)

        assert summary["median"] == pytest.approx(1.0)
        assert summary["trimmed_mean"] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_zero_trim_equals_the_plain_mean(self):
        errors = np.arange(1.0, 11.0)

        summary = robust_error_summary(squared_errors=errors, trim_fraction=0.0)

        assert summary["trimmed_mean"] == pytest.approx(errors.mean())


class TestPerChunkMse:
    @pytest.mark.unit
    def test_averages_each_chunk_over_horizon_and_dims(self):
        expert = np.zeros((2, 2, 2))
        generated = np.array([[[1.0, 1.0], [1.0, 1.0]], [[2.0, 0.0], [0.0, 0.0]]])

        assert per_chunk_mse(
            generated_chunks=generated, expert_chunks=expert
        ).tolist() == pytest.approx([1.0, 1.0])


class TestStandardError:
    @pytest.mark.unit
    def test_matches_sample_std_over_sqrt_n(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])

        assert standard_error(values=values) == pytest.approx(
            np.std(values, ddof=1) / 2.0
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("values", [np.array([]), np.array([3.0])])
    def test_fewer_than_two_values_is_zero(self, values):
        assert standard_error(values=values) == 0.0


class TestModeMatchRate:
    @pytest.mark.unit
    @pytest.mark.parametrize("swap_modes, expected", [(False, 1.0), (True, 0.0)])
    def test_assigns_generation_by_its_displacement_path(
        self, swap_modes: bool, expected: float
    ):
        upward = np.tile([[0.0, 1.0]], (4, 1))[None]
        expert = np.concatenate([upward, -upward])
        generated = np.concatenate([-upward, upward]) if swap_modes else expert

        rate = mode_match_rate(
            generated_chunks=generated,
            expert_chunks=expert,
            expert_mode_ids=np.array([0, 1]),
            num_modes=2,
        )

        assert rate == pytest.approx(expected)

    @pytest.mark.unit
    def test_empty_input_is_zero(self):
        rate = mode_match_rate(
            generated_chunks=np.zeros((0, 4, 2)),
            expert_chunks=np.zeros((0, 4, 2)),
            expert_mode_ids=np.array([], dtype=np.int64),
            num_modes=2,
        )

        assert rate == 0.0


class TestDecomposePredictionError:
    def test_cross_term_matches_the_mse_identity(self, action_arrays_factory):
        expert, round_trip, generation = action_arrays_factory()

        metrics = decompose_prediction_error(
            action_true=expert,
            reconstruction_from_gt_tokens=round_trip,
            reconstruction_from_generation=generation,
        )

        assert metrics["identity_gap"] == pytest.approx(0.0, abs=1e-9)

    def test_cross_term_equals_independent_covariance_form(self, action_arrays_factory):
        expert, round_trip, generation = action_arrays_factory()

        metrics = decompose_prediction_error(
            action_true=expert,
            reconstruction_from_gt_tokens=round_trip,
            reconstruction_from_generation=generation,
        )

        reconstruction_error = round_trip - expert
        learning_error = generation - round_trip
        expected_cross = 2.0 * float(np.mean(reconstruction_error * learning_error))
        assert metrics["cross"] == pytest.approx(expected_cross)

    def test_zero_learning_error_makes_total_equal_reconstruction(
        self, action_arrays_factory
    ):
        expert, round_trip, _ = action_arrays_factory()

        metrics = decompose_prediction_error(
            action_true=expert,
            reconstruction_from_gt_tokens=round_trip,
            reconstruction_from_generation=round_trip,
        )

        assert metrics["term2_mse"] == pytest.approx(0.0)
        assert metrics["cross"] == pytest.approx(0.0)
        assert metrics["total_mse"] == pytest.approx(metrics["term1_mse"])

    def test_rmse_is_sqrt_of_mse(self, action_arrays_factory):
        expert, round_trip, generation = action_arrays_factory()

        metrics = decompose_prediction_error(
            action_true=expert,
            reconstruction_from_gt_tokens=round_trip,
            reconstruction_from_generation=generation,
        )

        assert metrics["total_rmse"] == pytest.approx(np.sqrt(metrics["total_mse"]))


class TestCompareGenerationModes:
    def test_exposure_bias_is_argmax_minus_teacher_forced(self, action_arrays_factory):
        expert, teacher_forced, argmax_generation = action_arrays_factory()

        metrics = compare_generation_modes(
            stochastic_total_mse=0.5,
            action_true=expert,
            argmax_generation=argmax_generation,
            teacher_forced=teacher_forced,
        )

        expected_argmax = float(np.mean((argmax_generation - expert) ** 2))
        expected_teacher_forced = float(np.mean((teacher_forced - expert) ** 2))
        assert metrics["argmax_total_mse"] == pytest.approx(expected_argmax)
        assert metrics["teacher_forced_total_mse"] == pytest.approx(
            expected_teacher_forced
        )
        assert metrics["exposure_bias"] == pytest.approx(
            expected_argmax - expected_teacher_forced
        )

    def test_sampling_gap_is_stochastic_minus_argmax(self, action_arrays_factory):
        expert, _, argmax_generation = action_arrays_factory()

        metrics = compare_generation_modes(
            stochastic_total_mse=0.5,
            action_true=expert,
            argmax_generation=argmax_generation,
            teacher_forced=expert,
        )

        assert metrics["sampling_gap"] == pytest.approx(
            0.5 - metrics["argmax_total_mse"]
        )
        assert metrics["exposure_bias"] == pytest.approx(metrics["argmax_total_mse"])


class TestGenerateActionsWithArgmax:
    @pytest.mark.parametrize("deployed_mode", [False, True])
    def test_forces_argmax_during_generation_and_restores_mode(self, deployed_mode):
        policy = MagicMock()
        policy.decoder.deterministic = deployed_mode
        observation = {"agentview": torch.zeros(1, 1, 3, 4, 4)}
        modes_seen: list[bool] = []
        generated = torch.zeros(1, 2, 2)

        def record_mode(policy, observation):
            modes_seen.append(policy.decoder.deterministic)
            return generated, [[1, 2]]

        with patch(
            "versatil.analysis.tip2_tokenization.eval_hook.generate_actions_and_tokens",
            side_effect=record_mode,
        ) as generate:
            result = generate_actions_with_argmax(
                policy=policy, observation=observation
            )

        generate.assert_called_once_with(policy=policy, observation=observation)
        assert modes_seen == [True]
        assert policy.decoder.deterministic is deployed_mode
        assert result is generated


class TestFractionWithoutEos:
    def test_counts_sequences_missing_eos_for_variable_length(self):
        sequences = [[1, 2, 9], [1, 2, 3], [4, 9], [5, 6]]

        fraction = fraction_without_eos(
            token_sequences=sequences, eos_token_id=9, is_variable_length=True
        )

        assert fraction == pytest.approx(0.5)

    def test_fixed_length_tokenizer_reports_zero(self):
        sequences = [[1, 2, 3], [4, 5, 6]]

        fraction = fraction_without_eos(
            token_sequences=sequences, eos_token_id=9, is_variable_length=False
        )

        assert fraction == 0.0

    def test_no_eos_id_reports_zero(self):
        sequences = [[1, 2, 3]]

        fraction = fraction_without_eos(
            token_sequences=sequences, eos_token_id=None, is_variable_length=True
        )

        assert fraction == 0.0


class TestSequenceDiagnostics:
    def test_unique_sequence_count_collapses_to_one_for_constant_target(self):
        sequences = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]

        assert count_unique_sequences(token_sequences=sequences) == 1

    def test_unique_sequence_count_counts_distinct(self):
        sequences = [[1, 2], [1, 2], [3, 4], [5, 6]]

        assert count_unique_sequences(token_sequences=sequences) == 3

    def test_mean_sequence_length(self):
        sequences = [[1, 2], [1, 2, 3, 4]]

        assert mean_sequence_length(token_sequences=sequences) == pytest.approx(3.0)

    def test_mean_sequence_length_empty(self):
        assert mean_sequence_length(token_sequences=[]) == 0.0
