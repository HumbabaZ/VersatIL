"""Tests for versatil.analysis.token_usage.counter module."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from versatil.analysis.token_usage.counter import TokenUsageCounter


class TestTokenUsageCounterUpdate:
    def test_accumulates_counts_across_updates(self):
        counter = TokenUsageCounter(label="train")
        counter.update(np.array([1, 1, 2], dtype=np.int64))
        counter.update(np.array([2, 3], dtype=np.int64))
        tokens, counts = counter.counts_as_arrays()
        np.testing.assert_array_equal(tokens, np.array([1, 2, 3]))
        np.testing.assert_array_equal(counts, np.array([2, 2, 1]))

    def test_total_sums_all_counts(self):
        counter = TokenUsageCounter()
        counter.update(np.array([4, 4, 4, 9], dtype=np.int64))
        assert counter.total == 4

    def test_support_lists_observed_tokens(self):
        counter = TokenUsageCounter()
        counter.update(np.array([7, 7, 12], dtype=np.int64))
        assert counter.support == {7, 12}

    def test_counts_negative_coefficient_tokens(self):
        counter = TokenUsageCounter()
        counter.update(np.array([-5, -5, 0, 3], dtype=np.int64))
        assert counter.support == {-5, 0, 3}
        assert counter.probability(-5) == pytest.approx(0.5)

    def test_flattens_multidimensional_input(self):
        counter = TokenUsageCounter()
        counter.update(np.array([[1, 2], [2, 2]], dtype=np.int64))
        assert counter.probability(2) == pytest.approx(0.75)

    def test_empty_update_is_ignored(self):
        counter = TokenUsageCounter()
        counter.update(np.array([], dtype=np.int64))
        assert counter.total == 0
        assert counter.support == set()

    @pytest.mark.parametrize(
        "values, expectation",
        [
            (np.array([1, 2], dtype=np.int64), does_not_raise()),
            (
                np.array([1.5, 2.0], dtype=np.float64),
                pytest.raises(ValueError, match="Token IDs must be integers"),
            ),
        ],
    )
    def test_rejects_non_integer_tokens(self, values, expectation):
        counter = TokenUsageCounter()
        with expectation:
            counter.update(values)


class TestTokenUsageCounterProbability:
    def test_probability_is_relative_frequency(self):
        counter = TokenUsageCounter()
        counter.update(np.array([1, 1, 1, 2], dtype=np.int64))
        assert counter.probability(1) == pytest.approx(0.75)
        assert counter.probability(2) == pytest.approx(0.25)

    def test_probability_of_unseen_token_is_zero(self):
        counter = TokenUsageCounter()
        counter.update(np.array([1], dtype=np.int64))
        assert counter.probability(99) == 0.0

    def test_probability_on_empty_counter_is_zero(self):
        counter = TokenUsageCounter()
        assert counter.probability(1) == 0.0


class TestTokenUsageCounterSaveLoad:
    def test_round_trip_preserves_counts_and_label(self, tmp_path):
        counter = TokenUsageCounter(label="rollout")
        counter.update(np.array([-2, 5, 5, 5], dtype=np.int64))
        saved_path = counter.save(tmp_path / "counts")

        loaded = TokenUsageCounter.load(saved_path)
        assert loaded.label == "rollout"
        assert loaded.total == counter.total
        loaded_tokens, loaded_counts = loaded.counts_as_arrays()
        np.testing.assert_array_equal(loaded_tokens, np.array([-2, 5]))
        np.testing.assert_array_equal(loaded_counts, np.array([1, 3]))

    def test_save_enforces_npz_suffix(self, tmp_path):
        counter = TokenUsageCounter()
        counter.update(np.array([1], dtype=np.int64))
        saved_path = counter.save(tmp_path / "counts")
        assert saved_path.suffix == ".npz"
        assert saved_path.exists()
