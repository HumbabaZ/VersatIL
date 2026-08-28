"""Tests for versatil.data.tokenization.action_discretizer module."""

from unittest.mock import patch

import numpy as np
import pytest
from scipy.fft import idct

from versatil.data.tokenization.action_discretizer import FastActionDiscretizer


@pytest.fixture
def fast_discretizer_factory():
    """Factory for a FastActionDiscretizer with load_fast_processor mocked."""

    def factory(
        decoded_string: str,
        time_horizon: int,
        action_dim: int,
        scale: float = 10.0,
        min_token: float = 0.0,
    ) -> FastActionDiscretizer:
        with patch(
            "versatil.data.tokenization.action_discretizer.load_fast_processor"
        ) as mock_loader:
            processor = mock_loader.return_value
            processor.scale = scale
            processor.min_token = min_token
            processor.bpe_tokenizer.decode.return_value = decoded_string
            discretizer = FastActionDiscretizer(
                use_pretrained=True,
                time_horizon=time_horizon,
                action_dim=action_dim,
            )
        return discretizer

    return factory


class TestBpeIdsToCoefficientTokens:
    def test_maps_characters_to_ordinals(self, fast_discretizer_factory):
        discretizer = fast_discretizer_factory(
            decoded_string="abc", time_horizon=1, action_dim=3
        )
        coefficients = discretizer.bpe_ids_to_coefficient_tokens(bpe_local_ids=[1, 2])
        np.testing.assert_array_equal(
            coefficients, np.array([ord("a"), ord("b"), ord("c")], dtype=np.float32)
        )

    def test_applies_min_token_offset(self, fast_discretizer_factory):
        discretizer = fast_discretizer_factory(
            decoded_string="abc", time_horizon=1, action_dim=3, min_token=-10.0
        )
        coefficients = discretizer.bpe_ids_to_coefficient_tokens(bpe_local_ids=[5])
        np.testing.assert_array_equal(
            coefficients,
            np.array([ord("a") - 10, ord("b") - 10, ord("c") - 10], dtype=np.float32),
        )

    def test_zero_pads_short_coefficient_sequences(self, fast_discretizer_factory):
        discretizer = fast_discretizer_factory(
            decoded_string="ab", time_horizon=1, action_dim=3
        )
        coefficients = discretizer.bpe_ids_to_coefficient_tokens(bpe_local_ids=[1])
        np.testing.assert_array_equal(
            coefficients, np.array([ord("a"), ord("b"), 0], dtype=np.float32)
        )

    def test_truncates_long_coefficient_sequences(self, fast_discretizer_factory):
        discretizer = fast_discretizer_factory(
            decoded_string="abcd", time_horizon=1, action_dim=3
        )
        coefficients = discretizer.bpe_ids_to_coefficient_tokens(bpe_local_ids=[1])
        np.testing.assert_array_equal(
            coefficients, np.array([ord("a"), ord("b"), ord("c")], dtype=np.float32)
        )


@pytest.fixture
def fittable_fast_discretizer_factory():
    """Factory for a local (fittable) FastActionDiscretizer with a mock processor."""

    def factory(
        scale: float | None = None,
        vocab_size: int | None = None,
    ) -> FastActionDiscretizer:
        with patch(
            "versatil.data.tokenization.action_discretizer.load_fast_processor"
        ) as mock_loader:
            processor = mock_loader.return_value
            processor.fit.return_value = processor
            discretizer = FastActionDiscretizer(
                use_pretrained=False,
                scale=scale,
                vocab_size=vocab_size,
            )
        return discretizer

    return factory


class TestFastScaleVocabSizeForwarding:
    def test_stores_scale_and_vocab_size(self, fittable_fast_discretizer_factory):
        discretizer = fittable_fast_discretizer_factory(scale=25.0, vocab_size=512)

        assert discretizer.scale == 25.0
        assert discretizer.vocab_size == 512

    def test_fit_forwards_scale_and_vocab_size_when_set(
        self, fittable_fast_discretizer_factory
    ):
        discretizer = fittable_fast_discretizer_factory(scale=25.0, vocab_size=512)
        action_chunks = np.zeros((4, 3, 2), dtype=np.float32)

        discretizer.fit(action_chunks)

        discretizer.processor.fit.assert_called_once_with(
            action_chunks,
            time_horizon=3,
            action_dim=2,
            scale=25.0,
            vocab_size=512,
        )

    def test_fit_omits_knobs_when_none(self, fittable_fast_discretizer_factory):
        discretizer = fittable_fast_discretizer_factory(scale=None, vocab_size=None)
        action_chunks = np.zeros((4, 3, 2), dtype=np.float32)

        discretizer.fit(action_chunks)

        discretizer.processor.fit.assert_called_once_with(
            action_chunks,
            time_horizon=3,
            action_dim=2,
        )

    def test_token_count_follows_vocab_size(self, fittable_fast_discretizer_factory):
        discretizer = fittable_fast_discretizer_factory(scale=25.0, vocab_size=512)

        discretizer.fit(np.zeros((4, 3, 2), dtype=np.float32))

        assert discretizer.token_count == 512

    def test_token_count_keeps_default_when_vocab_size_none(
        self, fittable_fast_discretizer_factory
    ):
        discretizer = fittable_fast_discretizer_factory(scale=None, vocab_size=None)

        discretizer.fit(np.zeros((4, 3, 2), dtype=np.float32))

        assert discretizer.token_count == 1024


class TestFastDecodeUsesCoefficientTokens:
    def test_decode_matches_inverse_dct_of_coefficient_tokens(
        self, fast_discretizer_factory
    ):
        time_horizon = 2
        action_dim = 3
        scale = 10.0
        discretizer = fast_discretizer_factory(
            decoded_string="abcdef",
            time_horizon=time_horizon,
            action_dim=action_dim,
            scale=scale,
        )
        coefficients = discretizer.bpe_ids_to_coefficient_tokens(bpe_local_ids=[1, 2])
        expected = idct(
            coefficients.reshape(time_horizon, action_dim) / scale,
            axis=0,
            norm="ortho",
        )
        decoded = discretizer.decode(token_sequences=[[1, 2]])
        np.testing.assert_allclose(decoded[0], expected)
