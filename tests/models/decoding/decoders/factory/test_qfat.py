"""Tests for versatil.models.decoding.decoders.factory.qfat module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from versatil.models.decoding.action_heads.gaussian import GaussianHead
from versatil.models.decoding.constants import (
    DecoderOutputKey,
    GMMInitStrategy,
    MixtureSamplingMode,
)
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.qfat import QFATActionTransformer
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.normalization.constants import NormalizationType

EMBEDDING_DIMENSION = 32
NUMBER_OF_HEADS = 2
NUMBER_OF_LAYERS = 1
FEEDFORWARD_DIMENSION = 64
SPATIAL_HEIGHT = 4
SPATIAL_WIDTH = 4
BATCH_SIZE = 2
POSITION_DIM = 3
PREDICTION_HORIZON = 4
NUM_MIXTURE_COMPONENTS = 3


@pytest.fixture
def gaussian_qfat_factory(
    mock_action_space_factory: Callable[..., MagicMock],
    mock_observation_space_factory: Callable[..., MagicMock],
    gaussian_head_factory: Callable[..., GaussianHead],
) -> Callable[..., QFATActionTransformer]:
    """Factory for QFATActionTransformer using GaussianHead action heads."""

    def factory(
        input_keys: list[str] | None = None,
        position_dim: int = POSITION_DIM,
        has_orientation: bool = False,
        orientation_dim: int = 0,
        has_gripper: bool = False,
        gripper_dim: int = 0,
        observation_horizon: int = 1,
        prediction_horizon: int = PREDICTION_HORIZON,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        number_of_heads: int = NUMBER_OF_HEADS,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = FEEDFORWARD_DIMENSION,
        number_of_layers: int = NUMBER_OF_LAYERS,
        activation: str = ActivationFunction.GELU.value,
        normalization_type: str = NormalizationType.LAYER_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.0,
        positional_encoding_type: str | None = PositionalEncodingType.ROPE.value,
        num_mixture_components: int = NUM_MIXTURE_COMPONENTS,
        gmm_init_strategy: str = GMMInitStrategy.KMEANS_PLUS_PLUS.value,
        inference_sampling_mode: str = MixtureSamplingMode.DETERMINISTIC.value,
        device: str = "cpu",
    ) -> QFATActionTransformer:
        if input_keys is None:
            input_keys = ["rgb_features"]
        action_space = mock_action_space_factory(
            position_dim=position_dim,
            has_orientation=has_orientation,
            orientation_dim=orientation_dim,
            has_gripper=has_gripper,
            gripper_dim=gripper_dim,
        )
        action_heads = {}
        for key, meta in action_space.actions_metadata.items():
            if meta.requires_prediction_head:
                action_heads[key] = gaussian_head_factory(
                    input_dimension=embedding_dimension,
                )
        observation_space = mock_observation_space_factory()
        return QFATActionTransformer(
            action_heads=action_heads,
            input_keys=input_keys,
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            number_of_layers=number_of_layers,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            positional_encoding_type=positional_encoding_type,
            num_mixture_components=num_mixture_components,
            gmm_init_strategy=gmm_init_strategy,
            inference_sampling_mode=inference_sampling_mode,
        )

    return factory


@pytest.mark.unit
class TestQFATInitialization:
    def test_inherits_from_action_decoder(
        self,
        gaussian_qfat_factory: Callable[..., QFATActionTransformer],
    ):
        decoder = gaussian_qfat_factory()
        assert isinstance(decoder, ActionDecoder)

    @pytest.mark.parametrize("observation_horizon", [1, 2])
    def test_stores_configuration(
        self,
        gaussian_qfat_factory: Callable[..., QFATActionTransformer],
        observation_horizon: int,
    ):
        decoder = gaussian_qfat_factory(
            embedding_dimension=EMBEDDING_DIMENSION,
            num_mixture_components=NUM_MIXTURE_COMPONENTS,
            observation_horizon=observation_horizon,
        )
        assert decoder.embedding_dimension == EMBEDDING_DIMENSION
        assert decoder.num_mixture_components == NUM_MIXTURE_COMPONENTS
        assert decoder.observation_horizon == observation_horizon


@pytest.mark.integration
class TestQFATForwardWithGaussianHead:
    def test_training_returns_mixture_outputs(
        self,
        gaussian_qfat_factory: Callable[..., QFATActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gaussian_qfat_factory()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = noisy_actions_factory(
            prediction_horizon=PREDICTION_HORIZON,
            action_keys_to_dims={"position_action": POSITION_DIM},
        )
        predictions = decoder(features=features, actions=actions)
        assert f"position_action_{DecoderOutputKey.MEAN.value}" in predictions
        assert f"position_action_{DecoderOutputKey.LOGVAR.value}" in predictions
        assert DecoderOutputKey.ROUTING_WEIGHTS.value in predictions

    def test_training_output_shapes(
        self,
        gaussian_qfat_factory: Callable[..., QFATActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        noisy_actions_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gaussian_qfat_factory()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        actions = noisy_actions_factory(
            prediction_horizon=PREDICTION_HORIZON,
            action_keys_to_dims={"position_action": POSITION_DIM},
        )
        predictions = decoder(features=features, actions=actions)
        mean_key = f"position_action_{DecoderOutputKey.MEAN.value}"
        logvar_key = f"position_action_{DecoderOutputKey.LOGVAR.value}"
        expected_shape = (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            NUM_MIXTURE_COMPONENTS,
            POSITION_DIM,
        )
        assert predictions[mean_key].shape == expected_shape
        assert predictions[logvar_key].shape == expected_shape
        assert predictions[DecoderOutputKey.ROUTING_WEIGHTS.value].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            NUM_MIXTURE_COMPONENTS,
        )

    def test_inference_returns_action_chunk(
        self,
        gaussian_qfat_factory: Callable[..., QFATActionTransformer],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        decoder = gaussian_qfat_factory()
        features = spatial_feature_factory(
            batch_size=BATCH_SIZE,
            channels=EMBEDDING_DIMENSION,
            height=SPATIAL_HEIGHT,
            width=SPATIAL_WIDTH,
        )
        predictions = decoder(features=features, actions=None)
        assert "position_action" in predictions
        assert predictions["position_action"].shape == (
            BATCH_SIZE,
            PREDICTION_HORIZON,
            POSITION_DIM,
        )
