"""Tests for the conditional-circle configs used by versatil.analysis.tip1_noise.sweep.

Composes each conditional config exactly as the sweep does. Hydra composes
without error even when a variant silently drops its overrides, so the assertions
check the composed values rather than trusting that composition succeeded.
"""

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from versatil.analysis.tip1_noise.sweep import (
    CONDITIONAL_METHOD_CONFIG,
    FAST_MAX_TOKEN_LEN,
    FAST_OVERRIDES,
    MATCHED_OVERRIDES,
)
from versatil.configs.paths import get_hydra_configs_dir
from versatil.data.synthetic.constants import SyntheticTaskName

CONTEXT_COLUMN_KEYS = ["c0", "c1", "c2"]


def _compose(method: str) -> DictConfig:
    overrides = list(MATCHED_OVERRIDES)
    if method == "fast":
        overrides += list(FAST_OVERRIDES)
    with initialize_config_dir(
        config_dir=str(get_hydra_configs_dir()), version_base=None
    ):
        return compose(
            config_name=CONDITIONAL_METHOD_CONFIG[method], overrides=overrides
        )


@pytest.mark.integration
@pytest.mark.parametrize("method", sorted(CONDITIONAL_METHOD_CONFIG))
def test_conditional_config_feeds_context_to_the_decoder(method: str):
    config = _compose(method)

    observation_entries = config.task.observation_space.observations_metadata.values()
    context_entries = [
        entry
        for entry in observation_entries
        if list(entry.get("raw_data_column_keys", [])) == CONTEXT_COLUMN_KEYS
    ]
    assert len(context_entries) == 1
    assert (
        config.task.dataset_schema.task_name
        == SyntheticTaskName.CONDITIONAL_CIRCLE.value
    )
    assert "context" in config.policy.encoding_pipeline.encoders
    assert "context_proprio" in config.policy.decoder.input_keys


@pytest.mark.integration
@pytest.mark.parametrize("method", sorted(CONDITIONAL_METHOD_CONFIG))
def test_conditional_config_keeps_the_matched_backbone_overrides(method: str):
    config = _compose(method)

    assert config.policy.decoder.number_of_layers == 4
    assert config.policy.decoder.number_of_heads == 4
    assert config.policy.decoder.dropout_rate == 0.4
    assert config.policy.decoder.attention_dropout == 0.15
    assert config.training.optimizer.lr == 1e-4
    assert config.training.use_ema is True
    if method == "fast":
        tokenizer = config.task.dataloader.tokenization.action_tokenizer
        assert tokenizer.max_token_len == FAST_MAX_TOKEN_LEN
