"""Offline collection of the training-time action-token distribution.

Reuses the checkpoint's config and saved tokenizer so the counted local ID
space matches what the model was trained on and what it emits at rollout. Run
as ``python -m versatil.analysis.token_usage.collect_training``.
"""

import argparse
import enum
import logging
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from versatil.analysis.token_usage.counter import TokenUsageCounter
from versatil.common.logging import override_log_format
from versatil.configs import MainConfig
from versatil.data.constants import SampleKey
from versatil.data.dataloader import get_dataloaders
from versatil.data.tokenization.action_discretizer import FastActionDiscretizer
from versatil.data.tokenization.tokenizer import Tokenizer

CHECKPOINT_CONFIG_NAME = "config.yaml"
CHECKPOINT_TOKENIZER_DIR = "tokenizer"


class TokenUsageOutputName(enum.StrEnum):
    """Base filenames for training token-usage count files."""

    TRAIN_TOKENS = "train_tokens"
    TRAIN_COEFFICIENTS = "train_coefficients"


def collect_training_token_usage(
    checkpoint_path: str,
    output_dir: str,
    max_batches: int | None = None,
    device: str = "cpu",
) -> dict[str, Path]:
    """Count action-token usage over the training set for one checkpoint.

    For FAST it additionally counts reverse-BPE coefficient tokens, the
    metric-structured level used to compare against rollout usage.

    Args:
        checkpoint_path: Directory with the checkpoint config and saved tokenizer.
        output_dir: Destination directory for the count files.
        max_batches: Optional cap on the number of training batches scanned.
        device: Torch device string for tokenizer tensors.

    Returns:
        Mapping from output name to the written ``.npz`` path.

    Raises:
        FileNotFoundError: If the config or saved tokenizer is missing.
        ValueError: If the checkpoint tokenizer has no action tokenizer.
    """
    config_file = os.path.join(checkpoint_path, CHECKPOINT_CONFIG_NAME)
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config not found at {config_file}")
    config: MainConfig = hydra.utils.instantiate(OmegaConf.load(config_file))
    # Single-process loading so the tokenizer override below takes effect and
    # counting stays deterministic.
    config.task.dataloader.num_workers = 0

    train_loader, _, _, _, _ = get_dataloaders(config)

    tokenizer_path = os.path.join(checkpoint_path, CHECKPOINT_TOKENIZER_DIR)
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Saved tokenizer not found at {tokenizer_path}; token-usage counting "
            "needs the checkpoint tokenizer to match the rollout ID space."
        )
    tokenizer = Tokenizer.from_pretrained(tokenizer_path, device=torch.device(device))
    if tokenizer.action_tokenizer is None:
        raise ValueError("Checkpoint tokenizer has no action tokenizer to analyze.")
    train_loader.dataset.set_tokenizer(tokenizer)

    action_tokenizer = tokenizer.action_tokenizer
    discretizer = action_tokenizer.action_discretizer
    is_fast = isinstance(discretizer, FastActionDiscretizer)

    token_counter = TokenUsageCounter(label=TokenUsageOutputName.TRAIN_TOKENS.value)
    coefficient_counter = (
        TokenUsageCounter(label=TokenUsageOutputName.TRAIN_COEFFICIENTS.value)
        if is_fast
        else None
    )

    action_key = SampleKey.ACTION.value
    tokenized_key = SampleKey.TOKENIZED_ACTIONS.value
    for batch_index, batch in enumerate(train_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        tokenized_actions = batch[action_key][tokenized_key]
        for sample_tokens in tokenized_actions:
            local_ids = action_tokenizer.to_local_token_ids(sample_tokens)
            token_counter.update(local_ids)
            if coefficient_counter is not None:
                coefficients = discretizer.bpe_ids_to_coefficient_tokens(
                    bpe_local_ids=local_ids.tolist()
                )
                coefficient_counter.update(coefficients.astype(np.int64))

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {
        TokenUsageOutputName.TRAIN_TOKENS.value: token_counter.save(
            output_dir_path / TokenUsageOutputName.TRAIN_TOKENS.value
        )
    }
    if coefficient_counter is not None:
        written[TokenUsageOutputName.TRAIN_COEFFICIENTS.value] = (
            coefficient_counter.save(
                output_dir_path / TokenUsageOutputName.TRAIN_COEFFICIENTS.value
            )
        )
    logging.info(f"Wrote training token usage to {output_dir_path}")
    return written


def main() -> None:
    """Run the training token-usage collector from the command line."""
    override_log_format()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    collect_training_token_usage(
        checkpoint_path=arguments.checkpoint_path,
        output_dir=arguments.output_dir,
        max_batches=arguments.max_batches,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
