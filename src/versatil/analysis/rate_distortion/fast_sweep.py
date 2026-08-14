"""Fit FAST at each (scale, vocab_size) cell and measure rate and distortion.

The sweep drives ``UniversalActionProcessor`` directly rather than through
``FastActionDiscretizer``, because the discretizer wrapper hardcodes the scale
and vocabulary. Fitting the processor per cell is the only way to move the two
knobs independently.
"""

import logging
from typing import Any

import numpy as np
from transformers.processing_utils import ProcessorMixin

from versatil.analysis.rate_distortion.config import FastRateDistortionConfig
from versatil.analysis.rate_distortion.data import ActionChunkData
from versatil.analysis.rate_distortion.metrics import (
    check_near_lossless,
    dct_alphabet_size,
    rate_metrics,
    reconstruction_distortion,
)
from versatil.analysis.rate_distortion.replay import replay_success
from versatil.data.tokenization.fast import load_fast_processor

FAMILY = "fast"


def load_fast_class(model_path: str) -> type:
    """Return the ``UniversalActionProcessor`` class from cached remote code.

    Args:
        model_path: HuggingFace model id or local path supplying the processor.

    Returns:
        The processor class exposing the ``fit`` classmethod.
    """
    return type(load_fast_processor(model_path))


def fit_fast_processor(
    fast_class: type,
    chunks_normalized: np.ndarray,
    scale: float,
    vocab_size: int,
    time_horizon: int,
    action_dim: int,
) -> ProcessorMixin:
    """Fit a FAST processor on LIBERO chunks at one (scale, vocab_size) cell."""
    return fast_class.fit(
        list(chunks_normalized.astype(np.float32)),
        scale=scale,
        vocab_size=vocab_size,
        time_horizon=time_horizon,
        action_dim=action_dim,
    )


def run_fast_cell(
    fast_class: type,
    chunk_data: ActionChunkData,
    scale: float,
    vocab_size: int,
    sweep: str,
    is_operating_point: bool,
) -> dict[str, Any]:
    """Fit, round-trip, and score one (scale, vocab_size) configuration.

    Infeasible cells (vocabulary below the DCT alphabet floor) are recorded with
    NaN metrics instead of raising, so a full grid still produces a table.

    Args:
        fast_class: The ``UniversalActionProcessor`` class.
        chunk_data: Normalized chunks plus normalizer and layout.
        scale: DCT rounding scale for this cell.
        vocab_size: BPE vocabulary size for this cell.
        sweep: Which sweep this row belongs to (``scale``, ``vocab``, ``sanity``).
        is_operating_point: Whether this is the shared main-experiment config.

    Returns:
        A flat result row with rate, distortion, and bookkeeping fields.
    """
    chunks = chunk_data.chunks_normalized
    alphabet_size = dct_alphabet_size(chunks_normalized=chunks, scale=scale)
    row: dict[str, Any] = {
        "family": FAMILY,
        "sweep": sweep,
        "scale": scale,
        "vocab_size": vocab_size,
        "is_operating_point": is_operating_point,
        "alphabet_size": alphabet_size,
        "feasible": vocab_size >= alphabet_size,
        "replay_success": replay_success(),
    }
    if not row["feasible"]:
        logging.warning(
            "Skipping FAST cell scale=%s vocab_size=%s: vocabulary below DCT "
            "alphabet floor %s.",
            scale,
            vocab_size,
            alphabet_size,
        )
        return row

    processor = fit_fast_processor(
        fast_class=fast_class,
        chunks_normalized=chunks,
        scale=scale,
        vocab_size=vocab_size,
        time_horizon=chunk_data.time_horizon,
        action_dim=chunk_data.action_dim,
    )
    token_lists = processor(chunks.astype(np.float32))
    reconstruction = processor.decode(
        token_lists,
        time_horizon=chunk_data.time_horizon,
        action_dim=chunk_data.action_dim,
    )
    row.update(rate_metrics(token_lists=token_lists, vocab_size=vocab_size))
    row.update(
        reconstruction_distortion(
            ground_truth_normalized=chunks,
            reconstruction_normalized=reconstruction,
            normalizer=chunk_data.normalizer,
            layout=chunk_data.layout,
        )
    )
    logging.info(
        "FAST cell scale=%s vocab_size=%s: bits/chunk=%.1f continuous_rmse=%.5f",
        scale,
        vocab_size,
        row.get("bits_per_chunk", float("nan")),
        row.get("rmse_continuous", float("nan")),
    )
    return row


def run_fast_grid(
    fast_class: type,
    chunk_data: ActionChunkData,
    config: FastRateDistortionConfig,
) -> list[dict[str, Any]]:
    """Near-lossless check + scale sweep (G1) + vocab sweep (G2) at one horizon.

    Args:
        fast_class: The ``UniversalActionProcessor`` class (loaded once by caller).
        chunk_data: Normalized chunks plus normalizer and layout.
        config: Sweep grid.

    Returns:
        One result row per grid cell, most-important sweeps first.

    Raises:
        ValueError: If the near-lossless sanity check fails.
    """
    sanity_row = run_fast_cell(
        fast_class=fast_class,
        chunk_data=chunk_data,
        scale=config.near_lossless_scale,
        vocab_size=config.near_lossless_vocab,
        sweep="sanity",
        is_operating_point=False,
    )
    if not sanity_row["feasible"]:
        raise ValueError(
            "Near-lossless sanity cell is infeasible: increase near_lossless_vocab "
            f"above the DCT alphabet floor {sanity_row['alphabet_size']}."
        )
    check_near_lossless(
        distortion_rmse=sanity_row["rmse_continuous"],
        tolerance=config.near_lossless_tolerance,
    )

    rows = [sanity_row]
    for scale in config.scales:
        rows.append(
            run_fast_cell(
                fast_class=fast_class,
                chunk_data=chunk_data,
                scale=scale,
                vocab_size=config.scale_sweep_fixed_vocab,
                sweep="scale",
                is_operating_point=scale == config.vocab_sweep_fixed_scale,
            )
        )
    for vocab_size in config.vocab_sizes:
        rows.append(
            run_fast_cell(
                fast_class=fast_class,
                chunk_data=chunk_data,
                scale=config.vocab_sweep_fixed_scale,
                vocab_size=vocab_size,
                sweep="vocab",
                is_operating_point=vocab_size == config.scale_sweep_fixed_vocab,
            )
        )
    return rows


def run_sweep(
    chunk_data: ActionChunkData,
    config: FastRateDistortionConfig,
) -> list[dict[str, Any]]:
    """Load the FAST class and run the full grid (single-horizon Part-1 entry)."""
    return run_fast_grid(
        fast_class=load_fast_class(config.pretrained_fast_model),
        chunk_data=chunk_data,
        config=config,
    )
