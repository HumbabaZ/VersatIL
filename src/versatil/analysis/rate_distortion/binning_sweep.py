"""Fit per-value binning at each ``num_bins`` and measure rate and distortion.

Binning is the single-knob discrete baseline: the bin count moves rate and
distortion together (spec section 3), unlike FAST's decoupled knobs. This reuses
versatil's ``BinnedActionDiscretizer`` on the exact same normalized chunks and
normalizer the FAST sweep uses, so the two families' rate–distortion axes are
directly comparable.
"""

import logging
from typing import Any

from versatil.analysis.rate_distortion.data import ActionChunkData
from versatil.analysis.rate_distortion.metrics import (
    binning_rate,
    reconstruction_distortion,
)
from versatil.data.tokenization.action_discretizer import BinnedActionDiscretizer

FAMILY = "binning"
OPERATING_POINT_BINS = 256


def run_binning_cell(
    chunk_data: ActionChunkData,
    num_bins: int,
    binning_strategy: str,
) -> dict[str, Any]:
    """Fit, round-trip, and score one binning configuration.

    Args:
        chunk_data: Normalized chunks plus normalizer and layout.
        num_bins: Bins per action value for this cell.
        binning_strategy: Bin-edge strategy (``quantile`` or ``uniform``).

    Returns:
        A flat result row matching the FAST row schema (family, rate, distortion).
    """
    chunks = chunk_data.chunks_normalized
    discretizer = BinnedActionDiscretizer(
        num_bins=num_bins, binning_strategy=binning_strategy
    )
    discretizer.fit(chunks)
    token_lists = [discretizer.encode(chunk) for chunk in chunks]
    reconstruction = discretizer.decode(token_lists)

    row: dict[str, Any] = {
        "family": FAMILY,
        "sweep": "bins",
        "num_bins": num_bins,
        "is_operating_point": num_bins == OPERATING_POINT_BINS,
        "feasible": True,
    }
    row.update(
        binning_rate(
            num_bins=num_bins,
            time_horizon=chunk_data.time_horizon,
            action_dim=chunk_data.action_dim,
        )
    )
    row.update(
        reconstruction_distortion(
            ground_truth_normalized=chunks,
            reconstruction_normalized=reconstruction,
            normalizer=chunk_data.normalizer,
            layout=chunk_data.layout,
        )
    )
    logging.info(
        "Binning cell num_bins=%s: bits/chunk=%.1f continuous_rmse=%.5f",
        num_bins,
        row["bits_per_chunk"],
        row.get("rmse_continuous", float("nan")),
    )
    return row


def run_binning_sweep(
    chunk_data: ActionChunkData,
    bin_counts: list[int],
    binning_strategy: str,
) -> list[dict[str, Any]]:
    """Run one binning cell per bin count."""
    return [
        run_binning_cell(
            chunk_data=chunk_data, num_bins=num_bins, binning_strategy=binning_strategy
        )
        for num_bins in bin_counts
    ]
