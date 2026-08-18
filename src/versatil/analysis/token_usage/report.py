"""Training-vs-rollout token-usage report: variety, frequency, and shift.

Loads the offline training counts and the rollout token capture, maps both into
the same local ID space (and, for FAST, the reverse-BPE coefficient space),
then writes variety/shift metrics and frequency figures. Run as
``python -m versatil.analysis.token_usage.report``.
"""

import argparse
import enum
import json
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: E402, F401
import torch  # noqa: E402

from versatil.analysis.token_usage.collect_training import (  # noqa: E402
    CHECKPOINT_TOKENIZER_DIR,
    TokenUsageOutputName,
)
from versatil.analysis.token_usage.counter import TokenUsageCounter  # noqa: E402
from versatil.analysis.token_usage.rollout_sink import RolloutTokenField  # noqa: E402
from versatil.common.logging import override_log_format  # noqa: E402
from versatil.data.tokenization.action_discretizer import (  # noqa: E402
    ActionDiscretizer,
    FastActionDiscretizer,
)
from versatil.data.tokenization.action_tokenizer import ActionTokenizer  # noqa: E402
from versatil.data.tokenization.tokenizer import Tokenizer  # noqa: E402

PLOT_STYLE = ["science", "no-latex"]
QUALITATIVE_COLORMAP = "tab10"
METRICS_FILENAME = "token_usage_metrics.json"
LAPLACE_SMOOTHING = 1.0
SATURATION_BOOTSTRAP = 200
SATURATION_SEED = 0


class UsageLevel(enum.StrEnum):
    """Analysis levels for a tokenizer's usage."""

    TOKEN = "token_level"
    COEFFICIENT = "coefficient_level"


def _counter_to_dict(counter: TokenUsageCounter) -> dict[int, int]:
    """Return a ``{token_id: count}`` view of a counter via its public arrays."""
    tokens, counts = counter.counts_as_arrays()
    return {
        int(token): int(count)
        for token, count in zip(tokens.tolist(), counts.tolist(), strict=True)
    }


def variety_metrics(
    train_counter: TokenUsageCounter, rollout_counter: TokenUsageCounter
) -> dict[str, float]:
    """Compare support sets: sizes, overlap, coverage, and OOD mass.

    Args:
        train_counter: Training-time token counts.
        rollout_counter: Rollout-time token counts.

    Returns:
        Support sizes, Jaccard overlap, training coverage, out-of-distribution
        token mass, and the OOD token count.
    """
    train_support = train_counter.support
    rollout_support = rollout_counter.support
    intersection = train_support & rollout_support
    union = train_support | rollout_support
    rollout_map = _counter_to_dict(rollout_counter)
    rollout_total = rollout_counter.total
    out_of_distribution_ids = rollout_support - train_support
    out_of_distribution_mass = (
        sum(rollout_map[token] for token in out_of_distribution_ids) / rollout_total
        if rollout_total > 0
        else 0.0
    )
    return {
        "support_train": float(len(train_support)),
        "support_rollout": float(len(rollout_support)),
        "jaccard": len(intersection) / len(union) if union else 0.0,
        "train_coverage": len(intersection) / len(train_support)
        if train_support
        else 0.0,
        "ood_token_count": float(len(out_of_distribution_ids)),
        "ood_mass": out_of_distribution_mass,
    }


def shift_metrics(
    train_counter: TokenUsageCounter, rollout_counter: TokenUsageCounter
) -> dict[str, float]:
    """Compute KL(rollout||train) and JS divergence in bits over the union support."""
    train_map = _counter_to_dict(train_counter)
    rollout_map = _counter_to_dict(rollout_counter)
    support = sorted(set(train_map) | set(rollout_map))
    if not support:
        return {"kl_rollout_train_bits": 0.0, "js_divergence_bits": 0.0}
    train_vector = (
        np.array([train_map.get(token, 0) for token in support], dtype=np.float64)
        + LAPLACE_SMOOTHING
    )
    rollout_vector = (
        np.array([rollout_map.get(token, 0) for token in support], dtype=np.float64)
        + LAPLACE_SMOOTHING
    )
    train_probabilities = train_vector / train_vector.sum()
    rollout_probabilities = rollout_vector / rollout_vector.sum()
    mean_probabilities = 0.5 * (train_probabilities + rollout_probabilities)
    kl_divergence = float(
        np.sum(
            rollout_probabilities * np.log2(rollout_probabilities / train_probabilities)
        )
    )
    js_divergence = float(
        0.5
        * np.sum(
            train_probabilities * np.log2(train_probabilities / mean_probabilities)
        )
        + 0.5
        * np.sum(
            rollout_probabilities * np.log2(rollout_probabilities / mean_probabilities)
        )
    )
    return {"kl_rollout_train_bits": kl_divergence, "js_divergence_bits": js_divergence}


def _rollout_local_token_rows(
    rollout_jsonl: Path,
    action_tokenizer: ActionTokenizer,
    discretizer: ActionDiscretizer,
    is_fast: bool,
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    """Map captured rollout rows to per-chunk local (and FAST coefficient) IDs.

    Returns one array per captured chunk so both the pooled counts and the
    saturation (rarefaction) curve can be derived from the same rows.
    """
    token_rows: list[np.ndarray] = []
    coefficient_rows: list[np.ndarray] | None = [] if is_fast else None
    with rollout_jsonl.open("r", encoding="utf-8") as rollout_file:
        for line in rollout_file:
            row = json.loads(line)
            model_tokens = row[RolloutTokenField.TOKENS.value]
            local_ids = action_tokenizer.to_local_token_ids(model_tokens)
            token_rows.append(local_ids)
            if coefficient_rows is not None:
                coefficients = discretizer.bpe_ids_to_coefficient_tokens(
                    bpe_local_ids=local_ids.tolist()
                )
                coefficient_rows.append(coefficients.astype(np.int64))
    return token_rows, coefficient_rows


def _counter_from_rows(rows: list[np.ndarray], label: str) -> TokenUsageCounter:
    """Pool per-chunk ID arrays into one frequency counter."""
    counter = TokenUsageCounter(label=label)
    for row in rows:
        counter.update(row)
    return counter


def _plot_rank_frequency(
    train_counter: TokenUsageCounter,
    rollout_counter: TokenUsageCounter,
    title: str,
    output_path: Path,
) -> None:
    """Plot the log-log rank-frequency (Zipf) curves for train vs rollout."""
    colormap = matplotlib.colormaps[QUALITATIVE_COLORMAP]
    with plt.style.context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=(4.5, 3.5))
        for series_index, (counter, label) in enumerate(
            ((train_counter, "train"), (rollout_counter, "rollout"))
        ):
            _, counts = counter.counts_as_arrays()
            if counts.size == 0:
                continue
            sorted_counts = np.sort(counts)[::-1]
            probabilities = sorted_counts / sorted_counts.sum()
            ranks = np.arange(1, probabilities.size + 1)
            axis.loglog(
                ranks,
                probabilities,
                marker="o",
                markersize=3,
                linewidth=1.0,
                color=colormap(series_index),
                label=label,
            )
        axis.set_xlabel("rank")
        axis.set_ylabel("probability")
        axis.set_title(title, fontsize=11)
        axis.legend(fontsize=10)
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)


def _plot_distribution_over_ids(
    train_counter: TokenUsageCounter,
    rollout_counter: TokenUsageCounter,
    title: str,
    output_path: Path,
) -> None:
    """Plot probability vs token ID for train vs rollout on a metric ID axis."""
    colormap = matplotlib.colormaps[QUALITATIVE_COLORMAP]
    train_map = _counter_to_dict(train_counter)
    rollout_map = _counter_to_dict(rollout_counter)
    support = sorted(set(train_map) | set(rollout_map))
    if not support:
        return
    ids = np.array(support, dtype=np.int64)
    train_total = max(train_counter.total, 1)
    rollout_total = max(rollout_counter.total, 1)
    train_probabilities = (
        np.array([train_map.get(token, 0) for token in support], dtype=np.float64)
        / train_total
    )
    rollout_probabilities = (
        np.array([rollout_map.get(token, 0) for token in support], dtype=np.float64)
        / rollout_total
    )
    with plt.style.context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=(4.5, 3.5))
        axis.plot(
            ids,
            train_probabilities,
            drawstyle="steps-mid",
            linewidth=1.0,
            color=colormap(0),
            label="train",
        )
        axis.plot(
            ids,
            rollout_probabilities,
            drawstyle="steps-mid",
            linewidth=1.0,
            color=colormap(1),
            label="rollout",
        )
        axis.set_xlabel("token id")
        axis.set_ylabel("probability")
        axis.set_title(title, fontsize=11)
        axis.legend(fontsize=10)
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)


def _saturation_curve(
    rollout_rows: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Rarefaction of rollout support vs accumulated chunks.

    Averages, over many random chunk orderings, how the number of distinct IDs
    grows as chunks accumulate. A near-zero last-chunk marginal gain means the
    rollout support has saturated and coverage numbers can be trusted; a still
    positive gain means the support is under-sampled.

    Args:
        rollout_rows: Per-chunk local ID arrays.

    Returns:
        ``steps`` (1..num_chunks), mean and std of distinct-ID count at each
        step, and a summary dict with ``num_chunks``, ``final_support`` and
        ``last_chunk_marginal_gain``.
    """
    num_chunks = len(rollout_rows)
    if num_chunks == 0:
        empty = np.empty(0, dtype=np.float64)
        return (
            empty,
            empty,
            empty,
            {
                "num_chunks": 0.0,
                "final_support": 0.0,
                "last_chunk_marginal_gain": 0.0,
            },
        )
    generator = np.random.default_rng(SATURATION_SEED)
    support_by_step = np.zeros((SATURATION_BOOTSTRAP, num_chunks), dtype=np.int64)
    for bootstrap_index in range(SATURATION_BOOTSTRAP):
        order = generator.permutation(num_chunks)
        seen: set[int] = set()
        for step, chunk_index in enumerate(order):
            seen.update(int(token) for token in rollout_rows[chunk_index].tolist())
            support_by_step[bootstrap_index, step] = len(seen)
    mean_support = support_by_step.mean(axis=0)
    std_support = support_by_step.std(axis=0)
    steps = np.arange(1, num_chunks + 1)
    last_gain = (
        float(mean_support[-1] - mean_support[-2]) if num_chunks >= 2 else float("nan")
    )
    summary = {
        "num_chunks": float(num_chunks),
        "final_support": float(mean_support[-1]),
        "last_chunk_marginal_gain": last_gain,
    }
    return steps, mean_support, std_support, summary


def _plot_saturation(
    steps: np.ndarray,
    mean_support: np.ndarray,
    std_support: np.ndarray,
    train_support: int,
    title: str,
    output_path: Path,
) -> None:
    """Plot the rollout support rarefaction curve against training support."""
    if steps.size == 0:
        return
    colormap = matplotlib.colormaps[QUALITATIVE_COLORMAP]
    with plt.style.context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=(4.8, 3.6))
        axis.plot(
            steps,
            mean_support,
            color=colormap(1),
            linewidth=1.4,
            label="rollout support (mean)",
        )
        axis.fill_between(
            steps,
            mean_support - std_support,
            mean_support + std_support,
            color=colormap(1),
            alpha=0.25,
            linewidth=0,
        )
        axis.axhline(
            train_support,
            color=colormap(0),
            linestyle="--",
            linewidth=1.0,
            label=f"train support = {train_support}",
        )
        axis.set_xlabel("rollout chunks accumulated")
        axis.set_ylabel("distinct tokens")
        axis.set_title(title, fontsize=11)
        axis.legend(fontsize=9, loc="lower right")
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)


def _analyze_level(
    train_counter: TokenUsageCounter,
    rollout_counter: TokenUsageCounter,
    rollout_rows: list[np.ndarray],
    level: UsageLevel,
    ids_are_metric: bool,
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    """Compute metrics and write figures for one usage level."""
    steps, mean_support, std_support, saturation = _saturation_curve(
        rollout_rows=rollout_rows
    )
    metrics = {
        "variety": variety_metrics(train_counter, rollout_counter),
        "shift": shift_metrics(train_counter, rollout_counter),
        "saturation": saturation,
    }
    _plot_rank_frequency(
        train_counter=train_counter,
        rollout_counter=rollout_counter,
        title=f"{level.value} rank-frequency",
        output_path=output_dir / f"{level.value}_rank_frequency.png",
    )
    _plot_saturation(
        steps=steps,
        mean_support=mean_support,
        std_support=std_support,
        train_support=len(train_counter.support),
        title=f"{level.value} rollout saturation",
        output_path=output_dir / f"{level.value}_saturation.png",
    )
    if ids_are_metric:
        _plot_distribution_over_ids(
            train_counter=train_counter,
            rollout_counter=rollout_counter,
            title=f"{level.value} usage over ids",
            output_path=output_dir / f"{level.value}_distribution.png",
        )
    return metrics


def generate_report(
    checkpoint_path: str,
    train_counts_dir: str,
    rollout_jsonl: str,
    output_dir: str,
    device: str = "cpu",
) -> Path:
    """Produce the training-vs-rollout token-usage report.

    Args:
        checkpoint_path: Directory with the saved tokenizer used at rollout.
        train_counts_dir: Directory holding the offline training count files.
        rollout_jsonl: Rollout token capture JSONL from the capture hook.
        output_dir: Destination directory for metrics JSON and figures.
        device: Torch device string for the tokenizer.

    Returns:
        Path to the written metrics JSON file.

    Raises:
        FileNotFoundError: If the tokenizer, training counts, or rollout capture
            are missing.
        ValueError: If the checkpoint tokenizer has no action tokenizer.
    """
    tokenizer_path = os.path.join(checkpoint_path, CHECKPOINT_TOKENIZER_DIR)
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Saved tokenizer not found at {tokenizer_path}")
    tokenizer = Tokenizer.from_pretrained(tokenizer_path, device=torch.device(device))
    if tokenizer.action_tokenizer is None:
        raise ValueError("Checkpoint tokenizer has no action tokenizer to analyze.")
    action_tokenizer = tokenizer.action_tokenizer
    discretizer = action_tokenizer.action_discretizer
    is_fast = isinstance(discretizer, FastActionDiscretizer)

    train_counts_path = Path(train_counts_dir)
    train_token_counter = TokenUsageCounter.load(
        train_counts_path / TokenUsageOutputName.TRAIN_TOKENS.value
    )
    token_rows, coefficient_rows = _rollout_local_token_rows(
        rollout_jsonl=Path(rollout_jsonl),
        action_tokenizer=action_tokenizer,
        discretizer=discretizer,
        is_fast=is_fast,
    )
    rollout_token_counter = _counter_from_rows(rows=token_rows, label="rollout")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict[str, dict[str, float]]] = {
        UsageLevel.TOKEN.value: _analyze_level(
            train_counter=train_token_counter,
            rollout_counter=rollout_token_counter,
            rollout_rows=token_rows,
            level=UsageLevel.TOKEN,
            ids_are_metric=not is_fast,
            output_dir=output_dir_path,
        )
    }
    if is_fast and coefficient_rows is not None:
        train_coefficient_counter = TokenUsageCounter.load(
            train_counts_path / TokenUsageOutputName.TRAIN_COEFFICIENTS.value
        )
        rollout_coefficient_counter = _counter_from_rows(
            rows=coefficient_rows, label="rollout"
        )
        report[UsageLevel.COEFFICIENT.value] = _analyze_level(
            train_counter=train_coefficient_counter,
            rollout_counter=rollout_coefficient_counter,
            rollout_rows=coefficient_rows,
            level=UsageLevel.COEFFICIENT,
            ids_are_metric=True,
            output_dir=output_dir_path,
        )

    metrics_path = output_dir_path / METRICS_FILENAME
    metrics_path.write_text(json.dumps(report, indent=2))
    logging.info(f"Wrote token-usage report to {output_dir_path}")
    return metrics_path


def main() -> None:
    """Run the token-usage report from the command line."""
    override_log_format()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--train-counts-dir", required=True)
    parser.add_argument("--rollout-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    generate_report(
        checkpoint_path=arguments.checkpoint_path,
        train_counts_dir=arguments.train_counts_dir,
        rollout_jsonl=arguments.rollout_jsonl,
        output_dir=arguments.output_dir,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
