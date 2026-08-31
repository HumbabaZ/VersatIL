"""Plot the Tip 2 prediction-error decomposition from a collect_eval manifest.

Two columns, one per tokenizer family (FAST scale, binning num_bins). The top
row is the log-log prediction-error decomposition, three lines each: the total
error and its two arms (reconstruction, learning); a right-hand RMSE scale
accompanies the left MSE scale. The bottom row is closed-loop rollout success
(linear 0-1) on the same log x-axis, column-aligned so the reader can see
whether the error optimum lands where rollout succeeds. ``--space`` picks the
per-step action error or the integrated position-path error, ``--generation``
picks the stochastic (deployment) or argmax generation. Two marks keep the top
row honest: a horizontal line at the error of a "stand still" prediction, which
a collapsed tokenizer decodes to; and a vertical dashed line at any grid point
whose tokenizer collapsed (the reconstruction line still passes through it,
since that collapse is a genuine reconstruction reading). Runs of adjacent
grid points with zero rollout success are shaded as one band, explained in the
caption. Both x-axes are labelled at the actual grid values, coarse to fine.

    python -m versatil.analysis.tip2_tokenization.plot_decomposition \
        /data/horse/ws/qizh093f-versatil/tip2_results/tip2_eval_conditional_pilot.csv \
        --space position --generation argmax
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

matplotlib.use("Agg")

OUTPUT_DIR = Path("/home/qizh093f/versatil-github/outputs")
# unique_gt_sequence_count at or below this means the tokenizer mapped the whole
# validation set to (nearly) one sequence: the point is degenerate.
DEGENERATE_UNIQUE_COUNT = 2

TAB10 = matplotlib.colormaps["tab10"].colors
# (label, color, marker, linewidth) per arm, in draw order.
ARM_STYLES = (
    ("term1", "Reconstruction error", TAB10[0], "o", 1.4),
    ("term2", "Learning error", TAB10[1], "s", 1.4),
    ("total", "Total error", TAB10[3], "D", 2.6),
)
# Manifest column prefix and stand-still column per (space, generation).
COLUMN_LAYOUT = {
    ("action", "stochastic"): ("", "expert_mean_square"),
    ("action", "argmax"): ("argmax_", "expert_mean_square"),
    ("position", "stochastic"): ("position_stochastic_", "position_expert_mean_square"),
    ("position", "argmax"): ("position_argmax_", "position_expert_mean_square"),
}
# The action-space stochastic decomposition predates the prefixing scheme, and
# its argmax counterpart only prefixes the arms (the total is argmax_total_mse
# from compare_generation_modes); both resolve to the same column names here.
Y_LABELS = {
    "action": "MSE (denormalized action space, per step)",
    "position": "MSE (integrated position path, unit square)",
}
FAMILIES = (
    ("fast", "FAST rounding scale (coarse → fine)", "FAST"),
    ("binning", "Number of bins (coarse → fine)", "Binning"),
)


def series_columns(space: str, generation: str) -> tuple[list[tuple], str]:
    """Return the (column, label, color, marker, width) series and the
    stand-still column for one (space, generation) choice."""
    prefix, stand_still = COLUMN_LAYOUT[(space, generation)]
    series = [
        (f"{prefix}{arm}_mse", label, color, marker, width)
        for arm, label, color, marker, width in ARM_STYLES
    ]
    return series, stand_still


def failing_runs(failed: list[bool]) -> list[tuple[int, int]]:
    """Return (first, last) index pairs of each run of consecutive failures."""
    runs: list[tuple[int, int]] = []
    start = None
    for index, is_failed in enumerate(failed):
        if is_failed and start is None:
            start = index
        if not is_failed and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(failed) - 1))
    return runs


def load_family(csv_path: Path, method: str) -> list[dict[str, str]]:
    """Return the manifest rows of one family, sorted coarse to fine."""
    with open(csv_path, newline="") as manifest:
        rows = [row for row in csv.DictReader(manifest) if row["method"] == method]
    return sorted(rows, key=lambda row: float(row["param"]))


def is_degenerate(row: dict[str, str]) -> bool:
    """Whether the tokenizer collapsed at this grid point."""
    return int(float(row["unique_gt_sequence_count"])) <= DEGENERATE_UNIQUE_COUNT


def format_tick(value: float) -> str:
    """Format a grid value as a compact tick label."""
    if value >= 10.0 or value == round(value):
        return f"{int(round(value))}"
    return f"{value:g}"


def apply_grid_ticks(axis: plt.Axes, params: list[float]) -> None:
    """Label the x-axis at the actual grid values, no decade minor ticks."""
    axis.set_xticks(params)
    axis.set_xticklabels([format_tick(param) for param in params])
    axis.xaxis.set_minor_locator(NullLocator())


def add_rmse_axis(axis: plt.Axes) -> None:
    """Add a right-hand RMSE scale to a log MSE axis (RMSE = sqrt(MSE))."""
    secondary = axis.secondary_yaxis(
        "right", functions=(lambda mse: np.sqrt(mse), lambda rmse: rmse**2)
    )
    secondary.set_ylabel("RMSE (position path)")


def plot_rollout(axis: plt.Axes, rows: list[dict[str, str]], xlabel: str) -> None:
    """Draw closed-loop rollout success against the same granularity axis.

    Shares the column's log x-axis with the decomposition panel above it, so a
    reader can see at a glance whether the prediction-error optimum lands where
    rollout succeeds.

    Args:
        axis: Target axes (bottom row).
        rows: Manifest rows of one family, coarse to fine.
        xlabel: Granularity axis label.
    """
    params = [float(row["param"]) for row in rows]
    success = [float(row["rollout_success"]) for row in rows]
    axis.plot(
        params, success, marker="o", color="0.2", linewidth=1.6, markersize=5, zorder=3
    )
    axis.set_xscale("log")
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel(xlabel)
    axis.grid(True, alpha=0.3)
    apply_grid_ticks(axis=axis, params=params)


def plot_family(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    title: str,
    series: list[tuple],
    stand_still_column: str,
    break_at_degenerate: bool,
) -> None:
    """Draw every decomposition series of one family (top row).

    Args:
        axis: Target axes.
        rows: Manifest rows of one family, coarse to fine.
        title: Panel title.
        series: (column, label, color, marker, width) per line.
        stand_still_column: Column holding the stand-still error level.
        break_at_degenerate: Leave degenerate points out of the term2/total
            lines (the Tip 3 floor-figure convention) instead of connecting
            through them.
    """
    params = [float(row["param"]) for row in rows]
    degenerate = [is_degenerate(row) for row in rows]

    if "rollout_success" in rows[0]:
        failed = [float(row["rollout_success"]) == 0.0 for row in rows]
        # Shade runs of adjacent failing grid points as one band; both ends of a
        # family can fail (collapse at the coarse end, sampling explosion at the
        # fine end), so bands are per run, not min-to-max. The caption names them.
        for start, end in failing_runs(failed=failed):
            axis.axvspan(params[start] / 1.4, params[end] * 1.4, color="0.9", zorder=0)

    for column, label, color, marker, width in series:
        if column not in rows[0]:
            raise KeyError(f"Manifest has no column {column}; rerun collect_eval.")
        values = [float(row[column]) for row in rows]
        keep_degenerate = not break_at_degenerate or column.endswith("term1_mse")
        # An exact zero (learning error at a collapsed tokenizer) has no log-axis
        # value, so the learning line breaks there while reconstruction connects.
        connected = [
            (param, value)
            for param, value, bad in zip(params, values, degenerate, strict=True)
            if value > 0.0 and (keep_degenerate or not bad)
        ]
        axis.plot(
            [point[0] for point in connected],
            [point[1] for point in connected],
            marker=marker,
            color=color,
            linestyle="-",
            linewidth=width,
            markersize=6,
            label=label,
            zorder=3,
        )

    # Mark each collapsed grid point with a vertical line rather than a hollow
    # marker: at collapse the reconstruction and total points coincide, so
    # overlapping hollow markers are illegible.
    collapse_labelled = False
    for param, bad in zip(params, degenerate, strict=True):
        if not bad:
            continue
        axis.axvline(
            param,
            color="0.35",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
            label=None if collapse_labelled else "Tokenizer collapsed",
        )
        collapse_labelled = True

    if stand_still_column in rows[0]:
        axis.axhline(
            float(rows[0][stand_still_column]),
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="Stand-still baseline",
            zorder=2,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.grid(True, alpha=0.3)


def plot_manifest(
    csv_path: Path,
    output_path: Path,
    space: str,
    generation: str,
    break_at_degenerate: bool,
) -> None:
    """Render the two-panel decomposition figure for one manifest.

    Args:
        csv_path: collect_eval manifest.
        output_path: Destination PNG.
        space: ``"action"`` (per-step error) or ``"position"`` (integrated path).
        generation: ``"stochastic"`` (deployment sampling) or ``"argmax"``.
        break_at_degenerate: Leave collapsed grid points out of the lines.
    """
    series, stand_still_column = series_columns(space=space, generation=generation)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 6.6),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [3, 1]},
    )
    for column, (method, xlabel, title) in enumerate(FAMILIES):
        rows = load_family(csv_path=csv_path, method=method)
        if not rows:
            continue
        plot_family(
            axis=axes[0][column],
            rows=rows,
            title=title,
            series=series,
            stand_still_column=stand_still_column,
            break_at_degenerate=break_at_degenerate,
        )
        axes[0][column].tick_params(labelbottom=False)
        plot_rollout(axis=axes[1][column], rows=rows, xlabel=xlabel)

    axes[0][0].set_ylabel(Y_LABELS[space])
    axes[1][0].set_ylabel("Rollout success")
    axes[0][0].legend(fontsize=8, loc="best", framealpha=0.9)
    add_rmse_axis(axis=axes[0][1])
    figure.suptitle("Prediction error vs. tokenization granularity", fontsize=13)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Plot a Tip 2 eval manifest.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--space", default="action", choices=("action", "position"))
    parser.add_argument(
        "--generation", default="stochastic", choices=("stochastic", "argmax")
    )
    parser.add_argument("--break-at-degenerate", action="store_true")
    args = parser.parse_args()

    output_path = (
        args.out_dir / f"{args.csv_path.stem}_{args.space}_{args.generation}.png"
    )
    plot_manifest(
        csv_path=args.csv_path,
        output_path=output_path,
        space=args.space,
        generation=args.generation,
        break_at_degenerate=args.break_at_degenerate,
    )
    print(output_path)


if __name__ == "__main__":
    _main()
