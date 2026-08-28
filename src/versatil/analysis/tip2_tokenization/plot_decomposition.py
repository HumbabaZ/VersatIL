"""Plot the Tip 2 prediction-error decomposition from a collect_eval manifest.

One panel per tokenizer family (FAST scale, binning num_bins), log-log, coarse
to fine, three lines each: the total and its two arms (term1 reconstruction,
term2 learning). ``--space`` picks the per-step action error or the integrated
position-path error, ``--generation`` picks the stochastic (deployment) or
argmax generation. Two rules keep the figure honest: a horizontal line marks
the error of a "stand still" prediction, which a collapsed tokenizer decodes
to; and grid points whose ground-truth token sequences collapsed are drawn
hollow and left out of the connecting line.

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

matplotlib.use("Agg")

OUTPUT_DIR = Path("/home/qizh093f/versatil-github/outputs")
# unique_gt_sequence_count at or below this means the tokenizer mapped the whole
# validation set to (nearly) one sequence: the point is degenerate.
DEGENERATE_UNIQUE_COUNT = 2

TAB10 = matplotlib.colormaps["tab10"].colors
# (label, color, marker, linewidth, break_at_degenerate) per arm, in draw order.
ARM_STYLES = (
    ("term1", "term1 reconstruction", TAB10[0], "o", 1.4, False),
    ("term2", "term2 learning", TAB10[1], "s", 1.4, True),
    ("total", "total", TAB10[3], "D", 2.6, True),
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
    ("binning", "binning num_bins (coarse → fine)", "binning"),
)


def series_columns(space: str, generation: str) -> tuple[list[tuple], str]:
    """Return the (column, label, color, marker, width, break) series and the
    stand-still column for one (space, generation) choice."""
    prefix, stand_still = COLUMN_LAYOUT[(space, generation)]
    series = [
        (
            f"{prefix}{arm}_mse",
            f"{label} ({generation})" if arm == "total" else label,
            color,
            marker,
            width,
            break_line,
        )
        for arm, label, color, marker, width, break_line in ARM_STYLES
    ]
    return series, stand_still


def load_family(csv_path: Path, method: str) -> list[dict[str, str]]:
    """Return the manifest rows of one family, sorted coarse to fine."""
    with open(csv_path, newline="") as manifest:
        rows = [row for row in csv.DictReader(manifest) if row["method"] == method]
    return sorted(rows, key=lambda row: float(row["param"]))


def is_degenerate(row: dict[str, str]) -> bool:
    """Whether the tokenizer collapsed at this grid point."""
    return int(float(row["unique_gt_sequence_count"])) <= DEGENERATE_UNIQUE_COUNT


def plot_family(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    xlabel: str,
    title: str,
    series: list[tuple],
    stand_still_column: str,
) -> None:
    """Draw every series of one family, breaking lines at degenerate points."""
    params = [float(row["param"]) for row in rows]
    degenerate = [is_degenerate(row) for row in rows]

    if "rollout_success" in rows[0]:
        failed = [
            param
            for param, row in zip(params, rows, strict=True)
            if float(row["rollout_success"]) == 0.0
        ]
        # Shade each failing point on its own: both ends of a family can fail
        # (collapse at the coarse end, sampling explosion at the fine end), so a
        # single min-to-max band would hide the healthy middle.
        for index, param in enumerate(failed):
            axis.axvspan(
                param / 1.4,
                param * 1.4,
                color="0.9",
                zorder=0,
                label="task fails (rollout success = 0)" if index == 0 else None,
            )

    for column, label, color, marker, width, break_line in series:
        if column not in rows[0]:
            raise KeyError(f"Manifest has no column {column}; rerun collect_eval.")
        linestyle = "-"
        values = [float(row[column]) for row in rows]
        connected = [
            (param, value)
            for param, value, bad in zip(params, values, degenerate, strict=True)
            if not (break_line and bad)
        ]
        axis.plot(
            [point[0] for point in connected],
            [point[1] for point in connected],
            marker=marker,
            color=color,
            linestyle=linestyle,
            linewidth=width,
            markersize=6,
            label=label,
            zorder=3,
        )
        hollow = [
            (param, value)
            for param, value, bad in zip(params, values, degenerate, strict=True)
            if bad
        ]
        if hollow:
            axis.plot(
                [point[0] for point in hollow],
                [point[1] for point in hollow],
                marker=marker,
                color=color,
                markerfacecolor="none",
                linestyle="none",
                markersize=8,
                zorder=4,
            )

    if stand_still_column in rows[0]:
        axis.axhline(
            float(rows[0][stand_still_column]),
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="stand-still prediction",
            zorder=2,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(xlabel)
    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.grid(True, alpha=0.3)


def plot_manifest(
    csv_path: Path, output_path: Path, space: str, generation: str
) -> None:
    """Render the two-panel decomposition figure for one manifest.

    Args:
        csv_path: collect_eval manifest.
        output_path: Destination PNG.
        space: ``"action"`` (per-step error) or ``"position"`` (integrated path).
        generation: ``"stochastic"`` (deployment sampling) or ``"argmax"``.
    """
    series, stand_still_column = series_columns(space=space, generation=generation)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for axis, (method, xlabel, title) in zip(axes, FAMILIES, strict=True):
        rows = load_family(csv_path=csv_path, method=method)
        if rows:
            plot_family(
                axis=axis,
                rows=rows,
                xlabel=xlabel,
                title=title,
                series=series,
                stand_still_column=stand_still_column,
            )
    axes[0].set_ylabel(Y_LABELS[space])
    axes[0].legend(fontsize=8, loc="best", framealpha=0.9)
    figure.suptitle(
        f"Tip 2 · {csv_path.stem} · {space} space, {generation} generation "
        "(hollow = collapsed tokenizer, excluded from the line)",
        fontsize=11,
    )
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
    args = parser.parse_args()

    output_path = (
        args.out_dir / f"{args.csv_path.stem}_{args.space}_{args.generation}.png"
    )
    plot_manifest(
        csv_path=args.csv_path,
        output_path=output_path,
        space=args.space,
        generation=args.generation,
    )
    print(output_path)


if __name__ == "__main__":
    _main()
