"""Plot the Tip 2 prediction-error decomposition from a collect_eval manifest.

Two columns, one per tokenizer family (FAST scale, binning num_bins). The top
row is the log-log prediction-error decomposition in RMSE, three curves each:
the total error and its two arms (reconstruction, learning). The bottom row is
closed-loop rollout success (linear 0-1) on the same log x-axis,
column-aligned so the reader can see whether the error optimum lands where
rollout succeeds. With a multi-seed manifest every seed value is drawn as a
faint scatter point and the curve connects per-point centers: the geometric
mean for error curves (the arithmetic mean of a log-axis quantity is dominated
by a single diverged seed) and the arithmetic mean for the bounded success
rate. ``--space`` picks the per-step action error or the integrated
position-path error, ``--generation`` picks the stochastic (deployment) or
argmax generation.

Reference marks, labelled in place on the left panel (both panels share the
y-axis): a horizontal dotted line at the RMSE of a "stand still" prediction,
which a collapsed tokenizer decodes to; on the argmax figure a dash-dot line
at sqrt(2)*sigma, the irreducible floor of the position-path error under
sub-pixel (unobservable) demonstrator noise (the stochastic generation's floor
also carries a sampling random walk, so it is omitted there); and a shaded
band, aligned to midpoints between grid points, over runs of grid points whose
rollout success is zero. A collapsed tokenizer shows up as the learning curve
breaking there while the reconstruction curve passes through. Both x-axes are
labelled at the actual grid values, coarse to fine.

    python -m versatil.analysis.tip2_tokenization.plot_decomposition \
        /data/horse/ws/qizh093f-versatil/tip2_results/tip2_eval_conditional_main.csv \
        --space position --generation argmax
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator

matplotlib.use("Agg")

OUTPUT_DIR = Path("/home/qizh093f/versatil-github/outputs")
# unique_gt_sequence_count at or below this means the tokenizer mapped the whole
# validation set to (nearly) one sequence: the point is degenerate.
DEGENERATE_UNIQUE_COUNT = 2

# Cool/warm opposition keeps the data curves colour-blind friendly; every
# reference mark stays greyscale so the two layers never compete. Total sits
# under the learning curve over most of the grid, so learning is dashed and
# drawn on top of the thick total line. Per-seed scatter reuses each line's
# colour at lower alpha.
ARM_STYLES = (
    ("term1", "Reconstruction error", "#2878B5", "o", "-", 1.4, 3),
    ("total", "Total error", "#C82423", "D", "-", 2.6, 3),
    ("term2", "Learning error", "#b1a4ea", "s", "--", 1.6, 4),
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
    "action": "RMSE of per-step action (denormalized units)",
    "position": "RMSE of integrated position trajectory (unit-square coords)",
}
FAMILIES = (
    ("fast", "FAST rounding scale", "(a) FAST"),
    ("binning", "Number of bins", "(b) Binning"),
)
SCATTER_ALPHA = 0.45
SCATTER_SIZE = 4.0
FAILURE_BAND_COLOR = "0.9"
NOISE_FLOOR_COLOR = "0.35"
REFERENCE_LABEL_SIZE = 7.5


def series_columns(space: str, generation: str) -> tuple[list[tuple], str]:
    """Return the per-arm series definitions and the stand-still column.

    Each series is (column, label, color, marker, linestyle, width, zorder)
    for one (space, generation) choice.
    """
    prefix, stand_still = COLUMN_LAYOUT[(space, generation)]
    series = [
        (f"{prefix}{arm}_mse", label, color, marker, linestyle, width, zorder)
        for arm, label, color, marker, linestyle, width, zorder in ARM_STYLES
    ]
    return series, stand_still


def geometric_mean(values: list[float]) -> float | None:
    """Geometric mean of positive values; None when any value is non-positive.

    The center line of a log-axis curve must average in log space: with one
    diverged seed out of three, the arithmetic mean hugs the outlier while the
    geometric mean stays with the bulk. A non-positive value (the learning
    error of a collapsed tokenizer is exactly zero) has no log, so the point
    carries no center and is left out of the line.
    """
    if not values or any(value <= 0.0 for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def arithmetic_mean(values: list[float]) -> float:
    """Plain mean, for the bounded rollout success rate (which can be 0)."""
    return sum(values) / len(values)


def group_by_param(
    rows: list[dict[str, str]],
) -> list[tuple[float, list[dict[str, str]]]]:
    """Group manifest rows (one per seed) by granularity value, coarse to fine."""
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(float(row["param"]), []).append(row)
    return sorted(grouped.items())


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


def band_edges(params: list[float], start: int, end: int) -> tuple[float, float]:
    """Shading edges for a failing run: geometric midpoints to the neighbours.

    The outermost grid points extend to the panel margin instead, so the band
    stays aligned with the tick positions rather than an arbitrary factor.
    """
    left = (
        math.sqrt(params[start - 1] * params[start]) if start > 0 else params[0] / 1.6
    )
    right = (
        math.sqrt(params[end] * params[end + 1])
        if end < len(params) - 1
        else params[-1] * 1.6
    )
    return left, right


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


def label_granularity_axis(axis: plt.Axes, family_name: str) -> None:
    """Centre the family name under the axis with coarse/fine flanking the ends.

    Replaces an in-label arrow: "coarse" sits under the left (coarsest) end and
    "fine" under the right (finest) end, so the direction reads off the axis.
    """
    axis.text(0.5, -0.19, family_name, transform=axis.transAxes, ha="center", va="top")
    axis.text(
        0.0,
        -0.19,
        "coarse",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        style="italic",
        color="0.4",
    )
    axis.text(
        1.0,
        -0.19,
        "fine",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        style="italic",
        color="0.4",
    )


def annotate_reference_line(
    axis: plt.Axes, value: float, text: str, color: str
) -> None:
    """Label a horizontal reference line just below it, at the left edge."""
    axis.annotate(
        text,
        xy=(0.02, value),
        xycoords=axis.get_yaxis_transform(),
        xytext=(0, -5),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=REFERENCE_LABEL_SIZE,
        color=color,
    )


def draw_failure_band(
    axis: plt.Axes,
    params: list[float],
    failed: list[bool],
) -> None:
    """Shade runs of zero-success grid points on one axis.

    Drawn on both rows of a column so the panels read as one continuous story.
    """
    for start, end in failing_runs(failed=failed):
        left, right = band_edges(params=params, start=start, end=end)
        axis.axvspan(left, right, color=FAILURE_BAND_COLOR, zorder=0)


def scatter_and_center_line(
    axis: plt.Axes,
    points: list[tuple[float, list[float], float | None]],
    color: str | tuple,
    marker: str,
    linestyle: str,
    width: float,
    zorder: float,
) -> None:
    """Draw per-seed scatter plus the center line through per-param centers.

    Args:
        axis: Target axes.
        points: One (param, seed_values, center) per grid point; a None center
            leaves the point out of the line, and non-positive seed values are
            skipped (no position on a log axis).
        color: Shared colour for scatter and line.
        marker: Marker for the center line.
        linestyle: Linestyle for the center line.
        width: Center line width.
        zorder: Center line zorder; the scatter sits just below it.
    """
    scatter_x = [
        param
        for param, seed_values, _ in points
        for value in seed_values
        if value > 0.0
    ]
    scatter_y = [
        value for _, seed_values, _ in points for value in seed_values if value > 0.0
    ]
    if scatter_x:
        axis.plot(
            scatter_x,
            scatter_y,
            marker=marker,
            color=color,
            linestyle="none",
            markersize=SCATTER_SIZE,
            alpha=SCATTER_ALPHA,
            zorder=zorder - 0.5,
        )
    line = [(param, center) for param, _, center in points if center is not None]
    axis.plot(
        [point[0] for point in line],
        [point[1] for point in line],
        marker=marker,
        color=color,
        linestyle=linestyle,
        linewidth=width,
        markersize=6,
        zorder=zorder,
    )


def plot_rollout(
    axis: plt.Axes,
    groups: list[tuple[float, list[dict[str, str]]]],
    xlabel: str,
    failed: list[bool],
) -> None:
    """Draw closed-loop rollout success for one family (bottom row).

    Args:
        axis: Target axes.
        groups: Per-param seed rows, coarse to fine.
        xlabel: Granularity axis label.
        failed: Per-param zero-success flags, for the shared shading.
    """
    params = [param for param, _ in groups]
    points = [
        (
            param,
            [float(row["rollout_success"]) for row in seed_rows],
            arithmetic_mean(
                values=[float(row["rollout_success"]) for row in seed_rows]
            ),
        )
        for param, seed_rows in groups
    ]
    draw_failure_band(axis=axis, params=params, failed=failed)
    scatter_x = [param for param, seed_values, _ in points for _ in seed_values]
    scatter_y = [value for _, seed_values, _ in points for value in seed_values]
    axis.plot(
        scatter_x,
        scatter_y,
        marker="o",
        color="0.2",
        linestyle="none",
        markersize=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        zorder=2.5,
    )
    axis.plot(
        params,
        [center for _, _, center in points],
        marker="o",
        color="0.2",
        linewidth=1.6,
        markersize=5,
        zorder=3,
    )
    axis.set_xscale("log")
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlim(params[0] / 1.6, params[-1] * 1.6)
    axis.grid(True, alpha=0.3)
    apply_grid_ticks(axis=axis, params=params)
    label_granularity_axis(axis=axis, family_name=xlabel)


def plot_family(
    axis: plt.Axes,
    groups: list[tuple[float, list[dict[str, str]]]],
    title: str,
    series: list[tuple],
    stand_still_column: str,
    noise_std: float | None,
    failed: list[bool],
    degenerate: list[bool],
    break_at_degenerate: bool,
    show_reference_labels: bool,
) -> None:
    """Draw the RMSE decomposition of one family (top row).

    Args:
        axis: Target axes.
        groups: Per-param seed rows, coarse to fine.
        title: Panel title.
        series: Per-arm (column, label, color, marker, linestyle, width, zorder).
        stand_still_column: Column holding the stand-still MSE level.
        noise_std: Demonstrator noise sigma for the irreducible-floor line, or
            None to omit it (the stochastic figure, where sigma is not the floor).
        failed: Per-param zero-success flags.
        degenerate: Per-param collapse flags, used to break the learning/total
            lines at a collapsed point.
        break_at_degenerate: Leave collapsed points out of the learning/total
            lines (the Tip 3 floor-figure convention) instead of connecting
            through them.
        show_reference_labels: Annotate the horizontal reference lines in place
            (done in the left panel only, since both share the same y-axis).
    """
    params = [param for param, _ in groups]
    draw_failure_band(axis=axis, params=params, failed=failed)

    first_row = groups[0][1][0]
    for column, _, color, marker, linestyle, width, zorder in series:
        if column not in first_row:
            raise KeyError(f"Manifest has no column {column}; rerun collect_eval.")
        keep_degenerate = not break_at_degenerate or column.endswith("term1_mse")
        points = []
        for (param, seed_rows), bad in zip(groups, degenerate, strict=True):
            seed_values = [math.sqrt(float(row[column])) for row in seed_rows]
            center = geometric_mean(values=seed_values)
            if bad and not keep_degenerate:
                center = None
            points.append((param, seed_values, center))
        scatter_and_center_line(
            axis=axis,
            points=points,
            color=color,
            marker=marker,
            linestyle=linestyle,
            width=width,
            zorder=zorder,
        )

    standstill = math.sqrt(float(first_row[stand_still_column]))
    axis.axhline(standstill, color="black", linestyle=":", linewidth=1.2, zorder=2)
    if show_reference_labels:
        annotate_reference_line(
            axis=axis, value=standstill, text="stand-still", color="black"
        )
    if noise_std is not None:
        # Actions are differences of noisy positions, so the integrated path
        # error telescopes to eps[k] - eps[0] (variance 2 sigma^2); the
        # irreducible position-space floor is therefore sqrt(2)*sigma, not
        # sigma, since the sub-pixel noise is unobservable and cannot cancel.
        floor = math.sqrt(2.0) * noise_std
        axis.axhline(
            floor, color=NOISE_FLOOR_COLOR, linestyle="-.", linewidth=1.2, zorder=2
        )
        if show_reference_labels:
            annotate_reference_line(
                axis=axis, value=floor, text="noise floor", color=NOISE_FLOOR_COLOR
            )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(params[0] / 1.6, params[-1] * 1.6)
    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.grid(True, alpha=0.3)


def figure_legend_handles() -> list:
    """Proxy handles for the shared legend: the data curves and the failure band.

    The horizontal reference lines are labelled in place, so only the three
    error curves and the shaded band need a legend entry.
    """
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=width,
            markersize=6,
            label=label,
        )
        for _, label, color, marker, linestyle, width, _ in ARM_STYLES
    ]
    handles.append(Patch(color=FAILURE_BAND_COLOR, label="Rollout success = 0"))
    return handles


def plot_manifest(
    csv_path: Path,
    output_path: Path,
    space: str,
    generation: str,
    break_at_degenerate: bool,
) -> None:
    """Render the 2x2 decomposition-plus-rollout figure for one manifest.

    Args:
        csv_path: collect_eval manifest (one row per cell, seeds included).
        output_path: Destination PNG.
        space: ``"action"`` (per-step error) or ``"position"`` (integrated path).
        generation: ``"stochastic"`` (deployment sampling) or ``"argmax"``.
        break_at_degenerate: Leave collapsed grid points out of the lines.
    """
    series, stand_still_column = series_columns(space=space, generation=generation)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 6.9),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [3, 1]},
    )
    for column, (method, xlabel, title) in enumerate(FAMILIES):
        rows = load_family(csv_path=csv_path, method=method)
        if not rows:
            continue
        groups = group_by_param(rows=rows)
        # The tokenizer is fit on data shared across seeds, so collapse is a
        # per-param property; failure means every seed's rollout success is 0.
        degenerate = [is_degenerate(seed_rows[0]) for _, seed_rows in groups]
        failed = [
            arithmetic_mean(values=[float(row["rollout_success"]) for row in seed_rows])
            == 0.0
            for _, seed_rows in groups
        ]
        noise_std = (
            float(rows[0]["noise_std"])
            if generation == "argmax" and "noise_std" in rows[0]
            else None
        )
        plot_family(
            axis=axes[0][column],
            groups=groups,
            title=title,
            series=series,
            stand_still_column=stand_still_column,
            noise_std=noise_std,
            failed=failed,
            degenerate=degenerate,
            break_at_degenerate=break_at_degenerate,
            show_reference_labels=column == 0,
        )
        axes[0][column].tick_params(labelbottom=False)
        plot_rollout(
            axis=axes[1][column],
            groups=groups,
            xlabel=xlabel,
            failed=failed,
        )

    axes[0][0].set_ylabel(Y_LABELS[space])
    axes[1][0].set_ylabel("Rollout success rate")
    figure.legend(
        handles=figure_legend_handles(),
        loc="upper center",
        ncol=4,
        fontsize=8,
        framealpha=0.9,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
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
