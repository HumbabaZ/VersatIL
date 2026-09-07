"""Figures for the Tip 4 study: latency decomposition and rollout performance.

Two figures. The latency decomposition is the headline: where one action
chunk's inference time goes, per arm, with the measured sequential depth
annotated. The success figure is a plain performance comparison of the same
four arms (final-epoch rollout success from the Tip 1 sweep, n = 3 seeds) --
deliberately uncoupled from any latency axis.

Style follows the project's visualization config: the ``scienceplots`` toolkit
and the qualitative ``tab10`` palette assigned in fixed order, Title-Case axis
labels with units, and a light dashed grid.

Measurement notes surfaced in the figures:

- Bar segments are colored by *segment*, not by arm; arms are identified by
  the y-axis labels, so the legend is unambiguous.
- The bcat arm's latency was measured on a randomly initialized policy (its
  training sweep ran with checkpoint saving disabled); a single parallel
  forward has no sampling, no EOS and no data-dependent branching, so its
  timing is weight-independent. Its success comes from the trained Tip 1
  rollouts.
- The continuous arms have no detokenization; their recorded ``t_detok_ms``
  (~0.01 ms) is the segment boundary's own synchronize-and-read-clock cost
  and is invisible at this scale.

Runnable directly from the summary CSV (no versatil import, no re-measurement):

    python src/versatil/analysis/tip4_speed/plot_speed.py summary_speed_cuda.csv out_dir
"""

import csv
import sys
from importlib.util import find_spec
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if find_spec("scienceplots") is not None:
    import scienceplots  # noqa: F401,E402

    plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 300})

TAB10 = matplotlib.colormaps["tab10"].colors
# (arm, label, color, marker) in draw order: tokenized arms first, then continuous.
ARM_STYLES = (
    ("fast", "AR + FAST", TAB10[0], "o"),
    ("binned", "AR + Binning", TAB10[1], "s"),
    ("qfat", "Q-FAT (continuous)", TAB10[2], "^"),
    ("bcat", "BC transformer (continuous)", TAB10[3], "D"),
)
ARM_ORDER = tuple(arm for arm, _, _, _ in ARM_STYLES)
ARM_LABEL = {arm: label for arm, label, _, _ in ARM_STYLES}
ARM_COLOR = {arm: color for arm, _, color, _ in ARM_STYLES}
ARM_MARKER = {arm: marker for arm, _, _, marker in ARM_STYLES}
# Latency segments of one predict_action call, colored by segment so the
# legend applies to every bar identically.
SEGMENT_STYLES = (
    ("t_encode_ms", "Observation Encoding (shared)", "0.78"),
    ("t_generate_ms", "Autoregressive Generation", TAB10[0]),
    ("t_detok_ms", "Detokenization", "0.25"),
)
RANDOM_WEIGHT_ARM = "bcat"
DEFAULT_HORIZON = 60
ANCHOR_SIGMA = 1.0
GRID_STYLE = {"linestyle": "--", "linewidth": 0.5, "alpha": 0.4}


def read_summary(path: Path) -> list[dict[str, str]]:
    """Read the aggregated summary CSV into dict rows."""
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def select(
    rows: list[dict[str, str]],
    arm: str,
    horizon: int | None = None,
    sigma: float | None = None,
) -> list[dict[str, str]]:
    """Rows for one arm, optionally pinned to a horizon and a noise level."""
    selected = [row for row in rows if row["method"] == arm]
    if horizon is not None:
        selected = [row for row in selected if int(row["trajectory_length"]) == horizon]
    if sigma is not None:
        selected = [row for row in selected if float(row["sigma_multiplier"]) == sigma]
    return sorted(
        selected,
        key=lambda row: (
            int(row["trajectory_length"]),
            float(row["sigma_multiplier"]),
        ),
    )


def plot_decomposition(rows: list[dict[str, str]], output_path: Path) -> None:
    """Headline figure: where a chunk's latency goes, per arm.

    Horizontal stacked bars at the anchor noise level and the default horizon,
    with total latency and the measured sequential depth annotated per bar.
    The shared observation encoder is a fixed cost across arms, so the bars
    show directly how much of end-to-end latency the action representation
    actually governs.
    """
    figure, axis = plt.subplots(figsize=(7.4, 3.4))
    positions = range(len(ARM_ORDER))
    for position, arm in zip(positions, ARM_ORDER, strict=True):
        row = select(rows, arm=arm, horizon=DEFAULT_HORIZON, sigma=ANCHOR_SIGMA)[0]
        left = 0.0
        for column, label, color in SEGMENT_STYLES:
            width = float(row[column])
            axis.barh(
                position,
                width,
                left=left,
                height=0.6,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                label=label if position == 0 else None,
            )
            left += width
        depth = float(row["steps_mean"])
        axis.text(
            left + 6,
            position,
            f"{left:.0f} ms  (L = {depth:.0f})",
            va="center",
            fontsize=9,
        )
    axis.set_yticks(list(positions))
    axis.set_yticklabels([ARM_LABEL[arm] for arm in ARM_ORDER])
    axis.invert_yaxis()
    axis.set_xlabel("Latency per Action Chunk (ms)")
    axis.set_xlim(0, 300)
    axis.set_title(
        f"Latency Decomposition (H100, batch 1, sigma = {ANCHOR_SIGMA:.0f}x, "
        f"T = {DEFAULT_HORIZON})"
    )
    axis.grid(axis="x", **GRID_STYLE)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=True, fontsize=9)
    figure.text(
        0.5,
        -0.06,
        "L: measured sequential decoder steps. Continuous arms have no "
        "detokenization (recorded ~0.01 ms is the timing boundary itself). "
        f"{ARM_LABEL[RANDOM_WEIGHT_ARM]}: latency from randomly initialized "
        "weights (single forward, timing is weight-independent).",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_success(rows: list[dict[str, str]], output_path: Path) -> None:
    """Plain performance comparison: rollout success per arm and noise level.

    Final-epoch conditional success from the Tip 1 sweep (mean over 3 seed
    replicates, error bars are the standard deviation). No latency axis: this
    panel answers only which representation performs better.
    """
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    # Small horizontal dodge so exactly-overlapping series (both continuous
    # arms sit at 1.00 everywhere) stay individually visible.
    dodge = dict(zip(ARM_ORDER, (-0.06, -0.02, 0.02, 0.06), strict=True))
    for arm in ARM_ORDER:
        selected = [
            row
            for row in select(rows, arm=arm, horizon=DEFAULT_HORIZON)
            if row["success_mean"]
        ]
        sigmas = [float(row["sigma_multiplier"]) + dodge[arm] for row in selected]
        successes = [float(row["success_mean"]) for row in selected]
        errors = [
            float(row["success_sd"]) if row["success_sd"] else 0.0 for row in selected
        ]
        axis.errorbar(
            sigmas,
            successes,
            yerr=errors,
            color=ARM_COLOR[arm],
            marker=ARM_MARKER[arm],
            markersize=5.5,
            linewidth=1.6,
            capsize=2.5,
            label=ARM_LABEL[arm],
        )
    axis.set_xlabel("Noise Level (multiples of task default)")
    axis.set_ylabel("Rollout Success Rate")
    axis.set_xticks([1, 2, 3, 4])
    axis.set_ylim(-0.05, 1.08)
    axis.set_title(
        f"Rollout Success (conditional circle, T = {DEFAULT_HORIZON}, n = 3 seeds)"
    )
    axis.grid(**GRID_STYLE)
    axis.set_axisbelow(True)
    axis.legend(loc="lower left", frameon=True, fontsize=8.5)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: plot_speed.py <summary_speed_*.csv> <output_dir>")
    rows = read_summary(Path(sys.argv[1]))
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_decomposition(rows, output_dir / "figure_latency_decomposition.png")
    plot_success(rows, output_dir / "figure_rollout_success.png")
    print(f"Wrote 2 figures to {output_dir}")


if __name__ == "__main__":
    main()
