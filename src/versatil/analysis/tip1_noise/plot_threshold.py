"""Methodological figures for the Tip 1 noise study: threshold and error spread.

Two figures, both defending how the primary metric is read rather than
reporting a result.

C1, threshold sensitivity. The primary metric is a thresholded success rate and
the tolerance is fixed at tau = 0.043, two mean per-step displacements on the
noise-free reference. Each panel sweeps the tolerance for one condition at one
noise level, so a panel carries one line per arm and the eye compares arms
rather than untangling a bundle. The admissible window is shaded in every
panel: below 0.032 the weakest still-working policy fails, above 0.044 the
continuous arm under hysteresis passes and the hysteresis result disappears.
The figure answers, in one look, whether the threshold was chosen to flatter
the outcome -- the window is narrow and both edges are pinned by cells the
choice cannot move.

C2, error spread. Endpoint errors per arm and noise level, drawn as individual
rollouts rather than a summary. Two things are visible that a bar chart hides:
where each arm sits relative to tau, and that the greedily-decoded arms collapse
to a point mass while the sampling continuous arm spreads. The second is why
error bars in this study come from between replicates and never from within one.

Palette follows the project visualization skill. The four arms form two families
of two, so the family colors carry that structure directly: the discrete arms
take the blue pair and the continuous arms the red pair, dark for the primary
member of each family and light for the subordinate one. The tolerance is drawn
as a grey dashed reference so it does not spend colour budget.

Both read the CSVs written by the offline threshold scan, so neither needs a
GPU, a checkpoint, or a versatil import:

    python src/versatil/analysis/tip1_noise/plot_threshold.py scan_dir out_dir

On terminology in the labels: the two modes of the context-conditioned circle
share an endpoint, so the quantity plotted is loop-closure error, not distance
to a target. Mode correctness is a separate conjunct of the success criterion
and is not shown here.
"""

import csv
import glob
import os
import statistics
import sys
from collections import defaultdict
from importlib.util import find_spec

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if find_spec("scienceplots") is not None:
    import scienceplots  # noqa: F401,E402

    plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 12, "figure.dpi": 150, "savefig.dpi": 300})

# Two families of two. Dark is the primary member, light the subordinate one:
# FAST is the discrete arm the thesis studies and binning the supporting second,
# Q-FAT is the designated continuous reference and the BC transformer the
# descriptive external baseline. Warm and cool alternate between families.
ARM_STYLES = (
    ("fast", "AR + FAST", "#2878B5", "o"),
    ("binned", "AR + Binning", "#9AC9DB", "s"),
    ("qfat", "Q-FAT (continuous)", "#C82423", "^"),
    ("bcat", "BC transformer (continuous)", "#FF8884", "D"),
)
ARM_ORDER = tuple(arm for arm, _, _, _ in ARM_STYLES)
ARM_LABEL = {a: label for a, label, _, _ in ARM_STYLES}
ARM_COLOR = {a: c for a, _, c, _ in ARM_STYLES}
ARM_MARKER = {a: m for a, _, _, m in ARM_STYLES}

REFERENCE = "0.25"
TAU = 0.043
# Both edges are pinned by specific cells: the lower by the weakest arm that
# still works, the upper by the cell carrying the hysteresis result.
WINDOW = (0.032, 0.044)

STAGES = {
    "tremor_conditional_s0": ("Tremor", (1.0, 2.0, 3.0, 4.0)),
    "final_conditional_s0": ("Measurement noise", (1.0, 2.0, 3.0, 4.0)),
    "conditional_hysteresis_s0": ("Hysteresis", (4.0,)),
    "conditional_hysteresis_fast_win_s0": ("Hysteresis, calibrated", (10.0,)),
}


def load_errors(scan_dir: str) -> dict:
    """Endpoint errors keyed by (stage, method, sigma, trajectory_length)."""
    errors = defaultdict(list)
    for path in glob.glob(os.path.join(scan_dir, "*_endpoint_errors.csv")):
        stage = os.path.basename(path).replace("_endpoint_errors.csv", "")
        with open(path) as handle:
            for row in csv.DictReader(handle):
                key = (
                    stage,
                    row["method"],
                    float(row["sigma"]),
                    int(row["trajectory_length"]),
                )
                errors[key].append(float(row["endpoint_error"]))
    return errors


def success_at(values: list[float], threshold: float) -> float:
    """Fraction of rollouts inside the tolerance.

    Equal to the conditional success rate on this grid: context accuracy is
    1.00 and the collision and path-length conjuncts never bind, so the
    endpoint tolerance is the only active criterion. Checked against
    evaluate_rollouts before this shortcut was adopted.
    """
    return sum(1 for value in values if value < threshold) / len(values)


def _panel_cells(errors: dict) -> list[tuple[str, str, float]]:
    """Panels to draw, in reading order: one condition at one noise level."""
    cells = []
    for stage, (title, sigmas) in STAGES.items():
        for sigma in sigmas:
            if any(k[0] == stage and k[2] == sigma and k[3] == 60 for k in errors):
                cells.append((stage, title, sigma))
    return cells


def plot_threshold_sensitivity(errors: dict, out_dir: str) -> str:
    """C1: success against tolerance, one panel per condition and noise level."""
    panels = _panel_cells(errors)
    if not panels:
        raise ValueError("no cells found in the scan directory")
    columns = 4
    rows = -(-len(panels) // columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(3.1 * columns, 2.7 * rows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    sweep = np.linspace(0.015, 0.15, 300)

    for axis, (stage, title, sigma) in zip(axes, panels, strict=False):
        axis.axvspan(*WINDOW, color="0.88", zorder=0)
        axis.axvline(TAU, color=REFERENCE, linestyle="--", linewidth=1.1, zorder=1)
        for arm in ARM_ORDER:
            values = errors.get((stage, arm, sigma, 60))
            if not values:
                continue
            axis.plot(
                sweep,
                [success_at(values, t) for t in sweep],
                color=ARM_COLOR[arm],
                linewidth=1.7,
                label=ARM_LABEL[arm],
                zorder=2,
            )
        axis.set_title(rf"{title}, $\sigma={sigma:g}$", fontsize=11)
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        axis.set_xlim(sweep[0], sweep[-1])
        axis.set_ylim(-0.05, 1.05)
    for axis in axes[len(panels) :]:
        axis.set_visible(False)
    for axis in axes[len(panels) - columns : len(panels)]:
        axis.set_xlabel("Loop-Closure Tolerance")
    for index in range(0, len(panels), columns):
        axes[index].set_ylabel("Conditional Success")

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=REFERENCE, linestyle="--", linewidth=1.1))
    labels.append(rf"$\tau = {TAU}$")
    handles.append(plt.Rectangle((0, 0), 1, 1, color="0.88"))
    labels.append("Discriminating window")
    fig.tight_layout()
    # The grid usually leaves a trailing gap; parking the legend there beats
    # hanging it under the axis labels, where it collides with them.
    spare = axes[len(panels) :]
    if len(spare) >= 2:
        box = spare[0].get_position()
        last = spare[-1].get_position()
        fig.legend(
            handles,
            labels,
            loc="center",
            ncol=1,
            frameon=False,
            bbox_to_anchor=(
                (box.x0 + last.x1) / 2,
                (box.y0 + box.y1) / 2,
            ),
        )
    else:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
    path = os.path.join(out_dir, "c1_threshold_sensitivity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_error_spread(errors: dict, out_dir: str) -> str:
    """C2: per-rollout endpoint errors, showing point masses against spread."""
    per_stage: list[tuple[str, str, list[float]]] = []
    for stage, (title, sigmas) in STAGES.items():
        present = [
            s for s in sigmas if any((stage, arm, s, 60) in errors for arm in ARM_ORDER)
        ]
        if present:
            per_stage.append((stage, title, present))
    if not per_stage:
        raise ValueError("no cells found in the scan directory")

    widths = [len(sigmas) for _, _, sigmas in per_stage]
    fig, axes = plt.subplots(
        1,
        len(per_stage),
        figsize=(1.05 * sum(widths) + 2.0, 4.0),
        sharey=True,
        gridspec_kw={"width_ratios": widths},
    )
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)

    for axis, (stage, title, sigmas) in zip(axes, per_stage, strict=True):
        ticks, tick_labels = [], []
        position = 0
        for sigma in sigmas:
            for arm in ARM_ORDER:
                position += 1
                values = errors.get((stage, arm, sigma, 60))
                if not values:
                    continue
                # Jitter horizontally only: identical values must still read as
                # one flat band, which is the point of the panel.
                jitter = rng.uniform(-0.24, 0.24, size=len(values))
                axis.scatter(
                    position + jitter,
                    values,
                    s=11,
                    color=ARM_COLOR[arm],
                    marker=ARM_MARKER[arm],
                    alpha=0.55,
                    linewidths=0,
                )
                axis.hlines(
                    statistics.median(values),
                    position - 0.34,
                    position + 0.34,
                    color=ARM_COLOR[arm],
                    linewidth=1.8,
                )
            ticks.append(position - (len(ARM_ORDER) - 1) / 2)
            tick_labels.append(rf"$\sigma={sigma:g}$")
            position += 1
        axis.axhline(TAU, color=REFERENCE, linestyle="--", linewidth=1.1)
        axis.set_xticks(ticks)
        axis.set_xticklabels(tick_labels)
        axis.set_title(title, fontsize=11)
        axis.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
        # Linear, not log: the transform tokenizer's error under hysteresis is
        # exactly zero, which is a finding rather than a floor, and a log axis
        # would either drop those points or bury the informative band under
        # decades of empty space below them.
        axis.set_xlim(0.3, position - 0.3)

    axes[0].set_ylabel("Loop-Closure Error")
    handles = [
        plt.Line2D(
            [],
            [],
            color=ARM_COLOR[a],
            marker=ARM_MARKER[a],
            linestyle="",
            label=ARM_LABEL[a],
        )
        for a in ARM_ORDER
    ]
    handles.append(plt.Line2D([], [], color=REFERENCE, linestyle="--", linewidth=1.1))
    fig.legend(
        handles,
        [ARM_LABEL[a] for a in ARM_ORDER] + [rf"$\tau = {TAU}$"],
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "c2_error_spread.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """Write both methodological figures from a scan directory."""
    scan_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else scan_dir
    os.makedirs(out_dir, exist_ok=True)
    errors = load_errors(scan_dir)
    if not errors:
        raise ValueError(f"no endpoint-error CSVs under {scan_dir}")
    print(f"loaded {len(errors)} cells")
    print("wrote", plot_threshold_sensitivity(errors, out_dir))
    print("wrote", plot_error_spread(errors, out_dir))


if __name__ == "__main__":
    main()
