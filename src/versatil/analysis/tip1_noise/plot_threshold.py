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


def _gaussian_levels(stage: str) -> tuple:
    """The four Gaussian noise levels of a sigma-swept stage."""
    return tuple(
        (stage, sigma, rf"$\times {sigma:g}$") for sigma in (1.0, 2.0, 3.0, 4.0)
    )


# Panels in reading order, hysteresis first: it is the condition the chapter is
# built on, and the two Gaussian axes are what license reading it. A panel may
# span more than one stage -- the two hysteresis levels are one condition at two
# backlash thresholds, not two conditions, so they share an axis and a scale
# rather than sitting in separate panels that invite a cross-panel comparison.
# Levels are labelled by what they multiply: a backlash threshold for
# hysteresis, a noise standard deviation for the Gaussian conditions.
PANELS = (
    (
        "Cable hysteresis",
        (
            ("conditional_hysteresis_s0", 4.0, r"$\times 4$"),
            ("conditional_hysteresis_fast_win_s0", 10.0, r"$\times 10$"),
        ),
    ),
    ("Tremor", _gaussian_levels("tremor_conditional_s0")),
    ("Measurement noise", _gaussian_levels("final_conditional_s0")),
)


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


def _present(errors: dict, stage: str, sigma: float) -> bool:
    """Whether any arm was scanned for this level."""
    return any((stage, arm, sigma, 60) in errors for arm in ARM_ORDER)


def _panel_levels(errors: dict) -> list[tuple[str, list[tuple[str, float, str]]]]:
    """Panels with the levels actually present, dropping anything unscanned."""
    panels = []
    for title, levels in PANELS:
        found = [level for level in levels if _present(errors, level[0], level[1])]
        if found:
            panels.append((title, found))
    return panels


def _panel_cells(errors: dict) -> list[tuple[str, str, float, str]]:
    """Every (panel, level) pair flattened, for the one-level-per-panel figure."""
    return [
        (stage, title, sigma, label)
        for title, levels in _panel_levels(errors)
        for stage, sigma, label in levels
    ]


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

    for axis, (stage, title, _sigma, label) in zip(axes, panels, strict=False):
        axis.axvspan(*WINDOW, color="0.88", zorder=0)
        axis.axvline(TAU, color=REFERENCE, linestyle="--", linewidth=1.1, zorder=1)
        for arm in ARM_ORDER:
            values = errors.get((stage, arm, _sigma, 60))
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
        axis.set_title(f"{title}, {label}", fontsize=11)
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
    """C2: per-rollout endpoint errors, showing point masses against spread.

    One axis per condition, hysteresis first. Its two backlash levels share that
    axis rather than splitting into separate panels: they are one condition
    measured twice, and separating them would invite reading the calibrated
    level as an independent result.
    """
    per_panel = _panel_levels(errors)
    if not per_panel:
        raise ValueError("no cells found in the scan directory")

    widths = [len(levels) for _, levels in per_panel]
    fig, axes = plt.subplots(
        1,
        len(per_panel),
        figsize=(1.05 * sum(widths) + 2.0, 4.0),
        sharey=True,
        gridspec_kw={"width_ratios": widths},
    )
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)

    for axis, (title, levels) in zip(axes, per_panel, strict=True):
        ticks, tick_labels = [], []
        position = 0
        for stage, sigma, label in levels:
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
                    color="0.15",
                    linewidth=1.8,
                    zorder=4,
                )
            ticks.append(position - (len(ARM_ORDER) - 1) / 2)
            tick_labels.append(label)
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
    handles.append(plt.Line2D([], [], color="0.15", linewidth=1.8))
    handles.append(plt.Line2D([], [], color=REFERENCE, linestyle="--", linewidth=1.1))
    fig.legend(
        handles,
        [ARM_LABEL[a] for a in ARM_ORDER] + ["median", rf"$\tau = {TAU}$"],
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "c2_error_spread.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def print_caption(errors: dict) -> None:
    """Caption text for C2, printed rather than drawn.

    Two things a reader needs and the panel cannot carry at print size: that the
    x10 backlash level is a calibrated point rather than another sample of the
    condition, and that the flat bands are a property of greedy decoding rather
    than an unusually consistent policy.
    """
    print(
        "\nC2 caption: Loop-closure error of every rollout, by condition and "
        "level. Cable hysteresis is shown at two backlash thresholds on one "
        "axis. The x10 level is a calibrated point: the threshold and the FAST "
        "rounding scale were chosen analytically before training so that a "
        "policy reproducing the biased labels exactly would fall outside the "
        "success radius while FAST's coarse reconstruction stayed inside it. "
        "It demonstrates a predicted mechanism at one setting and is not an "
        "unbiased sample of the condition; the x4 level is. Bars are medians, "
        "the dashed line the tolerance tau = "
        f"{TAU}. The greedily-decoded arms produce one trajectory per context, "
        "so their points fall in flat bands and a single replicate carries no "
        "internal spread; error bars come from between replicates."
    )
    for title, levels in _panel_levels(errors):
        for stage, sigma, label in levels:
            parts = []
            for arm in ARM_ORDER:
                values = errors.get((stage, arm, sigma, 60))
                if values:
                    parts.append(f"{arm} {statistics.median(values):.4f}")
            plain = label.replace("$", "").replace("\\", "")
            print(f"  {title:18s} {plain:12s} median: {'  '.join(parts)}")


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
    print_caption(errors)


if __name__ == "__main__":
    main()
