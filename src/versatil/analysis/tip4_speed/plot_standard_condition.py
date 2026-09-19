"""Success under the standard condition, across the three benchmarks.

The headline performance figure of E.2.2: one bar group per benchmark
(synthetic unimodal, synthetic multimodal, LIBERO four-suite), one bar per
arm. Synthetic bars carry n=3 seed error bars; LIBERO is single-seed and the
multimodal bcat cell is single-seed, both stated in the caption. The bcat
collapse on the multimodal task (mode averaging of a unimodal regressor) is
annotated directly on its empty bar.

Numbers are inlined from the collected results (results_all.csv at sigma=1 for
the synthetic groups, the libero_all evaluation at the selected snapshot for
LIBERO); regenerate after any re-collection.

    python src/versatil/analysis/tip4_speed/plot_standard_condition.py <out_dir>
"""

import sys
from importlib.util import find_spec
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if find_spec("scienceplots") is not None:
    import scienceplots  # noqa: F401,E402

    plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 300})

# (arm, label, color) in draw order, matching the other Tip 4 figures.
ARMS = (
    ("fast", "AR + FAST", "#2878B5"),
    ("binned", "AR + Binning", "#C82423"),
    ("qfat", "Q-FAT (cont.)", "#EAA558"),
    ("bcat", "BC transformer (cont.)", "#9AC9DB"),
)
# Success in percent and its seed standard deviation (None = no error bar).
# Synthetic groups: sigma=1, n=3 (bcat multimodal is n=1). LIBERO: libero_all
# at the selected snapshot, single seed, each arm in its best decoding mode.
GROUPS = (
    (
        "Synthetic\nunimodal",
        {"fast": (94.7, 0.6), "binned": (81.3, 3.8), "qfat": (100.0, 0.0), "bcat": (100.0, 0.0)},
    ),
    (
        "Synthetic\nmultimodal",
        {"fast": (92.0, 2.0), "binned": (60.3, 5.5), "qfat": (100.0, 0.0), "bcat": (0.0, None)},
    ),
    (
        "LIBERO\nfour-suite",
        {"fast": (93.8, None), "binned": (61.8, None), "qfat": (71.2, None), "bcat": (93.5, None)},
    ),
)
GRID_STYLE = {"linestyle": "--", "linewidth": 0.5, "alpha": 0.4}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: plot_standard_condition.py <output_dir>")
    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.6, 3.9))
    group_positions = np.arange(len(GROUPS)) * 1.0
    bar_width = 0.19

    for arm_index, (arm, label, color) in enumerate(ARMS):
        offsets = group_positions + (arm_index - 1.5) * bar_width
        values = [group[1][arm][0] for group in GROUPS]
        errors = [group[1][arm][1] for group in GROUPS]
        yerr = [e if e is not None else 0.0 for e in errors]
        axis.bar(
            offsets,
            values,
            width=bar_width * 0.92,
            color=color,
            yerr=yerr,
            capsize=2.5,
            error_kw={"linewidth": 1.0},
            label=label,
        )
        for x, v, e in zip(offsets, values, errors, strict=True):
            if v > 0:
                axis.text(
                    x, v + (e or 0.0) + 2.5, f"{v:.0f}", ha="center", fontsize=8
                )

    # The empty multimodal bcat bar gets its explanation on the figure itself.
    bcat_x = group_positions[1] + (3 - 1.5) * bar_width
    axis.annotate(
        "0\n(mode\naveraging)",
        xy=(bcat_x, 0),
        xytext=(bcat_x + 0.06, 16),
        fontsize=7.5,
        ha="center",
        color="0.25",
        arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.8},
    )

    axis.set_xticks(group_positions)
    axis.set_xticklabels([group[0] for group in GROUPS])
    axis.set_ylabel("Rollout Success Rate (%)")
    axis.set_ylim(0, 124)
    axis.set_yticks([0, 20, 40, 60, 80, 100])
    axis.set_title("Task Success under the Standard Condition")
    axis.grid(axis="y", **GRID_STYLE)
    axis.set_axisbelow(True)
    axis.legend(loc="upper center", frameon=True, fontsize=8, ncols=4)
    output_path = output_dir / "figure_success_standard_condition.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
