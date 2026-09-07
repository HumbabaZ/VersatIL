"""LIBERO rollout success vs training epoch (spatial suite).

Model selection on LIBERO cannot use validation loss -- it decouples from
closed-loop success. This figure is the rollout-vs-epoch curve that replaces
val-loss selection: success rate on libero_spatial (10 tasks x 10 trials) at
the checkpoint_every=5 snapshots, per matched arm.

Data is passed inline (a handful of numbers collected from the eval logs);
regenerate with more epochs as training produces them.

    python src/versatil/analysis/tip4_speed/plot_libero_rollout.py <out_dir>
"""

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

# libero_spatial success (%), per arm and epoch snapshot. None = not yet run.
# Collected from logs/libero_eval_<jid>.log "Final success rate".
EPOCHS = (9, 29, 49, 69, 89)
SERIES = (
    # (arm, label, color, marker, linestyle, success-by-epoch)
    ("fast", "AR + FAST (argmax)", "#2878B5", "o", "-", (84.0, 85.0, 95.0, 94.0, 91.0)),
    ("binned", "AR + Binning (sample t=0.5)", "#C82423", "s", "-", (27.0, 57.0, 73.0, 64.0, 59.0)),
    ("qfat", "Q-FAT (cont.)", "#EAA558", "^", "--", (76.0, 79.0, 80.0, 81.0, 80.0)),
    ("bcat", "BC transformer (cont.)", "#9AC9DB", "D", "--", (83.0, 92.0, 95.0, 94.0, 95.0)),
)
GRID_STYLE = {"linestyle": "--", "linewidth": 0.5, "alpha": 0.4}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: plot_libero_rollout.py <output_dir>")
    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for _arm, label, color, marker, linestyle, values in SERIES:
        points = [(ep, v) for ep, v in zip(EPOCHS, values, strict=True) if v is not None]
        xs = [ep for ep, _ in points]
        ys = [v for _, v in points]
        axis.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=1.8,
            linestyle=linestyle,
            label=label,
        )
    axis.set_xlabel("Training Epoch (snapshot)")
    axis.set_ylabel("Rollout Success Rate (%)")
    axis.set_xticks(list(EPOCHS))
    axis.set_ylim(-3, 100)
    axis.set_title("LIBERO-Spatial Rollout Success vs Epoch (10 tasks x 10 trials)")
    axis.grid(**GRID_STYLE)
    axis.set_axisbelow(True)
    axis.legend(loc="center right", frameon=True, fontsize=9)
    figure.text(
        0.5,
        -0.03,
        "Matched-backbone arms, single seed. Each arm uses its native/best "
        "decoding: fast=argmax, binning=sampling t=0.5 (argmax collapses to 1%), "
        "Q-FAT/BC deterministic. Selection is by rollout, not validation loss "
        "(which plateaued by epoch 33 for every arm).",
        ha="center",
        fontsize=8,
        color="0.35",
    )
    output_path = output_dir / "figure_libero_rollout_vs_epoch.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
