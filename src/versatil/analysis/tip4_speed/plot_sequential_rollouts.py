"""The four-arm sequential-task rollout panel of E.2.2 (figure_sequential_rollouts).

This figure shows what the multimodal collapse looks like: on the sequential
task two equally good routes pass the central obstacle on opposite sides, and
the direct regressor commands their average, which runs through the obstacle.

The panels are the training-time rollout-callback visualizations (fifty rollouts
overlaid at the final training snapshot, drawn by
``versatil.training.callbacks.synthetic_rollout``), one per arm, pulled from the
wandb run directory inside each arm's sequential checkpoint at the standard
condition::

    <ckpt_root>/synthetic/<config>/sequential__inj-position__band-high__sig-1__dseed-42__<arm>__seed-0/
        wandb/run-*/files/media/images/synthetic/rollout_trajectories_<step>_*.png

The highest-step file in each is the fully trained policy. Those four PNGs are
kept under ``outputs/tip4_speed/sequential_sources/{fast,binned,qfat,bcat}.png``
so the panel can be recomposed without a GPU. All four are 840x886 with
identical axes, so they tile without reprojection; this script crops the baked
title strip, whites out each panel's internal legend, and draws one shared
legend. Red marks each rollout's endpoint (``final_positions`` in the callback),
so where an arm's rollouts coincide --- BCAT by determinism, Q-FAT by committing
to one mode --- they appear as a single curve.

    python -m versatil.analysis.tip4_speed.plot_sequential_rollouts \
        outputs/tip4_speed/sequential_sources outputs/tip4_speed/figure_sequential_rollouts.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

# (source stem, panel label) in the chapter's arm order.
PANELS = (
    ("fast", "AR + FAST"),
    ("binned", "AR + Binning"),
    ("qfat", "Q-FAT (continuous)"),
    ("bcat", "BC transformer (continuous)"),
)
TITLE_STRIP = 60                      # top pixels holding the source title
LEGEND_BOX = (556, 712, 840, 886)    # (left, top, right, bottom) internal legend

TRAJ_COLOR = "#4C89C4"
GOAL_COLOR = "#8DD08A"
OBST_COLOR = "#BEBEBE"
ENDPOINT_COLOR = "#C0392B"


def _load_panel(source_dir: Path, stem: str) -> Image.Image:
    path = source_dir / f"{stem}.png"
    image = Image.open(path).convert("RGB")
    image = image.crop((0, TITLE_STRIP, image.width, image.height))
    left, top, right, bottom = LEGEND_BOX
    white = Image.new("RGB", (right - left, bottom - top), "white")
    image.paste(white, (left, top - TITLE_STRIP))
    return image


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: plot_sequential_rollouts.py <source_dir> <out_png>")
    source_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    figure, axes = plt.subplots(2, 2, figsize=(8.2, 8.8))
    for axis, (stem, label) in zip(axes.flat, PANELS, strict=True):
        axis.imshow(_load_panel(source_dir, stem))
        axis.set_title(label, fontsize=12, pad=6)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    handles = [
        Line2D([], [], color=TRAJ_COLOR, linewidth=2.0, label="Rollout trajectory (50 per arm)"),
        Line2D([], [], color=ENDPOINT_COLOR, marker="o", linestyle="", label="Rollout endpoint"),
        Patch(facecolor=GOAL_COLOR, edgecolor="none", label="Goal"),
        Patch(facecolor=OBST_COLOR, edgecolor="none", label="Obstacle"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
