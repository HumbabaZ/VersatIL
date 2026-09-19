"""Closed-loop rollouts of the four arms, drawn against the clean target path.

The quantitative tables say by how much each arm misses; this figure says what
the miss looks like. Under cable hysteresis the difference is a shape, not a
number: the transform tokenizer's rollout returns to the clean circle while the
other three carry the backlash bias inward, and no error column conveys that as
directly as the trajectory does.

Each panel overlays one arm's rollout on the clean path it should reproduce, for
one context of the conditional circle. Columns are the arms in the order used
throughout the chapter; rows are the two contexts, coloured as every other
figure colours them (mode 0 blue, mode 1 red), so the panels read as the two
circles the task traces.

Split into two entry points, because the rollouts need a GPU and the drawing
does not:

    # once, on a GPU node (see scripts/tip1_scan.sbatch for the environment)
    python -m versatil.analysis.tip1_noise.plot_rollouts collect <stage> <cache.npz> [max_epoch]
    # then, anywhere
    python -m versatil.analysis.tip1_noise.plot_rollouts render <cache.npz> <out_dir>
"""

# ruff: noqa: PLC0415 -- the torch/inference imports live inside collect() on
# purpose: the render path must stay importable on the login node, where
# pulling in that stack takes minutes.

import os
import sys
from importlib.util import find_spec

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

# The task layout and palette are light and both paths need them. Everything
# that pulls in torch and the inference stack -- checkpoint loading, rollouts,
# episode generation -- is imported inside collect(), so the render path stays
# usable on the oversubscribed login node where importing them takes minutes.
from versatil.data.synthetic.task_layout import get_task_layout  # noqa: E402
from versatil.data.synthetic.visualization import (  # noqa: E402
    PLOT_MODE_COLORS,
    PLOT_OBSTACLE_COLOR,
)

if find_spec("scienceplots") is not None:
    import scienceplots  # noqa: F401,E402

    plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 300})

# The conditional-circle task value, inlined so the render path needs no import
# from the synthetic constants module.
TASK = "conditional_circle"
NUM_MODES = 2
IMAGE_SIZE = 64
# A deterministic arm gives one trajectory per context, so a handful of rollouts
# confirms that without cluttering the panel; the sampling continuous arm shows
# its spread within the same budget.
NUM_ROLLOUTS = 5

# The chapter order and labels, so this figure lines up with the tables.
ARM_ORDER = ("fast", "binned", "qfat", "bcat")
ARM_LABEL = {
    "fast": "AR + FAST",
    "binned": "AR + Binning",
    "qfat": "Q-FAT (continuous)",
    "bcat": "BC transformer (continuous)",
}
# Colour by context, matching every other conditional-circle figure.
MODE_COLOR = {mode: PLOT_MODE_COLORS[mode] for mode in range(NUM_MODES)}
CLEAN_COLOR = "0.78"
START_COLOR = "#2E8B57"


def collect(stage: str, cache_path: str, max_epoch: int | None = None) -> str:
    """Roll out every arm of a stage and cache the trajectories to an npz.

    One array per (method, context) plus the clean target paths, so the drawing
    step needs neither a checkpoint nor a GPU. Every heavy import lives here, so
    importing this module for the render path pulls in none of the torch stack.
    """
    import torch

    from versatil.analysis.tip1_noise.scan_endpoints import ckpt_dir, final_ckpt
    from versatil.analysis.tip1_noise.sweep import stage_cells
    from versatil.checkpoint_loading.float_policy import FloatCheckpointLoader
    from versatil.data.synthetic.generators import generate_task_episodes
    from versatil.inference.synthetic_rollout import run_rollouts

    def clean_paths(trajectory_length: int) -> np.ndarray:
        episodes = generate_task_episodes(
            task_name=TASK,
            num_episodes=NUM_MODES,
            seed=42,
            image_size=IMAGE_SIZE,
            num_modes=NUM_MODES,
            trajectory_length=trajectory_length,
            noise_std=0.0,
            render_images=False,
        )
        return np.stack([episode["position"] for episode in episodes])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Keyed by trajectory length as well as method: the control-rate stage holds
    # the same four arms at three lengths, and keying by method alone would keep
    # only whichever length came last.
    stored: dict[str, np.ndarray] = {}
    lengths: set[int] = set()
    methods: set[str] = set()
    for cell in stage_cells(stage):
        directory = ckpt_dir(cell)
        checkpoint = final_ckpt(directory, max_epoch=max_epoch) if directory else None
        if checkpoint is None:
            print(f"  MISS {cell.method}: no checkpoint for {cell.name}")
            continue
        length = cell.data.trajectory_length
        loader = FloatCheckpointLoader(
            device=device,
            checkpoint_path=directory,
            checkpoint_name=os.path.basename(checkpoint),
        )
        policy = loader.policy
        policy.eval()
        with torch.no_grad():
            for mode in range(NUM_MODES):
                stored[f"{cell.method}__T{length}__ctx{mode}"] = run_rollouts(
                    policy=policy,
                    task_name=TASK,
                    num_rollouts=NUM_ROLLOUTS,
                    image_size=IMAGE_SIZE,
                    context_mode=mode,
                    temporal_aggregation=False,
                )
        lengths.add(length)
        methods.add(cell.method)
        print(f"  {cell.method} T={length}: rolled out {NUM_MODES} contexts")

    if not lengths:
        raise ValueError(f"stage '{stage}' had no trainable checkpoints to roll out")
    for length in lengths:
        stored[f"clean__T{length}"] = clean_paths(length)
    stored["arms"] = np.array([m for m in ARM_ORDER if m in methods], dtype=object)
    stored["lengths"] = np.array(sorted(lengths))
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, **stored, allow_pickle=True)
    print(f"wrote {cache_path}")
    return cache_path


def _draw_background(axes) -> None:
    layout = get_task_layout(task_name=TASK, num_modes=NUM_MODES)
    for x_min, y_min, x_max, y_max in layout.obstacles:
        axes.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor=PLOT_OBSTACLE_COLOR,
                edgecolor="none",
                zorder=1,
            )
        )


def _draw_mode(axes, rollouts: np.ndarray, clean: np.ndarray, mode: int) -> None:
    """One context: its clean target path, its rollouts, and their endpoints."""
    axes.plot(clean[:, 0], clean[:, 1], color=CLEAN_COLOR, linewidth=2.6, zorder=2)
    color = MODE_COLOR[mode]
    for trajectory in rollouts:
        axes.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            linewidth=1.5,
            alpha=0.85,
            zorder=4,
        )
        axes.scatter(
            [trajectory[-1, 0]],
            [trajectory[-1, 1]],
            s=42,
            color=color,
            marker="X",
            zorder=5,
        )
    axes.scatter([clean[0, 0]], [clean[0, 1]], s=34, color=START_COLOR, zorder=6)


def _finish(axes, limits: tuple[float, float]) -> None:
    axes.set_xlim(*limits)
    axes.set_ylim(*limits)
    axes.set_aspect("equal")
    axes.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)


def _frame_limits(data, keys) -> tuple[float, float]:
    span = np.concatenate([data[key].reshape(-1, 2) for key in keys])
    margin = 0.04
    return (float(span.min()) - margin, float(span.max()) + margin)


def _legend_handles() -> list:
    return [
        Line2D([], [], color=CLEAN_COLOR, linewidth=2.6, label="Clean target path"),
        Line2D([], [], color="0.35", linewidth=1.5, label="Rollout (by context)"),
        Line2D(
            [], [], color="0.35", marker="X", linestyle="", label="Rollout endpoint"
        ),
        Line2D([], [], color=START_COLOR, marker="o", linestyle="", label="Start"),
        Patch(facecolor=PLOT_OBSTACLE_COLOR, edgecolor="none", label="Obstacle"),
    ]


def _save(figure, out_dir: str, stem: str, prefix: str) -> str:
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    path = os.path.join(out_dir, f"{prefix}_{stem}.png")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")
    return path


def render(cache_path: str, out_dir: str) -> str:
    """Draw a rollout grid, laid out to match what the cache varies.

    A stage at a single trajectory length compares the arms under one
    condition, so contexts become rows and arms columns. The control-rate stage
    varies the length instead, and there the question is how each arm moves
    along that axis, so arms become rows, lengths columns, and both contexts
    share a panel.

    No title: at print size a suptitle is illegible and belongs in the caption.
    """
    os.makedirs(out_dir, exist_ok=True)
    data = np.load(cache_path, allow_pickle=True)
    arms = [str(method) for method in data["arms"]]
    lengths = [int(length) for length in data["lengths"]]
    stem = os.path.splitext(os.path.basename(cache_path))[0]
    trajectory_keys = [key for key in data.files if "__ctx" in key]
    limits = _frame_limits(data, trajectory_keys + [f"clean__T{t}" for t in lengths])

    if len(lengths) == 1:
        length = lengths[0]
        figure, axes_grid = plt.subplots(
            NUM_MODES,
            len(arms),
            figsize=(2.9 * len(arms), 2.9 * NUM_MODES + 0.6),
            sharex=True,
            sharey=True,
        )
        axes_grid = np.atleast_2d(axes_grid)
        clean = data[f"clean__T{length}"]
        for column, method in enumerate(arms):
            for mode in range(NUM_MODES):
                axes = axes_grid[mode, column]
                _draw_background(axes)
                _draw_mode(
                    axes, data[f"{method}__T{length}__ctx{mode}"], clean[mode], mode
                )
                _finish(axes, limits)
                if mode == 0:
                    axes.set_title(ARM_LABEL[method], fontsize=10.5)
                if column == 0:
                    axes.set_ylabel(f"Context {mode}\n\nNormalised $y$", fontsize=10)
            axes_grid[-1, column].set_xlabel("Normalised $x$", fontsize=10)
        return _save(figure, out_dir, stem, "b3_rollouts")

    figure, axes_grid = plt.subplots(
        len(arms),
        len(lengths),
        figsize=(2.7 * len(lengths), 2.7 * len(arms) + 0.6),
        sharex=True,
        sharey=True,
    )
    axes_grid = np.atleast_2d(axes_grid)
    for row, method in enumerate(arms):
        for column, length in enumerate(lengths):
            axes = axes_grid[row, column]
            _draw_background(axes)
            clean = data[f"clean__T{length}"]
            # Both contexts share the panel: the axis being compared is the
            # length, so the pair of circles belongs together in each cell.
            for mode in range(NUM_MODES):
                _draw_mode(
                    axes, data[f"{method}__T{length}__ctx{mode}"], clean[mode], mode
                )
            _finish(axes, limits)
            if row == 0:
                axes.set_title(f"$T = {length}$", fontsize=11)
            if column == 0:
                axes.set_ylabel(f"{ARM_LABEL[method]}\n\nNormalised $y$", fontsize=9.5)
    for axes in axes_grid[-1]:
        axes.set_xlabel("Normalised $x$", fontsize=10)
    return _save(figure, out_dir, stem, "b4_rate")


def main() -> None:
    """Dispatch to the collect (GPU) or render (anywhere) path."""
    action = sys.argv[1]
    if action == "collect":
        max_epoch = int(sys.argv[4]) if len(sys.argv) > 4 else None
        collect(sys.argv[2], sys.argv[3], max_epoch=max_epoch)
    elif action == "render":
        cache_path = sys.argv[2]
        out_dir = sys.argv[3] if len(sys.argv) > 3 else "."
        render(cache_path, out_dir)
    else:
        raise SystemExit("usage: plot_rollouts.py {collect|render} ...")


if __name__ == "__main__":
    main()
