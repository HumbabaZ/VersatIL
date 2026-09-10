"""What each Tip 1 corruption does to the demonstrations, before any training.

Every panel draws one demonstration three ways: the clean path it was generated
from, the observed path the policy sees at rollout time, and the path its action
labels integrate to. Which of the three left the clean path is the whole point.
Where observed and label paths coincide, the corruption is in the world and the
demonstration is self-consistent; where they separate, the observations are
clean and only the supervision is wrong.

Two figures, because the conditions play different roles and inviting a
diagonal comparison between them would confuse the two variables the experiment
separates -- the injection point and the structure of the error.

- ``plot_measurement_conditions`` is the main figure for the cable-hysteresis
  section. Both columns inject at the action labels, so they share an injection
  point and differ only in the structure of the error: a structured
  play-operator lag on the left, its unstructured Gaussian control on the right.
  That pairing is the causal license for the hysteresis result, so it is the one
  the two panels are meant to be read across.
- ``plot_tremor`` is the demonstrator-side stochastic condition, a separate
  estimand rather than a control for hysteresis: the trajectory itself shakes,
  so observation and label move together and away from the clean path. It stands
  in its own small figure, not beside the label-side conditions.

Rows are the two contexts of the conditional circle, coloured as the rollout
figures colour them: mode 0 blue, mode 1 red. Under hysteresis the two modes lag
in mirrored directions, which is the visible form of "a structured bias does not
average away over repeated demonstrations".

No policy, no checkpoint, no GPU: this reads the generator, not a run.

    python -m versatil.analysis.tip1_noise.plot_conditions <out_dir>
"""

import os
import sys
from importlib.util import find_spec

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from versatil.data.synthetic.constants import (  # noqa: E402
    NoiseInjection,
    SyntheticNoiseModel,
    SyntheticTaskName,
)
from versatil.data.synthetic.generators import generate_task_episodes  # noqa: E402
from versatil.data.synthetic.task_layout import get_task_layout  # noqa: E402
from versatil.data.synthetic.visualization import (  # noqa: E402
    PLOT_MODE_COLORS,
    PLOT_OBSTACLE_COLOR,
)

if find_spec("scienceplots") is not None:
    import scienceplots  # noqa: F401,E402

    plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 300})

TASK = SyntheticTaskName.CONDITIONAL_CIRCLE.value
NUM_MODES = 2
TRAJECTORY_LENGTH = 60
SEED = 42
TASK_DEFAULT_NOISE_STD = 0.008

CLEAN_COLOR = "0.78"
OBSERVED_COLOR = "0.15"
START_COLOR = "#2E8B57"
# The label path carries the ground-truth mode, coloured as the rollout figures
# colour it, so the two contexts read as the two circles they trace.
MODE_COLOR = {mode: PLOT_MODE_COLORS[mode] for mode in range(NUM_MODES)}

# A modest illustrative level: the Gaussian conditions at 2x the task default
# make the label departure plainly visible while the circle stays recognisable,
# which a larger level does not. The hysteresis threshold is the calibrated
# 0.080 that produces a legible lag; both are stated in the captions.
GAUSSIAN_STD = 2.0 * TASK_DEFAULT_NOISE_STD
HYSTERESIS_THRESHOLD = 0.080

STRUCTURED = (
    "Structured: cable hysteresis (action)",
    HYSTERESIS_THRESHOLD,
    NoiseInjection.ACTION.value,
    SyntheticNoiseModel.CABLE_HYSTERESIS.value,
)
UNSTRUCTURED = (
    "Unstructured: Gaussian (action)",
    GAUSSIAN_STD,
    NoiseInjection.ACTION.value,
    SyntheticNoiseModel.GAUSSIAN.value,
)
TREMOR = (
    "Tremor: Gaussian (position)",
    GAUSSIAN_STD,
    NoiseInjection.POSITION.value,
    SyntheticNoiseModel.GAUSSIAN.value,
)


def label_path(episode: dict) -> np.ndarray:
    """Integrate the recorded action labels from the demonstration's start.

    This is the trajectory the policy is supervised towards, which equals the
    observed path only when the actions are differences of the observed
    positions. Under label-side injection it is not, and the gap between the two
    is what these panels show.
    """
    start = episode["position"][0]
    # The final action is a zero sentinel rather than a command, so integrating
    # every action would append a spurious stationary step.
    steps = episode["action"][:-1]
    return np.concatenate([[start], start + np.cumsum(steps, axis=0)])


def condition_episodes(noise_std: float, injection: str, noise_model: str) -> list:
    """One demonstration per mode under a single condition.

    Rendering is skipped: only the positions and action labels are read here,
    and rendering every frame dominates generation time.
    """
    return generate_task_episodes(
        task_name=TASK,
        num_episodes=NUM_MODES,
        seed=SEED,
        image_size=64,
        num_modes=NUM_MODES,
        trajectory_length=TRAJECTORY_LENGTH,
        noise_std=noise_std,
        noise_injection=injection,
        noise_model=noise_model,
        render_images=False,
    )


def frame_limits(episode_sets: list, margin: float = 0.04) -> tuple[float, float]:
    """A single square frame covering every path drawn, so excursions compare."""
    span = np.concatenate(
        [
            np.concatenate([episode["position"], label_path(episode)])
            for episodes in episode_sets
            for episode in episodes
        ]
    )
    return (float(span.min()) - margin, float(span.max()) + margin)


def draw_panel(
    axes,
    episode: dict,
    clean: np.ndarray,
    mode: int,
    limits: tuple[float, float],
) -> None:
    """Draw one demonstration against the clean path it was generated from."""
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
    observed = episode["position"]
    labels = label_path(episode)
    color = MODE_COLOR[mode]
    axes.plot(clean[:, 0], clean[:, 1], color=CLEAN_COLOR, linewidth=2.6, zorder=2)
    axes.plot(
        observed[:, 0], observed[:, 1], color=OBSERVED_COLOR, linewidth=1.4, zorder=3
    )
    axes.plot(
        labels[:, 0], labels[:, 1], color=color, linewidth=1.7, linestyle="--", zorder=4
    )
    axes.scatter([clean[0, 0]], [clean[0, 1]], s=34, color=START_COLOR, zorder=6)
    axes.scatter(
        [labels[-1, 0]], [labels[-1, 1]], s=46, color=color, marker="X", zorder=6
    )

    axes.set_xlim(*limits)
    axes.set_ylim(*limits)
    axes.set_aspect("equal")
    axes.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    # Two numbers, because one cannot tell the conditions apart: how far the
    # world was corrupted, and how far the supervision left the world. Tremor is
    # precisely the case where the first is large and the second is zero. They
    # are returned for the caption rather than drawn: at print size an in-panel
    # annotation is illegible, and the caption is where a reader looks for it.
    world = float(np.linalg.norm(observed - clean, axis=-1).mean())
    supervision = float(np.linalg.norm(labels - observed, axis=-1).mean())
    return world, supervision


def _legend_handles() -> list:
    return [
        Line2D([], [], color=CLEAN_COLOR, linewidth=2.6, label="Clean path"),
        Line2D([], [], color=OBSERVED_COLOR, linewidth=1.4, label="Observed path"),
        Line2D(
            [],
            [],
            color="0.35",
            linewidth=1.7,
            linestyle="--",
            label="Path integrated from action labels (by mode)",
        ),
        Line2D(
            [],
            [],
            color=START_COLOR,
            marker="o",
            linestyle="",
            label="Start / clean endpoint",
        ),
        Line2D(
            [], [], color="0.35", marker="X", linestyle="", label="Label-path endpoint"
        ),
        Patch(facecolor=PLOT_OBSTACLE_COLOR, edgecolor="none", label="Obstacle"),
    ]


def _clean_episodes() -> list:
    """The noise-free demonstrations every corrupted path is derived from.

    Same task, seed and length, so the comparison is per demonstration rather
    than against an average.
    """
    return condition_episodes(
        0.0, NoiseInjection.POSITION.value, SyntheticNoiseModel.GAUSSIAN.value
    )


def plot_measurement_conditions(out_dir: str) -> str:
    """Main figure: structured versus unstructured error at one injection point.

    Structured (hysteresis) on the left as the subject, its unstructured
    Gaussian control on the right.
    """
    columns = (STRUCTURED, UNSTRUCTURED)
    clean = [episode["position"] for episode in _clean_episodes()]
    by_column = [
        condition_episodes(noise_std, injection, model)
        for _, noise_std, injection, model in columns
    ]
    limits = frame_limits(by_column)

    figure, axes_grid = plt.subplots(
        NUM_MODES, len(columns), figsize=(3.2 * len(columns), 3.2 * NUM_MODES + 0.5)
    )
    axes_grid = np.atleast_2d(axes_grid)
    print(
        "caption numbers (mean over steps): observed-vs-clean / labels-vs-observed"
        f"  [hysteresis threshold {HYSTERESIS_THRESHOLD:g}, "
        f"Gaussian sigma {GAUSSIAN_STD / TASK_DEFAULT_NOISE_STD:g}x default]"
    )
    for column, (title, *_rest) in enumerate(columns):
        for mode in range(NUM_MODES):
            axes = axes_grid[mode, column]
            world, supervision = draw_panel(
                axes, by_column[column][mode], clean[mode], mode, limits
            )
            print(f"  {title:38s} context {mode}: {world:.3f} / {supervision:.3f}")
            if mode == 0:
                axes.set_title(title, fontsize=11)
            if column == 0:
                axes.set_ylabel(f"Context {mode}\n\nNormalised $y$", fontsize=10)
        axes_grid[-1, column].set_xlabel("Normalised $x$", fontsize=10)

    # Reserve a band under the axes so the legend clears the x-axis labels.
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    path = os.path.join(out_dir, "b2_measurement_error_structure.png")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_tremor(out_dir: str) -> str:
    """Tremor figure: the demonstrator-side stochastic condition, on its own."""
    _, noise_std, injection, model = TREMOR
    episodes = condition_episodes(noise_std, injection, model)
    clean = [episode["position"] for episode in _clean_episodes()]
    limits = frame_limits([episodes])

    figure, axes_row = plt.subplots(1, NUM_MODES, figsize=(3.4 * NUM_MODES, 4.4))
    axes_row = np.atleast_1d(axes_row)
    print(
        "caption numbers (mean over steps): observed-vs-clean / labels-vs-observed"
        f"  [tremor, Gaussian sigma {GAUSSIAN_STD / TASK_DEFAULT_NOISE_STD:g}x default]"
    )
    for mode in range(NUM_MODES):
        world, supervision = draw_panel(
            axes_row[mode], episodes[mode], clean[mode], mode, limits
        )
        print(f"  context {mode}: {world:.3f} / {supervision:.3f}")
        axes_row[mode].set_title(f"Context {mode}", fontsize=11)
        axes_row[mode].set_xlabel("Normalised $x$", fontsize=10)
    axes_row[0].set_ylabel("Normalised $y$", fontsize=10)

    # Reserve a band under the axes so the two-row legend clears the x labels.
    figure.tight_layout(rect=(0, 0.2, 1, 1))
    figure.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    path = os.path.join(out_dir, "b2_tremor.png")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    """Write both figures to the directory given on the command line."""
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    print("wrote", plot_measurement_conditions(out_dir))
    print("wrote", plot_tremor(out_dir))


if __name__ == "__main__":
    main()
