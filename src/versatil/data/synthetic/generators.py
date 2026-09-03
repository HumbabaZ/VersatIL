"""Trajectory generators for synthetic multimodal benchmark tasks.

Each task produces episodes with controlled multimodality in [0, 1]x[0, 1]
Cartesian space. Actions are fixed delta positions: action[t] = position[t+1] - position[t].
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from versatil.data.synthetic.constants import (
    CIRCLE_CENTER_BOTTOM,
    CIRCLE_CENTER_TOP,
    CIRCLE_CONTEXT_COLORS,
    CIRCLE_DEFAULT_NUM_MODES,
    CIRCLE_OBSTACLES,
    CIRCLE_RADIUS,
    CORRIDOR_DEFAULT_NUM_STYLES,
    CORRIDOR_GOAL,
    CORRIDOR_START,
    CORRIDOR_WALL_X1,
    CORRIDOR_WALL_X2,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_EPISODES,
    DEFAULT_SEED,
    GAUSSIAN_KERNEL_TRUNCATE,
    MAX_TRAJECTORY_RETRIES,
    MULTIPATH_DEFAULT_NOISE_STD,
    MULTIPATH_DEFAULT_NUM_MODES,
    MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,
    RADIAL_CENTER,
    RADIAL_RADIUS,
    REJECTION_ATTEMPTS_WARN_THRESHOLD,
    SEQUENTIAL_ENDPOINT_Y,
    SEQUENTIAL_FIRST_BRANCH_X_DELTA,
    SEQUENTIAL_FORK_TRANSITION_OFFSET,
    SEQUENTIAL_FORK_Y_1,
    SEQUENTIAL_FORK_Y_2,
    SEQUENTIAL_NUM_COMPOUND_MODES,
    SEQUENTIAL_OBSTACLES,
    SEQUENTIAL_SECOND_BRANCH_X_DELTA,
    SEQUENTIAL_START,
    NoiseInjection,
    SyntheticNoiseModel,
    SyntheticTaskName,
)
from versatil.data.synthetic.renderer import render_episode

logger = logging.getLogger(__name__)


@dataclass
class _RejectionStats:
    """Accumulate collision-rejection attempts across generated episodes.

    Rejection sampling conditions injected noise on being collision-free.
    That truncation strengthens as noise grows, so a noise sweep needs this
    recorded per configuration: if attempts saturate at the high-noise end,
    those cells carry less noise than requested and are not interpretable.
    """

    attempts_per_episode: list[int] = field(default_factory=list)

    def record(self, attempts: int) -> None:
        """Record the attempt count that produced one accepted episode.

        Args:
            attempts: Number of samples drawn, including the accepted one.
        """
        self.attempts_per_episode.append(attempts)

    @property
    def mean_attempts(self) -> float:
        """Mean attempts per accepted episode (1.0 means no rejection)."""
        if not self.attempts_per_episode:
            return 0.0
        return float(np.mean(self.attempts_per_episode))

    @property
    def rejection_rate(self) -> float:
        """Fraction of drawn samples that were rejected."""
        total = int(np.sum(self.attempts_per_episode))
        if total == 0:
            return 0.0
        return 1.0 - len(self.attempts_per_episode) / total

    def log_summary(self, task_name: str, noise_std: float) -> None:
        """Log the rejection summary, warning when truncation is material.

        Args:
            task_name: Task the statistics were collected for.
            noise_std: Noise level the statistics were collected at.
        """
        if not self.attempts_per_episode:
            return
        message = (
            "synthetic rejection sampling: task=%s noise_std=%.6g "
            "mean_attempts=%.3f rejection_rate=%.4f max_attempts=%d"
        )
        args = (
            task_name,
            noise_std,
            self.mean_attempts,
            self.rejection_rate,
            max(self.attempts_per_episode),
        )
        if self.mean_attempts > REJECTION_ATTEMPTS_WARN_THRESHOLD:
            logger.warning(
                message + " -- injected noise is materially truncated; treat this "
                "configuration as not interpretable in a noise sweep",
                *args,
            )
        else:
            logger.info(message, *args)


def generate_task_episodes(
    task_name: str = SyntheticTaskName.CIRCLE.value,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    seed: int = DEFAULT_SEED,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_modes: int = MULTIPATH_DEFAULT_NUM_MODES,
    trajectory_length: int = MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,
    noise_std: float = MULTIPATH_DEFAULT_NOISE_STD,
    num_styles: int = CORRIDOR_DEFAULT_NUM_STYLES,
    mode_weights: list[float] | None = None,
    noise_smoothing_sigma: float = 0.0,
    noise_injection: str = NoiseInjection.POSITION.value,
    noise_model: str = SyntheticNoiseModel.GAUSSIAN.value,
) -> list[dict[str, np.ndarray]]:
    """Generate synthetic episodes for a given task.

    Args:
        task_name: SyntheticTaskName.value string identifying which
            multimodal navigation task to generate.
        num_episodes: Total number of episodes to generate, balanced
            equally across all behavioral modes.
        seed: Random seed for reproducible generation.
        image_size: Side length in pixels of the rendered top-down RGB
            images (square).
        num_modes: Number of distinct behavioral modes for tasks that
            accept a variable mode count (radial, corridor_navigation).
        trajectory_length: Number of timesteps per episode.
        noise_std: Gaussian standard deviation, or the play-operator backlash
            threshold when ``noise_model`` is cable hysteresis.
        num_styles: Number of sinusoidal trajectory styles per corridor
            (corridor_navigation task only).
        mode_weights: Relative weights per mode for imbalanced generation.
            None for uniform distribution across modes.
        noise_smoothing_sigma: Gaussian temporal smoothing width in
            timesteps applied to the injected position noise. 0 (default)
            keeps the i.i.d. noise whose first difference is high-frequency
            action noise; larger values move that action noise to lower
            frequencies while holding its power fixed, which supports
            controlled high-band vs low-band robustness comparisons. See
            :func:`_sample_position_noise`.
        noise_injection: ``NoiseInjection`` value choosing where the noise
            enters. ``position`` (default) perturbs the trajectory, so the
            noise also changes images, clamping and rejection sampling.
            ``action`` leaves positions and images clean and perturbs only the
            action labels, at the same action-noise power. See
            :func:`_sample_action_noise`.
        noise_model: ``SyntheticNoiseModel`` value selecting the error process.
            ``gaussian`` preserves the stochastic spectral model.
            ``cable_hysteresis`` applies a systematic play operator to a hidden
            kinematic measurement and requires action-label injection.

    Returns:
        List of episode dicts. Each dict contains:
            "image": rendered top-down RGB, shape (T, image_size, image_size, 3), uint8
            "position": Cartesian (x, y) states, shape (T, 2), float32
            "action": delta (dx, dy) commands, shape (T, 2), float32
            "mode_id": ground-truth mode label, shape (T, 1), uint8
            "context": conditioning context vector, shape (T, C), float32
    """
    random_generator = np.random.default_rng(seed)
    # Shuffling is decoupled from the noise-sampling stream and derived from
    # `seed` on its own, so the pre-shuffle episode order -- fixed by the mode
    # loop, independent of noise_std/band/injection -- ends up permuted the same
    # way regardless of how many random draws generation consumed. Coupling them
    # (shuffling with the same generator right after sampling) let low-band
    # position noise, which pads its draw with `2 * radius` extra samples,
    # desync a noisy store's episode order from its zero-noise reference: index i
    # in one store no longer matched index i in the other, so any per-episode
    # diff between them -- exactly what SNR measurement needs -- compared
    # unrelated trajectories.
    shuffle_generator = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    episodes = _generate_task_episodes_unshuffled(
        task_name=task_name,
        num_episodes=num_episodes,
        random_generator=random_generator,
        image_size=image_size,
        num_modes=num_modes,
        trajectory_length=trajectory_length,
        noise_std=noise_std,
        num_styles=num_styles,
        mode_weights=mode_weights,
        noise_smoothing_sigma=noise_smoothing_sigma,
        noise_injection=noise_injection,
        noise_model=noise_model,
    )
    shuffle_generator.shuffle(episodes)
    return episodes


def _generate_task_episodes_unshuffled(
    task_name: str,
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    num_modes: int,
    trajectory_length: int,
    noise_std: float,
    num_styles: int,
    mode_weights: list[float] | None,
    noise_smoothing_sigma: float,
    noise_injection: str,
    noise_model: str,
) -> list[dict[str, np.ndarray]]:
    """Dispatch to the task generator; episode order still reflects the mode loop.

    Raises:
        ValueError: If ``task_name`` is not a known ``SyntheticTaskName``.
    """
    match task_name:
        case SyntheticTaskName.CIRCLE.value:
            return _generate_circle(
                num_episodes=num_episodes,
                random_generator=random_generator,
                image_size=image_size,
                trajectory_length=trajectory_length,
                noise_std=noise_std,
                mode_weights=mode_weights,
                noise_smoothing_sigma=noise_smoothing_sigma,
                noise_injection=noise_injection,
                noise_model=noise_model,
            )
        case SyntheticTaskName.CONDITIONAL_CIRCLE.value:
            return _generate_conditional_circle(
                num_episodes=num_episodes,
                random_generator=random_generator,
                image_size=image_size,
                trajectory_length=trajectory_length,
                noise_std=noise_std,
                mode_weights=mode_weights,
                noise_smoothing_sigma=noise_smoothing_sigma,
                noise_injection=noise_injection,
                noise_model=noise_model,
            )
        case SyntheticTaskName.SEQUENTIAL_DECISION.value:
            return _generate_sequential_decision(
                num_episodes=num_episodes,
                random_generator=random_generator,
                image_size=image_size,
                trajectory_length=trajectory_length,
                noise_std=noise_std,
                mode_weights=mode_weights,
                noise_smoothing_sigma=noise_smoothing_sigma,
                noise_injection=noise_injection,
                noise_model=noise_model,
            )
        case SyntheticTaskName.RADIAL.value:
            return _generate_radial(
                num_episodes=num_episodes,
                random_generator=random_generator,
                image_size=image_size,
                num_modes=num_modes,
                trajectory_length=trajectory_length,
                noise_std=noise_std,
                mode_weights=mode_weights,
                noise_smoothing_sigma=noise_smoothing_sigma,
                noise_injection=noise_injection,
                noise_model=noise_model,
            )
        case SyntheticTaskName.CORRIDOR_NAVIGATION.value:
            return _generate_corridor_navigation(
                num_episodes=num_episodes,
                random_generator=random_generator,
                image_size=image_size,
                num_modes=num_modes,
                num_styles=num_styles,
                trajectory_length=trajectory_length,
                noise_std=noise_std,
                mode_weights=mode_weights,
                noise_smoothing_sigma=noise_smoothing_sigma,
                noise_injection=noise_injection,
                noise_model=noise_model,
            )
        case _:
            raise ValueError(f"Unknown synthetic task: {task_name}")


def _generate_circle(
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    trajectory_length: int,
    noise_std: float,
    mode_weights: list[float] | None,
    noise_smoothing_sigma: float = 0.0,
    noise_injection: str = NoiseInjection.POSITION.value,
    noise_model: str = SyntheticNoiseModel.GAUSSIAN.value,
) -> list[dict[str, np.ndarray]]:
    """Traverse one of two tangent circles as a closed loop.

    Mode 0 = bottom circle, Mode 1 = top circle. The trajectory starts at
    the tangent point (0.5, 0.5) and traces a full clockwise loop around
    the selected circle, returning to the start.
    """
    return _generate_circle_episodes(
        num_episodes=num_episodes,
        random_generator=random_generator,
        image_size=image_size,
        trajectory_length=trajectory_length,
        noise_std=noise_std,
        mode_weights=mode_weights,
        noise_smoothing_sigma=noise_smoothing_sigma,
        noise_injection=noise_injection,
        noise_model=noise_model,
        use_context=False,
    )


def _generate_conditional_circle(
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    trajectory_length: int,
    noise_std: float,
    mode_weights: list[float] | None,
    noise_smoothing_sigma: float = 0.0,
    noise_injection: str = NoiseInjection.POSITION.value,
    noise_model: str = SyntheticNoiseModel.GAUSSIAN.value,
) -> list[dict[str, np.ndarray]]:
    """Same layout as circle but with a one-hot context signal per mode.

    Tests whether models exploit the context to resolve ambiguity.
    When conditioned on context, each mode becomes unimodal.
    """
    return _generate_circle_episodes(
        num_episodes=num_episodes,
        random_generator=random_generator,
        image_size=image_size,
        trajectory_length=trajectory_length,
        noise_std=noise_std,
        mode_weights=mode_weights,
        noise_smoothing_sigma=noise_smoothing_sigma,
        noise_injection=noise_injection,
        noise_model=noise_model,
        use_context=True,
    )


def _generate_circle_episodes(
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    trajectory_length: int,
    noise_std: float,
    mode_weights: list[float] | None,
    use_context: bool,
    noise_smoothing_sigma: float = 0.0,
    noise_injection: str = NoiseInjection.POSITION.value,
    noise_model: str = SyntheticNoiseModel.GAUSSIAN.value,
) -> list[dict[str, np.ndarray]]:
    """Shared implementation for circle and conditional_circle tasks.

    Args:
        num_episodes: Total episodes to generate.
        random_generator: NumPy random generator.
        image_size: Side length of rendered images.
        trajectory_length: Timesteps per episode.
        noise_std: Gaussian noise standard deviation.
        mode_weights: Per-mode weights or None for uniform.
        use_context: When True, set one-hot context and render context color.
    """
    num_modes = CIRCLE_DEFAULT_NUM_MODES
    episodes = []
    episodes_per_mode = _resolve_mode_counts(
        total_episodes=num_episodes,
        num_modes=num_modes,
        mode_weights=mode_weights,
    )
    centers = {0: CIRCLE_CENTER_BOTTOM, 1: CIRCLE_CENTER_TOP}

    for mode_index in range(num_modes):
        center = centers[mode_index]
        context_color = CIRCLE_CONTEXT_COLORS[mode_index] if use_context else None
        for _ in range(episodes_per_mode[mode_index]):
            positions = _parametric_circle(
                center=center,
                radius=CIRCLE_RADIUS,
                num_points=trajectory_length,
                clockwise=True,
            )
            positions, actions = _build_trajectory_signals(
                trajectory=positions,
                noise_std=noise_std,
                noise_smoothing_sigma=noise_smoothing_sigma,
                noise_injection=noise_injection,
                noise_model=noise_model,
                random_generator=random_generator,
            )
            images = render_episode(
                positions=positions,
                obstacles=CIRCLE_OBSTACLES,
                image_size=image_size,
                context_color=context_color,
            )
            if use_context:
                context_vector = np.zeros(num_modes, dtype=np.float32)
                context_vector[mode_index] = 1.0
                context = np.tile(context_vector, (trajectory_length, 1))
            else:
                context = np.zeros((trajectory_length, num_modes), dtype=np.float32)
            mode_label = np.full((trajectory_length, 1), mode_index, dtype=np.uint8)
            episodes.append(
                {
                    "image": images,
                    "position": positions,
                    "action": actions,
                    "mode_id": mode_label,
                    "context": context,
                }
            )
    return episodes


def _generate_sequential_decision(
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    trajectory_length: int,
    noise_std: float,
    mode_weights: list[float] | None,
    noise_smoothing_sigma: float = 0.0,
    noise_injection: str = NoiseInjection.POSITION.value,
    noise_model: str = SyntheticNoiseModel.GAUSSIAN.value,
) -> list[dict[str, np.ndarray]]:
    """Navigate upward from (0.5, 0) with two sequential left/right forks.

    First fork at y=0.4, second at y=0.7. Produces 4 compound modes
    (LL, LR, RL, RR) with obstacles at each fork point. Tests whether
    the model represents hierarchical sequential mode structure.
    """
    compound_modes = SEQUENTIAL_NUM_COMPOUND_MODES
    episodes = []
    episodes_per_mode = _resolve_mode_counts(
        total_episodes=num_episodes,
        num_modes=compound_modes,
        mode_weights=mode_weights,
    )
    mode_definitions = [
        ("left", "left"),
        ("left", "right"),
        ("right", "left"),
        ("right", "right"),
    ]
    start_x = float(SEQUENTIAL_START[0])
    start_y = float(SEQUENTIAL_START[1])

    for mode_index, (first_choice, second_choice) in enumerate(mode_definitions):
        first_x_delta = (
            -SEQUENTIAL_FIRST_BRANCH_X_DELTA
            if first_choice == "left"
            else SEQUENTIAL_FIRST_BRANCH_X_DELTA
        )
        second_x_delta = (
            -SEQUENTIAL_SECOND_BRANCH_X_DELTA
            if second_choice == "left"
            else SEQUENTIAL_SECOND_BRANCH_X_DELTA
        )
        waypoints = [
            (start_x, start_y),
            (start_x, SEQUENTIAL_FORK_Y_1),
            (
                start_x + first_x_delta,
                SEQUENTIAL_FORK_Y_1 + SEQUENTIAL_FORK_TRANSITION_OFFSET,
            ),
            (start_x + first_x_delta, SEQUENTIAL_FORK_Y_2),
            (
                start_x + first_x_delta + second_x_delta,
                SEQUENTIAL_FORK_Y_2 + SEQUENTIAL_FORK_TRANSITION_OFFSET,
            ),
            (start_x + first_x_delta + second_x_delta, SEQUENTIAL_ENDPOINT_Y),
        ]
        for _ in range(episodes_per_mode[mode_index]):
            positions = _interpolate_waypoints(
                waypoints=waypoints, num_points=trajectory_length
            )
            positions, actions = _build_trajectory_signals(
                trajectory=positions,
                noise_std=noise_std,
                noise_smoothing_sigma=noise_smoothing_sigma,
                noise_injection=noise_injection,
                noise_model=noise_model,
                random_generator=random_generator,
            )
            images = render_episode(
                positions=positions,
                obstacles=SEQUENTIAL_OBSTACLES,
                image_size=image_size,
            )
            context = np.zeros((trajectory_length, compound_modes), dtype=np.float32)
            mode_label = np.full((trajectory_length, 1), mode_index, dtype=np.uint8)
            episodes.append(
                {
                    "image": images,
                    "position": positions,
                    "action": actions,
                    "mode_id": mode_label,
                    "context": context,
                }
            )
    return episodes


def _generate_radial(
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    num_modes: int,
    trajectory_length: int,
    noise_std: float,
    mode_weights: list[float] | None,
    noise_smoothing_sigma: float = 0.0,
    noise_injection: str = NoiseInjection.POSITION.value,
    noise_model: str = SyntheticNoiseModel.GAUSSIAN.value,
) -> list[dict[str, np.ndarray]]:
    """K straight-line trajectories from center to K evenly-spaced points on a circle.

    Mode i travels to angle 2*pi*i/K at radius 0.4 from center.
    Obstacles are placed dynamically between each adjacent pair of radii.
    BC failure: mean action is zero displacement.
    """
    episodes = []
    episodes_per_mode = _resolve_mode_counts(
        total_episodes=num_episodes,
        num_modes=num_modes,
        mode_weights=mode_weights,
    )
    # The margin only needs to shrink when noise actually reaches positions.
    # Under action-only injection the trajectory stays exactly on its clean
    # path, so sizing obstacles off noise_std here would shrink the rendered
    # scene for a reason the trajectory never experiences -- the same "clean
    # observations" violation the action-noise path exists to avoid.
    obstacle_margin_noise_std = (
        noise_std if noise_injection == NoiseInjection.POSITION.value else 0.0
    )
    obstacles = _generate_radial_obstacles(
        num_modes=num_modes, noise_std=obstacle_margin_noise_std
    )
    rejection_stats = _RejectionStats()

    for mode_index in range(num_modes):
        angle = 2.0 * np.pi * mode_index / num_modes
        endpoint_x = float(RADIAL_CENTER[0]) + RADIAL_RADIUS * np.cos(angle)
        endpoint_y = float(RADIAL_CENTER[1]) + RADIAL_RADIUS * np.sin(angle)
        waypoints = [
            (float(RADIAL_CENTER[0]), float(RADIAL_CENTER[1])),
            (endpoint_x, endpoint_y),
        ]
        for _ in range(episodes_per_mode[mode_index]):
            base_positions = _interpolate_waypoints(
                waypoints=waypoints, num_points=trajectory_length
            )
            positions, actions = _build_trajectory_signals(
                trajectory=base_positions,
                noise_std=noise_std,
                noise_smoothing_sigma=noise_smoothing_sigma,
                noise_injection=noise_injection,
                noise_model=noise_model,
                random_generator=random_generator,
                obstacles=obstacles,
                rejection_stats=rejection_stats,
            )
            images = render_episode(
                positions=positions,
                obstacles=obstacles,
                image_size=image_size,
            )
            context = np.zeros((trajectory_length, num_modes), dtype=np.float32)
            mode_label = np.full((trajectory_length, 1), mode_index, dtype=np.uint8)
            episodes.append(
                {
                    "image": images,
                    "position": positions,
                    "action": actions,
                    "mode_id": mode_label,
                    "context": context,
                }
            )
    rejection_stats.log_summary(
        task_name=SyntheticTaskName.RADIAL.value, noise_std=noise_std
    )
    return episodes


def _generate_corridor_navigation(
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    num_modes: int,
    num_styles: int,
    trajectory_length: int,
    noise_std: float,
    mode_weights: list[float] | None,
    noise_smoothing_sigma: float = 0.0,
    noise_injection: str = NoiseInjection.POSITION.value,
    noise_model: str = SyntheticNoiseModel.GAUSSIAN.value,
) -> list[dict[str, np.ndarray]]:
    """Navigate through one of K gaps in a vertical wall, with S style variations.

    A vertical wall at x in [0.45, 0.55] has K gaps. Each gap defines
    a corridor mode. S sinusoidal style variations per corridor produce
    K*S total modes. Trajectory goes start -> gap center -> goal.

    K must be even so that no gap falls at y=0.5, ensuring the BC
    mean (which aims straight at y=0.5) always collides with the wall.
    """
    if num_modes % 2 != 0:
        raise ValueError(
            f"corridor_navigation requires even num_modes so no gap "
            f"falls at y=0.5 (BC must collide), got {num_modes}"
        )
    total_modes = num_modes * num_styles
    episodes = []
    episodes_per_mode = _resolve_mode_counts(
        total_episodes=num_episodes,
        num_modes=total_modes,
        mode_weights=mode_weights,
    )
    gap_centers = _compute_corridor_gap_centers(num_gaps=num_modes)
    obstacles = _generate_corridor_obstacles(gap_centers=gap_centers)
    rejection_stats = _RejectionStats()

    for gap_index in range(num_modes):
        gap_y = gap_centers[gap_index]
        for style_index in range(num_styles):
            flat_mode_index = gap_index * num_styles + style_index
            # Enter/exit wall x-range at gap_y so the full passage is horizontal
            waypoints = [
                (float(CORRIDOR_START[0]), float(CORRIDOR_START[1])),
                (float(CORRIDOR_WALL_X1), gap_y),
                (float(CORRIDOR_WALL_X2), gap_y),
                (float(CORRIDOR_GOAL[0]), float(CORRIDOR_GOAL[1])),
            ]
            for _ in range(episodes_per_mode[flat_mode_index]):
                base_positions = _interpolate_waypoints(
                    waypoints=waypoints, num_points=trajectory_length
                )
                if num_styles > 1:
                    base_positions = _apply_sinusoidal_style(
                        positions=base_positions,
                        style_index=style_index,
                        num_styles=num_styles,
                        gap_height=_compute_corridor_gap_height(num_gaps=num_modes),
                    )
                positions, actions = _build_trajectory_signals(
                    trajectory=base_positions,
                    noise_std=noise_std,
                    noise_smoothing_sigma=noise_smoothing_sigma,
                    noise_injection=noise_injection,
                    noise_model=noise_model,
                    random_generator=random_generator,
                    obstacles=obstacles,
                    rejection_stats=rejection_stats,
                )
                images = render_episode(
                    positions=positions,
                    obstacles=obstacles,
                    image_size=image_size,
                )
                context = np.zeros((trajectory_length, total_modes), dtype=np.float32)
                mode_label = np.full(
                    (trajectory_length, 1), flat_mode_index, dtype=np.uint8
                )
                episodes.append(
                    {
                        "image": images,
                        "position": positions,
                        "action": actions,
                        "mode_id": mode_label,
                        "context": context,
                    }
                )
    rejection_stats.log_summary(
        task_name=SyntheticTaskName.CORRIDOR_NAVIGATION.value, noise_std=noise_std
    )
    return episodes


def _parametric_circle(
    center: np.ndarray,
    radius: float,
    num_points: int,
    clockwise: bool,
) -> np.ndarray:
    """Generate positions along a parametric circle.

    Starts at the point on the circle closest to (0.5, 0.5) and traces
    a full loop. For the bottom circle this is the top of the circle,
    for the top circle this is the bottom.

    Args:
        center: Circle center (x, y). Shape (2,).
        radius: Circle radius in [0, 1] space.
        num_points: Number of trajectory positions.
        clockwise: Traverse clockwise if True, counter-clockwise otherwise.

    Returns:
        Cartesian positions, shape (num_points, 2), dtype float32.
    """
    start_angle = np.arctan2(0.5 - float(center[1]), 0.5 - float(center[0]))
    direction = -1.0 if clockwise else 1.0
    theta = start_angle + direction * np.linspace(
        0.0, 2.0 * np.pi, num_points, endpoint=True, dtype=np.float32
    )
    x_positions = float(center[0]) + radius * np.cos(theta)
    y_positions = float(center[1]) + radius * np.sin(theta)
    return np.stack(
        [x_positions.astype(np.float32), y_positions.astype(np.float32)],
        axis=-1,
    )


def _generate_radial_obstacles(
    num_modes: int,
    noise_std: float,
) -> list[tuple[float, float, float, float]]:
    """Generate obstacle rectangles between each adjacent pair of radii.

    Places an axis-aligned square in each sector at the midpoint angle
    between consecutive radii, at half the radial distance from center.
    Size is derived purely from sector geometry minus a 3-sigma noise
    margin so a noisy radial trajectory cannot enter it:

        perpendicular clearance = r_mid * sin(pi/K)
        available         = clearance - 3 * noise_std
        half_size         = available / sqrt(2)   (inscribed square)

    When available <= 0 the sector is too narrow to host any collision-free
    obstacle at this noise level, and an empty list is returned.

    Args:
        num_modes: Number of radial modes (K).
        noise_std: Standard deviation of trajectory noise (used as margin).

    Returns:
        List of (x_min, y_min, x_max, y_max) obstacle rectangles.
    """
    obstacles: list[tuple[float, float, float, float]] = []
    angular_gap = 2.0 * np.pi / num_modes
    midpoint_radius = RADIAL_RADIUS * 0.5
    clearance = midpoint_radius * np.sin(angular_gap / 2.0)
    available = clearance - 3.0 * noise_std
    if available <= 0.0:
        return obstacles
    obstacle_half_width = available / np.sqrt(2.0)
    obstacle_half_height = obstacle_half_width

    for mode_index in range(num_modes):
        angle_a = 2.0 * np.pi * mode_index / num_modes
        angle_b = 2.0 * np.pi * (mode_index + 1) / num_modes
        midpoint_angle = (angle_a + angle_b) / 2.0
        center_x = float(RADIAL_CENTER[0]) + midpoint_radius * np.cos(midpoint_angle)
        center_y = float(RADIAL_CENTER[1]) + midpoint_radius * np.sin(midpoint_angle)
        obstacles.append(
            (
                center_x - obstacle_half_width,
                center_y - obstacle_half_height,
                center_x + obstacle_half_width,
                center_y + obstacle_half_height,
            )
        )
    return obstacles


def _compute_corridor_gap_centers(
    num_gaps: int,
) -> list[float]:
    """Compute the y-coordinates of gap centers in the corridor wall.

    Gaps are evenly distributed across the wall height, excluding the
    top and bottom edges.

    Args:
        num_gaps: Number of gaps (K).

    Returns:
        List of y-coordinates for each gap center.
    """
    return [(index + 1) / (num_gaps + 1) for index in range(num_gaps)]


def _compute_corridor_gap_height(num_gaps: int) -> float:
    """Symmetric half-split of per-gap spacing between gap opening and wall.

    With K gaps evenly spaced across y=0..1, the spacing between gap
    centers is 1/(K+1). Splitting it evenly yields a gap opening of
    half the spacing and a wall of the other half.
    """
    gap_spacing = 1.0 / (num_gaps + 1)
    return gap_spacing / 2.0


def _generate_corridor_obstacles(
    gap_centers: list[float],
) -> list[tuple[float, float, float, float]]:
    """Generate wall segments between adjacent gaps in the corridor.

    Creates K-1 wall segments between K gaps. No wall segments at
    the top/bottom edges of the unit square.

    Args:
        gap_centers: Sorted y-coordinates of gap centers.

    Returns:
        List of (x_min, y_min, x_max, y_max) wall segment rectangles.
    """
    obstacles: list[tuple[float, float, float, float]] = []
    half_gap = _compute_corridor_gap_height(num_gaps=len(gap_centers)) / 2.0

    for index in range(len(gap_centers) - 1):
        wall_y_min = gap_centers[index] + half_gap
        wall_y_max = gap_centers[index + 1] - half_gap
        if wall_y_max > wall_y_min:
            obstacles.append(
                (CORRIDOR_WALL_X1, wall_y_min, CORRIDOR_WALL_X2, wall_y_max)
            )
    return obstacles


def _apply_sinusoidal_style(
    positions: np.ndarray,
    style_index: int,
    num_styles: int,
    gap_height: float,
) -> np.ndarray:
    """Add sinusoidal y-displacement to produce trajectory style variations.

    Each style uses a different frequency to create visually distinct
    curved trajectories through the same corridor.

    Args:
        positions: Base trajectory positions, shape (num_steps, 2).
        style_index: Index of the sinusoidal style (0-based).
        num_styles: Total number of styles for amplitude scaling.
        gap_height: Height of the corridor gap in y-coordinates. Used as
            the hard upper bound on amplitude so the trajectory cannot be
            pushed through the wall.

    Returns:
        Modified positions with sinusoidal y-displacement, shape (num_steps, 2).
    """
    num_steps = len(positions)
    normalized_time = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    envelope = 4.0 * normalized_time * (1.0 - normalized_time)
    frequency = 2.0 * (style_index + 1)
    # Each style shares the gap: amplitude scales inversely with num_styles
    # and is capped by the half-gap so no style can cross the wall.
    amplitude = min(gap_height / (4.0 * num_styles), gap_height / 2.0)
    y_offset = amplitude * np.sin(frequency * np.pi * normalized_time) * envelope
    modified = positions.copy()
    modified[:, 1] += y_offset.astype(np.float32)
    return modified


def _interpolate_waypoints(
    waypoints: list[tuple[float, float]],
    num_points: int,
) -> np.ndarray:
    """Linearly interpolate between ordered waypoints to produce a trajectory.

    Distributes num_points evenly along the piecewise-linear path defined
    by the waypoint sequence.

    Args:
        waypoints: Ordered Cartesian waypoints [(x0, y0), (x1, y1), ...].
        num_points: Total number of trajectory positions to produce.

    Returns:
        Cartesian positions of shape (num_points, 2), dtype float32.
    """
    waypoint_array = np.array(waypoints, dtype=np.float32)
    segment_lengths = np.linalg.norm(np.diff(waypoint_array, axis=0), axis=1)
    cumulative_distance = np.concatenate([np.array([0.0]), np.cumsum(segment_lengths)])
    total_distance = cumulative_distance[-1]

    uniform_distances = np.linspace(0.0, total_distance, num_points)
    interpolated_x = np.interp(
        uniform_distances, cumulative_distance, waypoint_array[:, 0]
    )
    interpolated_y = np.interp(
        uniform_distances, cumulative_distance, waypoint_array[:, 1]
    )
    return np.stack([interpolated_x, interpolated_y], axis=-1).astype(np.float32)


def _gaussian_kernel_1d(smoothing_sigma: float) -> np.ndarray:
    """Build a normalized 1-D Gaussian smoothing kernel.

    Args:
        smoothing_sigma: Standard deviation of the kernel, in timesteps.

    Returns:
        Kernel of shape (2 * radius + 1,) summing to 1, dtype float64.
    """
    radius = max(1, int(GAUSSIAN_KERNEL_TRUNCATE * smoothing_sigma + 0.5))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / smoothing_sigma) ** 2)
    return kernel / kernel.sum()


def _sample_position_noise(
    shape: tuple[int, ...],
    noise_std: float,
    smoothing_sigma: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Sample position noise with a controllable temporal frequency band.

    Actions are position differences, so a first difference -- a high-pass
    filter -- is applied to this noise downstream. With ``smoothing_sigma = 0``
    the noise is i.i.d. across timesteps and the resulting *action* noise has
    power spectrum proportional to ``4 sin^2(pi f / f_s)``, concentrated at the
    Nyquist frequency. Smoothing the position noise first multiplies that
    spectrum by the kernel response, moving action-noise energy to lower
    frequencies.

    The white-noise amplitude is rescaled so the variance of the resulting
    action noise is ``2 * noise_std ** 2`` regardless of ``smoothing_sigma``.
    Both bands therefore inject the same action-noise power and differ only in
    spectral shape, which is what makes a high-band/low-band comparison
    controlled. The identity used is
    ``Var(eta_{t+1} - eta_t) = 2 * white_std ** 2 * (r0 - r1)`` for
    ``eta = kernel * white``, ``r0 = sum(k^2)``, ``r1 = sum(k_i k_{i+1})``.

    Args:
        shape: Noise shape as (num_steps, num_dims).
        noise_std: Position-noise scale. Equals the i.i.d. standard deviation
            when ``smoothing_sigma`` is 0.
        smoothing_sigma: Gaussian smoothing width in timesteps. 0 disables
            smoothing and reproduces i.i.d. noise exactly.
        random_generator: NumPy random generator for reproducibility.

    Returns:
        Position noise of shape ``shape``, dtype float64.
    """
    if smoothing_sigma <= 0.0:
        return random_generator.normal(0.0, noise_std, size=shape)

    num_steps, num_dims = shape
    kernel = _gaussian_kernel_1d(smoothing_sigma=smoothing_sigma)
    radius = (kernel.size - 1) // 2
    autocorrelation_gap = float(kernel @ kernel) - float(kernel[:-1] @ kernel[1:])
    white_std = noise_std / np.sqrt(autocorrelation_gap)
    # Convolve in "valid" mode over a padded white sequence so every output
    # sample has identical statistics (no boundary variance artifacts).
    white = random_generator.normal(
        0.0, white_std, size=(num_steps + 2 * radius, num_dims)
    )
    smoothed = np.empty(shape, dtype=np.float64)
    for dim in range(num_dims):
        smoothed[:, dim] = np.convolve(white[:, dim], kernel, mode="valid")
    return smoothed


def _sample_action_noise(
    shape: tuple[int, ...],
    noise_std: float,
    smoothing_sigma: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Sample band-shaped noise to add directly to action labels.

    Unlike :func:`_sample_position_noise`, this noise is not differenced
    downstream, so the band shaping is applied here and the result is scaled to
    a fixed root-mean-square. Both bands target standard deviation
    ``sqrt(2) * noise_std`` before the caller zeroes the terminal sentinel step
    (see :func:`_build_trajectory_signals`), which makes the realized RMS over
    the full array slightly below that target; the two injection points are
    otherwise directly comparable on one sigma grid.

    ``smoothing_sigma = 0`` applies a first difference **with periodic (wrap-
    around) boundaries** via ``np.roll``, reproducing the ``4 sin^2(pi f / f_s)``
    spectrum the position path induces (high band) at the cost of an artificial
    correlation between the first and last step. A positive value applies
    Gaussian smoothing instead, also wrapped, moving the energy to low
    frequencies. Because the scaling to the target root-mean-square happens
    after shaping, the two bands carry identical power by construction, with no
    correction factor and no effect on positions, images or clamping.

    Both the draw and the shaping wrap around the episode, so exactly
    ``num_steps * num_dims`` samples are consumed at every noise level and in
    every band. This keeps the RNG state after sampling independent of band or
    sigma, but shuffling is a separate, independently-seeded step (see
    :func:`generate_task_episodes`) precisely so this draw-size invariant is not
    the only thing standing between a noisy store and its zero-noise reference.

    Args:
        shape: Noise shape as (num_steps, num_dims).
        noise_std: Position-path noise scale this action noise matches.
        smoothing_sigma: Gaussian smoothing width in timesteps. 0 selects the
            high band.
        random_generator: NumPy random generator for reproducibility.

    Returns:
        Action noise of shape ``shape``, dtype float64.
    """
    target_std = math.sqrt(2.0) * noise_std
    white = random_generator.normal(0.0, 1.0, size=shape)
    if smoothing_sigma <= 0.0:
        shaped = white - np.roll(white, shift=1, axis=0)
        shaped_std = math.sqrt(2.0)
    else:
        kernel = _gaussian_kernel_1d(smoothing_sigma=smoothing_sigma)
        radius = (kernel.size - 1) // 2
        shaped = np.zeros(shape, dtype=np.float64)
        for tap, weight in enumerate(kernel):
            shaped += weight * np.roll(white, shift=radius - tap, axis=0)
        shaped_std = math.sqrt(float(kernel @ kernel))
    # Analytic rather than sample standard deviation: smoothing leaves only
    # num_steps / smoothing_sigma effectively independent samples per episode, so
    # dividing by the sample estimate biases the delivered power upwards.
    return shaped * (target_std / shaped_std)


def _apply_cable_hysteresis(
    trajectory: np.ndarray,
    backlash_threshold: float,
) -> np.ndarray:
    """Apply an element-wise play operator to a kinematic trajectory.

    The output remains unchanged while the input moves inside a deadband of
    radius ``backlash_threshold`` around it. Once the input leaves that band,
    the output follows the corresponding boundary. This produces the lag and
    direction-dependent loop of a cable transmission with backlash.

    Args:
        trajectory: Ground-truth Cartesian positions with shape
            ``(num_steps, num_dims)``.
        backlash_threshold: Non-negative play-operator threshold in the same
            units as the trajectory coordinates.

    Returns:
        History-dependent kinematic measurements with the input shape and
        dtype.

    Raises:
        ValueError: If the threshold is negative or the trajectory is not a
            non-empty two-dimensional array.
    """
    if backlash_threshold < 0.0:
        raise ValueError(
            f"backlash_threshold must be non-negative, got {backlash_threshold}."
        )
    if trajectory.ndim != 2 or trajectory.shape[0] == 0:
        raise ValueError(
            "trajectory must be a non-empty two-dimensional array, got "
            f"shape {trajectory.shape}."
        )

    measured_trajectory = np.empty_like(trajectory)
    measured_trajectory[0] = trajectory[0]
    for step_index in range(1, trajectory.shape[0]):
        lower_bound = trajectory[step_index] - backlash_threshold
        upper_bound = trajectory[step_index] + backlash_threshold
        measured_trajectory[step_index] = np.maximum(
            lower_bound,
            np.minimum(upper_bound, measured_trajectory[step_index - 1]),
        )
    return measured_trajectory


def _build_trajectory_signals(
    trajectory: np.ndarray,
    noise_std: float,
    noise_smoothing_sigma: float,
    noise_injection: str,
    random_generator: np.random.Generator,
    noise_model: str = SyntheticNoiseModel.GAUSSIAN.value,
    obstacles: list[tuple[float, float, float, float]] | None = None,
    rejection_stats: _RejectionStats | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the positions and actions of one episode at the chosen noise point.

    Args:
        trajectory: Deterministic Cartesian path, shape (num_steps, 2).
        noise_std: Noise scale. For cable hysteresis this is the play-operator
            backlash threshold.
        noise_smoothing_sigma: Gaussian temporal smoothing width in timesteps
            selecting the noise band; 0 is the high band.
        noise_injection: ``NoiseInjection`` value choosing the injection point.
        random_generator: NumPy random generator for reproducibility.
        noise_model: ``SyntheticNoiseModel`` value choosing the error process.
            Cable hysteresis interprets ``noise_std`` as the play-operator
            backlash threshold rather than as a standard deviation.
        obstacles: Rectangles a position-noise trajectory must avoid. None
            skips rejection sampling.
        rejection_stats: Optional accumulator for rejection-sampling attempts.

    Returns:
        Positions clamped to the unit square and their action labels.

    Raises:
        ValueError: If the noise model or injection configuration is invalid.
    """
    valid_noise_models = [member.value for member in SyntheticNoiseModel]
    if noise_model not in valid_noise_models:
        raise ValueError(
            f"Unknown noise_model '{noise_model}'. Expected one of "
            f"{valid_noise_models}."
        )
    if noise_model == SyntheticNoiseModel.CABLE_HYSTERESIS.value:
        if noise_injection != NoiseInjection.ACTION.value:
            raise ValueError(
                "noise_model='cable_hysteresis' requires noise_injection='action'."
            )
        if noise_smoothing_sigma != 0.0:
            raise ValueError(
                "noise_smoothing_sigma must be 0 for noise_model='cable_hysteresis'."
            )

    if noise_injection == NoiseInjection.POSITION.value:
        if obstacles is None:
            positions = _add_noise_and_clamp(
                trajectory=trajectory,
                noise_std=noise_std,
                random_generator=random_generator,
                noise_smoothing_sigma=noise_smoothing_sigma,
            )
        else:
            positions = _sample_noisy_trajectory_no_collision(
                base_trajectory=trajectory,
                obstacles=obstacles,
                noise_std=noise_std,
                random_generator=random_generator,
                noise_smoothing_sigma=noise_smoothing_sigma,
                rejection_stats=rejection_stats,
            )
        return positions, _compute_actions(positions)

    if noise_injection != NoiseInjection.ACTION.value:
        raise ValueError(
            f"Unknown noise_injection '{noise_injection}'. Expected one of "
            f"{[member.value for member in NoiseInjection]}."
        )

    # Positions, and therefore the rendered images and the obstacle geometry,
    # stay exactly as the deterministic task defines them; only the labels move.
    positions = np.clip(trajectory, 0.0, 1.0).astype(np.float32)
    if noise_model == SyntheticNoiseModel.CABLE_HYSTERESIS.value:
        measured_positions = _apply_cable_hysteresis(
            trajectory=positions,
            backlash_threshold=noise_std,
        )
        actions = _compute_actions(measured_positions)
        if rejection_stats is not None:
            rejection_stats.record(attempts=1)
        return positions, actions

    noise = _sample_action_noise(
        shape=positions.shape,
        noise_std=noise_std,
        smoothing_sigma=noise_smoothing_sigma,
        random_generator=random_generator,
    ).astype(np.float32)
    # The final action is a zero sentinel, not a command: there is no position
    # after the last one to difference against. The position path leaves it at
    # zero because the noise reaches actions through that difference, so leaving
    # it noise-free here keeps the two injection points comparable and stops the
    # continuous arms -- whose horizon includes this step -- from being handed a
    # random terminal move the tokenized arms never see.
    noise[-1] = 0.0
    actions = _compute_actions(positions) + noise
    if rejection_stats is not None:
        rejection_stats.record(attempts=1)
    return positions, actions


def _add_noise_and_clamp(
    trajectory: np.ndarray,
    noise_std: float,
    random_generator: np.random.Generator,
    noise_smoothing_sigma: float = 0.0,
) -> np.ndarray:
    """Add isotropic Gaussian noise and clamp to [0, 1].

    Args:
        trajectory: Cartesian positions (x, y) of shape (num_steps, 2).
        noise_std: Standard deviation of the additive Gaussian noise.
        random_generator: NumPy random generator for reproducibility.
        noise_smoothing_sigma: Gaussian temporal smoothing width, in
            timesteps, applied to the noise before it is added. 0 keeps the
            i.i.d. (high-frequency) default; larger values shift the induced
            action noise to lower frequencies at matched power. See
            :func:`_sample_position_noise`.

    Returns:
        Noisy positions clamped to [0, 1], shape (num_steps, 2), float32.
    """
    noise = _sample_position_noise(
        shape=trajectory.shape,
        noise_std=noise_std,
        smoothing_sigma=noise_smoothing_sigma,
        random_generator=random_generator,
    ).astype(np.float32)
    noisy_trajectory = trajectory + noise
    return np.clip(noisy_trajectory, 0.0, 1.0)


def _trajectory_collides(
    trajectory: np.ndarray,
    obstacles: list[tuple[float, float, float, float]],
) -> bool:
    """Return True if any point of the trajectory lies inside any obstacle."""
    for x_min, y_min, x_max, y_max in obstacles:
        inside_x = (trajectory[:, 0] >= x_min) & (trajectory[:, 0] <= x_max)
        inside_y = (trajectory[:, 1] >= y_min) & (trajectory[:, 1] <= y_max)
        if (inside_x & inside_y).any():
            return True
    return False


def _sample_noisy_trajectory_no_collision(
    base_trajectory: np.ndarray,
    obstacles: list[tuple[float, float, float, float]],
    noise_std: float,
    random_generator: np.random.Generator,
    noise_smoothing_sigma: float = 0.0,
    rejection_stats: _RejectionStats | None = None,
) -> np.ndarray:
    """Sample noise until the resulting trajectory does not collide.

    Rejection sampling conditions the noise on being collision-free, which
    truncates the injected noise distribution. The truncation grows with
    ``noise_std``, so noise-sweep experiments must track how often it fires --
    otherwise a sweep can silently flatten because high-noise cells only keep
    their luckiest, least-noisy trajectories. Pass ``rejection_stats`` to
    record that.

    Args:
        base_trajectory: Deterministic Cartesian path, shape (num_steps, 2).
        obstacles: Axis-aligned (x_min, y_min, x_max, y_max) rectangles.
        noise_std: Gaussian noise standard deviation.
        random_generator: NumPy random generator for reproducibility.
        noise_smoothing_sigma: Gaussian temporal smoothing width for the
            injected noise, in timesteps. See :func:`_sample_position_noise`.
        rejection_stats: Optional accumulator recording attempts per episode.

    Returns:
        Noisy trajectory that does not collide with any obstacle.

    Raises:
        RuntimeError: If no collision-free trajectory is produced within
            MAX_TRAJECTORY_RETRIES attempts (indicates obstacle geometry
            is too tight for the given noise level).
    """
    for attempt in range(MAX_TRAJECTORY_RETRIES):
        candidate = _add_noise_and_clamp(
            trajectory=base_trajectory,
            noise_std=noise_std,
            random_generator=random_generator,
            noise_smoothing_sigma=noise_smoothing_sigma,
        )
        if not _trajectory_collides(trajectory=candidate, obstacles=obstacles):
            if rejection_stats is not None:
                rejection_stats.record(attempts=attempt + 1)
            return candidate
    raise RuntimeError(
        f"Failed to generate a collision-free trajectory after "
        f"{MAX_TRAJECTORY_RETRIES} attempts (obstacle geometry too tight "
        f"for noise_std={noise_std})."
    )


def _compute_actions(
    positions: np.ndarray,
) -> np.ndarray:
    """Compute delta-position actions from consecutive Cartesian positions.

    action[t] = position[t+1] - position[t], with the last action set to zeros.

    Args:
        positions: Cartesian positions (x, y) of shape (num_steps, 2).

    Returns:
        Delta actions (dx, dy) of shape (num_steps, 2), dtype float32.
    """
    actions = np.zeros_like(positions)
    actions[:-1] = positions[1:] - positions[:-1]
    return actions


def _balanced_mode_counts(
    total_episodes: int,
    num_modes: int,
) -> list[int]:
    """Distribute episodes as evenly as possible across modes.

    Any remainder from integer division is distributed one extra episode
    to the first modes.

    Args:
        total_episodes: Total number of episodes to distribute.
        num_modes: Number of behavioral modes.

    Returns:
        List of episode counts per mode, summing to total_episodes.
    """
    base_count = total_episodes // num_modes
    remainder = total_episodes % num_modes
    counts = [
        base_count + (1 if index < remainder else 0) for index in range(num_modes)
    ]
    return counts


def _weighted_mode_counts(
    total_episodes: int,
    mode_weights: list[float],
) -> list[int]:
    """Distribute episodes according to relative mode weights.

    Weights are normalized to sum to 1. Rounding remainders are assigned
    to modes with the largest fractional parts.

    Args:
        total_episodes: Total number of episodes to distribute.
        mode_weights: Relative weight per mode (must be positive).

    Returns:
        List of episode counts per mode, summing to total_episodes.
    """
    weight_sum = sum(mode_weights)
    normalized = [weight / weight_sum for weight in mode_weights]
    fractional_counts = [total_episodes * weight for weight in normalized]
    base_counts = [int(count) for count in fractional_counts]
    remainders = [
        fractional - base
        for fractional, base in zip(fractional_counts, base_counts, strict=True)
    ]
    deficit = total_episodes - sum(base_counts)
    sorted_indices = sorted(
        range(len(remainders)), key=lambda i: remainders[i], reverse=True
    )
    for rank in range(deficit):
        base_counts[sorted_indices[rank]] += 1
    return base_counts


def _resolve_mode_counts(
    total_episodes: int,
    num_modes: int,
    mode_weights: list[float] | None,
) -> list[int]:
    """Dispatch to balanced or weighted episode distribution.

    Args:
        total_episodes: Total number of episodes.
        num_modes: Number of behavioral modes.
        mode_weights: Relative weights or None for uniform.

    Returns:
        List of episode counts per mode.
    """
    if mode_weights is None:
        return _balanced_mode_counts(total_episodes=total_episodes, num_modes=num_modes)
    if len(mode_weights) != num_modes:
        raise ValueError(
            f"mode_weights length ({len(mode_weights)}) must match "
            f"num_modes ({num_modes})"
        )
    return _weighted_mode_counts(
        total_episodes=total_episodes, mode_weights=mode_weights
    )
