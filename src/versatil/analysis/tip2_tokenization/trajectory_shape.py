"""Open-loop trajectory-shape diagnostics for the Tip 2 tokenization sweep.

The decomposition manifest measures *how far* a prediction is from the expert
(position RMSE); it says nothing about the *shape* of the predicted motion. This
module reads the per-cell action arrays saved by the eval hook
(``<results>/arrays/<cell>.npz``, keys ``expert`` / ``round_trip`` / ``argmax`` /
``stochastic``, each ``(N, H, D)`` per-step displacements) and computes shape
statistics that characterise the prediction beyond its aggregate error:

* **jerk** -- RMS of the second time-difference of the displacements (the jerk of
  the integrated position path). High-frequency roughness: how much of the
  demonstrator's per-step variation the prediction reproduces.
* **turning angle** -- mean absolute angle between consecutive displacement
  vectors. Directional roughness, independent of magnitude.
* **step speed** -- mean displacement magnitude per step, for context.

Each statistic is reported for the argmax prediction against the expert and the
tokenizer round trip as references, per grid point, averaged over seeds. The
motivating question is why a family with lower position RMSE need not roll out
better: the shape statistics show whether the low-RMSE prediction reproduces the
expert's motion or merely a smoothed version of it.

    python -m versatil.analysis.tip2_tokenization.trajectory_shape \
        /data/horse/ws/qizh093f-versatil/tip2_results/position_main_v1/arrays
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTPUT_DIR = Path("/home/qizh093f/versatil-github/outputs")
# Displacement magnitude below this (unit-square coords) carries no reliable
# direction, so it is dropped from the turning-angle average.
MIN_STEP_NORM = 1e-6
CELL_PATTERN = re.compile(
    r"(?P<task>[^_]+)__(?P<method>fast|binning)__"
    r"(?:scale|bins)-(?P<param>[0-9p]+)__seed-(?P<seed>\d+)\.npz"
)
# Which saved array each reference draws from.
REFERENCE_KEYS = ("argmax", "expert", "round_trip")
FAMILIES = (("fast", "FAST rounding scale", "(a) FAST"),
            ("binning", "Number of bins", "(b) Binning"))
EXPERT_COLOR = "0.35"
ARGMAX_COLOR = "#C82423"


def rms_jerk(actions: np.ndarray) -> float:
    """RMS of the second time-difference of per-step displacements.

    Args:
        actions: ``(N, H, D)`` per-step displacements (H >= 3).

    Returns:
        Root-mean-square jerk of the integrated position path over all chunks,
        steps and dimensions.
    """
    jerk = np.diff(actions, n=2, axis=1)
    return float(np.sqrt(np.mean(jerk**2)))


def mean_step_speed(actions: np.ndarray) -> float:
    """Mean per-step displacement magnitude over all chunks and steps."""
    return float(np.mean(np.linalg.norm(actions, axis=-1)))


def mean_turning_angle(actions: np.ndarray) -> float:
    """Mean absolute angle (radians) between consecutive displacement vectors.

    Steps whose displacement magnitude is below ``MIN_STEP_NORM`` carry no
    reliable direction and are excluded. Returns 0.0 when no consecutive pair
    survives (e.g. a stand-still prediction).

    Args:
        actions: ``(N, H, D)`` per-step displacements (H >= 2).
    """
    current = actions[:, :-1, :]
    following = actions[:, 1:, :]
    current_norm = np.linalg.norm(current, axis=-1)
    following_norm = np.linalg.norm(following, axis=-1)
    valid = (current_norm > MIN_STEP_NORM) & (following_norm > MIN_STEP_NORM)
    if not valid.any():
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.sum(current * following, axis=-1) / (current_norm * following_norm)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return float(np.mean(angle[valid]))


def cell_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """Shape metrics for one cell, per reference array.

    Args:
        arrays: Mapping with the ``REFERENCE_KEYS`` arrays; ``stochastic`` is
            ignored here. Each is ``(N, H, D)``.

    Returns:
        ``{"<ref>_<metric>": value}`` for every reference in ``REFERENCE_KEYS``
        that is present and every metric (jerk, turning_angle, step_speed).
    """
    metrics: dict[str, float] = {}
    for key in REFERENCE_KEYS:
        if key not in arrays:
            continue
        actions = np.asarray(arrays[key], dtype=np.float64)
        metrics[f"{key}_jerk"] = rms_jerk(actions=actions)
        metrics[f"{key}_turning_angle"] = mean_turning_angle(actions=actions)
        metrics[f"{key}_step_speed"] = mean_step_speed(actions=actions)
    return metrics


def parse_cell_name(filename: str) -> tuple[str, float, int] | None:
    """Return ``(method, param, seed)`` for a cell npz name, or None if it does
    not match the sweep's naming."""
    match = CELL_PATTERN.fullmatch(filename)
    if match is None:
        return None
    param = float(match.group("param").replace("p", "."))
    return match.group("method"), param, int(match.group("seed"))


def arithmetic_mean(values: list[float]) -> float:
    """Plain mean over seeds; shape metrics are physical, not log-scaled."""
    return sum(values) / len(values)


def aggregate_rows(
    rows: list[dict[str, float | str]],
) -> dict[tuple[str, float], dict[str, float]]:
    """Average each metric over seeds, keyed by (method, param).

    Args:
        rows: One dict per cell with ``method`` / ``param`` and metric columns.

    Returns:
        ``{(method, param): {metric: seed-mean}}``.
    """
    grouped: dict[tuple[str, float], list[dict[str, float | str]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), float(row["param"])), []).append(row)
    aggregated: dict[tuple[str, float], dict[str, float]] = {}
    for key, cell_rows in grouped.items():
        metric_names = [
            name
            for name in cell_rows[0]
            if name not in {"method", "param", "seed"}
        ]
        aggregated[key] = {
            name: arithmetic_mean(values=[float(row[name]) for row in cell_rows])
            for name in metric_names
        }
    return aggregated


def load_cell_rows(arrays_dir: Path) -> list[dict[str, float | str]]:
    """Read every cell npz under ``arrays_dir`` into a metric row."""
    rows: list[dict[str, float | str]] = []
    for path in sorted(arrays_dir.glob("*.npz")):
        parsed = parse_cell_name(path.name)
        if parsed is None:
            continue
        method, param, seed = parsed
        with np.load(path) as arrays:
            metrics = cell_metrics(arrays={k: arrays[k] for k in arrays.files})
        rows.append({"method": method, "param": param, "seed": seed, **metrics})
    return rows


def write_table(
    aggregated: dict[tuple[str, float], dict[str, float]], csv_path: Path
) -> None:
    """Write the seed-averaged shape table, coarse to fine within each family."""
    metric_names = sorted(next(iter(aggregated.values())).keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "param", *metric_names])
        for method, _, _ in FAMILIES:
            for (cell_method, param) in sorted(aggregated):
                if cell_method != method:
                    continue
                values = aggregated[(cell_method, param)]
                writer.writerow(
                    [method, f"{param:g}", *[f"{values[m]:.6g}" for m in metric_names]]
                )


def plot_shape(
    aggregated: dict[tuple[str, float], dict[str, float]], output_path: Path
) -> None:
    """Two rows (jerk, turning angle) x two family columns, log-x, with the
    expert level drawn as a reference on each panel."""
    figure, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex="col", sharey="row")
    metric_rows = (
        ("jerk", "RMS jerk of position path"),
        ("turning_angle", "Mean turning angle (rad)"),
    )
    for column, (method, xlabel, title) in enumerate(FAMILIES):
        params = sorted(param for cell_method, param in aggregated if cell_method == method)
        if not params:
            continue
        for row, (metric, ylabel) in enumerate(metric_rows):
            axis = axes[row][column]
            argmax = [aggregated[(method, p)][f"argmax_{metric}"] for p in params]
            expert = [aggregated[(method, p)][f"expert_{metric}"] for p in params]
            axis.plot(params, argmax, marker="D", color=ARGMAX_COLOR,
                      linewidth=2.2, markersize=6, label="argmax prediction")
            axis.plot(params, expert, marker="o", color=EXPERT_COLOR,
                      linestyle="--", linewidth=1.4, markersize=4,
                      label="expert (noisy target)")
            axis.set_xscale("log")
            axis.set_xlim(params[0] / 1.6, params[-1] * 1.6)
            axis.set_xticks(params)
            axis.set_xticklabels(
                [f"{int(p)}" if p >= 10 or p == round(p) else f"{p:g}" for p in params]
            )
            axis.grid(True, alpha=0.3)
            if row == 0:
                axis.set_yscale("log")
                axis.set_title(title, fontsize=13, fontweight="bold")
            if column == 0:
                axis.set_ylabel(ylabel)
            if row == len(metric_rows) - 1:
                axis.set_xlabel(xlabel)
    axes[0][0].legend(fontsize=8, framealpha=0.9)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Tip 2 trajectory-shape diagnostics.")
    parser.add_argument("arrays_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    rows = load_cell_rows(arrays_dir=args.arrays_dir)
    aggregated = aggregate_rows(rows=rows)
    stem = args.arrays_dir.parent.name
    csv_path = args.out_dir / "tip2" / f"tip2_trajectory_shape_{stem}.csv"
    figure_path = args.out_dir / f"tip2_trajectory_shape_{stem}.png"
    write_table(aggregated=aggregated, csv_path=csv_path)
    plot_shape(aggregated=aggregated, output_path=figure_path)
    print(csv_path)
    print(figure_path)


if __name__ == "__main__":
    _main()
