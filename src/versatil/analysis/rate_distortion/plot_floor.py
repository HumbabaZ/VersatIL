"""Figures for the tokenizer floor study (FAST vs Binning, horizon effect).

Publication figures for the thesis. Style follows the project's visualization
config: the ``scienceplots`` toolkit and the qualitative ``tab10`` palette (``tab20``
when a figure needs more than ten series), assigned in fixed order. Axes carry
Title-Case labels with units and a light dashed grid; FAST's dominated
(Pareto-inferior) points are hollow off the frontier line and called out in the
legend. The rate-distortion frontier is coloured by chunk horizon; the
compression-distortion frontier is coloured by tokenizer family.

Runnable directly from a results CSV (no versatil import, no re-run of the sweep):

    python src/versatil/analysis/rate_distortion/plot_floor.py results.csv out_dir
"""

import csv
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

if find_spec("scienceplots") is not None:
    import scienceplots  # noqa: F401,E402

    plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 300})

# Qualitative palette per the project viz config: tab10 first, tab20 when a figure
# needs more than ten distinguishable series.
_TAB10_SIZE = 10

FAMILY_LABEL = {"fast": "FAST", "binning": "Binning"}
FAMILY_KNOB_KEY = {"fast": "scale", "binning": "num_bins"}
FAMILY_KNOB_PREFIX = {"fast": "s", "binning": "b"}
FAMILY_MARKER = {"fast": "o", "binning": "s"}
FAMILY_LINESTYLE = {"fast": "-", "binning": "--"}
FAMILY_ORDER = ("fast", "binning")
DOMINATED_LABEL = "Dominated (off Frontier)"
FULL_SWEEP_HORIZON = 10


def choose_colors(count: int) -> list:
    """Return ``count`` qualitative colors in fixed order (tab10, else tab20).

    Categorical hues are assigned in a fixed order so a series keeps its color
    across figures and re-runs.
    """
    colormap = plt.get_cmap("tab10" if count <= _TAB10_SIZE else "tab20")
    return [colormap(index) for index in range(count)]


def plot_all(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, dict]:
    """Write all floor-study figures; return the colors assigned to each series.

    Args:
        rows: Parsed study rows.
        output_dir: Directory to write the figures into.
    """
    horizons = sorted(
        {
            int(r["horizon"])
            for r in rows
            if r.get("feasible") and isinstance(r.get("horizon"), int | float)
        }
    )
    horizon_colors = (
        dict(zip(horizons, choose_colors(len(horizons)), strict=True))
        if horizons
        else {}
    )
    families = [
        family for family in FAMILY_ORDER if any(r["family"] == family for r in rows)
    ]
    family_colors = dict(zip(families, choose_colors(len(families)), strict=True))

    plot_frontier_by_horizon(rows, output_dir / "figure_a_frontier.png", horizon_colors)
    plot_family_frontier(
        rows,
        output_dir / "figure_a_frontier_h10.png",
        family_colors,
        x_key="bits_per_step",
        x_label="Rate (Bits per Timestep)",
        title="Tokenizer Rate-Distortion Frontier",
    )
    plot_tokens_frontier(rows, output_dir / "figure_tokens_frontier.png", family_colors)
    plot_horizon_effect(rows, output_dir / "figure_horizon.png", family_colors)
    plot_vocab_flat(rows, output_dir / "figure_vocab_flat.png", family_colors)
    return {
        "families": {f: to_hex(c) for f, c in family_colors.items()},
        "horizons": {h: to_hex(c) for h, c in horizon_colors.items()},
    }


def plot_frontier_by_horizon(
    rows: list[dict[str, Any]], output_path: Path, horizon_colors: dict
) -> None:
    """Rate-distortion frontier, one colour per chunk horizon.

    Marker/line-style encode the family (FAST circles + solid, Binning squares +
    dashed). At H = 10 the full scale/bins sweeps trace each frontier and are knob-
    annotated; other horizons contribute their operating point. FAST's dominated
    points are hollow.
    """
    if not horizon_colors:
        return
    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    for horizon in sorted(horizon_colors):
        color = horizon_colors[horizon]
        for family in FAMILY_ORDER:
            points = _labeled_points(
                rows,
                "bits_per_step",
                "rmse_continuous",
                FAMILY_KNOB_KEY[family],
                lambda r, fam=family, h=horizon: (
                    r["family"] == fam
                    and r.get("horizon") == h
                    and _in_frontier_sweep(r, fam)
                ),
            )
            if not points:
                continue
            _draw_line(axis, family, points, color, mark_dominated=(family == "fast"))
            if horizon == FULL_SWEEP_HORIZON:
                _annotate(axis, family, points, color)
    _style_axis(axis)
    axis.set_xlabel("Rate (Bits per Timestep)")
    axis.set_ylabel("Reconstruction RMSE (Normalized Action Units)")
    axis.set_title("Tokenizer Rate-Distortion Frontier")
    _frontier_by_horizon_legend(axis, horizon_colors)
    _save(figure, output_path)


def plot_tokens_frontier(
    rows: list[dict[str, Any]], output_path: Path, family_colors: dict
) -> None:
    """Compression-distortion frontier at H = 10, one colour per family.

    Binning is a fixed-length ``T*D`` vertical (dashed); its points are thinned so
    the bin-count labels never overlap. FAST is solid with its dominated point
    hollow. X axis is sequence length (tokens/chunk).
    """
    plot_family_frontier(
        rows,
        output_path,
        family_colors,
        x_key="mean_token_len",
        x_label="Sequence Length (Tokens per Chunk)",
        title="Tokenizer Compression-Distortion Frontier",
        thin_binning=True,
    )


def plot_family_frontier(
    rows: list[dict[str, Any]],
    output_path: Path,
    family_colors: dict,
    x_key: str,
    x_label: str,
    title: str,
    thin_binning: bool = False,
) -> None:
    """Single-horizon frontier coloured by tokenizer family (FAST blue, Binning orange).

    ``x_key`` selects the horizontal axis: ``bits_per_step`` (rate) or
    ``mean_token_len`` (sequence length). ``thin_binning`` drops crowded binning
    points so their bin-count labels stay legible on the fixed-length vertical.
    """
    horizon = FULL_SWEEP_HORIZON
    family_points = {}
    for family in FAMILY_ORDER:
        if family not in family_colors:
            continue
        points = _labeled_points(
            rows,
            x_key,
            "rmse_continuous",
            FAMILY_KNOB_KEY[family],
            lambda r, fam=family: (
                r["family"] == fam
                and _in_frontier_sweep(r, fam)
                and r.get("horizon") == horizon
            ),
        )
        if points:
            family_points[family] = points
    if not family_points:
        return

    all_y = [point[1] for points in family_points.values() for point in points]
    min_gap = 0.09 * (max(all_y) - min(all_y))
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    for family, points in family_points.items():
        color = family_colors[family]
        if thin_binning and family == "binning":
            points = _thin_by_gap(sorted(points, key=lambda p: p[1]), min_gap)
        _draw_line(axis, family, points, color, mark_dominated=(family == "fast"))
        _annotate(axis, family, points, color)
    _style_axis(axis)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Reconstruction RMSE (Normalized Action Units)")
    axis.set_title(f"{title} (H = {horizon})")
    handles = [
        _family_handle(family, family_colors[family]) for family in family_points
    ]
    handles.append(_dominated_handle())
    axis.legend(handles=handles, title="Tokenizer")
    _save(figure, output_path)


def plot_horizon_effect(
    rows: list[dict[str, Any]], output_path: Path, family_colors: dict
) -> None:
    """Horizon effect at operating points: RMSE (flat expected) and bits/step."""

    def operating(row: dict[str, Any]) -> bool:
        return bool(row.get("is_operating_point"))

    families = [
        family
        for family in FAMILY_ORDER
        if family in family_colors
        and any(
            r["family"] == family and operating(r) and r.get("feasible") for r in rows
        )
    ]
    if not families:
        return
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    for family in families:
        color = family_colors[family]
        for axis, key in ((left, "rmse_continuous"), (right, "bits_per_step")):
            points = _labeled_points(
                rows,
                "horizon",
                key,
                "horizon",
                lambda r, fam=family: r["family"] == fam and operating(r),
            )
            if not points:
                continue
            points.sort()
            axis.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                marker=FAMILY_MARKER[family],
                linestyle=FAMILY_LINESTYLE[family],
                color=color,
                label=FAMILY_LABEL[family],
            )
    left.set_xlabel("Chunk Horizon H")
    left.set_ylabel("Reconstruction RMSE (Normalized Action Units)")
    left.set_title("Reconstruction Floor vs Horizon")
    right.set_xlabel("Chunk Horizon H")
    right.set_ylabel("Rate (Bits per Timestep)")
    right.set_title("Rate vs Horizon")
    for axis in (left, right):
        _style_axis(axis)
        axis.legend(title="Tokenizer")
    _save(figure, output_path)


def plot_vocab_flat(
    rows: list[dict[str, Any]], output_path: Path, family_colors: dict
) -> None:
    """|V| sweep: distortion must be flat as rate moves (lossless rate knob)."""
    if "fast" not in family_colors:
        return
    horizon = FULL_SWEEP_HORIZON
    points = _labeled_points(
        rows,
        "bits_per_step",
        "rmse_continuous",
        "vocab_size",
        lambda r: (
            r["family"] == "fast"
            and r.get("sweep") == "vocab"
            and r.get("horizon") == horizon
        ),
    )
    if not points:
        return
    points.sort()
    color = family_colors["fast"]
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        marker="s",
        color=color,
        label="FAST |V| Sweep",
    )
    for x_value, y_value, knob in points:
        axis.annotate(
            f"|V|={int(knob)}",
            (x_value, y_value),
            color=color,
            fontsize=8,
            xytext=(4, 3),
            textcoords="offset points",
        )
    _style_axis(axis)
    axis.set_xlabel("Rate (Bits per Timestep)")
    axis.set_ylabel("Reconstruction RMSE (Normalized Action Units)")
    axis.set_title(f"Vocabulary Size Is a Lossless Rate Knob (H = {horizon})")
    axis.legend()
    _save(figure, output_path)


def _in_frontier_sweep(row: dict[str, Any], family: str) -> bool:
    """Whether a row is a frontier point for its family (scale/horizon vs bins)."""
    if family == "fast":
        return row.get("sweep") in ("scale", "horizon")
    return row.get("sweep") == "bins"


def _draw_line(
    axis,
    family: str,
    points: list[tuple[float, float, float]],
    color,
    mark_dominated: bool,
) -> None:
    """Draw a family's line + markers; hollow the dominated points when asked."""
    frontier, dominated = _pareto_split(points) if mark_dominated else (points, [])
    frontier_sorted = sorted(frontier)
    axis.plot(
        [point[0] for point in frontier_sorted],
        [point[1] for point in frontier_sorted],
        marker=FAMILY_MARKER[family],
        linestyle=FAMILY_LINESTYLE[family],
        color=color,
        zorder=3,
    )
    for point in dominated:
        axis.scatter(
            [point[0]],
            [point[1]],
            marker=FAMILY_MARKER[family],
            facecolors="none",
            edgecolors=color,
            zorder=3,
        )


def _annotate(
    axis, family: str, points: list[tuple[float, float, float]], color
) -> None:
    """Annotate each point with its knob value."""
    for x_value, y_value, knob in points:
        axis.annotate(
            _knob_text(family, knob),
            (x_value, y_value),
            color=color,
            fontsize=8,
            xytext=(4, 3),
            textcoords="offset points",
        )


def _knob_text(family: str, knob: float) -> str:
    """Label for a point's knob (FAST scale as-is, binning bin count as int)."""
    prefix = FAMILY_KNOB_PREFIX[family]
    return f"{prefix}={knob:g}" if family == "fast" else f"{prefix}={int(knob)}"


def _thin_by_gap(
    points: list[tuple[float, float, float]], min_gap: float
) -> list[tuple[float, float, float]]:
    """Drop crowded points so their labels don't overlap; keep the endpoints."""
    if len(points) <= 2:
        return points
    kept = [points[0]]
    for point in points[1:-1]:
        if abs(point[1] - kept[-1][1]) >= min_gap:
            kept.append(point)
    kept.append(points[-1])
    return kept


def _frontier_by_horizon_legend(axis, horizon_colors: dict) -> None:
    """Legend combining horizon colours, family markers, and the dominated cue."""
    handles = [
        Line2D([], [], color=horizon_colors[horizon], lw=2, label=f"H = {horizon}")
        for horizon in sorted(horizon_colors)
    ]
    handles += [_family_handle(family, "0.35") for family in FAMILY_ORDER]
    handles.append(_dominated_handle())
    axis.legend(handles=handles, fontsize=8, ncol=1)


def _family_handle(family: str, color) -> Line2D:
    """Legend handle showing a family's marker and line style."""
    return Line2D(
        [],
        [],
        color=color,
        marker=FAMILY_MARKER[family],
        linestyle=FAMILY_LINESTYLE[family],
        label=FAMILY_LABEL[family],
    )


def _dominated_handle() -> Line2D:
    """Legend handle explaining the hollow-marker convention."""
    return Line2D(
        [],
        [],
        color="0.35",
        marker="o",
        markerfacecolor="none",
        linestyle="",
        label=DOMINATED_LABEL,
    )


def _style_axis(axis) -> None:
    """Apply a light dashed background grid, drawn behind the data."""
    axis.grid(True, linestyle="--", linewidth=0.5, color="0.85")
    axis.set_axisbelow(True)


def _labeled_points(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    label_key: str,
    predicate,
) -> list[tuple[float, float, float]]:
    """Collect (x, y, knob) from feasible rows matching predicate."""
    points = []
    for row in rows:
        if not row.get("feasible") or not predicate(row):
            continue
        x_value, y_value, label = row.get(x_key), row.get(y_key), row.get(label_key)
        if (
            isinstance(x_value, int | float)
            and isinstance(y_value, int | float)
            and isinstance(label, int | float)
        ):
            points.append((float(x_value), float(y_value), float(label)))
    return points


def _pareto_split(
    points: list[tuple[float, float, float]],
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Split points into the lower-left Pareto frontier and dominated points."""
    frontier, dominated = [], []
    for point in points:
        x_value, y_value, _ = point
        is_dominated = any(
            other[0] <= x_value
            and other[1] <= y_value
            and (other[0] < x_value or other[1] < y_value)
            for other in points
            if other is not point
        )
        (dominated if is_dominated else frontier).append(point)
    return frontier, dominated


def _save(figure, output_path: Path) -> None:
    """Tight-layout and write a figure, then close it."""
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _load_rows_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Parse a results.csv into typed rows for the plotting functions."""
    numeric = {
        "horizon",
        "scale",
        "vocab_size",
        "num_bins",
        "alphabet_size",
        "mean_token_len",
        "bits_per_chunk",
        "bits_per_step",
        "rmse_continuous",
        "rmse_ee_pos_action",
        "mae_ee_pos_action",
        "rmse_ee_ori_action",
        "mae_ee_ori_action",
        "gripper_mismatch_rate",
    }
    boolean = {"is_operating_point", "feasible"}
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="") as csv_file:
        for raw in csv.DictReader(csv_file):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in boolean:
                    row[key] = value == "True"
                elif key in numeric and value not in ("", None):
                    row[key] = float(value)
                else:
                    row[key] = value
            rows.append(row)
    return rows


if __name__ == "__main__":
    results_csv = Path(sys.argv[1])
    figures_dir = Path(sys.argv[2])
    figures_dir.mkdir(parents=True, exist_ok=True)
    chosen = plot_all(rows=_load_rows_from_csv(results_csv), output_dir=figures_dir)
    print(f"colors={chosen}")
    print(f"Wrote figures to {figures_dir}")
