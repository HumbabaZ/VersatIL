"""Figures for the tokenizer floor study (FAST vs Binning, horizon effect).

Publication figures for the thesis. Style follows the project's visualization
config: the ``scienceplots`` toolkit and the qualitative ``tab10`` palette (``tab20``
when a figure needs more than ten series), assigned in fixed order. Axes carry
Title-Case labels with units and a light dashed grid; FAST's dominated
(Pareto-inferior) points are hollow off the frontier line and called out in the
legend.

A *series* is a tokenizer variant: FAST, plus one curve per binning strategy.
``uniform`` binning is the variant the trained policies and the open-loop replay
study use (OpenVLA-style, fixed [-1, 1] support), so it is the primary binning
curve; ``quantile`` binning places data-adaptive edges and is reported as the
stronger reference baseline.

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

SERIES_ORDER = ("fast", "binning:uniform", "binning:quantile")
SERIES_LABEL = {
    "fast": "FAST",
    "binning:uniform": "Binning (uniform)",
    "binning:quantile": "Binning (quantile)",
}
SERIES_MARKER = {"fast": "o", "binning:uniform": "s", "binning:quantile": "^"}
SERIES_LINESTYLE = {"fast": "-", "binning:uniform": "--", "binning:quantile": ":"}
FAMILY_KNOB_KEY = {"fast": "scale", "binning": "num_bins"}
FAMILY_KNOB_PREFIX = {"fast": "s", "binning": "b"}
PRIMARY_SERIES = ("fast", "binning:uniform")
# Bin count used as each binning variant's representative point in horizon plots.
REPRESENTATIVE_BINS = 256
DOMINATED_LABEL = "Dominated (off Frontier)"
FULL_SWEEP_HORIZON = 10


def choose_colors(count: int) -> list:
    """Return ``count`` qualitative colors in fixed order (tab10, else tab20)."""
    colormap = plt.get_cmap("tab10" if count <= _TAB10_SIZE else "tab20")
    return [colormap(index) for index in range(count)]


def series_key(row: dict[str, Any]) -> str:
    """Series identifier for a result row (``fast`` or ``binning:<strategy>``)."""
    if row["family"] != "binning":
        return row["family"]
    return f"binning:{row.get('binning_strategy') or 'uniform'}"


def _family_of(series: str) -> str:
    """Tokenizer family behind a series key."""
    return series.split(":", maxsplit=1)[0]


def plot_all(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, dict]:
    """Write all floor-study figures; return the colors assigned to each series."""
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
    present = [s for s in SERIES_ORDER if any(series_key(r) == s for r in rows)]
    series_colors = dict(zip(present, choose_colors(len(present)), strict=True))

    plot_frontier_by_horizon(rows, output_dir / "figure_a_frontier.png", horizon_colors)
    plot_series_frontier(
        rows,
        output_dir / "figure_a_frontier_h10.png",
        series_colors,
        x_key="bits_per_step",
        x_label="Rate (Bits per Timestep)",
        title="Tokenizer Rate-Distortion Frontier",
    )
    plot_series_frontier(
        rows,
        output_dir / "figure_tokens_frontier.png",
        series_colors,
        x_key="mean_token_len",
        x_label="Sequence Length (Tokens per Chunk)",
        title="Tokenizer Compression-Distortion Frontier",
        thin_binning=True,
    )
    plot_horizon_effect(rows, output_dir / "figure_horizon.png", series_colors)
    plot_vocab_flat(rows, output_dir / "figure_vocab_flat.png", series_colors)
    return {
        "series": {s: to_hex(c) for s, c in series_colors.items()},
        "horizons": {h: to_hex(c) for h, c in horizon_colors.items()},
    }


def plot_frontier_by_horizon(
    rows: list[dict[str, Any]], output_path: Path, horizon_colors: dict
) -> None:
    """Rate-distortion frontier, one colour per chunk horizon.

    Restricted to the primary series (FAST and uniform binning) so that horizon,
    not tokenizer variant, is the visual dimension. Marker/line-style encode the
    series; H = 10 additionally carries knob annotations.
    """
    if not horizon_colors:
        return
    figure, axis = plt.subplots(figsize=(6.8, 4.6))
    for horizon in sorted(horizon_colors):
        color = horizon_colors[horizon]
        for series in PRIMARY_SERIES:
            points = _series_points(rows, series, "bits_per_step", horizon)
            if not points:
                continue
            _draw_line(axis, series, points, color)
            if horizon == FULL_SWEEP_HORIZON:
                _annotate(axis, series, points, color)
    _style_axis(axis)
    axis.set_xlabel("Rate (Bits per Timestep)")
    axis.set_ylabel("Reconstruction RMSE (Normalized Action Units)")
    axis.set_title("Tokenizer Rate-Distortion Frontier")
    handles = [
        Line2D([], [], color=horizon_colors[h], lw=2, label=f"H = {h}")
        for h in sorted(horizon_colors)
    ]
    handles += [_series_handle(s, "0.35") for s in PRIMARY_SERIES]
    handles.append(_dominated_handle())
    axis.legend(handles=handles, fontsize=8)
    _save(figure, output_path)


def plot_series_frontier(
    rows: list[dict[str, Any]],
    output_path: Path,
    series_colors: dict,
    x_key: str,
    x_label: str,
    title: str,
    thin_binning: bool = False,
) -> None:
    """Single-horizon frontier with one colour per tokenizer variant.

    ``x_key`` selects the horizontal axis: ``bits_per_step`` (rate) or
    ``mean_token_len`` (sequence length). ``thin_binning`` drops crowded binning
    points so their bin-count labels stay legible on the fixed-length verticals.
    """
    horizon = FULL_SWEEP_HORIZON
    drawn = {}
    for series in SERIES_ORDER:
        if series not in series_colors:
            continue
        points = _series_points(rows, series, x_key, horizon)
        if points:
            drawn[series] = points
    if not drawn:
        return

    all_y = [p[1] for points in drawn.values() for p in points]
    min_gap = 0.09 * (max(all_y) - min(all_y))
    figure, axis = plt.subplots(figsize=(6.6, 4.4))
    for series, points in drawn.items():
        color = series_colors[series]
        if thin_binning and _family_of(series) == "binning":
            points = _thin_by_gap(sorted(points, key=lambda p: p[1]), min_gap)
        _draw_line(axis, series, points, color)
        _annotate(axis, series, points, color)
    _style_axis(axis)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Reconstruction RMSE (Normalized Action Units)")
    axis.set_title(f"{title} (H = {horizon})")
    handles = [_series_handle(s, series_colors[s]) for s in drawn]
    handles.append(_dominated_handle())
    axis.legend(handles=handles, title="Tokenizer", fontsize=8)
    _save(figure, output_path)


def plot_horizon_effect(
    rows: list[dict[str, Any]], output_path: Path, series_colors: dict
) -> None:
    """Horizon effect at each series' representative configuration.

    The series sit at their own operating points (FAST scale=10/|V|=1024, binning
    256 bins) and therefore at very different rates, so *levels are not
    comparable across series here*; only the trend in H is. The left panel
    consequently plots distortion **relative to each series' own H = 10 value**,
    which makes horizon-independence directly readable and structurally prevents
    a cross-series level comparison; absolute values appear in the legend and in
    the reconstruction table. Cross-tokenizer comparison belongs to the
    rate-distortion frontier, where rate is on the axis.

    The right panel shows rate vs H. The two binning curves coincide exactly,
    because ``D log2 B`` does not depend on the binning variant.
    """

    def operating(row: dict[str, Any]) -> bool:
        if row["family"] == "binning":
            return int(row.get("num_bins") or 0) == REPRESENTATIVE_BINS
        return bool(row.get("is_operating_point"))

    present = [
        series
        for series in SERIES_ORDER
        if series in series_colors
        and any(
            series_key(r) == series and operating(r) and r.get("feasible") for r in rows
        )
    ]
    if not present:
        return
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    for series in present:
        color = series_colors[series]
        for axis, key in ((left, "rmse_continuous"), (right, "bits_per_step")):
            points = _labeled_points(
                rows,
                "horizon",
                key,
                "horizon",
                lambda r, s=series: series_key(r) == s and operating(r),
            )
            if not points:
                continue
            points.sort()
            values = [p[1] for p in points]
            if axis is left:
                # Index to this series' own H = 10 value: the series sit at very
                # different rates, so only the trend in H is comparable.
                reference = _reference_value(points, FULL_SWEEP_HORIZON)
                label = f"{SERIES_LABEL[series]} (H=10: {reference:.4f})"
                values = [value / reference for value in values]
            else:
                label = SERIES_LABEL[series]
            axis.plot(
                [p[0] for p in points],
                values,
                marker=SERIES_MARKER[series],
                linestyle=SERIES_LINESTYLE[series],
                color=color,
                label=label,
            )
    left.axhline(1.0, color="0.6", linewidth=0.8, zorder=1)
    left.set_ylim(0.8, 1.2)
    left.set_xlabel("Chunk Horizon H")
    left.set_ylabel("Reconstruction RMSE, Relative to H = 10")
    left.set_title("Reconstruction Floor vs Horizon")
    right.set_xlabel("Chunk Horizon H")
    right.set_ylabel("Rate (Bits per Timestep)")
    right.set_title("Rate vs Horizon")
    for axis in (left, right):
        _style_axis(axis)
        axis.legend(title="Tokenizer", fontsize=7)
    _save(figure, output_path)


def _reference_value(points: list[tuple[float, float, float]], horizon: int) -> float:
    """Value at ``horizon`` if present, else the first point's value."""
    for x_value, y_value, _ in points:
        if int(x_value) == horizon:
            return y_value
    return points[0][1]


def plot_vocab_flat(
    rows: list[dict[str, Any]], output_path: Path, series_colors: dict
) -> None:
    """|V| sweep: distortion must be flat as rate moves (lossless rate knob)."""
    if "fast" not in series_colors:
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
    color = series_colors["fast"]
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


def _series_points(
    rows: list[dict[str, Any]], series: str, x_key: str, horizon: int
) -> list[tuple[float, float, float]]:
    """Frontier points (x, rmse, knob) of one series at one horizon."""
    family = _family_of(series)
    sweeps = ("scale", "horizon") if family == "fast" else ("bins",)
    return _labeled_points(
        rows,
        x_key,
        "rmse_continuous",
        FAMILY_KNOB_KEY[family],
        lambda r, s=series, sw=sweeps, h=horizon: (
            series_key(r) == s and r.get("sweep") in sw and r.get("horizon") == h
        ),
    )


def _draw_line(
    axis, series: str, points: list[tuple[float, float, float]], color
) -> None:
    """Draw a series' frontier line; dominated points are hollow and off the line."""
    frontier, dominated = _pareto_split(points)
    frontier_sorted = sorted(frontier)
    axis.plot(
        [p[0] for p in frontier_sorted],
        [p[1] for p in frontier_sorted],
        marker=SERIES_MARKER[series],
        linestyle=SERIES_LINESTYLE[series],
        color=color,
        zorder=3,
    )
    for point in dominated:
        axis.scatter(
            [point[0]],
            [point[1]],
            marker=SERIES_MARKER[series],
            facecolors="none",
            edgecolors=color,
            zorder=3,
        )


def _annotate(
    axis, series: str, points: list[tuple[float, float, float]], color
) -> None:
    """Annotate each point with its knob value."""
    prefix = FAMILY_KNOB_PREFIX[_family_of(series)]
    is_fast = _family_of(series) == "fast"
    for x_value, y_value, knob in points:
        text = f"{prefix}={knob:g}" if is_fast else f"{prefix}={int(knob)}"
        axis.annotate(
            text,
            (x_value, y_value),
            color=color,
            fontsize=8,
            xytext=(4, 3),
            textcoords="offset points",
        )


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


def _series_handle(series: str, color) -> Line2D:
    """Legend handle showing a series' marker and line style."""
    return Line2D(
        [],
        [],
        color=color,
        marker=SERIES_MARKER[series],
        linestyle=SERIES_LINESTYLE[series],
        label=SERIES_LABEL[series],
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
