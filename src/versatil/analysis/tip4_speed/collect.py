"""Aggregate the Tip 4 per-chunk latency rows into the E.5 summary table.

Reads the per-chunk CSV the benchmark appended, aggregates per grid point
(method x sigma x trajectory length x device), and joins the rollout success
already collected by Tip 1 so the accuracy-latency pairing needs no new
rollouts.

Aggregation protocol:

- Within a cell (one checkpoint), the per-chunk distribution is summarized by
  the median (robust to stragglers) and the p95.
- Across the seed replicates of a grid point, the cell medians are averaged
  and their standard deviation is the error bar.
- ``t_generate_share`` and ``t_detok_share`` are computed per chunk and then
  summarized the same way; the generate share is the quantity Tip 6's scaling
  prediction is about, the detok share tests the < 1 percent hypothesis.
- ``control_rate_hz`` is the executed actions per chunk divided by the
  end-to-end chunk latency: the ceiling on the closed-loop control frequency
  this method could sustain on this hardware.

Usage:

    python -m versatil.analysis.tip4_speed.collect \
        <speed_csv> <output_dir> [--tip1-results <results_all.csv>]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

TOKENIZED_METHODS = ("fast", "binned")

SUMMARY_FIELDS = (
    "device",
    "method",
    "sigma_multiplier",
    "trajectory_length",
    "num_cells",
    "num_chunks",
    "steps_mean",
    "steps_min",
    "steps_max",
    "t_pre_ms",
    "t_encode_ms",
    "t_generate_ms",
    "t_detok_ms",
    "t_unnorm_ms",
    "t_total_ms_mean",
    "t_total_ms_sd",
    "t_total_ms_p95",
    "t_generate_share",
    "t_detok_share",
    "control_rate_hz",
    "detok_warnings_total",
    "chunk_cv",
    "success_mean",
    "success_sd",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into dict rows."""
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def executed_steps(method: str, trajectory_length: int) -> int:
    """Actions executed per chunk: the method's prediction horizon."""
    if method in TOKENIZED_METHODS:
        return trajectory_length - 1
    return trajectory_length


def _cell_summary(rows: list[dict[str, str]]) -> dict[str, float]:
    """Per-cell summary: medians of segments, shares, p95, chunk CV."""
    totals = [float(row["t_total_ms"]) for row in rows]
    summary: dict[str, float] = {}
    for segment in (
        "t_pre_ms",
        "t_encode_ms",
        "t_generate_ms",
        "t_detok_ms",
        "t_unnorm_ms",
    ):
        summary[segment] = median(float(row[segment]) for row in rows)
    summary["t_total_ms"] = median(totals)
    summary["t_total_ms_p95"] = sorted(totals)[max(0, int(0.95 * len(totals)) - 1)]
    summary["t_generate_share"] = median(
        float(row["t_generate_ms"]) / float(row["t_total_ms"]) for row in rows
    )
    summary["t_detok_share"] = median(
        float(row["t_detok_ms"]) / float(row["t_total_ms"]) for row in rows
    )
    summary["steps_mean"] = mean(float(row["sequential_steps"]) for row in rows)
    summary["steps_min"] = min(float(row["sequential_steps"]) for row in rows)
    summary["steps_max"] = max(float(row["sequential_steps"]) for row in rows)
    summary["detok_warnings"] = sum(int(row["detok_warnings"]) for row in rows)
    summary["chunk_cv"] = (stdev(totals) / mean(totals)) if len(totals) > 1 else 0.0
    summary["num_chunks"] = len(rows)
    return summary


def read_tip1_success(path: Path) -> dict[str, float]:
    """Map Tip 1 cell name to its final conditional success."""
    success: dict[str, float] = {}
    for row in read_rows(path):
        value = row.get("final_conditional_success") or row.get("final_success")
        if value not in (None, ""):
            success[row["cell"]] = float(value)
    return success


def collect(
    speed_csv: Path,
    output_dir: Path,
    tip1_results: Path | None,
) -> Path:
    """Write the per-grid-point summary CSV and return its path."""
    rows = read_rows(speed_csv)
    if not rows:
        raise ValueError(f"No rows in {speed_csv}.")
    success_by_cell = read_tip1_success(tip1_results) if tip1_results else {}

    by_cell: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_cell[row["cell"]].append(row)

    grid: dict[tuple[str, str, float, int], list[tuple[str, dict[str, float]]]] = (
        defaultdict(list)
    )
    for cell, cell_rows in by_cell.items():
        first = cell_rows[0]
        key = (
            first["device"],
            first["method"],
            float(first["sigma_multiplier"]),
            int(first["trajectory_length"]),
        )
        grid[key].append((cell, _cell_summary(cell_rows)))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"summary_speed_{rows[0]['device']}.csv"
    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        for (device, method, sigma, length), cells in sorted(grid.items()):
            summaries = [summary for _, summary in cells]
            totals = [summary["t_total_ms"] for summary in summaries]
            successes = [
                success_by_cell[cell] for cell, _ in cells if cell in success_by_cell
            ]
            total_mean = mean(totals)
            writer.writerow(
                {
                    "device": device,
                    "method": method,
                    "sigma_multiplier": sigma,
                    "trajectory_length": length,
                    "num_cells": len(cells),
                    "num_chunks": sum(int(s["num_chunks"]) for s in summaries),
                    "steps_mean": round(mean(s["steps_mean"] for s in summaries), 2),
                    "steps_min": min(s["steps_min"] for s in summaries),
                    "steps_max": max(s["steps_max"] for s in summaries),
                    "t_pre_ms": round(mean(s["t_pre_ms"] for s in summaries), 4),
                    "t_encode_ms": round(mean(s["t_encode_ms"] for s in summaries), 4),
                    "t_generate_ms": round(
                        mean(s["t_generate_ms"] for s in summaries), 4
                    ),
                    "t_detok_ms": round(mean(s["t_detok_ms"] for s in summaries), 4),
                    "t_unnorm_ms": round(mean(s["t_unnorm_ms"] for s in summaries), 4),
                    "t_total_ms_mean": round(total_mean, 4),
                    "t_total_ms_sd": round(stdev(totals), 4)
                    if len(totals) > 1
                    else 0.0,
                    "t_total_ms_p95": round(
                        max(s["t_total_ms_p95"] for s in summaries), 4
                    ),
                    "t_generate_share": round(
                        mean(s["t_generate_share"] for s in summaries), 4
                    ),
                    "t_detok_share": round(
                        mean(s["t_detok_share"] for s in summaries), 6
                    ),
                    "control_rate_hz": round(
                        executed_steps(method=method, trajectory_length=length)
                        / (total_mean / 1000.0),
                        1,
                    ),
                    "detok_warnings_total": sum(
                        int(s["detok_warnings"]) for s in summaries
                    ),
                    "chunk_cv": round(max(s["chunk_cv"] for s in summaries), 4),
                    "success_mean": round(mean(successes), 4) if successes else None,
                    "success_sd": round(stdev(successes), 4)
                    if len(successes) > 1
                    else None,
                }
            )
    return output_csv


def _main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Tip 4 latency rows.")
    parser.add_argument("speed_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--tip1-results",
        type=Path,
        default=None,
        help="Tip 1 results_all.csv for the accuracy-latency join.",
    )
    arguments = parser.parse_args()
    output_csv = collect(
        speed_csv=arguments.speed_csv,
        output_dir=arguments.output_dir,
        tip1_results=arguments.tip1_results,
    )
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    _main()
