"""Reconstruction summary table + FAST-comparable compression ratio (H=10).

Self-contained (csv only, no versatil/torch import); runs from a floor-study
``results.csv`` in seconds. Emits:

- ``reconstruction_table.{csv,tex}`` — per config: Position MAE (mm), Rotation MAE
  (deg), tokens/chunk, bits/step, and the normalized-unit RMSE.
- ``compression_ratio.{csv,txt}`` — tokens/chunk ratio of Binning over FAST at
  matched reconstruction error (FAST tokens interpolated on its RMSE->tokens curve).

Physical units use LIBERO's robosuite OSC_POSE scaling (``osc_pose.json``:
``output_max = [0.05 m x3, 0.5 rad x3]``, ``control_delta=true``): a normalized
action of 1.0 maps to 0.05 m / 0.5 rad per control step. This assumes the default
OSC_POSE controller LIBERO loads; state it in the paper.

    python src/versatil/analysis/rate_distortion/summary_table.py results.csv out_dir
"""

import csv
import math
import sys
from pathlib import Path
from typing import Any

POS_OUTPUT_MAX_M = 0.05
ROT_OUTPUT_MAX_RAD = 0.5
POS_MM_PER_UNIT = POS_OUTPUT_MAX_M * 1000.0
ROT_DEG_PER_UNIT = ROT_OUTPUT_MAX_RAD * 180.0 / math.pi
FULL_SWEEP_HORIZON = 10
_NUMERIC = {
    "horizon",
    "scale",
    "num_bins",
    "mean_token_len",
    "bits_per_step",
    "rmse_continuous",
    "mae_ee_pos_action",
    "mae_ee_ori_action",
}


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Parse results.csv into typed rows (numeric fields as float, else str/bool)."""
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="") as csv_file:
        for raw in csv.DictReader(csv_file):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in {"is_operating_point", "feasible"}:
                    row[key] = value == "True"
                elif key in _NUMERIC and value not in ("", None):
                    row[key] = float(value)
                else:
                    row[key] = value
            rows.append(row)
    return rows


TRAINING_STRATEGY = "uniform"


def _knob_label(row: dict[str, Any]) -> str:
    """Human label for a config's knob (FAST scale / binning bin count)."""
    if row["family"] == "fast":
        return f"scale={row['scale']:g}"
    return f"bins={int(row['num_bins'])}"


def _variant_label(row: dict[str, Any]) -> str:
    """Tokenizer variant name: FAST, or binning tagged with its edge strategy."""
    if row["family"] != "binning":
        return row["family"]
    return f"binning ({row.get('binning_strategy') or TRAINING_STRATEGY})"


def _table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the H=10 FAST scale sweep and binning sweep rows for the table."""
    selected = [
        row
        for row in rows
        if row.get("feasible")
        and row.get("horizon") == FULL_SWEEP_HORIZON
        and row.get("sweep") in ("scale", "bins")
    ]
    order = {"fast": 0, f"binning ({TRAINING_STRATEGY})": 1}

    def sort_key(row: dict[str, Any]) -> tuple[int, float]:
        knob = row.get("scale") if row["family"] == "fast" else row.get("num_bins")
        return (order.get(_variant_label(row), 8), knob if knob is not None else 0.0)

    return sorted(selected, key=sort_key)


def write_reconstruction_table(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Write the per-config reconstruction table as CSV and LaTeX."""
    header = [
        "tokenizer",
        "config",
        "binning_strategy",
        "pos_mae_mm",
        "rot_mae_deg",
        "tokens_per_chunk",
        "bits_per_step",
        "rmse_norm",
        "operating_point",
    ]
    records = []
    for row in _table_rows(rows):
        records.append(
            {
                "tokenizer": _variant_label(row),
                "config": _knob_label(row),
                "binning_strategy": row.get("binning_strategy", ""),
                "pos_mae_mm": row["mae_ee_pos_action"] * POS_MM_PER_UNIT,
                "rot_mae_deg": row["mae_ee_ori_action"] * ROT_DEG_PER_UNIT,
                "tokens_per_chunk": row["mean_token_len"],
                "bits_per_step": row["bits_per_step"],
                "rmse_norm": row["rmse_continuous"],
                "operating_point": bool(row.get("is_operating_point")),
            }
        )

    with open(output_dir / "reconstruction_table.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    lines = [
        "% Reconstruction floor (libero_all, H=10, chunk=1 s @ 10 fps).",
        "% Pos/Rot MAE via default OSC_POSE scaling (0.05 m / 0.5 rad per unit).",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Tokenizer & Config & Pos MAE (mm) & Rot MAE ($^\\circ$) & "
        "Tokens/chunk & bits/step & RMSE (norm) \\\\",
        "\\midrule",
    ]
    for record in records:
        mark = "$^\\dagger$" if record["operating_point"] else ""
        lines.append(
            f"{record['tokenizer'].upper()}{mark} & {record['config']} & "
            f"{record['pos_mae_mm']:.2f} & {record['rot_mae_deg']:.3f} & "
            f"{record['tokens_per_chunk']:.1f} & {record['bits_per_step']:.1f} & "
            f"{record['rmse_norm']:.4f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (output_dir / "reconstruction_table.tex").write_text("\n".join(lines) + "\n")


def _fast_frontier(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """FAST (rmse, tokens) points on the efficient frontier at H=10, sorted by rmse."""
    points = [
        (row["rmse_continuous"], row["mean_token_len"])
        for row in rows
        if row.get("feasible")
        and row["family"] == "fast"
        and row.get("sweep") == "scale"
        and row.get("horizon") == FULL_SWEEP_HORIZON
    ]
    frontier = [
        point
        for point in points
        if not any(
            other[0] <= point[0] and other[1] <= point[1] and other != point
            for other in points
        )
    ]
    return sorted(frontier)


def _interp_tokens(rmse: float, frontier: list[tuple[float, float]]) -> float | None:
    """Linear-interpolate FAST tokens at a target rmse; None if outside the range."""
    if not frontier or rmse < frontier[0][0] or rmse > frontier[-1][0]:
        return None
    for (rmse_low, tokens_low), (rmse_high, tokens_high) in zip(
        frontier, frontier[1:], strict=False
    ):
        if rmse_low <= rmse <= rmse_high:
            span = rmse_high - rmse_low
            weight = 0.0 if span == 0 else (rmse - rmse_low) / span
            return tokens_low + weight * (tokens_high - tokens_low)
    return None


def write_compression_ratio(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Binning/FAST tokens-per-chunk ratio at each binning bin's reconstruction error."""
    frontier = _fast_frontier(rows)
    binning = sorted(
        (
            row
            for row in rows
            if row.get("feasible")
            and row["family"] == "binning"
            and row.get("horizon") == FULL_SWEEP_HORIZON
            and (row.get("binning_strategy") or TRAINING_STRATEGY) == TRAINING_STRATEGY
        ),
        key=lambda r: r["num_bins"],
    )
    header = [
        "bins",
        "binning_rmse",
        "binning_tokens",
        "fast_tokens_at_matched_rmse",
        "ratio",
    ]
    records = []
    for row in binning:
        fast_tokens = _interp_tokens(row["rmse_continuous"], frontier)
        records.append(
            {
                "bins": int(row["num_bins"]),
                "binning_rmse": row["rmse_continuous"],
                "binning_tokens": row["mean_token_len"],
                "fast_tokens_at_matched_rmse": fast_tokens,
                "ratio": (row["mean_token_len"] / fast_tokens) if fast_tokens else None,
            }
        )

    with open(output_dir / "compression_ratio.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {key: ("" if value is None else value) for key, value in record.items()}
            )

    text = [
        "FAST vs Binning compression at matched reconstruction "
        f"(libero_all, H=10, binning variant = {TRAINING_STRATEGY}, "
        "i.e. the variant the trained policies and the replay study use).",
        "Ratio = binning tokens/chunk / FAST tokens/chunk at equal RMSE "
        "(FAST tokens interpolated on its RMSE->tokens frontier).",
        "",
    ]
    for record in records:
        if record["ratio"] is None:
            text.append(
                f"  bins={record['bins']:<4d} rmse={record['binning_rmse']:.4f}  "
                "(FAST cannot reach this fidelity in the scale sweep; out of range)"
            )
        else:
            text.append(
                f"  bins={record['bins']:<4d} rmse={record['binning_rmse']:.4f}  "
                f"FAST~{record['fast_tokens_at_matched_rmse']:.1f} tok  "
                f"ratio={record['ratio']:.1f}x"
            )
    (output_dir / "compression_ratio.txt").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    results_csv = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed = load_rows(results_csv)
    write_reconstruction_table(parsed, out_dir)
    write_compression_ratio(parsed, out_dir)
    print(f"Wrote reconstruction_table.* and compression_ratio.* to {out_dir}")
