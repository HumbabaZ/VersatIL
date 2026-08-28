"""Print the Tip 2 pilot gate reading for a collect_eval manifest.

The gate decides whether a granularity sweep is worth its replicates. Per
tokenizer family it reports, coarse to fine: the stochastic total with its
standard error, whether the point is degenerate (collapsed token sequences),
whether it sits at the "stand still" level (informational), the mode-match
rate, the catastrophic-decode fraction and the rollout success; then whether
each pair of adjacent healthy points -- not collapsed and not failing the task
-- is resolved, i.e. their totals differ by more than twice the standard error
of the difference. Resolution is the power criterion: if neighbours are
indistinguishable at one seed, more replicates will not rescue a curve.
Resolution is necessary but not sufficient: with thousands of chunk draws a 5%
wiggle resolves statistically yet carries no granularity signal, so the spread
of the healthy totals (max/min - 1) is printed alongside as the effect size.

    python -m versatil.analysis.tip2_tokenization.gate_report \
        /data/horse/ws/qizh093f-versatil/tip2_results/tip2_eval_conditional_pilot.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from versatil.analysis.tip2_tokenization.plot_decomposition import (
    is_degenerate,
    load_family,
)

# A total within this relative band of mean(a^2) is flagged "at the stand-still
# level" for information only. A collapsed tokenizer lands there, but so can a
# healthy policy whose open-loop noise floor plus sampling variance happens to
# add up to mean(a^2) (the conditional-circle pilot did exactly that at rollout
# success 0.95), and a total *below* it is success, not degeneracy. Points are
# excluded from the optimum discussion only for collapsed token sequences or a
# failed task.
STAND_STILL_TOLERANCE = 0.1
# Adjacent totals are resolved when they differ by more than this many standard
# errors of their difference (about a 95% two-sided criterion).
RESOLUTION_SIGMAS = 2.0
# Points whose catastrophic-decode fraction is at or above this are in the
# sampling-explosion regime; they stay in the main curve but are left out of
# the second spread line so the granularity effect can be read without them.
CATASTROPHIC_LIMIT = 0.05
REQUIRED_COLUMNS = (
    "total_mse",
    "total_mse_se",
    "unique_gt_sequence_count",
    "expert_mean_square",
    "mode_match_rate",
    "catastrophic_fraction",
    "rollout_success",
    "identity_gap",
    "no_eos_rate",
)
FAMILIES = ("fast", "binning")


def is_at_stand_still(total_mse: float, expert_mean_square: float) -> bool:
    """Whether a total sits within the band around an all-zero prediction's MSE."""
    return abs(total_mse - expert_mean_square) <= (
        STAND_STILL_TOLERANCE * expert_mean_square
    )


def adjacent_resolution(
    totals: list[float], standard_errors: list[float]
) -> list[bool]:
    """Whether each adjacent pair of totals is separated beyond its noise.

    Args:
        totals: Stochastic totals ordered coarse to fine.
        standard_errors: Standard error of each total, same order.

    Returns:
        One flag per adjacent pair (length ``len(totals) - 1``).
    """
    flags = []
    for index in range(len(totals) - 1):
        difference = abs(totals[index + 1] - totals[index])
        noise = math.sqrt(standard_errors[index] ** 2 + standard_errors[index + 1] ** 2)
        flags.append(difference > RESOLUTION_SIGMAS * noise)
    return flags


def missing_columns(row: dict[str, str]) -> list[str]:
    """Required manifest columns absent from ``row``."""
    return [column for column in REQUIRED_COLUMNS if column not in row]


def family_report(rows: list[dict[str, str]]) -> list[str]:
    """Render one family's gate table and adjacent-pair resolution lines."""
    lines = [
        f"{'param':>10} {'total':>10} {'se':>9} {'degen':>5} {'stand':>5} "
        f"{'mode':>5} {'catas':>5} {'roll':>5}"
    ]
    healthy_totals: list[float] = []
    healthy_errors: list[float] = []
    healthy_params: list[float] = []
    healthy_catastrophic: list[float] = []
    for row in rows:
        total = float(row["total_mse"])
        error = float(row["total_mse_se"])
        degenerate = is_degenerate(row)
        stand_still = is_at_stand_still(
            total_mse=total, expert_mean_square=float(row["expert_mean_square"])
        )
        lines.append(
            f"{float(row['param']):>10.3f} {total:>10.3e} {error:>9.2e} "
            f"{'yes' if degenerate else '-':>5} {'yes' if stand_still else '-':>5} "
            f"{float(row['mode_match_rate']):>5.2f} "
            f"{float(row['catastrophic_fraction']):>5.2f} "
            f"{float(row['rollout_success']):>5.2f}"
        )
        task_failed = float(row["rollout_success"]) == 0.0
        if not degenerate and not task_failed:
            healthy_totals.append(total)
            healthy_errors.append(error)
            healthy_params.append(float(row["param"]))
            healthy_catastrophic.append(float(row["catastrophic_fraction"]))
    resolved = adjacent_resolution(
        totals=healthy_totals, standard_errors=healthy_errors
    )
    for index, flag in enumerate(resolved):
        lines.append(
            f"  {healthy_params[index]:.3f} -> {healthy_params[index + 1]:.3f}: "
            f"{'resolved' if flag else 'NOT resolved'}"
        )
    if resolved:
        lines.append(
            f"  resolved adjacent pairs: {sum(resolved)}/{len(resolved)} "
            f"(healthy points: {len(healthy_totals)})"
        )
        spread = max(healthy_totals) / min(healthy_totals) - 1.0
        lines.append(f"  healthy-point spread (max/min - 1): {spread:.1%}")
        calm = [
            total
            for total, fraction in zip(
                healthy_totals, healthy_catastrophic, strict=True
            )
            if fraction < CATASTROPHIC_LIMIT
        ]
        if len(calm) >= 2:
            calm_spread = max(calm) / min(calm) - 1.0
            lines.append(
                f"  spread among healthy points with catastrophic_fraction < "
                f"{CATASTROPHIC_LIMIT}: {calm_spread:.1%} ({len(calm)} points)"
            )
    else:
        lines.append("  fewer than two healthy points; nothing to resolve")
    return lines


def gate_report(csv_path: Path) -> str:
    """Build the full gate reading for a manifest.

    Raises:
        ValueError: If the manifest lacks a required column.
    """
    sections: list[str] = []
    all_rows: list[dict[str, str]] = []
    for family in FAMILIES:
        rows = load_family(csv_path=csv_path, method=family)
        if not rows:
            continue
        missing = missing_columns(rows[0])
        if missing:
            raise ValueError(f"{csv_path} lacks columns {missing}.")
        all_rows.extend(rows)
        sections.append(f"== {family} ==")
        sections.extend(family_report(rows=rows))
    identity = max(abs(float(row["identity_gap"])) for row in all_rows)
    no_eos = max(float(row["no_eos_rate"]) for row in all_rows)
    mode_match = min(float(row["mode_match_rate"]) for row in all_rows)
    sections.append("== checks ==")
    sections.append(f"max |identity_gap| = {identity:.2e} (want ~0)")
    sections.append(f"max no_eos_rate = {no_eos:.3f}")
    sections.append(
        f"min mode_match_rate = {mode_match:.2f} (observable-mode task: ~1)"
    )
    return "\n".join(sections)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Tip 2 pilot gate reading.")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    print(gate_report(csv_path=args.csv_path))


if __name__ == "__main__":
    _main()
