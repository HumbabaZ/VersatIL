"""Collect Tip 1 training results from the sweep's SLURM logs.

Each array element prints its cell name and then one ``Synthetic rollout:`` line
per evaluation epoch. That log is the only place a run's success rate is written
where a later analysis can reach it without a network call, so this module reads
the logs rather than the wandb project: a collected result stays reproducible
from the files the job itself left behind.

Deliberately imports nothing from ``versatil``. Importing the package pulls in
torch and transformers, which takes minutes on a login node, and none of it is
needed to read text.

    python src/versatil/analysis/tip1_noise/collect.py logs outputs/tip1_noise
"""

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

CELL_PATTERN = re.compile(r"^\[\d+/\d+\]\s+(\S+)\s*$")
# valid_entropy is logged as -0.00 once the policy collapses to a single mode, so
# every numeric field allows a leading minus. Without it the final evaluation is
# silently skipped and the run is scored by an earlier, positive-entropy epoch --
# usually epoch 0, which reads as success 0.00. The context fields exist only on
# conditional runs, so they are an optional tail and older logs still parse.
ROLLOUT_PATTERN = re.compile(
    r"Synthetic rollout: epoch (?P<epoch>\d+), "
    r"success=(?P<success>-?[\d.]+), "
    r"collision=(?P<collision>-?[\d.]+), "
    r"endpoint_reach=(?P<endpoint_reach>-?[\d.]+), "
    r"path_length=(?P<path_length>-?[\d.]+), "
    r"valid_mode_coverage=(?P<valid_mode_coverage>-?[\d.]+), "
    r"valid_entropy=(?P<valid_entropy>-?[\d.]+)"
    r"(?:.*?context_accuracy=(?P<context_accuracy>-?[\d.]+), "
    r"conditional_success=(?P<conditional_success>-?[\d.]+))?"
)
NAME_PATTERN = re.compile(
    r"^(?P<task>[^_]+)__inj-(?P<injection>[^_]+)__band-(?P<band>[^_]+)"
    r"__sig-(?P<sigma_multiplier>[\d.]+)__dseed-(?P<data_seed>\d+)"
    r"(?:__ep-(?P<num_episodes>\d+))?"
    r"__(?P<method>[^_]+)__seed-(?P<train_seed>\d+)$"
)
FAILURE_MARKERS = ("Traceback", "CANCELLED", "Out of memory", "CUDA out of memory")


@dataclass
class RunResult:
    """One training run as recovered from its log."""

    cell: str
    log: str
    evaluations: list[dict[str, float]]
    finished: bool
    failed: bool


def parse_log(path: Path) -> RunResult | None:
    """Read one array element's log.

    Returns:
        The run's cell name and evaluation history, or None when the log has no
        cell line yet, which is how a job that died during import looks.
    """
    cell = None
    evaluations: list[dict[str, float]] = []
    finished = False
    failed = False
    for line in path.read_text(errors="replace").splitlines():
        if cell is None:
            match = CELL_PATTERN.match(line)
            if match:
                cell = match.group(1)
                continue
        rollout = ROLLOUT_PATTERN.search(line)
        if rollout:
            evaluations.append(
                {
                    key: float(value)
                    for key, value in rollout.groupdict().items()
                    if value is not None
                }
            )
            continue
        if line.startswith("Finished at:"):
            finished = True
        if any(marker in line for marker in FAILURE_MARKERS):
            failed = True
    if cell is None:
        return None
    return RunResult(
        cell=cell,
        log=path.name,
        evaluations=evaluations,
        finished=finished,
        failed=failed,
    )


def summarize(result: RunResult) -> dict[str, float | str | int]:
    """Flatten one run into a manifest row.

    Both the last and the best evaluation are reported. The last is what a
    fixed-epoch protocol commits to; the best is carried alongside so that a run
    which peaked and then degraded is visible instead of being silently
    represented by its worst moment.

    Raises:
        ValueError: If the cell name does not parse, since an unparsed name
            means the row cannot be placed on the noise grid.
    """
    fields = NAME_PATTERN.match(result.cell)
    if not fields:
        raise ValueError(
            f"Cell name '{result.cell}' from {result.log} does not match the "
            "sweep's naming scheme. A row that cannot be placed on the noise "
            "grid is worse than a missing one."
        )
    row: dict[str, float | str | int] = {
        "cell": result.cell,
        "task": fields.group("task"),
        "injection": fields.group("injection"),
        "band": fields.group("band"),
        "sigma_multiplier": float(fields.group("sigma_multiplier")),
        "data_seed": int(fields.group("data_seed")),
        "method": fields.group("method"),
        "train_seed": int(fields.group("train_seed")),
        "log": result.log,
        "num_evaluations": len(result.evaluations),
        "finished": result.finished,
        "failed": result.failed,
    }
    if not result.evaluations:
        row.update(
            {
                "final_epoch": "",
                "final_success": "",
                "best_success": "",
                "final_collision": "",
                "final_endpoint_reach": "",
                "final_valid_mode_coverage": "",
                "final_context_accuracy": "",
                "final_conditional_success": "",
            }
        )
        return row
    last = result.evaluations[-1]
    row.update(
        {
            "final_epoch": int(last["epoch"]),
            "final_success": last["success"],
            "best_success": max(item["success"] for item in result.evaluations),
            "final_collision": last["collision"],
            "final_endpoint_reach": last["endpoint_reach"],
            "final_valid_mode_coverage": last["valid_mode_coverage"],
            # Conditional runs only: success on the route the context asked for.
            "final_context_accuracy": last.get("context_accuracy", ""),
            "final_conditional_success": last.get("conditional_success", ""),
        }
    )
    return row


def collect(log_dir: Path, pattern: str) -> list[dict[str, float | str | int]]:
    """Summarize every training log matching the pattern, newest run per cell.

    A cell trained more than once keeps the most recently modified log, so a
    rerun after a failure supersedes the failure instead of appearing beside it.
    """
    by_cell: dict[str, tuple[float, RunResult]] = {}
    for path in sorted(log_dir.glob(pattern)):
        result = parse_log(path)
        if result is None:
            continue
        stamp = path.stat().st_mtime
        previous = by_cell.get(result.cell)
        if previous is None or stamp > previous[0]:
            by_cell[result.cell] = (stamp, result)
    rows = [summarize(result) for _, result in by_cell.values()]
    rows.sort(
        key=lambda row: (
            str(row["task"]),
            str(row["method"]),
            float(row["sigma_multiplier"]),
            int(row["train_seed"]),
        )
    )
    return rows


def report(rows: list[dict[str, float | str | int]]) -> None:
    """Print the completion state and the success grid to stdout."""
    missing = [row for row in rows if row["final_success"] == ""]
    failed = [row for row in rows if row["failed"]]
    unfinished = [row for row in rows if not row["finished"] and not row["failed"]]
    print(
        f"{len(rows)} runs found: {len(rows) - len(missing)} with results, "
        f"{len(failed)} failed, {len(unfinished)} still running or interrupted"
    )
    for row in failed:
        print(f"  FAILED   {row['cell']}  ({row['log']})")
    for row in unfinished:
        print(f"  RUNNING  {row['cell']}  ({row['num_evaluations']} evals so far)")

    graded = [row for row in rows if row["final_success"] != ""]
    if not graded:
        return
    print(f"\n{'task':<11}{'method':<9}{'sigma':>7}{'seed':>6}{'final':>8}{'best':>8}")
    for row in graded:
        print(
            f"{str(row['task']):<11}{str(row['method']):<9}"
            f"{float(row['sigma_multiplier']):>7.1f}{int(row['train_seed']):>6}"
            f"{float(row['final_success']):>8.2f}{float(row['best_success']):>8.2f}"
        )


def main() -> None:
    """Parse arguments, collect the logs and write the results manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", help="Directory holding the sbatch logs.")
    parser.add_argument("output_dir", help="Where to write results_<stage>.csv.")
    parser.add_argument("--stage", default="main")
    parser.add_argument(
        "--pattern",
        default="tip1_train_*.log",
        help="Glob selecting the logs to read.",
    )
    arguments = parser.parse_args()

    rows = collect(Path(arguments.log_dir), arguments.pattern)
    if not rows:
        print(f"No parsable logs matched {arguments.pattern} in {arguments.log_dir}")
        return

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / f"results_{arguments.stage}.csv"
    with open(manifest, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report(rows)
    print(f"\nWrote {manifest}")


if __name__ == "__main__":
    main()
