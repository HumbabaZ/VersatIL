"""Run the Tip 2 eval hook over a stage's trained checkpoints into a manifest.

Each cell is evaluated with the eval overrides derived from the same
``TrainCell`` that trained it, so the eval split, normalizer and tokenizer match
training. The argmax prediction-error decomposition and the diagnostics are
written one row per cell to a manifest CSV, appended as each cell finishes so a
mid-run failure keeps the rows already computed.

    python -m versatil.analysis.tip2_tokenization.collect_eval \
        --stage pilot --task sequential out_dir
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch

from versatil.analysis.tip2_tokenization.eval_hook import run_from_config_name
from versatil.analysis.tip2_tokenization.rollout_success import final_success_by_cell
from versatil.analysis.tip2_tokenization.sweep import (
    STAGES,
    STORES,
    TrainCell,
    stage_cells,
)

CHECKPOINT_DIR_ENV = "VERSATIL_CHECKPOINT_DIR"
CHECKPOINT_SUBDIR = "synthetic"
# src/versatil/analysis/tip2_tokenization/collect_eval.py -> repository root.
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[4] / "logs"


def checkpoint_root() -> Path:
    """Return the synthetic checkpoint root from the environment.

    Raises:
        ValueError: If the checkpoint directory is not configured.
    """
    configured = os.environ.get(CHECKPOINT_DIR_ENV)
    if not configured:
        raise ValueError(f"{CHECKPOINT_DIR_ENV} is not set.")
    return Path(configured) / CHECKPOINT_SUBDIR


def find_checkpoint_dir(root: Path, cell_name: str) -> str:
    """Return the unique checkpoint directory for a cell, found by its name.

    Args:
        root: Checkpoint root to search under.
        cell_name: The cell's experiment name (unique per grid point).

    Returns:
        The checkpoint directory path.

    Raises:
        FileNotFoundError: If no directory matches the cell name.
    """
    matches = [path for path in root.glob(f"**/{cell_name}") if path.is_dir()]
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint directory named {cell_name} under {root}."
        )
    return str(sorted(matches)[0])


def cell_row(
    cell: TrainCell, metrics: dict[str, float], rollout_success: float
) -> dict[str, float | str]:
    """Prefix a metrics row with the cell's identity and its rollout success."""
    return {
        "cell": cell.name,
        "task": cell.store.task,
        "method": cell.method,
        "param": cell.param,
        "train_seed": cell.train_seed,
        "rollout_success": rollout_success,
        **metrics,
    }


def collect(
    stage: str,
    task: str,
    output_csv: Path,
    device: torch.device,
    eval_seed: int,
    num_generation_samples: int,
    log_dir: Path,
    array_dir: Path,
) -> None:
    """Evaluate every cell of a stage and append its row to ``output_csv``.

    Args:
        stage: The sweep stage whose cells to evaluate.
        task: The task key selecting the fixed store.
        output_csv: Manifest path; rewritten from scratch, one row per cell.
        device: Device to run each policy on.
        eval_seed: Torch seed forwarded to the eval hook.
        num_generation_samples: Stochastic samples for the fragility diagnostic.
        log_dir: Directory of the sbatch training logs, read for each cell's
            final rollout success rate (NaN when no log is found).
        array_dir: Directory receiving one ``<cell>.npz`` of per-chunk arrays
            per cell, so later metrics need no GPU pass.
    """
    root = checkpoint_root()
    successes = final_success_by_cell(log_dir=log_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    with open(output_csv, "w", newline="") as manifest:
        for cell in stage_cells(stage=stage, task=task):
            checkpoint_path = find_checkpoint_dir(root=root, cell_name=cell.name)
            metrics = run_from_config_name(
                config_name=cell.config_name,
                checkpoint_path=checkpoint_path,
                overrides=cell.eval_overrides(),
                device=device,
                eval_seed=eval_seed,
                num_generation_samples=num_generation_samples,
                array_path=array_dir / f"{cell.name}.npz",
            )
            row = cell_row(
                cell=cell,
                metrics=metrics,
                rollout_success=successes.get(cell.name, float("nan")),
            )
            if writer is None:
                writer = csv.DictWriter(manifest, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            manifest.flush()
            print(f"{cell.name}: total_mse={metrics['total_mse']:.6g}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Tip 2 eval-hook manifest collector.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--stage", default="pilot", choices=sorted(STAGES))
    parser.add_argument("--task", default="sequential", choices=sorted(STORES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-seed", type=int, default=0)
    # Stochastic is the main metric and its fine-scale tail is high variance, so
    # the default draws many samples per observation.
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--array-dir", type=Path, default=None)
    args = parser.parse_args()

    output_csv = args.output_dir / f"tip2_eval_{args.task}_{args.stage}.csv"
    collect(
        stage=args.stage,
        task=args.task,
        output_csv=output_csv,
        device=torch.device(args.device),
        eval_seed=args.eval_seed,
        num_generation_samples=args.num_samples,
        log_dir=args.log_dir,
        array_dir=args.array_dir or args.output_dir / "arrays",
    )
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    _main()
