"""Tip 2 tokenization-granularity sweep driver.

Tip 2 fixes the noise at each task's default and sweeps tokenization
granularity: FAST's rounding ``scale`` and binning's ``num_bins``. The two
families are reported as independent curves (no point-for-point matching), each
spanning its own coarse->fine range. Every cell trains on the same fixed
default-noise store (``stores.SEQUENTIAL`` / ``stores.CONDITIONAL_CIRCLE``) with
``data_seed`` fixed, so the only thing that moves within a family is the
granularity knob, and the only thing that moves across a replicate is the train
seed.

The FAST scale grid and its ``max_token_len`` come from ``calibrate_fast_scale``
on the 0.012 store: all seven scales are architecturally feasible, spanning a
fully degenerate coarse end to a well-resolved fine end, with a worst-case token
length of 118 (so a 119 cap fits every cell). The cap is injected only on FAST
cells; binning emits a fixed ``horizon * action_dim`` tokens whose own default
cap already fits.

    python -m versatil.analysis.tip2_tokenization.sweep list --stage main
    python -m versatil.analysis.tip2_tokenization.sweep train --stage main --dry-run
    python -m versatil.analysis.tip2_tokenization.sweep train --stage main --index 0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from versatil.analysis.tip1_noise.sweep import TASK_SCHEMA_GROUP
from versatil.analysis.tip2_tokenization.stores import STORES, Tip2Store

# Seven feasible FAST scales per task from calibrate_fast_scale on each task's
# fixed default-noise store, degenerate (every coefficient rounds to zero) to
# fine. The grid is task-specific because it is derived from the task's own
# action coefficient distribution; full precision so the fitted grid matches
# calibration. sequential: 0.012 store; conditional: 0.008 store.
FAST_SCALES = {
    "sequential": (
        0.41631974215059914,
        1.1529983532183592,
        3.193231230536625,
        8.843660238726597,
        24.49253460573554,
        67.832123263425,
        187.86120017758375,
    ),
    "conditional": (
        0.28010148101104365,
        0.8787208593846062,
        2.7566807070440906,
        8.64812578355214,
        27.130483184733638,
        85.11244358136324,
        267.01065377512424,
    ),
}
# Binning coarse->fine per the master plan: {4, 16} coarse probes, {64, 256,
# 1024} a 4x-spaced middle where 256 is the operating point the trained
# policies and the floor study use (the cross-Tip anchor), {2048, 4096} fine
# probes. Binning always emits horizon * action_dim tokens plus EOS, so
# num_bins only sizes the output vocabulary and the shared cap of 120 fits all.
NUM_BINS_GRID = (4, 16, 64, 256, 1024, 2048, 4096)
# Per-task FAST token cap: calibrated worst-case token length plus one. Injected
# on FAST cells only; the shared yaml stays 64. conditional circles carry richer
# DCT content, so their finest scale runs longer (182 -> 183) than sequential's
# (118 -> 119). Both leave ample room under the 512 decoder budget.
FAST_MAX_TOKEN_LEN = {"sequential": 119, "conditional": 183}

DATA_SEED = 42
TRAIN_SEEDS = (0, 1, 2)
# Rollouts are a secondary metric and each is a full autoregressive generation,
# so the cadence is coarsened from the config defaults (50 rollouts / 50 epochs)
# to keep the sweep's rollout cost bounded.
NUM_ROLLOUTS = 50
VAL_EVERY = 250

FAST = "fast"
BINNING = "binning"

METHOD_CONFIG = {
    FAST: "end_to_end_training_runs/synthetic/gpt_transformer",
    BINNING: "end_to_end_training_runs/synthetic/gpt_transformer_binned",
}
CONDITIONAL_METHOD_CONFIG = {
    FAST: "end_to_end_training_runs/synthetic/gpt_transformer_conditional",
    BINNING: "end_to_end_training_runs/synthetic/gpt_transformer_binned_conditional",
}
CONDITIONAL_TASK = "conditional"

_DISCRETIZER = "task.dataloader.tokenization.action_tokenizer.action_discretizer"
_MAX_TOKEN_LEN = "task.dataloader.tokenization.action_tokenizer.max_token_len"

STAGES = {
    # Single seed, both families, full granularity grid: the pipeline / power
    # gate before committing the replicates.
    "pilot": {"train_seeds": (TRAIN_SEEDS[0],)},
    # Three seeds; the pilot's seed-0 cells are a subset and are reused.
    "main": {"train_seeds": TRAIN_SEEDS},
}


def method_config(task: str, method: str) -> str:
    """Config name for a method on a task.

    Raises:
        KeyError: If the method has no config for the task.
    """
    configs = CONDITIONAL_METHOD_CONFIG if task == CONDITIONAL_TASK else METHOD_CONFIG
    return configs[method]


@dataclass(frozen=True)
class TrainCell:
    """One training run: a granularity point on one family at one train seed."""

    store: Tip2Store
    method: str
    param: float
    train_seed: int

    @property
    def param_tag(self) -> str:
        """Filename-safe granularity tag (scale for FAST, num_bins for binning)."""
        if self.method == FAST:
            return f"scale-{self.param:g}".replace(".", "p")
        return f"bins-{int(self.param)}"

    @property
    def name(self) -> str:
        """Identifier carrying task, family, granularity and seed."""
        return f"{self.store.task}__{self.method}__{self.param_tag}__seed-{self.train_seed}"

    @property
    def config_name(self) -> str:
        """Training config name for this cell's method and task."""
        return method_config(task=self.store.task, method=self.method)

    def granularity_overrides(self) -> list[str]:
        """Hydra overrides setting this cell's tokenization granularity."""
        if self.method == FAST:
            return [
                f"{_DISCRETIZER}.scale={self.param:g}",
                f"{_MAX_TOKEN_LEN}={FAST_MAX_TOKEN_LEN[self.store.task]}",
            ]
        return [f"{_DISCRETIZER}.num_bins={int(self.param)}"]

    def data_overrides(self) -> list[str]:
        """Overrides pinning the store, split seed and tokenizer granularity.

        Shared by training and evaluation so the two build the same normalizer,
        validation split and tokenizer; the absolute ``zarr_path`` and fixed
        ``data_seed`` make the split and normalizer identical across replicates.
        """
        return [
            f"task/dataset_schema={TASK_SCHEMA_GROUP[self.store.task]}",
            f"task.dataset_schema.zarr_path={self.store.zarr_path}",
            f"task.dataset_schema.noise_std={self.store.noise_std:g}",
            f"task.dataset_schema.num_episodes={self.store.num_episodes}",
            f"task.dataset_schema.seed={self.store.seed}",
            f"experiment.data_seed={DATA_SEED}",
            *self.granularity_overrides(),
        ]

    def eval_overrides(self) -> list[str]:
        """Hydra overrides for the offline eval-hook pass on this checkpoint.

        Only the data/tokenizer pins matter: the eval hook loads the trained
        weights and forces argmax itself, so training-only knobs (rollout
        cadence, train seed, decoder sampling mode) are irrelevant.
        """
        return self.data_overrides()

    def command(self) -> list[str]:
        """Full training command for this cell.

        The store is pinned by absolute ``zarr_path`` and the generation seed is
        pinned as ``data_seed`` so the split, normalizer and tokenizer are
        identical across replicates; only ``experiment.seed`` moves. The cell
        name becomes the experiment name so results map back to the granularity
        point and replicate.
        """
        overrides = [
            *self.data_overrides(),
            # The rollout reference must stay clean so the success threshold does
            # not drift with the training noise (consistency with Tip 1's eval).
            "task.dataset_schema.eval_reference_noise_std=0.0",
            f"task.dataset_schema.num_rollouts={NUM_ROLLOUTS}",
            f"experiment.val_every={VAL_EVERY}",
            f"experiment.seed={self.train_seed}",
            f"experiment.name={self.name}",
        ]
        return [
            sys.executable,
            "-m",
            "versatil.endpoints.train",
            "--config-name",
            self.config_name,
            *overrides,
        ]


def stage_cells(stage: str, task: str) -> list[TrainCell]:
    """All training cells for one stage on one task.

    Args:
        stage: ``STAGES`` key selecting the train-seed set.
        task: ``STORES`` key selecting the fixed store.

    Returns:
        FAST cells over every scale then binning cells over every num_bins,
        each at every train seed of the stage.

    Raises:
        KeyError: If the stage or task is unknown.
    """
    store = STORES[task]
    seeds = STAGES[stage]["train_seeds"]
    cells = [
        TrainCell(store=store, method=FAST, param=scale, train_seed=seed)
        for seed in seeds
        for scale in FAST_SCALES[task]
    ]
    cells += [
        TrainCell(store=store, method=BINNING, param=float(num_bins), train_seed=seed)
        for seed in seeds
        for num_bins in NUM_BINS_GRID
    ]
    return cells


def check_paths_unique(cells: list[TrainCell]) -> None:
    """Verify every cell has a distinct name (hence output directory).

    Raises:
        ValueError: If two cells share a name.
    """
    seen: dict[str, int] = {}
    for cell in cells:
        seen[cell.name] = seen.get(cell.name, 0) + 1
    duplicates = {name: count for name, count in seen.items() if count > 1}
    if duplicates:
        raise ValueError(f"Duplicate cell names: {duplicates}")


def _require_store(task: str) -> None:
    """Fail early if the task's fixed store has not been generated.

    Raises:
        FileNotFoundError: If the store is missing.
    """
    store = STORES[task]
    if not Path(store.zarr_path).exists():
        raise FileNotFoundError(
            f"Store {store.zarr_path} is missing. Generate it first: "
            f"python -m versatil.analysis.tip2_tokenization.stores {task}"
        )


def _list(stage: str, task: str) -> None:
    cells = stage_cells(stage=stage, task=task)
    check_paths_unique(cells)
    for index, cell in enumerate(cells):
        print(f"{index:>3}  {cell.name}")
    print(f"\n{len(cells)} cells for stage '{stage}' on task '{task}'.")


def _train(stage: str, task: str, index: int | None, dry_run: bool) -> None:
    _require_store(task)
    cells = stage_cells(stage=stage, task=task)
    check_paths_unique(cells)
    selected = cells if index is None else [cells[index]]
    for cell in selected:
        command = cell.command()
        if dry_run:
            print(" ".join(command))
            continue
        subprocess.run(command, check=True)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Tip 2 tokenization sweep driver.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    list_parser = subparsers.add_parser("list", help="List cells for a stage.")
    list_parser.add_argument("--stage", default="main", choices=sorted(STAGES))
    list_parser.add_argument("--task", default="sequential", choices=sorted(STORES))

    train_parser = subparsers.add_parser("train", help="Launch or preview training.")
    train_parser.add_argument("--stage", default="main", choices=sorted(STAGES))
    train_parser.add_argument("--task", default="sequential", choices=sorted(STORES))
    train_parser.add_argument(
        "--index", type=int, default=None, help="Run only this cell (SLURM array)."
    )
    train_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.mode == "list":
        _list(stage=args.stage, task=args.task)
    else:
        _train(stage=args.stage, task=args.task, index=args.index, dry_run=args.dry_run)


if __name__ == "__main__":
    _main()
