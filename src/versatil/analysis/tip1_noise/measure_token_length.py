"""Measure the FAST token length each sweep cell actually needs.

FAST does not emit a fixed number of tokens: noisier action labels survive more
DCT coefficients through rounding, and a denser-sampled episode has more
coefficients to begin with, so the post-BPE sequence grows with the noise level
and with the chunk length. The shared ``max_token_len`` of 64 is therefore not a
safe cap for this sweep -- several cells overflowed it in job 3966651 and crashed.
This module finds the true maximum across a grid so the cap can be set once per
trajectory length, high enough to fit every cell yet no higher than needed, since
padded positions still lengthen every sequence the decoder attends over.

The measurement reuses the real training pipeline rather than reimplementing the
tokenizer: it composes each cell's FAST config with the cap raised out of the
way and every other override the training command would carry (in particular
the chunk horizon), builds the dataloaders exactly as training does, and reads
back the token count from the tokenizer's own padding mask. That count is what
training would have produced, normaliser and fit included.

    export VERSATIL_NOISY_ZARR_DIR=/data/horse/ws/qizh093f-versatil/noisy_zarr
    python src/versatil/analysis/tip1_noise/measure_token_length.py outputs/tip1_noise
    python src/versatil/analysis/tip1_noise/measure_token_length.py \\
        outputs/tip1_noise --stage rate_conditional_s0
"""

import argparse
import csv
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from versatil.analysis.tip1_noise.sweep import (
    ACTION_TOKENIZER_MAX_TOKEN_LEN_KEY,
    DATA_SEEDS,
    SIGMA_MULTIPLIERS,
    STAGES,
    DataCell,
    TrainCell,
    method_config,
    stage_cells,
)
from versatil.configs.paths import get_hydra_configs_dir
from versatil.data.constants import SampleKey
from versatil.data.dataloader import get_dataloaders

# Raised far above any plausible FAST length so encoding never rejects a chunk
# while measuring; the real cap is chosen from the measured maximum afterwards.
# A 240-step chunk has 478 coefficients before BPE, and BPE can expand rather
# than compress a noisy chunk, so the ceiling leaves room above that.
MEASUREMENT_CAP = 1024
FAST_METHOD = "fast"
TASKS = ("sequential", "radial")
# Read from the main stage so the measured stores can never drift from the ones
# the sweep will actually train on; the stage carries a single injection value.
MAIN_INJECTION = STAGES["main"]["injections"][0]
# Only the tokenizer runs; the policy is built because the training config
# instantiates it, and it must not reach for a GPU on a login node.
MEASUREMENT_OVERRIDES = (
    f"{ACTION_TOKENIZER_MAX_TOKEN_LEN_KEY}={MEASUREMENT_CAP}",
    "experiment.use_wandb=false",
    "experiment.device=cpu",
)


def grid_cells() -> list[TrainCell]:
    """The FAST cells of the current noise grid, one per store."""
    return [
        TrainCell(
            data=DataCell(
                task=task,
                injection=MAIN_INJECTION,
                smoothing_sigma=0.0,
                sigma_multiplier=multiplier,
                data_seed=seed,
            ),
            method=FAST_METHOD,
            seed=0,
        )
        for task in TASKS
        for multiplier in SIGMA_MULTIPLIERS
        for seed in DATA_SEEDS
    ]


def stage_fast_cells(stage: str) -> list[TrainCell]:
    """The FAST cells of a stage, one per store.

    Replicates share a store only when they share a data seed, so deduplicating
    on the store name keeps exactly the cells whose token count can differ.
    """
    by_store: dict[str, TrainCell] = {}
    for cell in stage_cells(stage):
        if cell.method == FAST_METHOD:
            by_store.setdefault(cell.data.name, cell)
    return list(by_store.values())


def measurement_overrides(cell: TrainCell) -> list[str]:
    """Overrides composing the cell exactly as training would, cap aside."""
    return (
        cell.data.schema_overrides()
        + cell.length_overrides()
        + list(MEASUREMENT_OVERRIDES)
    )


def measure_cell(cell: TrainCell) -> dict[str, float | str | int]:
    """Return the largest FAST action-token count over the cell's whole dataset.

    Both splits are covered: the tokenizer is fit on the training split, but the
    validation chunks pass through the same tokenizer at evaluation time and
    could be the longest, so a cap that ignored them would still crash.
    """
    with initialize_config_dir(
        config_dir=str(get_hydra_configs_dir()), version_base=None
    ):
        config = compose(
            config_name=method_config(task=cell.data.task, method=FAST_METHOD),
            overrides=measurement_overrides(cell),
        )
    instantiated = instantiate(config)
    train_loader, val_loader, _, _, _ = get_dataloaders(instantiated)

    datasets = [train_loader.dataset]
    if val_loader is not None:
        datasets.append(val_loader.dataset)

    max_tokens = 0
    total = 0
    for dataset in datasets:
        for index in range(len(dataset)):
            is_pad = dataset[index][SampleKey.ACTION.value][
                SampleKey.IS_PAD_ACTION.value
            ]
            # The kept positions are the action tokens plus one EOS; the action
            # count is what the cap is compared against, so drop the EOS.
            action_tokens = int((~is_pad.bool()).sum().item()) - 1
            max_tokens = max(max_tokens, action_tokens)
            total += 1
    return {
        "cell": cell.data.name,
        "task": cell.data.task,
        "sigma_multiplier": cell.data.sigma_multiplier,
        "trajectory_length": cell.data.trajectory_length,
        "data_seed": cell.data.data_seed,
        "chunks": total,
        "max_action_tokens": max_tokens,
    }


def suggested_cap(max_action_tokens: int, margin: int) -> int:
    """Cap clearing a measured maximum.

    The tokenizer rejects a chunk when its action-token count is not strictly
    below the cap, and one EOS is appended on top, so the cap must exceed the
    maximum by at least one before any margin.
    """
    return max_action_tokens + 1 + margin


def main() -> None:
    """Measure every cell of a grid or stage and report the caps to use."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Where to write token_lengths.csv.")
    parser.add_argument(
        "--stage",
        default=None,
        choices=sorted(STAGES),
        help="Measure the FAST cells of this stage instead of the main grid.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=8,
        help="Headroom added over the measured maximum when suggesting a cap.",
    )
    arguments = parser.parse_args()

    cells = (
        grid_cells() if arguments.stage is None else stage_fast_cells(arguments.stage)
    )
    rows = []
    for index, cell in enumerate(cells, start=1):
        print(f"[{index}/{len(cells)}] {cell.data.name}", flush=True)
        row = measure_cell(cell)
        print(
            f"    max_action_tokens={row['max_action_tokens']} "
            f"over {row['chunks']} chunks",
            flush=True,
        )
        rows.append(row)

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if arguments.stage is None else f"_{arguments.stage}"
    manifest = output_dir / f"token_lengths{suffix}.csv"
    with open(manifest, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # A stage on the control-rate axis needs one cap per trajectory length; the
    # main grid shares one length and is read by noise level instead.
    group_key = "trajectory_length" if arguments.stage else "sigma_multiplier"
    print(f"\nby {group_key} (suggested cap = max + 1 + margin {arguments.margin}):")
    for value in sorted({row[group_key] for row in rows}):
        level = [row for row in rows if row[group_key] == value]
        worst = max(int(row["max_action_tokens"]) for row in level)
        print(
            f"  {group_key}={value:>5}: max {worst}, "
            f"cap {suggested_cap(worst, arguments.margin)}"
        )
    print(f"\nWrote {manifest}")


if __name__ == "__main__":
    main()
