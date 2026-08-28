"""Measure the FAST token length each sweep cell actually needs.

FAST does not emit a fixed number of tokens: noisier action labels survive more
DCT coefficients through rounding, so the post-BPE sequence grows with the noise
level. The shared ``max_token_len`` of 64 is therefore not a safe cap for this
sweep -- several cells overflowed it in job 3966651 and crashed. This module
finds the true maximum across the grid so the cap can be set once, high enough to
fit every cell yet no higher than needed, since padded positions still lengthen
every sequence the decoder attends over.

The measurement reuses the real training pipeline rather than reimplementing the
tokenizer: it composes each cell's FAST config with the cap raised out of the
way, builds the dataloaders exactly as training does, and reads back the token
count from the tokenizer's own padding mask. That count is what training would
have produced, normaliser and fit included.

    export VERSATIL_NOISY_ZARR_DIR=/data/horse/ws/qizh093f-versatil/noisy_zarr
    python src/versatil/analysis/tip1_noise/measure_token_length.py outputs/tip1_noise
"""

import argparse
import csv
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from versatil.analysis.tip1_noise.sweep import (
    DATA_SEEDS,
    METHOD_CONFIG,
    SIGMA_MULTIPLIERS,
    STAGES,
    DataCell,
)
from versatil.configs.paths import get_hydra_configs_dir
from versatil.data.constants import SampleKey
from versatil.data.dataloader import get_dataloaders

# Raised far above any plausible FAST length so encoding never rejects a chunk
# while measuring; the real cap is chosen from the measured maximum afterwards.
MEASUREMENT_CAP = 256
MAX_TOKEN_LEN_KEY = "task.dataloader.tokenization.action_tokenizer.max_token_len"
TASKS = ("sequential", "radial")
# Read from the main stage so the measured stores can never drift from the ones
# the sweep will actually train on; the stage carries a single injection value.
MAIN_INJECTION = STAGES["main"]["injections"][0]


def grid_cells() -> list[DataCell]:
    """The FAST data cells of the current noise grid, one per store."""
    return [
        DataCell(
            task=task,
            injection=MAIN_INJECTION,
            smoothing_sigma=0.0,
            sigma_multiplier=multiplier,
            data_seed=seed,
        )
        for task in TASKS
        for multiplier in SIGMA_MULTIPLIERS
        for seed in DATA_SEEDS
    ]


def measure_cell(cell: DataCell) -> dict[str, float | str | int]:
    """Return the largest FAST action-token count over the cell's whole dataset.

    Both splits are covered: the tokenizer is fit on the training split, but the
    validation chunks pass through the same tokenizer at evaluation time and
    could be the longest, so a cap that ignored them would still crash.
    """
    overrides = cell.schema_overrides() + [
        f"{MAX_TOKEN_LEN_KEY}={MEASUREMENT_CAP}",
        "experiment.use_wandb=false",
    ]
    with initialize_config_dir(
        config_dir=str(get_hydra_configs_dir()), version_base=None
    ):
        config = compose(config_name=METHOD_CONFIG["fast"], overrides=overrides)
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
        "cell": cell.name,
        "task": cell.task,
        "sigma_multiplier": cell.sigma_multiplier,
        "data_seed": cell.data_seed,
        "chunks": total,
        "max_action_tokens": max_tokens,
    }


def main() -> None:
    """Measure every grid cell and report the cap the sweep should use."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Where to write token_lengths.csv.")
    parser.add_argument(
        "--margin",
        type=int,
        default=8,
        help="Headroom added over the measured maximum when suggesting a cap.",
    )
    arguments = parser.parse_args()

    cells = grid_cells()
    rows = []
    for index, cell in enumerate(cells, start=1):
        print(f"[{index}/{len(cells)}] {cell.name}", flush=True)
        row = measure_cell(cell)
        print(
            f"    max_action_tokens={row['max_action_tokens']} "
            f"over {row['chunks']} chunks",
            flush=True,
        )
        rows.append(row)

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "token_lengths.csv"
    with open(manifest, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grid_max = max(int(row["max_action_tokens"]) for row in rows)
    # The tokenizer rejects a chunk when its action-token count is not strictly
    # below the cap, and one EOS is appended on top, so the cap must exceed the
    # maximum by at least one before any margin.
    suggested = grid_max + 1 + arguments.margin
    print(f"\ngrid maximum action tokens: {grid_max}")
    print(f"suggested max_token_len (max + 1 + margin {arguments.margin}): {suggested}")
    print("\nby noise level:")
    for multiplier in SIGMA_MULTIPLIERS:
        level = [r for r in rows if r["sigma_multiplier"] == multiplier]
        worst = max(int(r["max_action_tokens"]) for r in level)
        print(f"  sigma={multiplier:>4}: max {worst}")
    print(f"\nWrote {manifest}")


if __name__ == "__main__":
    main()
