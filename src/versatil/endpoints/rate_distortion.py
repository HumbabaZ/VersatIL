"""Hydra endpoint for the action-tokenizer rate-distortion / floor study.

Reconstruction layer: no model training. Sweeps FAST (scale, |V|) and per-value
Binning (num_bins) over LIBERO action chunks across chunk horizons, and writes one
unified results table plus the FAST-vs-Binning and horizon figures. Replay success
is measured separately (scripts/libero_replay_floor/).
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from versatil.analysis.rate_distortion import plot_floor
from versatil.analysis.rate_distortion.binning_sweep import run_binning_sweep
from versatil.analysis.rate_distortion.config import FastRateDistortionConfig
from versatil.analysis.rate_distortion.data import (
    chunk_data_at_horizon,
    load_per_step_context,
)
from versatil.analysis.rate_distortion.fast_sweep import (
    load_fast_class,
    run_fast_cell,
    run_fast_grid,
)
from versatil.analysis.rate_distortion.metrics import bits_per_step
from versatil.common.logging import override_log_format
from versatil.configs.paths import get_hydra_configs_dir

EXPERIMENTS_DIR = get_hydra_configs_dir()
FULL_SWEEP_HORIZON = 10

_CSV_COLUMNS = [
    "family",
    "horizon",
    "sweep",
    "scale",
    "vocab_size",
    "num_bins",
    "binning_strategy",
    "is_operating_point",
    "feasible",
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
    "replay_success",
]


@hydra.main(
    version_base=None,
    config_path=str(EXPERIMENTS_DIR),
    config_name="rate_distortion/fast_libero",
)
def main(config: DictConfig) -> None:
    """Run the reconstruction floor study and write results and figures.

    Args:
        config: Hydra config selecting the LIBERO task and the sweep grids.
    """
    override_log_format()
    if not config:
        raise ValueError(
            "No configuration specified! Provide --config-name, e.g. "
            "--config-name rate_distortion/floor_study"
        )

    sweep_config: FastRateDistortionConfig = hydra.utils.instantiate(
        config.rate_distortion
    )
    dataset_schema = hydra.utils.instantiate(config.task.dataset_schema)
    action_space = hydra.utils.instantiate(config.task.action_space)
    observation_space = hydra.utils.instantiate(config.task.observation_space)
    dataloader_config = hydra.utils.instantiate(config.task.dataloader)
    seed = int(config.experiment.seed)

    horizons = [int(horizon) for horizon in sweep_config.horizons]
    full_sweep_horizon = (
        FULL_SWEEP_HORIZON if FULL_SWEEP_HORIZON in horizons else min(horizons)
    )
    context = load_per_step_context(
        dataset_schema=dataset_schema,
        action_space=action_space,
        observation_space=observation_space,
        dataloader_config=dataloader_config,
        observation_horizon=int(config.task.observation_horizon),
        seed=seed,
        base_horizon=min(horizons),
    )
    fast_class = (
        load_fast_class(sweep_config.pretrained_fast_model)
        if "fast" in sweep_config.families
        else None
    )

    rows = _run_study(
        context=context,
        sweep_config=sweep_config,
        horizons=horizons,
        full_sweep_horizon=full_sweep_horizon,
        fast_class=fast_class,
        seed=seed,
    )

    output_dir = (
        Path(hydra.utils.get_original_cwd())
        / "outputs"
        / "rate_distortion"
        / ("floor_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_results_csv(rows=rows, output_path=output_dir / "results.csv")
    plot_floor.plot_all(rows=rows, output_dir=output_dir)
    logging.info("Floor study written to %s", output_dir)


def _run_study(
    context,
    sweep_config: FastRateDistortionConfig,
    horizons: list[int],
    full_sweep_horizon: int,
    fast_class: type | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Sweep every family's full knob grid at every horizon.

    Running the full scale/bins sweep per horizon (not just the operating point)
    is what lets each horizon draw its own rate-distortion frontier; the
    reconstruction layer is cheap enough to afford it. ``full_sweep_horizon`` marks
    which horizon also carries the |V| confirmation sweep and the near-lossless
    sanity check, which only need to run once.
    """
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        chunk_data = chunk_data_at_horizon(
            context=context,
            horizon=horizon,
            max_chunks=sweep_config.max_chunks,
            seed=seed,
        )
        logging.info(
            "horizon=%d: %d chunks", horizon, chunk_data.chunks_normalized.shape[0]
        )
        horizon_rows: list[dict[str, Any]] = []
        if fast_class is not None:
            if horizon == full_sweep_horizon:
                horizon_rows += run_fast_grid(
                    fast_class=fast_class, chunk_data=chunk_data, config=sweep_config
                )
            else:
                horizon_rows += [
                    run_fast_cell(
                        fast_class=fast_class,
                        chunk_data=chunk_data,
                        scale=scale,
                        vocab_size=sweep_config.scale_sweep_fixed_vocab,
                        sweep="scale",
                        is_operating_point=scale
                        == sweep_config.vocab_sweep_fixed_scale,
                    )
                    for scale in sweep_config.scales
                ]
        if "binning" in sweep_config.families:
            for binning_strategy in sweep_config.binning_strategies:
                horizon_rows += run_binning_sweep(
                    chunk_data=chunk_data,
                    bin_counts=sweep_config.bin_counts,
                    binning_strategy=binning_strategy,
                )
        for row in horizon_rows:
            row["horizon"] = horizon
            if row.get("feasible") and "bits_per_chunk" in row:
                row["bits_per_step"] = bits_per_step(row["bits_per_chunk"], horizon)
            rows.append(row)
    return rows


def _write_results_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write study rows to CSV with a stable column order."""
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=_CSV_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in _CSV_COLUMNS})


if __name__ == "__main__":
    main()
