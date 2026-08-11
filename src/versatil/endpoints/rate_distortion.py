"""Hydra endpoint for the FAST tokenizer rate-distortion analysis.

Measures the FAST tokenizer floor on LIBERO demonstration action chunks: no model
training, reconstruction-error distortion only (replay success is stubbed). Writes
a results table and the rate-distortion figure.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import matplotlib

from versatil.analysis.rate_distortion.config import FastRateDistortionConfig
from versatil.analysis.rate_distortion.data import load_libero_action_chunks
from versatil.analysis.rate_distortion.fast_sweep import run_sweep
from versatil.common.logging import override_log_format
from versatil.configs.paths import get_hydra_configs_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

EXPERIMENTS_DIR = get_hydra_configs_dir()

_CSV_COLUMNS = [
    "family",
    "sweep",
    "scale",
    "vocab_size",
    "is_operating_point",
    "feasible",
    "alphabet_size",
    "mean_token_len",
    "bits_per_chunk",
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
    """Run the FAST rate-distortion sweep and write results.

    Args:
        config: Hydra config selecting the LIBERO task and the sweep grid.
    """
    override_log_format()
    if not config:
        raise ValueError(
            "No configuration specified! Provide --config-name, e.g. "
            "--config-name rate_distortion/fast_libero"
        )

    sweep_config: FastRateDistortionConfig = hydra.utils.instantiate(
        config.rate_distortion
    )
    dataset_schema = hydra.utils.instantiate(config.task.dataset_schema)
    action_space = hydra.utils.instantiate(config.task.action_space)
    observation_space = hydra.utils.instantiate(config.task.observation_space)
    dataloader_config = hydra.utils.instantiate(config.task.dataloader)

    chunk_data = load_libero_action_chunks(
        dataset_schema=dataset_schema,
        action_space=action_space,
        observation_space=observation_space,
        dataloader_config=dataloader_config,
        prediction_horizon=int(config.task.prediction_horizon),
        observation_horizon=int(config.task.observation_horizon),
        seed=int(config.experiment.seed),
        max_chunks=sweep_config.max_chunks,
    )
    logging.info(
        "Loaded %d LIBERO action chunks (time_horizon=%d, action_dim=%d).",
        chunk_data.chunks_normalized.shape[0],
        chunk_data.time_horizon,
        chunk_data.action_dim,
    )

    rows = run_sweep(chunk_data=chunk_data, config=sweep_config)

    output_dir = (
        Path(hydra.utils.get_original_cwd())
        / "outputs"
        / "rate_distortion"
        / ("fast_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_results_csv(rows=rows, output_path=output_dir / "results.csv")
    _plot_rate_distortion(rows=rows, output_path=output_dir / "figure_a.png")
    logging.info("Rate-distortion analysis written to %s", output_dir)


def _write_results_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write sweep rows to CSV with a stable column order."""
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=_CSV_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in _CSV_COLUMNS})


def _plot_rate_distortion(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Plot bits-per-chunk versus continuous reconstruction RMSE.

    The scale sweep traces the genuine rate-distortion frontier; the vocabulary
    sweep is expected to move horizontally (rate changes, distortion flat).
    """
    figure, axis = plt.subplots(figsize=(6, 4))
    for sweep, label, marker in (
        ("scale", "rounding-scale sweep (|V| fixed)", "o"),
        ("vocab", "BPE |V| sweep (scale fixed)", "s"),
    ):
        points = [
            (row["bits_per_chunk"], row["rmse_continuous"])
            for row in rows
            if row["sweep"] == sweep and row.get("feasible") and "bits_per_chunk" in row
        ]
        points.sort()
        if points:
            x_values, y_values = zip(*points, strict=True)
            axis.plot(x_values, y_values, marker=marker, label=label)
    axis.set_xlabel("rate (bits per chunk)")
    axis.set_ylabel("continuous reconstruction RMSE (original units)")
    axis.set_title("FAST tokenizer rate-distortion (LIBERO)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
