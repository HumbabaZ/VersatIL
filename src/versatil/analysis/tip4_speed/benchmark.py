"""Tip 4 inference-latency benchmark over trained Tip 1 checkpoints.

Measures, per ``Policy.predict_action`` call, the wall-clock split into
pre-processing / observation encoding / generation / detokenization /
unnormalization, plus the emitted token count, for every trained cell of a
Tip 1 stage. No training happens here: checkpoints, stores and configs are the
ones the Tip 1 sweep produced.

Measurement protocol (fixed up front, reported in M.5):

- batch size 1: the autoregressive loop only stops early when *every*
  sequence in the batch has emitted EOS, so any batching would erase exactly
  the variable-length advantage under test. Batch 1 is also the deployment
  shape (one control loop, one observation).
- KV cache on, ``torch.compile`` off, precision and sampling mode exactly as
  the checkpoint's own config (the autocast context is the same no-op-for-fp32
  wrapper the training-time rollout callback used).
- Warm-up calls are discarded, then ``num_chunks`` calls are timed, each on a
  different validation observation (cycled if the split is smaller).
- Every segment boundary synchronizes the CUDA device (see ``timing.py``).
- Detokenizer warnings are counted per call and kept out of the timed path.

Usage (Tip 1 synthetic stages):

    python -m versatil.analysis.tip4_speed.benchmark <out_dir> \
        --stage final_conditional --stage rate_conditional_s0 \
        --device cuda [--num-chunks 100] [--warmup 10] [--cells SUBSTR]

Usage (direct checkpoint directories, e.g. the LIBERO matched arms):

    python -m versatil.analysis.tip4_speed.benchmark <out_dir> \
        --checkpoint-dir <ckpt_dir> [--checkpoint-dir ...] \
        [--checkpoint-name last.ckpt] [--task-label libero] --device cuda

Rows are appended to ``<out_dir>/results_speed_<device>.csv`` as each cell
finishes, so a mid-run failure keeps the rows already computed. Generated
token sequences are dumped to ``<out_dir>/tokens/<cell>.jsonl`` for the
offline detokenization micro-benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import hydra.utils
import torch
from omegaconf import OmegaConf

from versatil.analysis.tip1_noise.sweep import TrainCell, stage_cells
from versatil.analysis.tip4_speed.timing import (
    SegmentTimerSink,
    SilencedDetokWarnings,
)
from versatil.checkpoint_loading.base import BaseCheckpointLoader
from versatil.configs import MainConfig
from versatil.data.constants import SampleKey
from versatil.data.dataloader import get_dataloaders
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.endpoints.deploy import load_policy
from versatil.models.policy import Policy
from versatil.training.constants import CheckpointFilename, PrecisionType

CHECKPOINT_DIR_ENV = "VERSATIL_CHECKPOINT_DIR"
CHECKPOINT_SUBDIR = "synthetic"

# Sequential depth of the continuous heads as a function of the prediction
# horizon, assigned per method because their predictions carry no token
# dimension: qfat runs one cached decoder step per action timestep, bcat emits
# the whole chunk in a single parallel forward.
CONTINUOUS_STEPS = {
    "qfat": lambda horizon: horizon,
    "bcat": lambda _horizon: 1,
    "act": lambda _horizon: 1,
}

CSV_FIELDS = (
    "cell",
    "task",
    "method",
    "sigma_multiplier",
    "data_seed",
    "train_seed",
    "trajectory_length",
    "device",
    "device_name",
    "precision",
    "param_count",
    "chunk_index",
    "sequential_steps",
    "generated_tokens",
    "t_pre_ms",
    "t_encode_ms",
    "t_generate_ms",
    "t_detok_ms",
    "t_unnorm_ms",
    "t_total_ms",
    "detok_warnings",
    "weights",
)


def checkpoint_root() -> Path:
    """Return the synthetic checkpoint root from the environment.

    Raises:
        ValueError: If the checkpoint directory is not configured.
    """
    configured = os.environ.get(CHECKPOINT_DIR_ENV)
    if not configured:
        raise ValueError(f"{CHECKPOINT_DIR_ENV} is not set.")
    return Path(configured) / CHECKPOINT_SUBDIR


def find_checkpoint_dir(root: Path, cell_name: str) -> Path:
    """Return the unique checkpoint directory whose name equals ``cell_name``.

    Args:
        root: Checkpoint root to search under.
        cell_name: The cell's experiment name (unique per grid point).

    Raises:
        FileNotFoundError: If no directory matches the cell name.
        ValueError: If several directories match.
    """
    matches = [path for path in root.glob(f"**/{cell_name}") if path.is_dir()]
    if not matches:
        raise FileNotFoundError(f"No checkpoint directory named {cell_name!r}.")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous checkpoint dirs for {cell_name!r}: {matches}.")
    return matches[0]


class _CapturingTokenSink:
    """Buffers predicted model-vocab token id rows in memory."""

    def __init__(self) -> None:
        self.recorded: list[list[int]] = []

    def record(self, action_tokens: torch.Tensor) -> None:
        squeezed = action_tokens
        if squeezed.ndim == 3 and squeezed.shape[-1] == 1:
            squeezed = squeezed.squeeze(-1)
        for row in squeezed:
            self.recorded.append([int(value) for value in row.tolist()])


@dataclass
class CellMeasurement:
    """Per-call rows and the captured token sequences for one cell."""

    rows: list[dict[str, float | int | str | None]]
    token_sequences: list[list[int]]


@dataclass(frozen=True)
class Target:
    """One checkpoint to measure, with the identity fields for its CSV rows.

    Produced either from a Tip 1 sweep cell (synthetic grids) or directly from
    a checkpoint directory (LIBERO / arbitrary checkpoints), so the
    measurement loop is shared between both entry points.
    """

    name: str
    method: str
    checkpoint_dir: Path
    checkpoint_name: str
    task: str
    sigma_multiplier: float | str
    data_seed: int | str
    train_seed: int | str
    trajectory_length: int | str
    # Sequential depth for continuous heads; None for tokenized heads, whose
    # depth is the measured generated length.
    continuous_steps: int | None
    allow_random_init: bool


def target_from_cell(cell: TrainCell, checkpoint_dir: Path) -> Target:
    """Target for a Tip 1 synthetic sweep cell."""
    steps_for = CONTINUOUS_STEPS.get(cell.method)
    return Target(
        name=cell.name,
        method=cell.method,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=CheckpointFilename.DEFAULT_CHECKPOINT.value,
        task=cell.data.task,
        sigma_multiplier=cell.data.sigma_multiplier,
        data_seed=cell.data.data_seed,
        train_seed=cell.seed,
        trajectory_length=cell.data.trajectory_length,
        continuous_steps=None
        if steps_for is None
        else steps_for(cell.prediction_horizon),
        allow_random_init=cell.method in CONTINUOUS_STEPS,
    )


def infer_method(raw_config: MainConfig) -> str:
    """Infer the arm name from a checkpoint's raw (uninstantiated) config."""
    decoder_target = str(
        OmegaConf.select(raw_config, "policy.decoder._target_", default="")
    )
    if "qfat" in decoder_target.lower():
        return "qfat"
    if "gpt_action_transformer" in decoder_target.lower():
        discretizer_type = str(
            OmegaConf.select(
                raw_config,
                "task.dataloader.tokenization.action_tokenizer.action_discretizer.type",
                default="",
            )
        )
        return "binned" if "binned" in discretizer_type.lower() else "fast"
    if "action_transformer" in decoder_target.lower():
        return "bcat"
    raise ValueError(f"Cannot infer arm from decoder target {decoder_target!r}.")


def target_from_checkpoint_dir(
    checkpoint_dir: Path, checkpoint_name: str, task_label: str
) -> Target:
    """Target for an explicit checkpoint directory (the LIBERO path).

    The arm and prediction horizon come from the checkpoint's own config;
    grid fields that only exist for the synthetic sweeps stay empty. Direct
    targets always require trained weights: they are named checkpoints, not
    grid cells whose training may simply not have saved.
    """
    raw_config = OmegaConf.load(checkpoint_dir / CheckpointFilename.CONFIG.value)
    method = infer_method(raw_config)
    horizon = int(OmegaConf.select(raw_config, "task.prediction_horizon"))
    steps_for = CONTINUOUS_STEPS.get(method)
    return Target(
        name=checkpoint_dir.name,
        method=method,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        task=task_label,
        sigma_multiplier="",
        data_seed="",
        train_seed="",
        trajectory_length=horizon,
        continuous_steps=None if steps_for is None else steps_for(horizon),
        allow_random_init=False,
    )


def collect_observations(
    config: MainConfig, max_observations: int
) -> tuple[list[dict[str, torch.Tensor]], LinearNormalizer, Tokenizer | None]:
    """Return observation dicts plus the fitted normalizer and tokenizer.

    The validation loader is rebuilt with batch size 1, no shuffling, and
    trailing padding disabled (full-horizon chunks only, the rollout shape).
    Observations stay on the CPU: the device transfer belongs to the timed
    ``pre`` segment, as in deployment. The normalizer and tokenizer are fitted
    on the training split exactly as at training time; a policy loaded from a
    checkpoint carries its own and ignores them.

    Raises:
        ValueError: If the config has no validation split.
    """
    config.task.dataloader.batch_size = 1
    config.task.dataloader.shuffle = False
    config.task.dataloader.trailing_padded_actions = 0
    # Worker processes buy nothing for ~100 pre-collected observations and
    # their multiprocessing must stay out of a timing benchmark.
    config.task.dataloader.num_workers = 0
    _, val_loader, normalizer, tokenizer, _ = get_dataloaders(config=config)
    if val_loader is None:
        raise ValueError("The benchmark needs a validation split; val_ratio > 0.")
    observations: list[dict[str, torch.Tensor]] = []
    for batch in val_loader:
        observations.append(dict(batch[SampleKey.OBSERVATION.value]))
        if len(observations) >= max_observations:
            break
    if not observations:
        raise ValueError("Validation loader yielded no batches.")
    return observations, normalizer, tokenizer


def load_measurement_policy(
    target: Target,
    device: torch.device,
) -> tuple[Policy, MainConfig, bool]:
    """Load the target's policy for timing; returns whether weights were loaded.

    Synthetic cells whose training ran with ``save_checkpoints: false`` (the
    bcat arm) have a config but no weights on disk. For continuous heads
    latency is weight-independent -- generation is a fixed number of decoder
    steps and no EOS decision depends on what the model learned -- so those
    cells fall back to a randomly initialized policy built from the saved
    config, flagged in the results. Tokenized heads and direct checkpoint
    targets never fall back.

    Raises:
        FileNotFoundError: If the checkpoint file is missing and the target
            does not allow random initialization.
    """
    checkpoint_file = target.checkpoint_dir / target.checkpoint_name
    if checkpoint_file.exists():
        runtime = load_policy(
            checkpoint_path=str(target.checkpoint_dir),
            device=device,
            checkpoint_name=target.checkpoint_name,
            compile_model=False,
        )
        return runtime.policy, runtime.config, True
    if not target.allow_random_init:
        raise FileNotFoundError(
            f"{target.name}: no {checkpoint_file.name} and method "
            f"{target.method!r} needs trained weights."
        )
    raw_config = OmegaConf.load(target.checkpoint_dir / CheckpointFilename.CONFIG.value)
    for device_key in BaseCheckpointLoader.CONFIG_DEVICE_KEYS:
        if OmegaConf.select(raw_config, device_key) is not None:
            OmegaConf.update(raw_config, device_key, str(device))
    config = hydra.utils.instantiate(raw_config)
    policy = config.policy
    policy.to(device).eval()
    policy.device = device
    return policy, config, False


def measure_policy(
    policy: Policy,
    observations: list[dict[str, torch.Tensor]],
    device: torch.device,
    precision_type: PrecisionType,
    num_chunks: int,
    warmup: int,
    eval_seed: int,
) -> tuple[SegmentTimerSink, list[list[int]], list[int]]:
    """Time ``num_chunks`` predict_action calls after ``warmup`` discarded ones.

    Returns the filled timer sink, the captured token sequences (empty for
    continuous heads), and the per-call detokenizer warning counts.
    """
    torch.manual_seed(eval_seed)
    autocast = precision_type.autocast(device_type=device.type)

    def call(observation: dict[str, torch.Tensor]) -> None:
        with torch.no_grad(), autocast:
            policy.predict_action(obs_dict=dict(observation))

    for index in range(warmup):
        call(observations[index % len(observations)])

    timer = SegmentTimerSink(device=device)
    token_sink = _CapturingTokenSink()
    warning_counts: list[int] = []
    policy.set_latency_sink(sink=timer)
    policy.set_token_usage_sink(sink=token_sink)
    try:
        with SilencedDetokWarnings() as counter:
            for index in range(num_chunks):
                call(observations[index % len(observations)])
                warning_counts.append(counter.take())
    finally:
        policy.set_latency_sink(sink=None)
        policy.set_token_usage_sink(sink=None)
    return timer, token_sink.recorded, warning_counts


def benchmark_target(
    target: Target,
    device: torch.device,
    num_chunks: int,
    warmup: int,
    eval_seed: int,
) -> CellMeasurement:
    """Measure one checkpoint and return its per-call rows."""
    policy, config, weights_loaded = load_measurement_policy(
        target=target, device=device
    )
    observations, normalizer, tokenizer = collect_observations(
        config=config, max_observations=max(num_chunks, warmup)
    )
    if not weights_loaded:
        policy.set_normalizer(normalizer=normalizer)
        if tokenizer is not None:
            tokenizer.to(device)
        policy.set_tokenizer(tokenizer=tokenizer)
    policy.eval()
    # Deployment decodes by sampling; pin it the way the Tip 2 hook does so a
    # checkpoint saved with another mode cannot silently change the protocol.
    if hasattr(policy.decoder, "deterministic"):
        policy.decoder.deterministic = False
    precision_type = PrecisionType(str(config.experiment.precision))
    param_count = sum(parameter.numel() for parameter in policy.parameters())
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    timer, token_sequences, warning_counts = measure_policy(
        policy=policy,
        observations=observations,
        device=device,
        precision_type=precision_type,
        num_chunks=num_chunks,
        warmup=warmup,
        eval_seed=eval_seed,
    )
    records = timer.records
    if len(records) != num_chunks:
        raise RuntimeError(
            f"{target.name}: expected {num_chunks} latency records, got {len(records)}."
        )

    rows: list[dict[str, float | int | str | None]] = []
    for chunk_index, record in enumerate(records):
        if record.generated_tokens is not None:
            sequential_steps = record.generated_tokens
        elif target.continuous_steps is not None:
            sequential_steps = target.continuous_steps
        else:
            raise ValueError(
                f"{target.name}: method {target.method!r} produced no tokens "
                "and has no assigned sequential depth."
            )
        rows.append(
            {
                "cell": target.name,
                "task": target.task,
                "method": target.method,
                "sigma_multiplier": target.sigma_multiplier,
                "data_seed": target.data_seed,
                "train_seed": target.train_seed,
                "trajectory_length": target.trajectory_length,
                "device": device.type,
                "device_name": device_name,
                "precision": str(config.experiment.precision),
                "param_count": param_count,
                "chunk_index": chunk_index,
                "sequential_steps": sequential_steps,
                "generated_tokens": record.generated_tokens,
                "t_pre_ms": record.segments_ms.get("pre"),
                "t_encode_ms": record.segments_ms.get("encode"),
                "t_generate_ms": record.segments_ms.get("generate"),
                "t_detok_ms": record.segments_ms.get("detok"),
                "t_unnorm_ms": record.segments_ms.get("unnorm"),
                "t_total_ms": record.total_ms,
                "detok_warnings": warning_counts[chunk_index],
                "weights": "trained" if weights_loaded else "random",
            }
        )
    return CellMeasurement(rows=rows, token_sequences=token_sequences)


def unique_stage_cells(
    stages: list[str], cell_filter: str | None = None
) -> list[TrainCell]:
    """Cells of the given stages, deduplicated by name in stable order.

    The control-rate stage shares its T=60 cells with the noise grid; each
    shared cell is measured once.
    """
    cells: list[TrainCell] = []
    seen: set[str] = set()
    for stage in stages:
        for cell in stage_cells(stage=stage):
            if cell.name in seen:
                continue
            if cell_filter and cell_filter not in cell.name:
                continue
            seen.add(cell.name)
            cells.append(cell)
    return cells


def run_targets(
    targets: list[Target],
    output_dir: Path,
    device: torch.device,
    num_chunks: int,
    warmup: int,
    eval_seed: int,
) -> Path:
    """Benchmark every target, appending rows to the CSV as each finishes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir = output_dir / "tokens"
    tokens_dir.mkdir(exist_ok=True)
    output_csv = output_dir / f"results_speed_{device.type}.csv"
    write_header = not output_csv.exists()

    with open(output_csv, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(CSV_FIELDS))
        if write_header:
            writer.writeheader()
        for index, target in enumerate(targets):
            print(f"[{index + 1}/{len(targets)}] {target.name}", flush=True)
            try:
                measurement = benchmark_target(
                    target=target,
                    device=device,
                    num_chunks=num_chunks,
                    warmup=warmup,
                    eval_seed=eval_seed,
                )
            except FileNotFoundError as error:
                # A still-training or never-launched cell must not kill the
                # sweep over the others; the collector reports coverage.
                print(f"SKIP {target.name}: {error}", flush=True)
                continue
            writer.writerows(measurement.rows)
            csv_file.flush()
            if measurement.token_sequences:
                token_path = tokens_dir / f"{target.name}.jsonl"
                with open(token_path, "w") as token_file:
                    for sequence in measurement.token_sequences:
                        token_file.write(json.dumps(sequence) + "\n")
    return output_csv


def stage_targets(stages: list[str], cell_filter: str | None = None) -> list[Target]:
    """Targets for the Tip 1 sweep stages, skipping cells without a directory."""
    targets: list[Target] = []
    root = checkpoint_root()
    for cell in unique_stage_cells(stages=stages, cell_filter=cell_filter):
        try:
            checkpoint_dir = find_checkpoint_dir(root=root, cell_name=cell.name)
        except FileNotFoundError as error:
            print(f"SKIP {cell.name}: {error}", flush=True)
            continue
        targets.append(target_from_cell(cell=cell, checkpoint_dir=checkpoint_dir))
    return targets


def _main() -> None:
    parser = argparse.ArgumentParser(description="Tip 4 inference-speed benchmark.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--stage",
        action="append",
        default=None,
        help="Tip 1 sweep stage(s) whose checkpoints to measure; repeatable. "
        "Defaults to final_conditional + rate_conditional_s0.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        type=Path,
        default=None,
        help="Measure these checkpoint directories directly instead of a Tip 1 "
        "stage (the LIBERO path); repeatable. Arm and horizon are read from "
        "each checkpoint's config.yaml.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=CheckpointFilename.DEFAULT_CHECKPOINT.value,
        help="Checkpoint file to load in --checkpoint-dir mode.",
    )
    parser.add_argument(
        "--task-label",
        default="libero",
        help="Task column value for --checkpoint-dir rows.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-chunks", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument(
        "--cells", default=None, help="Only cells whose name contains this substring."
    )
    arguments = parser.parse_args()

    if arguments.checkpoint_dir:
        targets = [
            target_from_checkpoint_dir(
                checkpoint_dir=checkpoint_dir,
                checkpoint_name=arguments.checkpoint_name,
                task_label=arguments.task_label,
            )
            for checkpoint_dir in arguments.checkpoint_dir
        ]
    else:
        stages = arguments.stage or ["final_conditional", "rate_conditional_s0"]
        targets = stage_targets(stages=stages, cell_filter=arguments.cells)

    output_csv = run_targets(
        targets=targets,
        output_dir=arguments.output_dir,
        device=torch.device(arguments.device),
        num_chunks=arguments.num_chunks,
        warmup=arguments.warmup,
        eval_seed=arguments.eval_seed,
    )
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    _main()
