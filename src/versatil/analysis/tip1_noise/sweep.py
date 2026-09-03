"""Enumerate and launch the Tip 1 noise sweep.

Every cell of the sweep needs its own zarr store. ``_ensure_zarr_exists`` decides
whether to regenerate by looking at the path and the key set only -- it never
inspects ``noise_std`` -- so two cells that share a path silently share data and
the noise curve comes out flat with nothing in the logs to say why. This module
derives a unique path from the generation parameters so that cannot happen, and
verifies uniqueness before anything runs.

Data cells are keyed by generation parameters alone. The tokenizer and the
training seed do not change the dataset, so every method and seed at one noise
setting reuses one store.

Stores go to $VERSATIL_NOISY_ZARR_DIR, never to VERSATIL_ZARR_DIR: every dataset
here carries injected noise, and a normal experiment resolving its dataset through
the shared zarr directory must not be able to reach them.

    export VERSATIL_NOISY_ZARR_DIR=/data/horse/ws/qizh093f-versatil/noisy_zarr

    # generate the stores and the per-cell diagnostics, no GPU needed
    python src/versatil/analysis/tip1_noise/sweep.py data --stage pilot out_dir

    # print the training commands the stage would run
    python src/versatil/analysis/tip1_noise/sweep.py train --stage pilot --dry-run
"""

import argparse
import csv
import itertools
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from versatil.configs.paths import get_hydra_configs_dir
from versatil.data.constants import ProprioKey
from versatil.data.preprocessing.create_zarr_from_synthetic import (
    create_replay_buffer_from_synthetic,
)
from versatil.data.preprocessing.replay_buffer import ReplayBuffer
from versatil.data.synthetic.constants import NoiseInjection, SyntheticNoiseModel

# Noise is expressed as a multiple of each task's own default, because the tasks
# differ severalfold in action magnitude and a shared absolute sigma would put
# them at very different signal-to-noise ratios.
TASK_DEFAULT_NOISE_STD = {
    "sequential": 0.012,
    "radial": 0.006,
    "corridor": 0.005,
    "conditional": 0.008,
}
TASK_SCHEMA_GROUP = {
    "sequential": "synthetic/sequential",
    "radial": "synthetic/radial",
    "corridor": "synthetic/corridor_navigation",
    "conditional": "synthetic/conditional_circle",
}
# The conditional task feeds a one-hot context that selects the mode, so each
# mode is unimodal given its context and a policy's success under noise measures
# execution accuracy rather than whether it kept its modes. That needs the
# context-aware configs, which add the context encoder and decoder input.
CONDITIONAL_TASK = "conditional"
# The grid anchors at 1.0, the benchmark's own default noise, and steps up from
# there. Zero noise is excluded on purpose: with the benchmark's single style and
# fixed start, noise_std is the only source of episode-to-episode variation, so
# under action-label injection a sigma of 0 collapses the dataset to a handful of
# identical demonstrations -- a categorically different learning problem, not a
# cleaner version of the same one. The 0.5 step is dropped as uninformative. The
# levels are equally spaced on the noise-to-signal axis the curves are drawn on,
# since that ratio grows in proportion to sigma.
SIGMA_MULTIPLIERS = (1.0, 2.0, 3.0, 4.0)
# 0 keeps the high band; 2.0 is the low band the band-migration sweep settled on.
HIGH_BAND_SMOOTHING = 0.0
LOW_BAND_SMOOTHING = 2.0

METHOD_CONFIG = {
    "fast": "end_to_end_training_runs/synthetic/gpt_transformer",
    "binned": "end_to_end_training_runs/synthetic/gpt_transformer_binned",
    "qfat": "end_to_end_training_runs/synthetic/qfat",
    "act": "end_to_end_training_runs/synthetic/kl_cvae_fixed_gaussian",
    # Plain BC action transformer (direct regression, no CVAE). Candidate to
    # replace the CVAE "act" arm as the external continuous baseline.
    "bcat": "end_to_end_training_runs/synthetic/bcat",
}
CONDITIONAL_METHOD_CONFIG = {
    "fast": "end_to_end_training_runs/synthetic/gpt_transformer_conditional",
    "binned": "end_to_end_training_runs/synthetic/gpt_transformer_binned_conditional",
    "qfat": "end_to_end_training_runs/synthetic/qfat_conditional",
    "bcat": "end_to_end_training_runs/synthetic/bcat_conditional",
}
CORE_METHODS = ("fast", "binned", "qfat")
# The finalized comparison: two discrete arms (fast, binned) and two continuous
# ones (qfat, and the plain action-transformer bcat as the external baseline).
FINAL_METHODS = ("qfat", "fast", "binned", "bcat")


def method_config(task: str, method: str) -> str:
    """Config name for a method on a task.

    Raises:
        KeyError: If the method has no config for the task.
    """
    configs = CONDITIONAL_METHOD_CONFIG if task == CONDITIONAL_TASK else METHOD_CONFIG
    return configs[method]


# Single source of truth for the matched-capacity arm. Every shared
# hyperparameter is pinned so the action representation is the only factor that
# moves; prediction_horizon is left alone because 59 vs 60 follows from how each
# family frames the chunk. Depth is pinned to 4 layers: the qfat and
# bcat_conditional configs already sit there, and the GPT configs fell back to
# the decoder default of 6, so the multimodal grids ran with unequal depth.
MATCHED_OVERRIDES = (
    "policy.decoder.number_of_layers=4",
    "policy.decoder.number_of_heads=4",
    "policy.decoder.dropout_rate=0.4",
    "policy.decoder.attention_dropout=0.15",
    "training.optimizer.lr=1e-4",
    "training.use_ema=true",
)
MATCHED_ACT_EXTRA = (
    "policy.algorithm.posterior_encoder.number_of_heads=4",
    "policy.algorithm.posterior_encoder.dropout_rate=0.4",
    "policy.algorithm.posterior_encoder.attention_dropout=0.15",
)

# FAST emits a variable number of tokens that grows with the injected noise, so
# the shared cap of 64 overflows across most of this grid (an early action-noise
# run crashed eight FAST cells that way). measure_token_length.py found the
# largest chunk on the position-injection grid to be 94 action tokens; the cap
# clears that plus the appended EOS with a small margin guarding the tail we did
# not sample, and also covers the action-injection stores (max 97) kept as an
# ablation. Applied only to the FAST arm and only here, rather than by editing
# the shared action_fast.yaml default that other experiments rely on.
FAST_MAX_TOKEN_LEN = 106
FAST_OVERRIDES = (
    f"task.dataloader.tokenization.action_tokenizer.max_token_len={FAST_MAX_TOKEN_LEN}",
)

# A targeted rate-distortion condition for cable hysteresis.  The continuous
# oracle exactly reproduces the biased kinematics, whereas scale 0.2 retains
# only the dominant low-frequency DCT components and reconstructs the clean path
# closely enough to remain well within the clean endpoint
# tolerance. 
CABLE_HYSTERESIS_FAST_SCALE = 0.2
CABLE_HYSTERESIS_COMMON_OVERRIDES = (
    "task.prediction_horizon=59",
    "task.dataloader.trailing_padded_actions=0",
    "task.dataloader.num_workers=0",
    "task.dataset_schema.num_rollouts=10",
    "training.num_epochs=400",
    "experiment.val_every=50",
)
CABLE_HYSTERESIS_FAST_OVERRIDES = (
    "task.dataloader.tokenization.action_tokenizer.action_discretizer.scale="
    f"{CABLE_HYSTERESIS_FAST_SCALE:g}",
)

BINNING_NUM_BINS = 64

# A replicate index picks both the demonstration-noise draw and the training
# seed, so the two vary together and the spread across replicates covers both.
DATA_SEEDS = (42, 43, 44)
TRAIN_SEEDS = (0, 1, 2)

# Deliberately NOT VERSATIL_ZARR_DIR. Every store this sweep writes holds
# noise-corrupted demonstrations, and a normal experiment that resolved its
# dataset through the shared zarr directory must have no way of reaching them.
# Keeping them behind a separate variable makes the separation structural rather
# than a naming convention, and the variable is required rather than defaulted so
# an unset environment fails loudly instead of writing somewhere plausible.
NOISY_ZARR_DIR_ENV = "VERSATIL_NOISY_ZARR_DIR"
NOISY_ZARR_SUBDIR = "tip1_noisy_synthetic"


def noisy_zarr_root() -> str:
    """Return the directory holding this sweep's noise-corrupted stores.

    Raises:
        ValueError: If the environment variable is unset, rather than falling
            back to a working directory or to the shared clean store.
    """
    root = os.environ.get(NOISY_ZARR_DIR_ENV)
    if not root:
        raise ValueError(
            f"{NOISY_ZARR_DIR_ENV} is not set. The Tip 1 sweep writes "
            "noise-corrupted datasets and keeps them out of the shared clean "
            f"store on purpose, so point {NOISY_ZARR_DIR_ENV} at a directory "
            "reserved for them, for example "
            "/data/horse/ws/qizh093f-versatil/noisy_zarr."
        )
    return root


@dataclass(frozen=True)
class DataCell:
    """One generated dataset: everything that changes the episodes, nothing else."""

    task: str
    injection: str
    smoothing_sigma: float
    sigma_multiplier: float
    noise_model: str = SyntheticNoiseModel.GAUSSIAN.value
    data_seed: int = 42
    num_episodes: int | None = None

    @property
    def noise_std(self) -> float:
        """Absolute noise scale for this task's default and this multiplier."""
        return self.sigma_multiplier * TASK_DEFAULT_NOISE_STD[self.task]

    @property
    def band(self) -> str:
        """Band label used in paths and reports."""
        if self.noise_model == SyntheticNoiseModel.CABLE_HYSTERESIS.value:
            return "hysteresis"
        return "high" if self.smoothing_sigma <= 0.0 else "low"

    @property
    def name(self) -> str:
        """Filesystem-safe identifier carrying every generation parameter.

        ``num_episodes`` joins the name when it is overridden, so a smoke-test
        store can never be mistaken for the full one at the same noise setting.
        """
        model_suffix = (
            ""
            if self.noise_model == SyntheticNoiseModel.GAUSSIAN.value
            else f"__model-{self.noise_model}"
        )
        episode_suffix = (
            "" if self.num_episodes is None else f"__ep-{self.num_episodes}"
        )
        return (
            f"{self.task}__inj-{self.injection}__band-{self.band}"
            f"__sig-{self.sigma_multiplier:g}__dseed-{self.data_seed}"
            f"{model_suffix}{episode_suffix}"
        )

    @property
    def zarr_path(self) -> str:
        """Absolute store path; unique per generation parameter combination.

        Raises:
            ValueError: If the noisy-store directory is not configured.
        """
        return str(Path(noisy_zarr_root()) / NOISY_ZARR_SUBDIR / f"{self.name}.zarr")

    def schema_overrides(self) -> list[str]:
        """Hydra overrides selecting this cell's dataset."""
        episode_override = (
            []
            if self.num_episodes is None
            else [f"task.dataset_schema.num_episodes={self.num_episodes}"]
        )
        return [
            f"task/dataset_schema={TASK_SCHEMA_GROUP[self.task]}",
            f"task.dataset_schema.zarr_path={self.zarr_path}",
            f"task.dataset_schema.noise_std={self.noise_std:g}",
            f"task.dataset_schema.noise_smoothing_sigma={self.smoothing_sigma:g}",
            f"task.dataset_schema.noise_injection={self.injection}",
            f"task.dataset_schema.noise_model={self.noise_model}",
            f"task.dataset_schema.seed={self.data_seed}",
            *episode_override,
            # The rollout reference must not follow the training noise: it sets the
            # mode centroids, the success threshold and the radial obstacle
            # geometry, so letting it drift would loosen the bar exactly where
            # performance is supposed to degrade.
            "task.dataset_schema.eval_reference_noise_std=0.0",
        ]


@dataclass(frozen=True)
class TrainCell:
    """One training run: a data cell plus the method and seed that consume it."""

    data: DataCell
    method: str
    seed: int

    @property
    def name(self) -> str:
        """Identifier extending the data cell with method and seed."""
        experiment_name = f"{self.data.name}__{self.method}__seed-{self.seed}"
        if self.data.noise_model == SyntheticNoiseModel.CABLE_HYSTERESIS.value:
            experiment_name += "__horizon-59__full-windows"
            if self.method == "fast":
                experiment_name += f"__fast-scale-{CABLE_HYSTERESIS_FAST_SCALE:g}"
        return experiment_name

    def command(
        self, matched: bool, extra_overrides: tuple[str, ...] = ()
    ) -> list[str]:
        """Build the full training command for this cell.

        The cell name is passed as the experiment name so the run carries its
        full identity into the output directory and the wandb run name. Without
        it every cell of a stage logs under the config's own name and the
        results cannot be matched back to a noise level or a replicate.

        Args:
            matched: Append the matched-backbone overrides.
            extra_overrides: Hydra overrides appended last, so they win. Meant
                for shortening a smoke test, never for a reported comparison.
        """
        overrides = self.data.schema_overrides() + [
            f"experiment.seed={self.seed}",
            f"experiment.name={self.name}",
        ]
        if matched:
            overrides += list(MATCHED_OVERRIDES)
            if self.method == "act":
                overrides += list(MATCHED_ACT_EXTRA)
        if self.method == "fast":
            overrides += list(FAST_OVERRIDES)
        if self.data.noise_model == SyntheticNoiseModel.CABLE_HYSTERESIS.value:
            overrides += list(CABLE_HYSTERESIS_COMMON_OVERRIDES)
            if self.method == "fast":
                overrides += list(CABLE_HYSTERESIS_FAST_OVERRIDES)
        overrides += list(extra_overrides)
        return [
            sys.executable,
            "-m",
            "versatil.endpoints.train",
            "--config-name",
            method_config(task=self.data.task, method=self.method),
            *overrides,
        ]


def _cells(
    tasks: tuple[str, ...],
    injections: tuple[str, ...],
    smoothings: tuple[float, ...],
    multipliers: tuple[float, ...],
    methods: tuple[str, ...],
    replicates: tuple[int, ...],
    noise_models: tuple[str, ...] = (SyntheticNoiseModel.GAUSSIAN.value,),
    num_episodes: int | None = None,
) -> list[TrainCell]:
    """Expand the axes into training cells, one replicate per noise realization.

    A replicate redraws the demonstration noise and re-seeds training together.
    Repeating only the training seed would put error bars around optimization
    stochasticity alone, which cannot answer whether a discrete-continuous gap
    survives a different draw of noise -- the question Tip 1 is asking. Within a
    replicate every method reads the same store, so comparisons between methods
    stay paired and the run count is unchanged.
    """
    train_cells = []
    for (
        task,
        injection,
        smoothing,
        multiplier,
        noise_model,
        replicate,
    ) in itertools.product(
        tasks, injections, smoothings, multipliers, noise_models, replicates
    ):
        # At zero noise the band is a no-op, so only keep the high-band copy.
        if multiplier == 0.0 and smoothing != HIGH_BAND_SMOOTHING:
            continue
        data = DataCell(
            task=task,
            injection=injection,
            smoothing_sigma=smoothing,
            sigma_multiplier=multiplier,
            noise_model=noise_model,
            data_seed=DATA_SEEDS[replicate],
            num_episodes=num_episodes,
        )
        for method in methods:
            train_cells.append(
                TrainCell(data=data, method=method, seed=TRAIN_SEEDS[replicate])
            )
    return train_cells


ACTION = NoiseInjection.ACTION.value
POSITION = NoiseInjection.POSITION.value
GAUSSIAN = SyntheticNoiseModel.GAUSSIAN.value
CABLE_HYSTERESIS = SyntheticNoiseModel.CABLE_HYSTERESIS.value

STAGES = {
    # Stage A: does the sigma grid span "no effect" to "collapse" at all?
    "pilot": {
        "tasks": ("sequential",),
        "injections": (POSITION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": SIGMA_MULTIPLIERS,
        "methods": CORE_METHODS,
        "replicates": (0,),
    },
    # Stage B: the main estimand, both tasks, three seeds. Noise is injected on
    # the trajectory (position injection), the standard way to make a noisy
    # demonstration: the whole demonstration is a coherent noisy trajectory, both
    # the rendered observation and the differenced action label carry it, and both
    # representations face identical inputs so the comparison stays fair.
    "main": {
        "tasks": ("sequential", "radial"),
        "injections": (POSITION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": SIGMA_MULTIPLIERS,
        "methods": CORE_METHODS,
        "replicates": (0, 1, 2),
    },
    # Stage C1: band contrast at matched action-noise power.
    "band": {
        "tasks": ("sequential",),
        "injections": (ACTION,),
        "smoothings": (LOW_BAND_SMOOTHING,),
        "multipliers": (1.0, 4.0),
        "methods": CORE_METHODS,
        "replicates": (0, 1, 2),
    },
    # Stage C2: the external baseline, kept out of the discrete/continuous claim.
    # Same position injection as the main stage so it is comparable.
    "act": {
        "tasks": ("sequential",),
        "injections": (POSITION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": SIGMA_MULTIPLIERS,
        "methods": ("act",),
        "replicates": (0, 1, 2),
    },
    # Ecological-validity layer: noise on the trajectory, so images, clamping and
    # rejection sampling move with it. Diagnostic, not the main estimand.
    "ecological": {
        "tasks": ("sequential", "radial"),
        "injections": (POSITION,),
        "smoothings": (HIGH_BAND_SMOOTHING, LOW_BAND_SMOOTHING),
        "multipliers": SIGMA_MULTIPLIERS,
        "methods": CORE_METHODS,
        "replicates": (0,),
    },
    # Difficulty probe: one continuous run on corridor at the anchor noise, to
    # see whether corridor sits between the saturated sequential and the floored
    # radial. Single seed, one method -- a diagnostic, not an estimand.
    "probe_corridor": {
        "tasks": ("corridor",),
        "injections": (POSITION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (1.0,),
        "methods": ("qfat",),
        "replicates": (0,),
    },
    # External-baseline probe: check whether the plain action transformer (bcat)
    # is a usable continuous baseline where the CVAE "act" arm collapsed.
    "probe_bcat": {
        "tasks": ("sequential",),
        "injections": (POSITION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (1.0,),
        "methods": ("bcat",),
        "replicates": (0,),
    },
    # Range-finding probe: corridor qfat saturates at sigma=1 (success 1.00), so
    # the separation between continuous and discrete has to come from higher
    # noise. Run both arms at sigma=2 and 3 to locate where qfat starts to fall.
    "probe_corridor_hi": {
        "tasks": ("corridor",),
        "injections": (POSITION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (2.0, 3.0),
        "methods": ("qfat", "fast"),
        "replicates": (0,),
    },
    # Range-finding probe: radial floors by sigma=2 on the main grid, so a usable
    # window sits below the current anchor. Run both arms at sigma=1.2 and 1.4 to
    # find noise low enough that the continuous arm is not already on the floor.
    "probe_radial_lo": {
        "tasks": ("radial",),
        "injections": (POSITION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (1.2, 1.4),
        "methods": ("qfat", "fast"),
        "replicates": (0,),
    },
    # The finalized design: one stage per task so each carries its own sigma grid
    # on the measured-SNR axis. Sequential and corridor span 1-4; radial floors the
    # discrete arm quickly, so it uses a compressed low grid (range-finding showed
    # both arms dead by sigma=2). All four methods, three replicates.
    "final_sequential": {
        "tasks": ("sequential",),
        "injections": (POSITION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (1.0, 2.0, 3.0, 4.0),
        "methods": FINAL_METHODS,
        "replicates": (0, 1, 2),
    },
    "final_radial": {
        "tasks": ("radial",),
        "injections": (POSITION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (1.0, 1.2, 1.4, 1.6),
        "methods": FINAL_METHODS,
        "replicates": (0, 1, 2),
    },
    "final_corridor": {
        "tasks": ("corridor",),
        "injections": (POSITION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (1.0, 2.0, 3.0, 4.0),
        "methods": FINAL_METHODS,
        "replicates": (0, 1, 2),
    },
    # Single-seed pass of the three settled arms (the external baseline is held
    # out pending its head choice). Methods are ordered qfat, fast, binned so a
    # cell's index is sigma_position*3 + method_position; the submitter runs only
    # the indices not already completed. Multi-seed is the same stages with more
    # replicates, added once the design is locked.
    "final_sequential_s0": {
        "tasks": ("sequential",),
        "injections": (POSITION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (1.0, 2.0, 3.0, 4.0),
        "methods": ("qfat", "fast", "binned"),
        "replicates": (0,),
    },
    "final_radial_s0": {
        "tasks": ("radial",),
        "injections": (POSITION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (1.0, 1.2, 1.4, 1.6),
        "methods": ("qfat", "fast", "binned"),
        "replicates": (0,),
    },
    "final_corridor_s0": {
        "tasks": ("corridor",),
        "injections": (POSITION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (1.0, 2.0, 3.0, 4.0),
        "methods": ("qfat", "fast", "binned"),
        "replicates": (0,),
    },
    # The primary comparison. On the multimodal tasks the continuous arm's
    # success tracked which modes it kept rather than how accurately it executed
    # under noise, so the estimand moves to the context-conditioned circle, where
    # the mode is given and each arm is unimodal. Noise reaches training action
    # labels only; positions and rendered observations stay on the clean path.
    # The plain action transformer is a valid arm here because there is nothing
    # to mode-average. Its reported number is conditional success: success on the
    # route the context asked for.
    "final_conditional_s0": {
        "tasks": (CONDITIONAL_TASK,),
        "injections": (ACTION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (1.0, 2.0, 3.0, 4.0),
        "methods": FINAL_METHODS,
        "replicates": (0,),
    },
    "final_conditional": {
        "tasks": (CONDITIONAL_TASK,),
        "injections": (ACTION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (1.0, 2.0, 3.0, 4.0),
        "methods": FINAL_METHODS,
        "replicates": (0, 1, 2),
    },
    # One strong, systematic kinematic-error condition. The play operator models
    # cable backlash: labels come from the lagged internal kinematic state while
    # images and stored positions remain ground truth. Unlike independent
    # Gaussian draws, this history-dependent error does not disappear by
    # repeating the same demonstration.
    "conditional_hysteresis_s0": {
        "tasks": (CONDITIONAL_TASK,),
        "injections": (ACTION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (4.0,),
        "methods": FINAL_METHODS,
        "replicates": (0,),
        "noise_models": (CABLE_HYSTERESIS,),
    },
    # Deliberately targeted stress test, calibrated analytically before
    # training. At threshold 0.080 the exact continuous label trajectory misses
    # the clean endpoint, while FAST's scale-0.2 reconstruction remains inside the
    # fixed clean-reference success radius. This stage tests that predicted
    # model behavior at one setting; it is not an unbiased robustness sweep.
    "conditional_hysteresis_fast_win_s0": {
        "tasks": (CONDITIONAL_TASK,),
        "injections": (ACTION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (10.0,),
        "methods": ("fast", "qfat", "bcat"),
        "replicates": (0,),
        "noise_models": (CABLE_HYSTERESIS,),
    },
}


def stage_cells(stage: str, num_episodes: int | None = None) -> list[TrainCell]:
    """Training cells for a named stage.

    Args:
        stage: Stage name from ``STAGES``.
        num_episodes: Override the per-cell episode count. Smoke testing only;
            it becomes part of the store name so a reduced store cannot be
            mistaken for the full one.

    Raises:
        ValueError: If the stage name is unknown.
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Expected one of {sorted(STAGES)}.")
    return _cells(**STAGES[stage], num_episodes=num_episodes)


def data_cells(train_cells: list[TrainCell]) -> list[DataCell]:
    """Distinct data cells behind a set of training cells, in stable order."""
    seen: dict[str, DataCell] = {}
    for cell in train_cells:
        seen.setdefault(cell.data.name, cell.data)
    return list(seen.values())


def check_paths_unique(cells: list[DataCell]) -> None:
    """Fail loudly if two different cells would share a zarr store.

    Raises:
        ValueError: If any store path is claimed by more than one cell.
    """
    by_path: dict[str, list[str]] = {}
    for cell in cells:
        by_path.setdefault(cell.zarr_path, []).append(cell.name)
    collisions = {path: names for path, names in by_path.items() if len(names) > 1}
    if collisions:
        raise ValueError(
            "Distinct sweep cells map to the same zarr store, which would make "
            f"them silently share data: {collisions}"
        )


class _RejectionCapture(logging.Handler):
    """Collect the generator's rejection-sampling summary for the manifest.

    The generator reports this summary at warning level only when rejection is
    heavy enough to be a threat, and at info level otherwise. Capturing just the
    warnings would leave every moderate rejection rate indistinguishable from no
    rejection at all, so the caller lowers the logger's level for the duration of
    generation and this handler accepts both.
    """

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.mean_attempts: float | None = None
        self.rejection_rate: float | None = None
        self.captured = False

    def emit(self, record: logging.LogRecord) -> None:
        """Record the summary arguments when the generator logs one."""
        if record.args and "rejection sampling" in str(record.msg):
            self.mean_attempts = float(record.args[2])
            self.rejection_rate = float(record.args[3])
            self.captured = True


def generate_cell(cell: DataCell) -> dict[str, float | str]:
    """Create one cell's zarr store and measure what the sweep must disclose.

    Returns:
        Manifest row with the store path, the fitted action range that min-max
        normalization will use, and the rejection statistics when the task uses
        rejection sampling.
    """
    with initialize_config_dir(
        config_dir=str(get_hydra_configs_dir()), version_base=None
    ):
        config = compose(
            config_name=method_config(task=cell.task, method="fast"),
            overrides=cell.schema_overrides(),
        )
    schema = instantiate(config.task.dataset_schema)

    capture = _RejectionCapture()
    generator_logger = logging.getLogger("versatil.data.synthetic.generators")
    previous_level = generator_logger.level
    generator_logger.addHandler(capture)
    generator_logger.setLevel(logging.INFO)
    regenerated = not Path(cell.zarr_path).exists()
    try:
        if regenerated:
            create_replay_buffer_from_synthetic(schema=schema)
    finally:
        generator_logger.removeHandler(capture)
        generator_logger.setLevel(previous_level)

    buffer = ReplayBuffer.create_from_path(cell.zarr_path)
    actions = np.asarray(buffer[ProprioKey.SYNTHETIC_POSITION_ACTION.value][:])
    return {
        "measured_snr": measured_snr(cell=cell, actions=actions),
        "cell": cell.name,
        "task": cell.task,
        "injection": cell.injection,
        "noise_model": cell.noise_model,
        "band": cell.band,
        "sigma_multiplier": cell.sigma_multiplier,
        "noise_std": cell.noise_std,
        "zarr_path": cell.zarr_path,
        "num_timesteps": int(actions.shape[0]),
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
        "action_range": float(actions.max() - actions.min()),
        "action_std": float(actions.std()),
        # Distinguish "measured and found to be zero" from "not measured": a
        # store that already existed was not regenerated, so no summary was
        # emitted and reporting 0.0 would claim a rejection rate that was never
        # observed.
        "rejection_measured": bool(capture.captured),
        "mean_attempts": capture.mean_attempts if capture.captured else float("nan"),
        "rejection_rate": capture.rejection_rate if capture.captured else float("nan"),
    }


def measured_snr(cell: DataCell, actions: np.ndarray) -> float:
    """Signal-to-noise ratio measured on the generated actions themselves.

    Defined as the per-element root-mean-square of the clean action divided by
    the per-element root-mean-square of the injected perturbation, both taken
    over timesteps and dimensions:

        SNR = RMS(a_clean) / RMS(a_noisy - a_clean)

    This replaces the earlier mixture of "mean action norm over per-dimension
    noise standard deviation", which the plan quoted two mutually inconsistent
    ways. It is measured rather than derived, so it stays correct whatever the
    injection point and band do, and it is comparable across tasks whose action
    magnitudes differ -- unlike the sigma multiplier, which is task-relative by
    construction. Report it on the primary axis and keep the multiplier for the
    tables.

    Args:
        cell: The noisy cell whose actions were passed in.
        actions: Flattened action array read back from that cell's store.

    Returns:
        Measured ratio, or infinity for the noise-free cell.
    """
    if cell.sigma_multiplier == 0.0:
        return float("inf")
    clean_cell = DataCell(
        task=cell.task,
        injection=cell.injection,
        smoothing_sigma=HIGH_BAND_SMOOTHING,
        sigma_multiplier=0.0,
        data_seed=cell.data_seed,
        num_episodes=cell.num_episodes,
    )
    clean_actions = np.asarray(
        ReplayBuffer.create_from_path(clean_cell.zarr_path)[
            ProprioKey.SYNTHETIC_POSITION_ACTION.value
        ][:]
    )
    perturbation = actions - clean_actions
    signal_rms = float(np.sqrt(np.mean(clean_actions**2)))
    noise_rms = float(np.sqrt(np.mean(perturbation**2)))
    return signal_rms / noise_rms if noise_rms > 0.0 else float("inf")


def clean_reference_range(cell: DataCell) -> float | None:
    """Action range of the zero-noise store matching this cell's task and seed.

    Read from disk rather than from the current manifest, because a stage need
    not contain a zero-noise cell of its own -- the band contrast, for instance,
    only enumerates noisy levels -- and would otherwise have no reference at all.

    Returns:
        The fitted action range, or None when that store has not been generated.
    """
    reference_cell = DataCell(
        task=cell.task,
        injection=cell.injection,
        smoothing_sigma=HIGH_BAND_SMOOTHING,
        sigma_multiplier=0.0,
        data_seed=cell.data_seed,
        num_episodes=cell.num_episodes,
    )
    if not Path(reference_cell.zarr_path).exists():
        return None
    actions = np.asarray(
        ReplayBuffer.create_from_path(reference_cell.zarr_path)[
            ProprioKey.SYNTHETIC_POSITION_ACTION.value
        ][:]
    )
    return float(actions.max() - actions.min())


def add_effective_bins(
    rows: list[dict[str, float | str]], cells: list[DataCell]
) -> None:
    """Annotate each row with how many bins the clean signal still occupies.

    Min-max normalization is refitted per cell, so a wider noisy range squeezes
    the clean signal into a smaller part of [-1, 1] and the fixed bin budget is
    spent representing noise rather than signal.
    """
    by_name = {cell.name: cell for cell in cells}
    reference_cache: dict[tuple[str, str, int], float | None] = {}
    for row in rows:
        cell = by_name[str(row["cell"])]
        key = (cell.task, cell.injection, cell.data_seed)
        if key not in reference_cache:
            reference_cache[key] = clean_reference_range(cell)
        reference = reference_cache[key]
        if not reference or not row["action_range"]:
            row["range_inflation"] = float("nan")
            row["effective_bins"] = float("nan")
            continue
        inflation = float(row["action_range"]) / reference
        row["range_inflation"] = inflation
        row["effective_bins"] = BINNING_NUM_BINS / inflation


def reference_cells(cells: list[DataCell]) -> list[DataCell]:
    """Zero-noise cells the given cells measure against but may not enumerate.

    Signal-to-noise ratio and range inflation are both defined relative to the
    noise-free store of the same task, injection point and data seed. A stage
    need not contain that store: the band contrast enumerates only noisy levels.
    Deriving the references here makes every stage self-sufficient, instead of
    depending on another stage having been generated first -- a dependency that
    is invisible until two stages run concurrently and one reads a store the
    other has not finished writing.
    """
    needed = {
        DataCell(
            task=cell.task,
            injection=cell.injection,
            smoothing_sigma=HIGH_BAND_SMOOTHING,
            sigma_multiplier=0.0,
            data_seed=cell.data_seed,
            num_episodes=cell.num_episodes,
        )
        for cell in cells
    }
    enumerated = {cell.name for cell in cells}
    return [cell for cell in needed if cell.name not in enumerated]


def run_data(stage: str, output_dir: Path, num_episodes: int | None) -> None:
    """Generate every store the stage needs and write the manifest."""
    cells = data_cells(stage_cells(stage, num_episodes=num_episodes))
    check_paths_unique(cells)
    output_dir.mkdir(parents=True, exist_ok=True)

    # References are generated but not reported: they belong to the stage that
    # enumerates them, and repeating them here would double-count in the manifest.
    references = reference_cells(cells)
    for index, cell in enumerate(references, start=1):
        print(f"[reference {index}/{len(references)}] {cell.name}", flush=True)
        generate_cell(cell)

    # Sorted so each replicate's zero-noise store is built before the noisy
    # cells that measure their signal-to-noise ratio against it.
    cells = sorted(
        cells, key=lambda item: (item.task, item.data_seed, item.sigma_multiplier)
    )
    rows = []
    for index, cell in enumerate(cells, start=1):
        print(f"[{index}/{len(cells)}] {cell.name}", flush=True)
        rows.append(generate_cell(cell))
    add_effective_bins(rows, cells)

    manifest = output_dir / f"manifest_{stage}.csv"
    with open(manifest, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'cell':<58}{'SNR':>8}{'inflation':>11}{'eff.bins':>10}{'reject':>9}")
    for row in rows:
        print(
            f"{str(row['cell']):<58}{float(row['measured_snr']):>8.2f}"
            f"{float(row['range_inflation']):>11.2f}"
            f"{float(row['effective_bins']):>10.1f}"
            f"{float(row['rejection_rate']):>9.3f}"
        )
    print(f"\nWrote {manifest}")


def run_train(
    stage: str,
    matched: bool,
    dry_run: bool,
    index: int | None,
    extra_overrides: tuple[str, ...] = (),
) -> None:
    """Run (or print) the stage's training commands.

    Args:
        stage: Stage name from ``STAGES``.
        matched: Apply the matched-backbone overrides.
        dry_run: Print the commands without running them.
        index: Zero-based cell to run alone, for one element of a job array.
            ``None`` runs the whole stage sequentially. The enumeration is
            deterministic, so an array element and a sequential run at the same
            position resolve to the same cell.
        extra_overrides: Hydra overrides appended to every command.

    Raises:
        IndexError: If ``index`` falls outside the stage.
    """
    cells = stage_cells(stage)
    check_paths_unique(data_cells(cells))
    if index is not None:
        if not 0 <= index < len(cells):
            raise IndexError(
                f"Cell index {index} is outside stage '{stage}', which has "
                f"{len(cells)} cells (valid indices 0-{len(cells) - 1})."
            )
        cells = [cells[index]]
    for position, cell in enumerate(cells, start=1):
        command = cell.command(matched=matched, extra_overrides=extra_overrides)
        print(f"[{position}/{len(cells)}] {cell.name}", flush=True)
        if dry_run:
            print("  " + " ".join(command))
            continue
        subprocess.run(command, check=True)


def main() -> None:
    """Parse arguments and dispatch to the data or training path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["data", "train", "list"])
    parser.add_argument("output_dir", nargs="?", default=None)
    parser.add_argument("--stage", default="pilot", choices=sorted(STAGES))
    parser.add_argument(
        "--tuned",
        action="store_true",
        help="Use each method's own tuned config instead of the matched arm.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Zero-based cell to train alone, for one SLURM array element.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Hydra override, repeatable. For smoke tests, not results.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Shrink each cell for a smoke test; recorded in the store name.",
    )
    arguments = parser.parse_args()

    if arguments.action == "list":
        cells = stage_cells(arguments.stage, num_episodes=arguments.num_episodes)
        stores = data_cells(cells)
        check_paths_unique(stores)
        print(f"stage={arguments.stage}: {len(cells)} runs over {len(stores)} stores")
        for cell in stores:
            print(f"  {cell.name}")
        return

    if arguments.action == "data":
        if arguments.output_dir is None:
            parser.error("data requires an output_dir")
        run_data(
            stage=arguments.stage,
            output_dir=Path(arguments.output_dir),
            num_episodes=arguments.num_episodes,
        )
        return

    run_train(
        stage=arguments.stage,
        matched=not arguments.tuned,
        dry_run=arguments.dry_run,
        index=arguments.index,
        extra_overrides=tuple(arguments.override),
    )


if __name__ == "__main__":
    main()
