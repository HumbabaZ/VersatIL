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
from versatil.data.synthetic.constants import (
    MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,
    NoiseInjection,
    SyntheticNoiseModel,
)

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
# The tokenized arms frame the chunk one step shorter than the continuous ones:
# an episode of T positions has T-1 real actions plus a zero sentinel, and the
# tokenizer never sees the sentinel.
TOKENIZED_METHODS = ("fast", "binned")


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
ACTION_TOKENIZER_MAX_TOKEN_LEN_KEY = (
    "task.dataloader.tokenization.action_tokenizer.max_token_len"
)
FAST_OVERRIDES = (f"{ACTION_TOKENIZER_MAX_TOKEN_LEN_KEY}={FAST_MAX_TOKEN_LEN}",)
# The cap depends on the chunk length: a denser-sampled loop has more DCT
# coefficients, and FAST's post-BPE length grows with them in a data-dependent
# way, so each trajectory length gets its own measured value. A length missing
# here fails at command time rather than on the first training batch.
# The 120 and 240 entries are measured, not guessed: job 4126443 ran
# measure_token_length.py over every chunk of both control-rate stores (2000 of
# 2000, so the maximum is exact rather than sampled) and found 92 and 172 action
# tokens. Each cap clears its measured maximum by the tool's margin of 8 plus
# the EOS. The provisional bounds they replace, 400 and 800, were roughly 4x
# oversized; padding is masked out of the loss so that cost correctness nothing,
# but attention is quadratic in the padded length, so the 240-step FAST cell was
# set to attend about 20x the compute it needs.
# The 60 entry stays at the value measured across the whole sigma grid: the
# control-rate stores sit at sigma=1, where FAST emits only 49 tokens, but
# final_conditional spans sigma up to 4 and FAST's length grows with the noise.
FAST_MAX_TOKEN_LEN_BY_LENGTH = {
    MULTIPATH_DEFAULT_TRAJECTORY_LENGTH: FAST_MAX_TOKEN_LEN,
    120: 101,
    240: 181,
    # Provisional, pending measurement on the stores once they exist. The three
    # measured points sit close to 0.7 tokens per step (49, 92, 172 at 60, 120
    # and 240), so these are that fit with roughly 40% headroom. Measure with
    # measure_token_length.py and tighten: padding is masked out of the loss, so
    # an oversized cap costs only compute, but it costs it quadratically.
    400: 400,
    1000: 1000,
}
# The GPT decoders size a precomputed positional table from this; the default of
# 512 leaves a binned chunk at 240 steps (478 tokens plus prefix) a margin of a
# few tokens, so longer chunks raise it. It costs no parameters.
# Episodes at or beyond this length stream from disk instead of preloading.
PRELOAD_LENGTH_LIMIT = 400
GPT_MAX_SEQ_LEN_KEY = "policy.decoder.max_seq_len"
# Every non-default length trained so far used this table size, so it stays the
# floor: shrinking it for the shorter lengths would give a replicate a smaller
# positional table than the replicate it is meant to sit beside, which is a
# different model rather than a different seed.
GPT_LONG_MAX_SEQ_LEN = 1024
# Room for the observation prefix and the EOS on top of the action tokens.
GPT_SEQ_LEN_MARGIN = 64


def gpt_max_seq_len(action_tokens: int) -> int:
    """Positional-table size for a tokenized arm at a non-default length.

    Grows past the historical 1024 only when the chunk needs it, which the
    control-rate axis reaches once episodes get long: binning emits two tokens
    per step, so a 1000-step chunk needs about 2000 positions and would
    silently exceed a fixed 1024. Powers of two keep the count off the critical
    path of any kernel that prefers them, and the table is precomputed, so a
    larger one costs no parameters.
    """
    needed = action_tokens + GPT_SEQ_LEN_MARGIN
    size = GPT_LONG_MAX_SEQ_LEN
    while size < needed:
        size *= 2
    return size


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


def fast_max_token_len(trajectory_length: int) -> int:
    """FAST token cap measured for a trajectory length.

    Raises:
        ValueError: If no measurement exists for the length, naming the tool
            that produces one.
    """
    if trajectory_length not in FAST_MAX_TOKEN_LEN_BY_LENGTH:
        raise ValueError(
            "FAST max_token_len is not measured for "
            f"trajectory_length={trajectory_length}; run "
            "measure_token_length.py --stage <stage> and add the value to "
            "FAST_MAX_TOKEN_LEN_BY_LENGTH."
        )
    return FAST_MAX_TOKEN_LEN_BY_LENGTH[trajectory_length]


# Set this when a change makes previously trained cells incomparable with new
# ones -- a different injection point, decoding rule, normaliser setting, or
# anything else that alters what a cell means rather than only how it is scored.
# The tag joins every experiment name, and therefore every checkpoint directory,
# so the two batches cannot land in the same place.
#
# Without it they do, and the failure is silent in a way worth spelling out: the
# framework does not overwrite on a name collision, it writes the newer files
# with a "-v1" suffix beside the older ones. A directory then holds two runs,
# and the *un-suffixed* files are the OLD ones -- so the obvious way to find the
# final checkpoint, the highest epoch among "latest-<epoch>.ckpt", silently
# returns stale weights. That happened to the tremor cells, which reuse the
# names of an earlier position-injection run, and it reversed the reported
# ordering of the two discrete arms at the highest noise level.
#
# Empty is correct for the current grid: everything trained since the two-error-
# source rework shares one revision, and tagging now would orphan it. Bump this
# (to "r2", "r3", ...) as part of the change that invalidates the old runs, not
# afterwards.
#
# Note this is deliberately not a per-stage suffix. Stages are meant to share
# cells -- the control-rate axis reuses the anchor cells the noise sweep already
# trained, and a stage tag would make those two different experiments.
SWEEP_REVISION = ""

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


CHECKPOINT_DIR_ENV = "VERSATIL_CHECKPOINT_DIR"
CHECKPOINT_SUBDIR = "synthetic"


def checkpoint_dir(cell: "TrainCell") -> Path:
    """Directory a cell's weights are written to and read back from.

    Single source of truth for that path, so the driver's collision check and
    any offline re-scoring resolve a cell to the same place. The layout follows
    the training workspace: the configured checkpoint root, the config's own
    directory, then the experiment name.
    """
    root = os.environ.get(CHECKPOINT_DIR_ENV, ".")
    config_dir = method_config(task=cell.data.task, method=cell.method).split("/")[-1]
    return Path(root) / CHECKPOINT_SUBDIR / config_dir / cell.name


def existing_checkpoints(cell: "TrainCell") -> list[Path]:
    """Checkpoints already sitting where this cell would write.

    Non-empty means an earlier run used this name. Training over it does not
    replace those files -- the newer ones land beside them with a "-v1" suffix,
    leaving one directory holding two runs whose un-suffixed files are the older
    of the two. Callers should refuse rather than merge the two runs.
    """
    directory = checkpoint_dir(cell)
    return sorted(directory.glob("*.ckpt")) if directory.is_dir() else []


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
    # Timesteps per episode. The path is fixed, so a longer episode samples the
    # same geometry denser: this is the control-rate axis.
    trajectory_length: int = MULTIPATH_DEFAULT_TRAJECTORY_LENGTH

    @property
    def noise_std(self) -> float:
        """Absolute noise scale for this task's default and this multiplier.

        Scaled inversely with the trajectory length so the per-step
        signal-to-noise ratio stays fixed along the control-rate axis: the
        clean per-step displacement shrinks with denser sampling while the
        noise is drawn per step either way, so an unscaled noise would confound
        the rate with the noise level. The scaling holds under both injection
        points, since both put a per-step draw of fixed scale against a
        displacement that shrinks with the length.

        Under ``CABLE_HYSTERESIS`` this is the play-operator backlash threshold
        rather than a standard deviation, so the length scaling keeps the
        threshold a fixed fraction of the per-step displacement.
        """
        return (
            self.sigma_multiplier
            * TASK_DEFAULT_NOISE_STD[self.task]
            * MULTIPATH_DEFAULT_TRAJECTORY_LENGTH
            / self.trajectory_length
        )

    @property
    def band(self) -> str:
        """Band label used in paths and reports."""
        if self.noise_model == SyntheticNoiseModel.CABLE_HYSTERESIS.value:
            return "hysteresis"
        return "high" if self.smoothing_sigma <= 0.0 else "low"

    @property
    def has_default_length(self) -> bool:
        """Whether the episode length is the benchmark default."""
        return self.trajectory_length == MULTIPATH_DEFAULT_TRAJECTORY_LENGTH

    @property
    def name(self) -> str:
        """Filesystem-safe identifier carrying every generation parameter.

        A non-default ``noise_model``, ``num_episodes`` and a non-default
        ``trajectory_length`` join the name when set, so a hysteresis store, a
        smoke-test store or a denser-sampled store can never be mistaken for
        the default one at the same noise setting.
        """
        model_suffix = (
            ""
            if self.noise_model == SyntheticNoiseModel.GAUSSIAN.value
            else f"__model-{self.noise_model}"
        )
        suffix = "" if self.num_episodes is None else f"__ep-{self.num_episodes}"
        if not self.has_default_length:
            suffix += f"__T-{self.trajectory_length}"
        return (
            f"{self.task}__inj-{self.injection}__band-{self.band}"
            f"__sig-{self.sigma_multiplier:g}__dseed-{self.data_seed}"
            f"{model_suffix}{suffix}"
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
        length_override = (
            []
            if self.has_default_length
            else [f"task.dataset_schema.trajectory_length={self.trajectory_length}"]
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
            *length_override,
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
        """Identifier extending the data cell with method, seed and revision.

        The name is the experiment name, and therefore the checkpoint directory,
        so two runs that share it share a directory. That is deliberate between
        stages -- the control-rate axis reuses the anchor cells the noise sweep
        already trained -- and wrong between revisions, which is what
        ``SWEEP_REVISION`` separates.
        """
        experiment_name = f"{self.data.name}__{self.method}__seed-{self.seed}"
        if self.data.noise_model == SyntheticNoiseModel.CABLE_HYSTERESIS.value:
            experiment_name += "__horizon-59__full-windows"
            if self.method == "fast":
                experiment_name += f"__fast-scale-{CABLE_HYSTERESIS_FAST_SCALE:g}"
        if SWEEP_REVISION:
            experiment_name += f"__{SWEEP_REVISION}"
        return experiment_name

    @property
    def prediction_horizon(self) -> int:
        """Chunk length covering the whole episode for this method's family."""
        if self.method in TOKENIZED_METHODS:
            return self.data.trajectory_length - 1
        return self.data.trajectory_length

    def length_overrides(self) -> list[str]:
        """Hydra overrides that follow a non-default episode length.

        The chunk must still cover the whole episode, so the horizon moves with
        the length. Binning emits a fixed two tokens per step and its cap must
        clear that count plus the EOS; both GPT arms get a longer positional
        table so a long chunk plus its observation prefix fits.
        """
        if self.data.has_default_length:
            return []
        overrides = [f"task.prediction_horizon={self.prediction_horizon}"]
        if self.method == "binned":
            action_tokens = 2 * self.prediction_horizon + 2
            overrides.append(f"{ACTION_TOKENIZER_MAX_TOKEN_LEN_KEY}={action_tokens}")
        elif self.method == "fast":
            action_tokens = fast_max_token_len(self.data.trajectory_length)
        if self.method in TOKENIZED_METHODS:
            overrides.append(f"{GPT_MAX_SEQ_LEN_KEY}={gpt_max_seq_len(action_tokens)}")
        if self.data.trajectory_length >= PRELOAD_LENGTH_LIMIT:
            # The synthetic default preloads the whole store into RAM, which its
            # own documentation limits to small datasets. A 1000-step store is
            # 12 GB of frames before the float conversion, and the cluster caps
            # a single-GPU job below what that needs once copies are counted.
            # Shorter lengths keep preloading, so the cells already trained stay
            # reproducible.
            overrides.append("task.dataloader.preload_data_in_memory=false")
        return overrides

    def overrides(
        self, matched: bool, extra_overrides: tuple[str, ...] = ()
    ) -> list[str]:
        """Every Hydra override the training command carries, in order.

        Args:
            matched: Append the matched-backbone overrides.
            extra_overrides: Hydra overrides appended last, so they win. Meant
                for shortening a smoke test, never for a reported comparison.

        Raises:
            ValueError: If the FAST arm has no measured token cap for this
                cell's trajectory length.
        """
        overrides = self.data.schema_overrides() + [
            f"experiment.seed={self.seed}",
            f"experiment.name={self.name}",
        ]
        if matched:
            overrides += list(MATCHED_OVERRIDES)
            if self.method == "act":
                overrides += list(MATCHED_ACT_EXTRA)
        overrides += self.length_overrides()
        if self.method in TOKENIZED_METHODS:
            # Only synthetic_default.yaml turns off min-max range clamping; the
            # tokenized dataloaders inherit clamp_kinematics_range=True with
            # min_kinematics_range=0.01 from the structured default. Per-step
            # deltas shrink as 1/T, so on the rate axis the clamp starts
            # attenuating exactly the discrete arms under test once a delta
            # range falls under 0.01 (clean T=400 range is 0.0063). Turning it
            # off is a no-op for every cell trained so far: the smallest range
            # in any training store is 0.0105 (sigma-0, T=240).
            overrides.append("task.dataloader.clamp_kinematics_range=false")
        if self.data.sigma_multiplier == 0.0:
            # Clean data converges long before the noisy-protocol 2000 epochs:
            # the external clean reference trained 400 epochs on 400 episodes,
            # and 800 epochs on our 1000 episodes is five times that training
            # volume. Checkpoints land every 100 epochs, so cells already
            # trained to 2000 are scored at the same epoch offline instead of
            # being retrained.
            overrides.append("training.num_epochs=800")
        if self.method == "fast":
            cap = fast_max_token_len(self.data.trajectory_length)
            overrides.append(f"{ACTION_TOKENIZER_MAX_TOKEN_LEN_KEY}={cap}")
        if self.data.noise_model == SyntheticNoiseModel.CABLE_HYSTERESIS.value:
            overrides += list(CABLE_HYSTERESIS_COMMON_OVERRIDES)
            if self.method == "fast":
                overrides += list(CABLE_HYSTERESIS_FAST_OVERRIDES)
        overrides += list(extra_overrides)
        return overrides

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
        return [
            sys.executable,
            "-m",
            "versatil.endpoints.train",
            "--config-name",
            method_config(task=self.data.task, method=self.method),
            *self.overrides(matched=matched, extra_overrides=extra_overrides),
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
    trajectory_lengths: tuple[int, ...] = (MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,),
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
    # The noise model and the length sit inside the noise loop and outside the
    # replicate loop, so a stage that keeps the Gaussian default and the default
    # length enumerates exactly as before.
    for (
        task,
        injection,
        smoothing,
        multiplier,
        noise_model,
        length,
        replicate,
    ) in itertools.product(
        tasks,
        injections,
        smoothings,
        multipliers,
        noise_models,
        trajectory_lengths,
        replicates,
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
            trajectory_length=length,
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
    # The stochastic error source: surgeon-side physiological tremor. Position
    # injection is its physical model -- the demonstrated trajectory itself
    # shakes, so the rendered observations carry the noise and differencing
    # makes the action noise high-frequency by construction. This is a separate
    # estimand from the stages above (per-method G(sigma) under world noise),
    # not a control for the hysteresis condition, so it does not need to share
    # that condition's injection point. The generation parameters match the
    # stores prepared for the original position-injection comparison, so the
    # existing stores are reused and no data is generated.
    "tremor_conditional_s0": {
        "tasks": (CONDITIONAL_TASK,),
        "injections": (POSITION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (1.0, 2.0, 3.0, 4.0),
        "methods": FINAL_METHODS,
        "replicates": (0,),
    },
    "tremor_conditional": {
        "tasks": (CONDITIONAL_TASK,),
        "injections": (POSITION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (1.0, 2.0, 3.0, 4.0),
        "methods": FINAL_METHODS,
        "replicates": (0, 1, 2),
    },
    # The control-rate axis: the same loop sampled 2x and 4x denser at the
    # anchor noise, with noise_std scaled down to hold the per-step SNR. This is
    # the regime FAST's own claim is about (a high control rate makes per-step
    # binning long and low-information), separate from the noise axis above.
    # Action injection, matching the Gaussian control and hysteresis stages, so
    # the noise model is the only thing that varies between them.
    # The first four cells coincide with final_conditional_s0's sigma=1 cells,
    # so a submission that already has those starts at index 4.
    "rate_conditional_s0": {
        "tasks": (CONDITIONAL_TASK,),
        "injections": (ACTION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (1.0,),
        "trajectory_lengths": (60, 120, 240, 400, 1000),
        "methods": FINAL_METHODS,
        "replicates": (0,),
    },
    "rate_conditional": {
        "tasks": (CONDITIONAL_TASK,),
        "injections": (ACTION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (1.0,),
        "trajectory_lengths": (60, 120, 240, 400, 1000),
        "methods": FINAL_METHODS,
        "replicates": (0, 1, 2),
    },
    # The clean control-rate axis: the same T grid with no noise at all. The
    # noisy grid above holds per-step SNR fixed, so total label corruption
    # falls as 1/sqrt(T) and a high-T gain cannot be attributed to the control
    # rate alone. Zero noise removes that confound: what remains is the pure
    # effect of sampling density on each representation (the supervisor's
    # frequency_control design, and the thesis mainline for this axis). With
    # sigma=0 there is no noise draw to redraw, so replicates vary the
    # training seed; the paired data seeds are kept for naming consistency.
    "rate_conditional_clean": {
        "tasks": (CONDITIONAL_TASK,),
        "injections": (ACTION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (0.0,),
        "trajectory_lengths": (60, 120, 240, 400, 1000),
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
    # Three replicates of the same condition. Replicate 0 reuses the cells the
    # single-seed stage already trained, so a submission that has those starts
    # at index 4.
    "conditional_hysteresis": {
        "tasks": (CONDITIONAL_TASK,),
        "injections": (ACTION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (4.0,),
        "methods": FINAL_METHODS,
        "replicates": (0, 1, 2),
        "noise_models": (CABLE_HYSTERESIS,),
    },
    "conditional_hysteresis_fast_win_s0": {
        "tasks": (CONDITIONAL_TASK,),
        "injections": (ACTION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (10.0,),
        # Binning was added after the first three arms had trained. It sits
        # last so those cells keep their array indices: only index 3 is new.
        "methods": ("fast", "qfat", "bcat", "binned"),
        "replicates": (0,),
        "noise_models": (CABLE_HYSTERESIS,),
    },
    "conditional_hysteresis_fast_win": {
        "tasks": (CONDITIONAL_TASK,),
        "injections": (ACTION,),
        "smoothings": (HIGH_BAND_SMOOTHING,),
        "multipliers": (10.0,),
        "methods": ("fast", "qfat", "bcat", "binned"),
        "replicates": (0, 1, 2),
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
        "trajectory_length": cell.trajectory_length,
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
        trajectory_length=cell.trajectory_length,
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
        trajectory_length=cell.trajectory_length,
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
    reference_cache: dict[tuple[str, str, int, int], float | None] = {}
    for row in rows:
        cell = by_name[str(row["cell"])]
        key = (cell.task, cell.injection, cell.data_seed, cell.trajectory_length)
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
            trajectory_length=cell.trajectory_length,
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
        cells,
        key=lambda item: (
            item.task,
            item.trajectory_length,
            item.data_seed,
            item.sigma_multiplier,
        ),
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
    allow_existing: bool = False,
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
        allow_existing: Train even though the cell's checkpoint directory
            already holds an earlier run. Only for deliberately continuing one.

    Raises:
        IndexError: If ``index`` falls outside the stage.
        FileExistsError: If a cell's checkpoint directory already holds an
            earlier run and ``allow_existing`` is not set.
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
        # Training into a directory that already holds a run does not replace
        # it, and the mixture that results reads as one run while returning the
        # older weights. Refuse here, where the fix is still cheap, rather than
        # at analysis time where it has already changed a reported number.
        existing = existing_checkpoints(cell)
        if existing and not allow_existing:
            raise FileExistsError(
                f"{checkpoint_dir(cell)} already holds {len(existing)} "
                f"checkpoints from an earlier run of '{cell.name}'. Training "
                "again would write beside them, not over them, leaving one "
                "directory with two runs. Bump SWEEP_REVISION if the new run "
                "is not comparable with the old one, clear the directory if "
                "the old one is disposable, or pass --allow-existing to "
                "continue the run that is already there."
            )
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
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help=(
            "Train even though the cell already has checkpoints. Only for "
            "continuing that run; a fresh one belongs under a new "
            "SWEEP_REVISION."
        ),
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
        allow_existing=arguments.allow_existing,
    )


if __name__ == "__main__":
    main()
