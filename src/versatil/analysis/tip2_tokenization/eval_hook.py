"""Offline prediction-error decomposition for a tokenized-action policy.

Tip 2 measures how a trained policy's prediction error depends on tokenization
granularity, decomposing it into a reconstruction arm and a learning arm to
explain the shape (not to assume any particular shape). All three quantities
live in the same denormalized action space so they compare across configs:

    term1 (reconstruction) = g(q(a)) - a      # tokenizer round-trip vs expert
    term2 (learning)       = g(z_gen) - g(q(a))  # generation vs tokenizer target
    total                  = g(z_gen) - a      # generation vs expert

where a is the expert action, q the tokenizer encode, g the decode, and z_gen
the model's free-running autoregressive generation (not teacher forcing).

Three decisions make the measurement honest:

- The main metric uses free-running generation via ``Policy.predict_action``,
  the same path deployment uses, because teacher forcing hides autoregressive
  error accumulation and the variable-length-sequence cost that Tip 2 is trying
  to see. A teacher-forced pass is kept only to report exposure bias.
- Evaluation runs on full-horizon chunks only (the val loader is built with
  ``trailing_padded_actions=0``). At rollout the policy is handed the start
  observation and emits the whole horizon, so the full chunk is the natural
  and deployment-faithful unit; trailing-padded partial windows are a
  teacher-forcing training construct whose decode length does not align with
  their valid length.
- A no-EOS generation is a real model failure, so it is decoded with the
  deployment fallback and kept in the main error; excluding it would flatter
  exactly the coarse-token failures Tip 2 wants to expose. ``no_eos_rate`` is
  reported separately as a diagnostic.

Deployment decodes by sampling tokens (``deterministic: false``, temperature 1),
so the main metric is the stochastic free-running generation and the
decomposition is measured on it. Because sampling is high variance at a fine
scale -- one mis-sampled low-frequency coefficient is amplified by the inverse
DCT into a large action error -- the main total averages ``num_generation_samples``
draws per observation, and torch is seeded first so a checkpoint's numbers are
reproducible. A single argmax (greedy) pass is kept only for explanation: it is
deterministic (one pass, no samples needed) and lets ``sampling_gap`` (stochastic
minus argmax) separate the decoding fragility from the model's systematic error.
The FAST global DCT makes that fragility much larger than binning's independent
per-value quantization, which is itself a finding. Exposure bias is the argmax
free-running generation minus the argmax teacher-forced pass, so the two differ
only in free-running versus teacher-forced prefixes.

The cross term is computed independently as 2*mean(e_rec . e_learn) and checked
against MSE(total) - MSE(term1) - MSE(term2); the two agreeing is what verifies
the three errors share one set of elements, mask, and denormalization.
"""

import csv
from pathlib import Path

import hydra.utils
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from versatil.configs.paths import get_hydra_configs_dir
from versatil.data.constants import SampleKey, SyntheticObsKey
from versatil.data.dataloader import get_dataloaders
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.processing.transform import detokenize_actions, unnormalize_actions
from versatil.data.task import ActionSpace
from versatil.endpoints.deploy import load_policy
from versatil.metrics.synthetic_metrics import assign_rollout_modes
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.policy import Policy

# Fragility companions for the stochastic main metric: a chunk is catastrophic
# when any decoded action exceeds CATASTROPHIC_FACTOR times the largest expert
# action in the validation set, and the trimmed total drops the largest
# TRIM_FRACTION of per-element squared errors so a few exploded chunks cannot
# flatten the rest of the curve on a log axis.
CATASTROPHIC_FACTOR = 10.0
TRIM_FRACTION = 0.1


def decompose_prediction_error(
    action_true: np.ndarray,
    reconstruction_from_gt_tokens: np.ndarray,
    reconstruction_from_generation: np.ndarray,
) -> dict[str, float]:
    """Decompose total prediction error into reconstruction and learning arms.

    Args:
        action_true: Expert actions in denormalized units, shape
            (num_elements, action_dim). One row per (chunk, timestep).
        reconstruction_from_gt_tokens: ``g(q(a))`` in the same units and rows,
            the tokenizer round-trip of the expert action.
        reconstruction_from_generation: ``g(z_gen)`` in the same units and
            rows, the decode of the model's free-running generation.

    Returns:
        Per-arm MSE and RMSE (``term1``/``term2``/``total``), the independently
        computed ``cross`` (= 2*mean(e_rec . e_learn)), and ``identity_gap`` =
        total_mse - term1_mse - term2_mse - cross, which is zero up to floating
        point when the three errors share elements, mask, and denormalization.
    """
    reconstruction_error = reconstruction_from_gt_tokens - action_true
    learning_error = reconstruction_from_generation - reconstruction_from_gt_tokens
    total_error = reconstruction_from_generation - action_true

    term1_mse = float(np.mean(reconstruction_error**2))
    term2_mse = float(np.mean(learning_error**2))
    total_mse = float(np.mean(total_error**2))
    cross = 2.0 * float(np.mean(reconstruction_error * learning_error))

    return {
        "term1_mse": term1_mse,
        "term2_mse": term2_mse,
        "total_mse": total_mse,
        "term1_rmse": float(np.sqrt(term1_mse)),
        "term2_rmse": float(np.sqrt(term2_mse)),
        "total_rmse": float(np.sqrt(total_mse)),
        "cross": cross,
        "identity_gap": total_mse - term1_mse - term2_mse - cross,
    }


def decompose_in_position_space(
    expert_chunks: np.ndarray,
    round_trip_chunks: np.ndarray,
    generated_chunks: np.ndarray,
) -> dict[str, float]:
    """Decompose the prediction error on the integrated (position) trajectory.

    Actions are per-step displacements, so their cumulative sum over the
    horizon is the open-loop path from the chunk start. Differencing is a
    high-pass filter that turns i.i.d. demonstrator position noise into action
    noise of the same power as the signal; integrating undoes that, so the
    path error has headroom the per-step error lacks while the vector identity
    total = term1 + term2 survives (the cumulative sum is linear).

    Args:
        expert_chunks: Expert actions, shape (num_chunks, horizon, action_dim).
        round_trip_chunks: ``g(q(a))`` aligned chunk for chunk.
        generated_chunks: ``g(z)`` aligned chunk for chunk.

    Returns:
        The :func:`decompose_prediction_error` keys measured on the integrated
        paths, plus ``expert_mean_square`` = mean square of the expert path,
        the level a stand-still prediction sits at in this space.
    """
    expert_paths = np.cumsum(expert_chunks, axis=1)
    action_dim = expert_chunks.shape[-1]
    metrics = decompose_prediction_error(
        action_true=expert_paths.reshape(-1, action_dim),
        reconstruction_from_gt_tokens=np.cumsum(round_trip_chunks, axis=1).reshape(
            -1, action_dim
        ),
        reconstruction_from_generation=np.cumsum(generated_chunks, axis=1).reshape(
            -1, action_dim
        ),
    )
    metrics["expert_mean_square"] = float(np.mean(expert_paths**2))
    return metrics


def prefixed_metrics(
    metrics: dict[str, float], prefix: str, keys: tuple[str, ...]
) -> dict[str, float]:
    """Return ``{prefix + key: metrics[key]}`` for the selected keys."""
    return {f"{prefix}{key}": metrics[key] for key in keys}


DECOMPOSITION_KEYS = (
    "term1_mse",
    "term2_mse",
    "total_mse",
    "cross",
    "identity_gap",
)


def save_eval_arrays(
    array_path: Path,
    expert_chunks: np.ndarray,
    round_trip_chunks: np.ndarray,
    argmax_chunks: np.ndarray,
    stochastic_chunks: np.ndarray,
    mode_ids: np.ndarray,
) -> None:
    """Persist the per-chunk arrays so later metrics need no GPU pass.

    Args:
        array_path: Destination ``.npz`` file; parent directories are created.
        expert_chunks: Expert actions, shape (num_chunks, horizon, action_dim).
        round_trip_chunks: ``g(q(a))`` aligned chunk for chunk.
        argmax_chunks: Argmax generation aligned chunk for chunk.
        stochastic_chunks: Stochastic generations, shape
            (num_samples, num_chunks, horizon, action_dim).
        mode_ids: Expert mode id per chunk, shape (num_chunks,).
    """
    array_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        array_path,
        expert=expert_chunks,
        round_trip=round_trip_chunks,
        argmax=argmax_chunks,
        stochastic=stochastic_chunks,
        mode_ids=mode_ids,
    )


def compare_generation_modes(
    stochastic_total_mse: float,
    action_true: np.ndarray,
    argmax_generation: np.ndarray,
    teacher_forced: np.ndarray,
) -> dict[str, float]:
    """Explain the stochastic main total with argmax and teacher-forced passes.

    The main total (``stochastic_total_mse``) is the deployment-faithful
    stochastic generation. This adds the single argmax pass (to separate the
    sampling fragility) and the teacher-forced pass (for exposure bias), both
    aligned one row per validation element.

    Args:
        stochastic_total_mse: Total MSE of the stochastic free-running
            generation, i.e. the main metric's ``total_mse``.
        action_true: Expert actions, one row per validation element, aligned
            with ``argmax_generation`` and ``teacher_forced``.
        argmax_generation: Free-running generation with argmax token selection.
        teacher_forced: Argmax prediction under ground-truth token prefixes.

    Returns:
        ``argmax_total_mse``, ``teacher_forced_total_mse``, ``exposure_bias`` =
        argmax generation minus teacher forced (both argmax, so the difference
        is only free-running error accumulation), and ``sampling_gap`` =
        stochastic minus argmax (the cost of sampling tokens instead of taking
        the mode, which the FAST global DCT amplifies at fine scales).
    """
    argmax_total = float(np.mean((argmax_generation - action_true) ** 2))
    teacher_forced_total = float(np.mean((teacher_forced - action_true) ** 2))
    return {
        "argmax_total_mse": argmax_total,
        "teacher_forced_total_mse": teacher_forced_total,
        "exposure_bias": argmax_total - teacher_forced_total,
        "sampling_gap": stochastic_total_mse - argmax_total,
    }


def fraction_without_eos(
    token_sequences: list[list[int]],
    eos_token_id: int | None,
    is_variable_length: bool,
) -> float:
    """Return the share of generated sequences that never emit EOS.

    Only variable-length tokenizers (FAST) terminate with EOS; fixed-length
    tokenizers (binning) emit a fixed payload and are reported as zero.

    Args:
        token_sequences: Model-vocabulary token id lists, one per chunk.
        eos_token_id: The tokenizer's EOS id, or None when it has none.
        is_variable_length: Whether the discretizer is variable-length
            (``fixed_token_count is None``), i.e. relies on EOS to terminate.

    Returns:
        Fraction of sequences with no EOS, or 0.0 for fixed-length tokenizers.
    """
    if not is_variable_length or eos_token_id is None or not token_sequences:
        return 0.0
    missing = sum(1 for tokens in token_sequences if eos_token_id not in tokens)
    return missing / len(token_sequences)


def count_unique_sequences(token_sequences: list[list[int]]) -> int:
    """Return the number of distinct token sequences.

    A tokenizer whose target has collapsed to a constant shows up as 1, which
    marks the point as degenerate for term2 interpretation.
    """
    return len({tuple(tokens) for tokens in token_sequences})


def mean_sequence_length(token_sequences: list[list[int]]) -> float:
    """Return the mean token-sequence length (a FAST granularity confound)."""
    if not token_sequences:
        return 0.0
    return float(np.mean([len(tokens) for tokens in token_sequences]))


def catastrophic_fraction(
    generated_chunks: np.ndarray,
    action_bound: float,
    factor: float,
) -> float:
    """Share of generated chunks with an action beyond ``factor * action_bound``.

    At a fine FAST scale a single mis-sampled low-frequency coefficient decodes
    to an action orders of magnitude outside the expert range; this counts how
    often that happens, per chunk, so the stochastic total can be read next to
    its failure rate.

    Args:
        generated_chunks: Denormalized generations, shape (num_chunks, horizon,
            action_dim).
        action_bound: Largest absolute expert action in the validation set.
        factor: Multiple of ``action_bound`` past which a chunk is catastrophic.

    Returns:
        Fraction of chunks in [0, 1]; 0.0 for an empty input.
    """
    num_chunks = generated_chunks.shape[0]
    if num_chunks == 0:
        return 0.0
    per_chunk_max = np.abs(generated_chunks).reshape(num_chunks, -1).max(axis=1)
    return float(np.mean(per_chunk_max > factor * action_bound))


def robust_error_summary(
    squared_errors: np.ndarray,
    trim_fraction: float,
) -> dict[str, float]:
    """Median and upper-trimmed mean of per-element squared errors.

    The stochastic total is a mean, so a few catastrophic chunks dominate it;
    these companions show where the bulk of the distribution sits. The trim is
    one-sided because the pathology is an upper tail.

    Args:
        squared_errors: Per-element squared errors, any shape.
        trim_fraction: Share of the largest errors dropped before averaging.

    Returns:
        ``median`` and ``trimmed_mean``.
    """
    ordered = np.sort(squared_errors.reshape(-1))
    keep = max(1, int(round(ordered.size * (1.0 - trim_fraction))))
    return {
        "median": float(np.median(ordered)),
        "trimmed_mean": float(np.mean(ordered[:keep])),
    }


def per_chunk_mse(
    generated_chunks: np.ndarray, expert_chunks: np.ndarray
) -> np.ndarray:
    """MSE of each chunk over its (horizon, action_dim) elements.

    Args:
        generated_chunks: Denormalized generations, shape (num_chunks, horizon,
            action_dim).
        expert_chunks: Expert actions aligned chunk for chunk.

    Returns:
        One MSE per chunk, shape (num_chunks,).
    """
    num_chunks = generated_chunks.shape[0]
    squared = (generated_chunks - expert_chunks).reshape(num_chunks, -1) ** 2
    return np.mean(squared, axis=1)


def standard_error(values: np.ndarray) -> float:
    """Standard error of the mean of ``values``; 0.0 with fewer than two."""
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def mode_match_rate(
    generated_chunks: np.ndarray,
    expert_chunks: np.ndarray,
    expert_mode_ids: np.ndarray,
    num_modes: int,
) -> float:
    """Share of generations whose displacement path lands on the expert's mode.

    Actions are per-step displacements, so their cumulative sum is the path
    relative to the chunk start. Each generated path is assigned to the nearest
    per-mode mean expert path and compared with the mode id of the expert chunk
    it was generated for. On a task whose mode is observable from the input
    this should sit near 1; a low value means the generation bet on the wrong
    mode, which inflates term2 for reasons unrelated to granularity.

    Args:
        generated_chunks: Denormalized generations, shape (num_chunks, horizon,
            action_dim).
        expert_chunks: Expert actions aligned chunk for chunk.
        expert_mode_ids: Mode id of each expert chunk, shape (num_chunks,).
        num_modes: Number of behavioral modes in the task.

    Returns:
        Fraction in [0, 1]; 0.0 for an empty input.
    """
    if generated_chunks.shape[0] == 0:
        return 0.0
    assigned = assign_rollout_modes(
        generated_trajectories=np.cumsum(generated_chunks, axis=1),
        expert_trajectories=np.cumsum(expert_chunks, axis=1),
        expert_mode_ids=expert_mode_ids,
        num_modes=num_modes,
    )
    return float(np.mean(assigned == expert_mode_ids))


def mask_after_ground_truth_eos(
    predicted_tokens: torch.Tensor,
    ground_truth_tokens: torch.Tensor,
    eos_token_id: int | None,
) -> torch.Tensor:
    """Replace predictions from the ground-truth EOS onward with the GT tail.

    The teacher-forced logits cover the whole sequence padded to
    ``max_token_len``. Positions after the ground-truth EOS were never trained
    (the loss masks them), so their argmax is arbitrary and, for a
    variable-length tokenizer, gets decoded as coefficients unless the model
    happens to predict EOS at exactly the right position. Splicing in the GT
    tail (EOS followed by padding) keeps the payload prediction and fixes the
    termination. Rows without an EOS are returned unchanged.

    Args:
        predicted_tokens: Argmax tokens, shape (batch, token_len).
        ground_truth_tokens: Tokenized targets, same shape.
        eos_token_id: The tokenizer's EOS id, or None when it has none.

    Returns:
        Tokens with the ground-truth tail spliced in from its EOS position.
    """
    if eos_token_id is None:
        return predicted_tokens
    is_eos = ground_truth_tokens == eos_token_id
    has_eos = is_eos.any(dim=1)
    first_eos = torch.argmax(is_eos.to(torch.int64), dim=1)
    positions = torch.arange(predicted_tokens.shape[1], device=predicted_tokens.device)
    tail = (positions[None, :] >= first_eos[:, None]) & has_eos[:, None]
    return torch.where(tail, ground_truth_tokens, predicted_tokens)


class _CapturingTokenSink:
    """In-memory ``TokenUsageSink`` that keeps each generation's token ids."""

    def __init__(self) -> None:
        self.recorded: list[torch.Tensor] = []

    def record(self, action_tokens: torch.Tensor) -> None:
        """Store one prediction's model-vocab action token ids."""
        self.recorded.append(action_tokens.detach().cpu())


def concatenate_predicted_actions(
    action_dict: dict[str, torch.Tensor],
    action_space: ActionSpace,
) -> torch.Tensor:
    """Concatenate per-key action tensors in the canonical prediction order.

    Mirrors the order ``tokenize_actions``/``detokenize_actions`` use, so the
    ground-truth, tokenizer round-trip, and generation tensors line up
    component-for-component.

    Args:
        action_dict: Per-key action tensors, each (batch, horizon, key_dim).
        action_space: Provides the canonical metadata key order and which keys
            carry a prediction head.

    Returns:
        Concatenated actions of shape (batch, horizon, action_dim).
    """
    components = []
    for key, meta in action_space.actions_metadata.items():
        if meta.is_numerical and meta.requires_prediction_head and key in action_dict:
            tensor = action_dict[key]
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(-1)
            components.append(tensor)
    return torch.cat(components, dim=-1)


def ground_truth_actions(
    batch: dict[str, dict[str, torch.Tensor]],
    action_space: ActionSpace,
    normalizer: LinearNormalizer,
) -> torch.Tensor:
    """Return denormalized expert actions ``a`` for one batch.

    Args:
        batch: One collated batch; ``batch['action']`` holds normalized
            per-key expert actions.
        action_space: Canonical key order and prediction-head filter.
        normalizer: Fitted normalizer used to denormalize.

    Returns:
        Denormalized expert actions, shape (batch, horizon, action_dim).
    """
    normalized = batch[SampleKey.ACTION.value]
    denormalized = unnormalize_actions(
        normalized_actions=normalized,
        normalizer=normalizer,
        action_space=action_space,
    )
    return concatenate_predicted_actions(
        action_dict=denormalized, action_space=action_space
    )


def decode_denormalized(
    action_tokens: torch.Tensor,
    policy: Policy,
) -> torch.Tensor:
    """Decode model tokens to denormalized actions via the policy's tokenizer.

    Args:
        action_tokens: Model-vocab token ids, shape (batch, token_len).
        policy: Loaded policy carrying the trained tokenizer and normalizer.

    Returns:
        Denormalized actions, shape (batch, horizon, action_dim).
    """
    normalized = detokenize_actions(
        action_tokens=action_tokens,
        action_tokenizer=policy.tokenizer.action_tokenizer,
        action_space=policy.action_space,
    )
    denormalized = unnormalize_actions(
        normalized_actions=normalized,
        normalizer=policy.normalizer,
        action_space=policy.action_space,
    )
    return concatenate_predicted_actions(
        action_dict=denormalized, action_space=policy.action_space
    )


def generate_actions_and_tokens(
    policy: Policy,
    observation: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, list[list[int]]]:
    """Free-running generation: denormalized actions plus raw token sequences.

    Uses ``Policy.predict_action`` (the deployment path) for the actions and a
    capturing token sink for the model-vocab token ids, so the no-EOS check
    sees exactly the sequences generation produced.

    Args:
        policy: Loaded policy in eval mode.
        observation: Raw (un-normalized) batched observation dict.

    Returns:
        Denormalized generated actions (batch, horizon, action_dim) and the
        per-chunk model-vocab token id lists.
    """
    sink = _CapturingTokenSink()
    policy.set_token_usage_sink(sink=sink)
    action_dict = policy.predict_action(obs_dict=observation)
    policy.set_token_usage_sink(sink=None)

    generated = concatenate_predicted_actions(
        action_dict=action_dict, action_space=policy.action_space
    )
    token_sequences: list[list[int]] = []
    for recorded in sink.recorded:
        squeezed = recorded
        if squeezed.ndim == 3 and squeezed.shape[-1] == 1:
            squeezed = squeezed.squeeze(-1)
        for row in squeezed:
            token_sequences.append([int(value) for value in row.tolist()])
    return generated, token_sequences


def generate_actions_with_argmax(
    policy: Policy,
    observation: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Free-running generation with argmax token selection.

    Temporarily switches the discrete decoder to deterministic decoding so the
    explanatory argmax total differs from the stochastic main only by token
    selection, then restores the deployed sampling mode.

    Args:
        policy: Loaded policy in eval mode.
        observation: Raw (un-normalized) batched observation dict.

    Returns:
        Denormalized generated actions, shape (batch, horizon, action_dim).
    """
    deployed_mode = policy.decoder.deterministic
    policy.decoder.deterministic = True
    generated, _ = generate_actions_and_tokens(policy=policy, observation=observation)
    policy.decoder.deterministic = deployed_mode
    return generated


def teacher_forced_actions(
    policy: Policy,
    batch: dict[str, dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Denormalized actions from the teacher-forced argmax, for exposure bias.

    The training forward feeds ground-truth token prefixes and predicts every
    position in parallel; its argmax is the teacher-forced prediction. The
    untrained positions after the ground-truth EOS are replaced with the GT
    tail before decoding (see :func:`mask_after_ground_truth_eos`). This is a
    diagnostic (exposure bias = total_argmax - total_teacher_forced), not the
    main metric.

    Args:
        policy: Loaded policy in eval mode.
        batch: One collated batch with tokenized ground-truth actions.

    Returns:
        Denormalized teacher-forced actions, shape (batch, horizon, action_dim).
    """
    output = policy.forward(batch)
    logits = output[DecoderOutputKey.ACTION_LOGITS.value]
    predicted_tokens = mask_after_ground_truth_eos(
        predicted_tokens=torch.argmax(logits, dim=-1),
        ground_truth_tokens=batch[SampleKey.ACTION.value][
            SampleKey.TOKENIZED_ACTIONS.value
        ],
        eos_token_id=policy.tokenizer.action_tokenizer.eos_token_id,
    )
    return decode_denormalized(action_tokens=predicted_tokens, policy=policy)


def move_batch_to_device(
    batch: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return a copy of the nested batch with every tensor on ``device``."""
    moved: dict[str, dict[str, torch.Tensor]] = {}
    for group_key, group in batch.items():
        moved[group_key] = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in group.items()
        }
    return moved


def evaluate_checkpoint(
    config: DictConfig,
    checkpoint_path: str,
    device: torch.device,
    eval_seed: int,
    num_generation_samples: int,
    array_path: Path | None,
) -> dict[str, float]:
    """Run the full-horizon prediction-error decomposition for one checkpoint.

    Builds the validation loader from ``config`` with trailing padding disabled
    (full-horizon chunks only), loads the trained policy, and accumulates the
    denormalized expert action, tokenizer round-trip, stochastic generation, and
    a single argmax and teacher-forced pass across the validation set before
    decomposing the error once over all elements. The decomposition is measured
    on the stochastic generation; argmax is the explanatory companion. Both are
    also decomposed on the integrated position path
    (:func:`decompose_in_position_space`).

    Args:
        config: Instantiated training config (task already built). Its
            ``task.dataloader.trailing_padded_actions`` is forced to 0 and
            ``shuffle`` to False so evaluation is deterministic and full-horizon.
        checkpoint_path: Directory of the trained checkpoint.
        device: Device to run the policy on.
        eval_seed: Torch seed set before generation so the stochastic main
            metric gives the same numbers on every run of the same checkpoint.
        num_generation_samples: Stochastic generations drawn per observation; the
            main decomposition averages over all of them, each paired with its
            own copy of the expert and round-trip rows. Sampling is high variance
            at a fine scale, so this is raised well above 1.
        array_path: When given, the per-chunk expert, round-trip, argmax and
            stochastic arrays are saved there as ``.npz`` (see
            :func:`save_eval_arrays`) so later metrics are pure post-processing.

    Returns:
        A flat metrics row: the error decomposition over the stochastic
        generation, the argmax/teacher-forced comparison (``argmax_total_mse``,
        ``exposure_bias``, ``sampling_gap``), the argmax decomposition arms
        (``argmax_term1_mse`` ...), the position-space decompositions
        (``position_stochastic_*``, ``position_argmax_*``,
        ``position_expert_mean_square``), no-EOS rate, unique ground-truth
        sequence count, mean generated token length, the sampling-fragility
        companions (``catastrophic_fraction``, ``total_mse_median``,
        ``total_mse_trimmed``), ``total_mse_se`` (standard error of the
        per-chunk total over ``num_eval_chunks`` chunk draws, for the
        adjacent-point power check), ``expert_mean_square`` (the MSE of an
        all-zero prediction, the level a collapsed tokenizer sits at), and
        ``mode_match_rate`` (share of generations landing on the expert chunk's
        mode; near 1 when the mode is observable from the input).

    Raises:
        ValueError: If the config has no validation split to evaluate, or
            ``num_generation_samples`` is below 1.
    """
    if num_generation_samples < 1:
        raise ValueError(
            f"num_generation_samples must be >= 1, got {num_generation_samples}."
        )
    config.task.dataloader.trailing_padded_actions = 0
    config.task.dataloader.shuffle = False
    _, val_loader, _, _, _ = get_dataloaders(config=config)
    if val_loader is None:
        raise ValueError(
            "evaluate_checkpoint needs a validation split; set val_ratio > 0."
        )

    policy = load_policy(
        checkpoint_path=checkpoint_path, device=device, compile_model=False
    ).policy
    policy.eval()
    # The checkpoint stores whatever sampling mode training used for its own
    # rollouts; the main metric must follow deployment (stochastic) regardless,
    # so the mode is pinned here rather than inherited. The argmax pass toggles
    # it on and restores this value.
    policy.decoder.deterministic = False
    torch.manual_seed(eval_seed)

    action_space = policy.action_space
    discretizer = policy.tokenizer.action_tokenizer.action_discretizer
    is_variable_length = discretizer.fixed_token_count is None
    eos_token_id = policy.tokenizer.action_tokenizer.eos_token_id

    expert_batches: list[np.ndarray] = []
    round_trip_batches: list[np.ndarray] = []
    argmax_batches: list[np.ndarray] = []
    teacher_forced_batches: list[np.ndarray] = []
    mode_id_batches: list[np.ndarray] = []
    stochastic_batches: list[list[np.ndarray]] = [
        [] for _ in range(num_generation_samples)
    ]
    stochastic_tokens: list[list[int]] = []
    ground_truth_tokens: list[list[int]] = []

    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch_to_device(batch=batch, device=device)
            expert = ground_truth_actions(
                batch=batch, action_space=action_space, normalizer=policy.normalizer
            )
            round_trip = decode_denormalized(
                action_tokens=batch[SampleKey.ACTION.value][
                    SampleKey.TOKENIZED_ACTIONS.value
                ],
                policy=policy,
            )
            observation = batch[SampleKey.OBSERVATION.value]
            mode_id_batches.append(
                observation[SyntheticObsKey.MODE_ID.value]
                .reshape(expert.shape[0], -1)[:, 0]
                .to(torch.int64)
                .cpu()
                .numpy()
            )
            expert_batches.append(expert.cpu().numpy())
            round_trip_batches.append(round_trip.cpu().numpy())

            for sample_index in range(num_generation_samples):
                sampled, batch_tokens = generate_actions_and_tokens(
                    policy=policy, observation=observation
                )
                stochastic_batches[sample_index].append(sampled.cpu().numpy())
                stochastic_tokens.extend(batch_tokens)

            argmax_batches.append(
                generate_actions_with_argmax(policy=policy, observation=observation)
                .cpu()
                .numpy()
            )
            teacher_forced_batches.append(
                teacher_forced_actions(policy=policy, batch=batch).cpu().numpy()
            )
            for row in batch[SampleKey.ACTION.value][SampleKey.TOKENIZED_ACTIONS.value]:
                ground_truth_tokens.append([int(value) for value in row.tolist()])

    # (N, H, D) once per validation chunk; stochastic is (S, N, H, D).
    expert_once = np.concatenate(expert_batches, axis=0)
    round_trip_once = np.concatenate(round_trip_batches, axis=0)
    argmax_once = np.concatenate(argmax_batches, axis=0)
    teacher_forced_once = np.concatenate(teacher_forced_batches, axis=0)
    mode_ids = np.concatenate(mode_id_batches, axis=0)
    stochastic_samples = np.stack(
        [np.concatenate(batches, axis=0) for batches in stochastic_batches], axis=0
    )
    if array_path is not None:
        save_eval_arrays(
            array_path=array_path,
            expert_chunks=expert_once,
            round_trip_chunks=round_trip_once,
            argmax_chunks=argmax_once,
            stochastic_chunks=stochastic_samples,
            mode_ids=mode_ids,
        )

    # Every stochastic draw is paired with its own copy of the expert and
    # round-trip chunks so the three errors share one element set.
    action_dim = expert_once.shape[-1]
    expert_chunk_array = np.concatenate([expert_once] * num_generation_samples)
    round_trip_chunk_array = np.concatenate([round_trip_once] * num_generation_samples)
    generated_chunks = stochastic_samples.reshape(-1, *stochastic_samples.shape[2:])
    expert_actions = expert_chunk_array.reshape(-1, action_dim)
    stochastic_actions = generated_chunks.reshape(-1, action_dim)
    metrics = decompose_prediction_error(
        action_true=expert_actions,
        reconstruction_from_gt_tokens=round_trip_chunk_array.reshape(-1, action_dim),
        reconstruction_from_generation=stochastic_actions,
    )
    metrics.update(
        compare_generation_modes(
            stochastic_total_mse=metrics["total_mse"],
            action_true=expert_once.reshape(-1, action_dim),
            argmax_generation=argmax_once.reshape(-1, action_dim),
            teacher_forced=teacher_forced_once.reshape(-1, action_dim),
        )
    )
    argmax_decomposition = decompose_prediction_error(
        action_true=expert_once.reshape(-1, action_dim),
        reconstruction_from_gt_tokens=round_trip_once.reshape(-1, action_dim),
        reconstruction_from_generation=argmax_once.reshape(-1, action_dim),
    )
    metrics.update(
        prefixed_metrics(
            metrics=argmax_decomposition,
            prefix="argmax_",
            keys=("term1_mse", "term2_mse", "cross", "identity_gap"),
        )
    )
    position_stochastic = decompose_in_position_space(
        expert_chunks=expert_chunk_array,
        round_trip_chunks=round_trip_chunk_array,
        generated_chunks=generated_chunks,
    )
    position_argmax = decompose_in_position_space(
        expert_chunks=expert_once,
        round_trip_chunks=round_trip_once,
        generated_chunks=argmax_once,
    )
    metrics.update(
        prefixed_metrics(
            metrics=position_stochastic,
            prefix="position_stochastic_",
            keys=DECOMPOSITION_KEYS,
        )
    )
    metrics.update(
        prefixed_metrics(
            metrics=position_argmax, prefix="position_argmax_", keys=DECOMPOSITION_KEYS
        )
    )
    metrics["position_expert_mean_square"] = position_argmax["expert_mean_square"]
    metrics["eval_seed"] = eval_seed
    metrics["num_generation_samples"] = num_generation_samples
    metrics["no_eos_rate"] = fraction_without_eos(
        token_sequences=stochastic_tokens,
        eos_token_id=eos_token_id,
        is_variable_length=is_variable_length,
    )
    metrics["unique_gt_sequence_count"] = count_unique_sequences(
        token_sequences=ground_truth_tokens
    )
    metrics["mean_generated_token_length"] = mean_sequence_length(
        token_sequences=stochastic_tokens
    )
    metrics["catastrophic_fraction"] = catastrophic_fraction(
        generated_chunks=generated_chunks,
        action_bound=float(np.abs(expert_actions).max()),
        factor=CATASTROPHIC_FACTOR,
    )
    # Standard error of the total across chunk draws, for the pilot gate's
    # adjacent-granularity power check.
    chunk_mse = per_chunk_mse(
        generated_chunks=generated_chunks, expert_chunks=expert_chunk_array
    )
    metrics["total_mse_se"] = standard_error(values=chunk_mse)
    metrics["num_eval_chunks"] = int(chunk_mse.size)
    metrics["mode_match_rate"] = mode_match_rate(
        generated_chunks=generated_chunks,
        expert_chunks=expert_chunk_array,
        expert_mode_ids=np.concatenate([mode_ids] * num_generation_samples),
        num_modes=int(config.task.dataset_schema.num_modes),
    )
    robust = robust_error_summary(
        squared_errors=(stochastic_actions - expert_actions) ** 2,
        trim_fraction=TRIM_FRACTION,
    )
    metrics["total_mse_median"] = robust["median"]
    metrics["total_mse_trimmed"] = robust["trimmed_mean"]
    # MSE of an all-zero ("stand still") prediction: a collapsed tokenizer
    # decodes to this, and on a multimodal expert it scores below any prediction
    # that commits to a mode, so a total at or under it marks degeneracy.
    metrics["expert_mean_square"] = float(np.mean(expert_actions**2))
    return metrics


def run_from_config_name(
    config_name: str,
    checkpoint_path: str,
    overrides: list[str],
    device: torch.device,
    eval_seed: int,
    num_generation_samples: int,
    array_path: Path | None,
) -> dict[str, float]:
    """Compose a training config by name, instantiate it, and evaluate.

    Args:
        config_name: Hydra config name the checkpoint was trained with.
        checkpoint_path: Directory of the trained checkpoint.
        overrides: Hydra overrides pinning the same data as training (the fixed
            ``zarr_path``, ``experiment.data_seed``, and the FAST
            ``scale``/``vocab_size``/``max_token_len``).
        device: Device to run the policy on.
        eval_seed: Torch seed for the stochastic generation path.
        num_generation_samples: Sampled generations drawn per observation.
        array_path: Optional ``.npz`` destination for the per-chunk arrays.

    Returns:
        The metrics row from :func:`evaluate_checkpoint`.
    """
    with initialize_config_dir(
        config_dir=str(get_hydra_configs_dir()), version_base=None
    ):
        raw_config = compose(config_name=config_name, overrides=overrides)
        config = hydra.utils.instantiate(raw_config)
    return evaluate_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        eval_seed=eval_seed,
        num_generation_samples=num_generation_samples,
        array_path=array_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tip 2 prediction-error eval hook.")
    parser.add_argument("config_name")
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_csv")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--array-path", type=Path, default=None)
    arguments = parser.parse_args()

    result = run_from_config_name(
        config_name=arguments.config_name,
        checkpoint_path=arguments.checkpoint_path,
        overrides=arguments.override,
        device=torch.device(arguments.device),
        eval_seed=arguments.eval_seed,
        num_generation_samples=arguments.num_samples,
        array_path=arguments.array_path,
    )

    output_path = Path(arguments.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)
    for metric_name, metric_value in result.items():
        print(f"{metric_name:>28}: {metric_value}")
    print(f"Wrote {output_path}")
