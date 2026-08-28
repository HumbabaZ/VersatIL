"""Calibrate the FAST scale grid for Tip 2 from the real training data.

Tip 2 sweeps FAST's DCT rounding scale to characterize how prediction error
depends on tokenization granularity. A blindly chosen scale range can land the
whole grid in the fully degenerate regime (every coefficient rounds to zero) or
the saturated regime (no coefficient sits near a rounding boundary, so a finer
scale changes nothing). The sweep also needs to know how long the FAST token
sequences actually get before it picks a ``max_token_len`` for its own configs.

This reads the selected task's fixed default-noise store the sweep trains
against (``--task sequential`` -> ``stores.SEQUENTIAL``, ``--task conditional``
-> ``stores.CONDITIONAL_CIRCLE``), reconstructs the train split with the store's
generation seed, and fits both the normalizer and the FAST processor on that
split -- matching the training pipeline instead of regenerating episodes and
fitting a second normalizer. The scale grid is task-specific: it is derived from
the task's own action coefficient distribution, so the two tasks calibrate to
different grids. Token length is measured over chunks of every length the
dataloader actually encodes: the training pipeline fits FAST on full-horizon
chunks only, but the
dataloader emits trailing-padded windows too, whose padded rows are dropped so
the encoder sees variable-length chunks. Post-BPE length is not monotone in
chunk length at a large scale, so the worst case has to be measured, not
assumed from the full-length chunk alone.

Feasibility is an architectural question, not an arbitrary token cap: a scale is
usable when its DCT alphabet fits the vocabulary and its longest token sequence
plus the observation prefix fits the decoder's ``max_seq_len``. The sweep then
sets its own ``max_token_len`` above the measured global maximum; scales are not
dropped for exceeding a cap that was itself picked arbitrarily.

    python -m versatil.analysis.tip2_tokenization.calibrate_fast_scale out_dir \
        --task sequential
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import zarr
from scipy.fft import dct

from versatil.analysis.rate_distortion.fast_sweep import (
    fit_fast_processor,
    load_fast_class,
)
from versatil.analysis.rate_distortion.metrics import dct_alphabet_size
from versatil.analysis.tip2_tokenization.stores import STORES, Tip2Store
from versatil.data.normalization.normalizer import SingleFieldLinearNormalizer
from versatil.data.preprocessing.sampler import get_val_mask

ACTION_KEY = "synthetic_position_action"
EPISODE_ENDS_KEY = "episode_ends"
# Both synthetic tasks share the discrete-method horizon and the sweep's data
# seed / val ratio; only the store and the observation prefix differ per task.
PREDICTION_HORIZON = 59
VAL_RATIO = 0.05

PRETRAINED_FAST_MODEL = "physical-intelligence/fast"
VOCAB_SIZE = 1024
# gpt_action_transformer.yaml sets the decoder max_seq_len; the action tokens
# share it with the observation prefix (image feature tokens plus, for the
# conditional task, one context token) and EOS. The budget is conservative
# headroom, not an exact count; mono_rgb_context adds a single context token, so
# the sequential headroom already covers it and feasibility is far from the cap.
DECODER_MAX_SEQ_LEN = 512
TASK_PREFIX_BUDGET = {"sequential": 96, "conditional": 96}
NUM_SCALE_CANDIDATES = 7
LOWER_TAIL_QUANTILE = 0.01
# Full-length chunks dominate the token-length maximum, so they are checked on
# every train episode. The all-length scan (to test the monotonicity
# assumption) runs on a subsample; it exists to catch a shorter chunk that
# tokenizes longer, not to be exhaustive.
VARIABLE_LENGTH_SCAN_EPISODES = 150


def load_train_episode_actions(store: Tip2Store) -> list[np.ndarray]:
    """Return per-episode action arrays for the store's train split.

    Opens the stored ``synthetic_position_action`` and ``episode_ends`` arrays
    directly (the store is on a slow filesystem where a full group walk stalls)
    and reconstructs the train split with the store's generation seed and the
    synthetic dataloader's ``val_ratio``.

    Args:
        store: The fixed default-noise store to calibrate against.

    Returns:
        One (episode_length, action_dim) array per train episode.
    """
    root = Path(store.zarr_path)
    actions = zarr.open_array(str(root / "data" / ACTION_KEY), mode="r")[:]
    episode_ends = zarr.open_array(str(root / "meta" / EPISODE_ENDS_KEY), mode="r")[:]
    val_mask = get_val_mask(
        n_episodes=len(episode_ends), val_ratio=VAL_RATIO, seed=store.seed
    )
    starts = np.concatenate([[0], episode_ends[:-1]])
    return [
        actions[start:end]
        for index, (start, end) in enumerate(zip(starts, episode_ends, strict=True))
        if not val_mask[index]
    ]


def full_length_chunks(episodes: list[np.ndarray]) -> np.ndarray:
    """Stack every full-horizon window, matching the pipeline's tokenizer fit.

    The training pipeline fits FAST on windows of exactly ``prediction_horizon``
    steps (``_create_action_chunks_for_tokenizer``); the same windows fit the
    normalizer here.

    Args:
        episodes: Per-episode action arrays.

    Returns:
        Chunks of shape (num_chunks, PREDICTION_HORIZON, action_dim).
    """
    chunks = [
        episode[start : start + PREDICTION_HORIZON]
        for episode in episodes
        for start in range(len(episode) - PREDICTION_HORIZON + 1)
    ]
    return np.stack(chunks)


def normalize_chunks(
    chunks: np.ndarray, normalizer: SingleFieldLinearNormalizer
) -> np.ndarray:
    """Return chunks scaled to [-1, 1] with the fitted train-split normalizer."""
    return normalizer.normalize(chunks).numpy()


def variable_length_chunks(
    episodes: list[np.ndarray],
    normalizer: SingleFieldLinearNormalizer,
) -> dict[int, np.ndarray]:
    """Group the dataloader's variable-length encoded chunks by valid length.

    Trailing-padded windows drop their padded rows before encoding, so the
    encoder sees each episode's length-k action suffixes for k in 1..horizon.
    Grouping by length lets a single processor call encode each length batch.

    Args:
        episodes: Per-episode action arrays (subsampled by the caller).
        normalizer: Train-split normalizer applied before DCT.

    Returns:
        Mapping from valid length to a (num_chunks, length, action_dim) batch of
        normalized chunks.
    """
    by_length: dict[int, list[np.ndarray]] = {}
    for episode in episodes:
        length = len(episode)
        for valid_length in range(1, PREDICTION_HORIZON + 1):
            suffix = episode[length - valid_length :]
            by_length.setdefault(valid_length, []).append(suffix)
    return {
        valid_length: normalizer.normalize(np.stack(batch)).numpy()
        for valid_length, batch in by_length.items()
    }


def scale_candidates(chunks_normalized: np.ndarray) -> np.ndarray:
    """Return log-spaced scale candidates spanning full degeneracy to fine.

    The lower bound is the scale at which the largest-magnitude coefficient
    first survives rounding; below it every coefficient rounds to zero. The
    upper bound comes from the 1st percentile of nonzero coefficient
    magnitudes, past which finer scales resolve gaps smaller than all but the
    most extreme outliers.

    Args:
        chunks_normalized: Normalized full-length chunks, shape
            (num_chunks, PREDICTION_HORIZON, action_dim).

    Returns:
        NUM_SCALE_CANDIDATES scale values, log-spaced.

    Raises:
        ValueError: If every DCT coefficient in the chunk set is zero.
    """
    coefficients = np.concatenate(
        [dct(chunk, axis=0, norm="ortho").flatten() for chunk in chunks_normalized]
    )
    max_abs_coefficient = float(np.max(np.abs(coefficients)))
    if max_abs_coefficient == 0.0:
        raise ValueError("All DCT coefficients are zero; cannot calibrate scale.")
    lower_bound = 0.5 / max_abs_coefficient
    nonzero_abs = np.abs(coefficients[coefficients != 0.0])
    smallest_meaningful_gap = float(np.quantile(nonzero_abs, LOWER_TAIL_QUANTILE))
    upper_bound = 0.5 / smallest_meaningful_gap
    return np.geomspace(lower_bound, upper_bound, num=NUM_SCALE_CANDIDATES)


def evaluate_scale(
    fast_class: type,
    full_chunks_normalized: np.ndarray,
    variable_batches: dict[int, np.ndarray],
    scale: float,
    prefix_budget: int,
) -> dict[str, float | int | bool]:
    """Measure degeneracy, token length, and feasibility at one scale.

    Args:
        fast_class: The UniversalActionProcessor class.
        full_chunks_normalized: Normalized full-length train chunks, used for
            the tokenizer fit and the full-length token-length maximum.
        variable_batches: Normalized chunks grouped by valid length, for the
            all-length token-length scan.
        scale: DCT rounding scale to test.
        prefix_budget: Observation prefix token headroom reserved in the decoder
            sequence for this task.

    Returns:
        One result row. ``degenerate_fraction`` is the dead-zone measure (share
        of zeroed coefficients); ``unique_sequence_count`` counts distinct
        full-length token sequences, so a tokenizer that has collapsed its
        target to a constant shows up as 1. ``feasible`` reflects the
        architectural constraints (alphabet within vocabulary, longest token
        sequence plus observation prefix within the decoder), not a token cap.
    """
    action_dim = full_chunks_normalized.shape[2]
    rounded = np.around(dct(full_chunks_normalized, axis=1, norm="ortho") * scale)
    degenerate_fraction = float(np.mean(rounded == 0.0))
    alphabet_size = dct_alphabet_size(
        chunks_normalized=full_chunks_normalized, scale=scale
    )
    alphabet_feasible = alphabet_size <= VOCAB_SIZE

    row: dict[str, float | int | bool] = {
        "scale": scale,
        "degenerate_fraction": degenerate_fraction,
        "alphabet_size": alphabet_size,
        "alphabet_feasible": alphabet_feasible,
        "unique_sequence_count": 0,
        "max_token_len_full": float("nan"),
        "max_token_len_any_length": float("nan"),
        "decoder_budget": DECODER_MAX_SEQ_LEN - prefix_budget,
        "feasible": False,
    }
    if not alphabet_feasible:
        return row

    processor = fit_fast_processor(
        fast_class=fast_class,
        chunks_normalized=full_chunks_normalized,
        scale=scale,
        vocab_size=VOCAB_SIZE,
        time_horizon=PREDICTION_HORIZON,
        action_dim=action_dim,
    )

    full_token_lists = processor(full_chunks_normalized.astype(np.float32))
    full_lengths = [len(tokens) for tokens in full_token_lists]
    unique_sequence_count = len({tuple(tokens) for tokens in full_token_lists})
    max_token_len_full = float(max(full_lengths))

    max_token_len_any = max_token_len_full
    for batch in variable_batches.values():
        token_lists = processor(batch.astype(np.float32))
        batch_max = max(len(tokens) for tokens in token_lists)
        max_token_len_any = max(max_token_len_any, float(batch_max))

    row["unique_sequence_count"] = unique_sequence_count
    row["max_token_len_full"] = max_token_len_full
    row["max_token_len_any_length"] = max_token_len_any
    row["feasible"] = max_token_len_any + prefix_budget < DECODER_MAX_SEQ_LEN
    return row


def run(store: Tip2Store, prefix_budget: int) -> list[dict[str, float | int | bool]]:
    """Evaluate the calibrated scale grid on the store's real train split.

    Args:
        store: The fixed default-noise store to calibrate against.
        prefix_budget: Observation prefix token headroom for this task.
    """
    episodes = load_train_episode_actions(store=store)
    full_chunks = full_length_chunks(episodes)
    normalizer = SingleFieldLinearNormalizer.create_fit(full_chunks, last_n_dims=1)
    full_chunks_normalized = normalize_chunks(full_chunks, normalizer)
    variable_batches = variable_length_chunks(
        episodes=episodes[:VARIABLE_LENGTH_SCAN_EPISODES],
        normalizer=normalizer,
    )
    scales = scale_candidates(full_chunks_normalized)
    fast_class = load_fast_class(PRETRAINED_FAST_MODEL)
    return [
        evaluate_scale(
            fast_class=fast_class,
            full_chunks_normalized=full_chunks_normalized,
            variable_batches=variable_batches,
            scale=scale,
            prefix_budget=prefix_budget,
        )
        for scale in scales
    ]


def _main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the FAST scale grid.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--task", default="sequential", choices=sorted(STORES))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run(store=STORES[args.task], prefix_budget=TASK_PREFIX_BUDGET[args.task])

    header = list(results[0].keys())
    csv_path = output_dir / "tip2_fast_scale_calibration.csv"
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)

    feasible_max = [
        row["max_token_len_any_length"] for row in results if row["feasible"]
    ]
    recommended_cap = int(max(feasible_max)) + 1 if feasible_max else None

    print(
        f"{'scale':>10} {'degen':>7} {'alphabet':>9} {'unique':>7} "
        f"{'max_full':>9} {'max_any':>8} {'feasible':>9}"
    )
    for row in results:
        print(
            f"{row['scale']:>10.3f} {row['degenerate_fraction']:>7.3f} "
            f"{row['alphabet_size']:>9} {row['unique_sequence_count']:>7} "
            f"{row['max_token_len_full']:>9.1f} {row['max_token_len_any_length']:>8.1f} "
            f"{'yes' if row['feasible'] else 'no':>9}"
        )
    feasible_count = sum(1 for row in results if row["feasible"])
    print(
        f"\n{feasible_count}/{len(results)} scales feasible; a sweep max_token_len "
        f"of {recommended_cap} covers every feasible scale. Wrote {csv_path}"
    )


if __name__ == "__main__":
    _main()
