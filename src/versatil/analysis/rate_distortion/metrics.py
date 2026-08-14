"""Rate and distortion metrics for the FAST tokenizer floor.

Distortion is reported in the original action space and split by component nature
(position, orientation, gripper), never averaged across natures. Rate is reported
as bits per chunk, the only axis comparable across tokenizer families.
"""

import numpy as np
import torch
from scipy.fft import dct

from versatil.analysis.rate_distortion.data import ActionComponentLayout
from versatil.data.normalization.normalizer import LinearNormalizer


def dct_alphabet_size(chunks_normalized: np.ndarray, scale: float) -> int:
    """Return the number of distinct rounded DCT symbols FAST would need.

    This is the lower bound on ``vocab_size``: fitting FAST with a smaller
    vocabulary cannot even hold the initial alphabet. Computing it directly lets
    the sweep skip infeasible cells without triggering the library's assertion.

    Args:
        chunks_normalized: Normalized action chunks with shape
            (num_chunks, time_horizon, action_dim).
        scale: DCT rounding scale.

    Returns:
        Alphabet size ``max_token - min_token + 1``.
    """
    coefficients = np.concatenate(
        [dct(chunk, axis=0, norm="ortho").flatten() for chunk in chunks_normalized]
    )
    rounded = np.around(coefficients * scale)
    return int(rounded.max() - rounded.min()) + 1


def rate_metrics(token_lists: list[list[int]], vocab_size: int) -> dict[str, float]:
    """Compute the FAST rate for a set of encoded chunks.

    Args:
        token_lists: BPE token id lists, one per chunk.
        vocab_size: BPE vocabulary size, for the bit cost per token.

    Returns:
        ``mean_token_len`` and ``bits_per_chunk``
        (``mean_token_len * log2(vocab_size)``).
    """
    lengths = np.array([len(tokens) for tokens in token_lists], dtype=np.float64)
    mean_token_len = float(lengths.mean())
    bits_per_chunk = mean_token_len * float(np.log2(vocab_size))
    return {"mean_token_len": mean_token_len, "bits_per_chunk": bits_per_chunk}


def binning_rate(num_bins: int, time_horizon: int, action_dim: int) -> dict[str, float]:
    """Analytic rate of per-value binning (fixed token length T·D).

    Args:
        num_bins: Bins per action value.
        time_horizon: Chunk length.
        action_dim: Action dimension.

    Returns:
        ``mean_token_len`` (= T·D) and ``bits_per_chunk`` (= T·D·log2(num_bins)).
    """
    token_length = float(time_horizon * action_dim)
    return {
        "mean_token_len": token_length,
        "bits_per_chunk": token_length * float(np.log2(num_bins)),
    }


def bits_per_step(bits_per_chunk: float, time_horizon: int) -> float:
    """Convert per-chunk bits to the cross-family per-timestep rate axis."""
    return bits_per_chunk / time_horizon


def reconstruction_distortion(
    ground_truth_normalized: np.ndarray,
    reconstruction_normalized: np.ndarray,
    normalizer: LinearNormalizer,
    layout: list[ActionComponentLayout],
) -> dict[str, float]:
    """Measure per-component reconstruction distortion in original units.

    Continuous components are de-normalized before the error is computed, so RMSE
    and MAE are in original action units. The binary gripper is scored as a
    classification mismatch after thresholding the reconstruction at zero.

    Args:
        ground_truth_normalized: FAST input chunks, shape
            (num_chunks, time_horizon, action_dim).
        reconstruction_normalized: FAST round-trip output, same shape.
        normalizer: Fitted normalizer used to de-normalize continuous components.
        layout: Component placement inside ``action_dim``.

    Returns:
        Per-component ``rmse_<key>``/``mae_<key>`` for continuous components,
        ``gripper_mismatch_rate`` for the gripper, and ``rmse_continuous`` as a
        combined original-space RMSE over all continuous components.
    """
    metrics: dict[str, float] = {}
    continuous_squared_errors = []
    for component in layout:
        ground_truth = _flatten_component(ground_truth_normalized, component)
        reconstruction = _flatten_component(reconstruction_normalized, component)
        if component.needs_normalization:
            ground_truth = _to_numpy(
                normalizer[component.key].unnormalize(ground_truth)
            )
            reconstruction = _to_numpy(
                normalizer[component.key].unnormalize(reconstruction)
            )
            error = reconstruction - ground_truth
            metrics[f"rmse_{component.key}"] = float(np.sqrt(np.mean(error**2)))
            metrics[f"mae_{component.key}"] = float(np.mean(np.abs(error)))
            continuous_squared_errors.append(error.reshape(-1) ** 2)
        else:
            ground_truth_sign = np.where(ground_truth > 0.0, 1.0, -1.0)
            reconstruction_sign = np.where(reconstruction > 0.0, 1.0, -1.0)
            metrics["gripper_mismatch_rate"] = float(
                np.mean(ground_truth_sign != reconstruction_sign)
            )
    if continuous_squared_errors:
        metrics["rmse_continuous"] = float(
            np.sqrt(np.mean(np.concatenate(continuous_squared_errors)))
        )
    return metrics


def check_near_lossless(distortion_rmse: float, tolerance: float) -> None:
    """Raise when a near-lossless FAST round trip fails to reproduce the input.

    Args:
        distortion_rmse: Combined continuous original-space RMSE of the
            near-lossless cell.
        tolerance: Maximum RMSE allowed before the sweep is considered untrusted.

    Raises:
        ValueError: If ``distortion_rmse`` exceeds ``tolerance``.
    """
    if distortion_rmse > tolerance:
        raise ValueError(
            "Near-lossless FAST round trip did not reproduce the input "
            f"(continuous RMSE {distortion_rmse:.4f} > tolerance {tolerance:.4f}); "
            "the reconstruction pipeline is misconfigured and sweep numbers cannot "
            "be trusted."
        )


def _flatten_component(
    chunks: np.ndarray, component: ActionComponentLayout
) -> np.ndarray:
    """Slice one component and flatten it to (num_rows, component_dim)."""
    sliced = chunks[..., component.start : component.end]
    return sliced.reshape(-1, component.end - component.start)


def _to_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return a NumPy view of a tensor or array."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)
