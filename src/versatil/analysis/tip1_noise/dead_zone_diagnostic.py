"""Measure how much injected action noise FAST's rounding step destroys.

FAST quantizes with a single global scale, so ``round(coefficient * scale)``
maps every coefficient below ``0.5 / scale`` to zero. That dead zone is a
magnitude threshold applied identically at every frequency: whether it behaves
like a low-pass filter is a property of the signal's spectrum, not of the
tokenizer. The prediction this script tests is that noise power spread over
many coefficients -- high-band noise -- lands under the threshold more often
than the same power concentrated in few coefficients.

The measurement needs no trained model: generate clean and noisy episodes from
the same seed, transform both, and compare which coefficients survive rounding.
Running it before any training can falsify the mechanism outright.

    python src/versatil/analysis/tip1_noise/dead_zone_diagnostic.py out_dir
"""

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct

from versatil.data.synthetic.constants import NoiseInjection, SyntheticTaskName
from versatil.data.synthetic.generators import generate_task_episodes

TASK_NAME = SyntheticTaskName.SEQUENTIAL_DECISION.value
TASK_DEFAULT_NOISE_STD = 0.012
SIGMA_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)
# 0 is the high band (first-difference shaped); 2.0 is the low band chosen in
# the plan after the band-migration sweep saturated there.
SMOOTHING_SIGMAS = (0.0, 2.0)
# The dead zone is 0.5 / scale wide, so the rounding scale sets the amplitude
# below which noise is annihilated. Sweeping it over Tip 3's scale grid shows how
# far the denoising threshold moves with the tokenizer's one lossy knob.
FAST_SCALES = (2.0, 5.0, 10.0, 20.0, 50.0)
CHUNK_HORIZON = 10
NUM_EPISODES = 60
TRAJECTORY_LENGTH = 60
IMAGE_SIZE = 32
SEED = 0


def chunk_actions(actions: np.ndarray, horizon: int) -> np.ndarray:
    """Split (num_episodes, num_steps, dim) actions into non-overlapping chunks.

    Args:
        actions: Per-episode action arrays.
        horizon: Chunk length; a trailing remainder is dropped.

    Returns:
        Chunks of shape (num_chunks, horizon, dim).
    """
    num_episodes, num_steps, action_dim = actions.shape
    usable = (num_steps // horizon) * horizon
    trimmed = actions[:, :usable, :]
    return trimmed.reshape(num_episodes * (usable // horizon), horizon, action_dim)


def dead_zone_metrics(
    clean_chunks: np.ndarray,
    noisy_chunks: np.ndarray,
    scale: float,
) -> dict[str, float]:
    """Compare rounded DCT coefficients of clean and noisy chunks.

    Args:
        clean_chunks: Noise-free chunks, shape (num_chunks, horizon, dim).
        noisy_chunks: Same chunks with injected action noise.
        scale: FAST rounding scale; the dead zone is ``0.5 / scale`` wide.

    Returns:
        Fraction of noisy coefficients rounding to zero, the fraction whose
        rounded symbol the noise leaves untouched, and the ratio of surviving to
        injected noise amplitude. The ratio can exceed one: a coefficient
        sitting near a rounding boundary crosses it under a perturbation far
        smaller than the step it then jumps, so quantization removes noise from
        some coefficients while amplifying it on others. Reporting the raw ratio
        keeps both effects visible instead of hiding them in a "removed" number.
    """
    clean_coefficients = dct(clean_chunks, axis=1, norm="ortho")
    noisy_coefficients = dct(noisy_chunks, axis=1, norm="ortho")

    clean_rounded = np.around(clean_coefficients * scale)
    noisy_rounded = np.around(noisy_coefficients * scale)

    injected = noisy_coefficients - clean_coefficients
    surviving = (noisy_rounded - clean_rounded) / scale
    injected_rms = float(np.sqrt(np.mean(injected**2)))

    return {
        "zero_coefficient_fraction": float(np.mean(noisy_rounded == 0.0)),
        "unchanged_symbol_fraction": float(np.mean(noisy_rounded == clean_rounded)),
        "surviving_noise_ratio": (
            float(np.sqrt(np.mean(surviving**2)) / injected_rms)
            if injected_rms > 0.0
            else 0.0
        ),
    }


def run() -> list[dict[str, float | str]]:
    """Sweep noise level and band, returning one metrics row per cell."""
    shared = {
        "task_name": TASK_NAME,
        "num_episodes": NUM_EPISODES,
        "seed": SEED,
        "trajectory_length": TRAJECTORY_LENGTH,
        "image_size": IMAGE_SIZE,
    }
    # Generated on the action path too: the generator shuffles episodes with the
    # same random stream the noise is drawn from, so a clean baseline taken on a
    # different path would not line up chunk for chunk with the noisy runs.
    clean_episodes = generate_task_episodes(
        noise_std=0.0,
        noise_injection=NoiseInjection.ACTION.value,
        **shared,
    )
    clean_chunks = chunk_actions(
        np.stack([episode["action"] for episode in clean_episodes]), CHUNK_HORIZON
    )

    rows: list[dict[str, float | str]] = []
    for smoothing_sigma in SMOOTHING_SIGMAS:
        for multiplier in SIGMA_MULTIPLIERS:
            # Episode generation is the expensive part and does not depend on the
            # rounding scale, so every scale reuses the same noisy chunks.
            noisy_episodes = generate_task_episodes(
                noise_std=multiplier * TASK_DEFAULT_NOISE_STD,
                noise_smoothing_sigma=smoothing_sigma,
                noise_injection=NoiseInjection.ACTION.value,
                **shared,
            )
            noisy_chunks = chunk_actions(
                np.stack([episode["action"] for episode in noisy_episodes]),
                CHUNK_HORIZON,
            )
            for scale in FAST_SCALES:
                metrics = dead_zone_metrics(
                    clean_chunks=clean_chunks,
                    noisy_chunks=noisy_chunks,
                    scale=scale,
                )
                rows.append(
                    {
                        "band": "low" if smoothing_sigma > 0.0 else "high",
                        "smoothing_sigma": smoothing_sigma,
                        "sigma_multiplier": multiplier,
                        "noise_std": multiplier * TASK_DEFAULT_NOISE_STD,
                        "scale": scale,
                        "dead_zone_half_width": 0.5 / scale,
                        **metrics,
                    }
                )
    return rows


if __name__ == "__main__":
    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run()

    header = list(results[0].keys())
    with open(output_dir / "dead_zone_diagnostic.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)

    print(
        f"{'band':>5} {'scale':>7} {'deadzone':>9} {'sigma':>7} "
        f"{'zeroed':>8} {'unchanged':>10} {'survive':>8}"
    )
    for row in results:
        print(
            f"{row['band']:>5} {row['scale']:>7.0f} "
            f"{row['dead_zone_half_width']:>9.3f} {row['sigma_multiplier']:>7.2f} "
            f"{row['zero_coefficient_fraction']:>8.3f} "
            f"{row['unchanged_symbol_fraction']:>10.3f} "
            f"{row['surviving_noise_ratio']:>8.3f}"
        )
    print(f"\nWrote dead_zone_diagnostic.csv to {output_dir}")
