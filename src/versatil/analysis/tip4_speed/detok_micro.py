"""Offline micro-benchmark of the FAST detokenization sub-stages.

The main benchmark times detokenization as one coarse segment. This script
answers "where inside detok does the time go" without touching the timed
protocol: it replays the token sequences the benchmark dumped through the
three FAST decode sub-stages -- special-token strip/unmap, reverse BPE
(including the per-character ``ord`` conversion), and the inverse DCT -- and
reports per-stage microseconds. Pure CPU, no GPU needed; run it only when the
coarse ``t_detok_share`` is worth explaining.

Usage:

    python -m versatil.analysis.tip4_speed.detok_micro \
        <checkpoint_dir> <tokens_jsonl> <output_csv>
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.fft import idct

from versatil.checkpoint_loading.float_policy import FloatCheckpointLoader
from versatil.training.constants import CheckpointFilename


def _main() -> None:
    parser = argparse.ArgumentParser(description="FAST detok sub-stage timing.")
    parser.add_argument("checkpoint_dir")
    parser.add_argument("tokens_jsonl", type=Path)
    parser.add_argument("output_csv", type=Path)
    arguments = parser.parse_args()

    loader = FloatCheckpointLoader(
        device=torch.device("cpu"),
        checkpoint_path=arguments.checkpoint_dir,
        checkpoint_name=CheckpointFilename.DEFAULT_CHECKPOINT.value,
    )
    action_tokenizer = loader.tokenizer.action_tokenizer
    discretizer = action_tokenizer.action_discretizer
    scale = discretizer._processor_scale()

    sequences = [
        json.loads(line)
        for line in arguments.tokens_jsonl.read_text().splitlines()
        if line.strip()
    ]
    if not sequences:
        raise ValueError(f"No token sequences in {arguments.tokens_jsonl}.")

    rows: list[dict[str, float | int]] = []
    for index, sequence in enumerate(sequences):
        start = time.perf_counter()
        local_tokens = action_tokenizer._strip_and_unmap_tokens(
            np.asarray(sequence, dtype=np.int64)
        )
        after_strip = time.perf_counter()
        coefficients = discretizer.bpe_ids_to_coefficient_tokens(
            bpe_local_ids=local_tokens.tolist()
        )
        after_bpe = time.perf_counter()
        matrix = coefficients.reshape(discretizer.time_horizon, discretizer.action_dim)
        idct(matrix / scale, axis=0, norm="ortho")
        after_idct = time.perf_counter()
        rows.append(
            {
                "sequence_index": index,
                "token_count": len(sequence),
                "strip_unmap_us": (after_strip - start) * 1e6,
                "reverse_bpe_us": (after_bpe - after_strip) * 1e6,
                "idct_us": (after_idct - after_bpe) * 1e6,
                "total_us": (after_idct - start) * 1e6,
            }
        )

    arguments.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(arguments.output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for stage in ("strip_unmap_us", "reverse_bpe_us", "idct_us", "total_us"):
        values = sorted(row[stage] for row in rows)
        print(f"{stage}: median {values[len(values) // 2]:.1f} us")


if __name__ == "__main__":
    _main()
