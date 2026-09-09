"""Re-score a finished Tip 1 stage offline, without retraining it.

The success threshold and the decoding mode affect only evaluation, never the
weights, so revisiting either is a re-scoring job rather than a retraining one.
This module makes that cheap: it enumerates a stage's cells through sweep.py, so
method, trajectory length, sigma and noise model come from the single source of
truth, loads each cell's final checkpoint, rolls out, and records the raw
Euclidean loop-closure error alongside a success sweep over the tolerance.

Recording the error and not only the thresholded success matters. A thresholded
rate cannot distinguish an arm sitting just inside the tolerance from one far
inside it, nor rank two arms that have both fallen outside; the raw error
answers both, and it is what fixed the tolerance in the first place.

Two hazards this module handles rather than inherits:

- A cell retrained under a name an earlier run already used does not replace
  that run's checkpoints. Lightning writes "-v1" beside them, so the directory
  holds both runs and the un-suffixed files are the OLD ones. ``final_ckpt``
  ranks by epoch and then modification time, and names any directory holding
  more than one run.
- The task is a closed loop and both modes share an endpoint, so what is
  measured here is loop-closure error, not target acquisition. Mode correctness
  is a separate conjunct and comes from ``evaluate_rollouts``.

    python -m versatil.analysis.tip1_noise.scan_endpoints <stage> <out_dir>

Uses the GPU when one is visible. The login node is routinely oversubscribed,
where a single cell has taken ~20 minutes; prefer scripts/tip1_scan.sbatch.
"""

import csv
import glob
import os
import re
import sys
import time

import numpy as np
import torch

from versatil.analysis.tip1_noise.sweep import method_config, stage_cells
from versatil.checkpoint_loading.float_policy import FloatCheckpointLoader
from versatil.data.synthetic.constants import SyntheticTaskName
from versatil.data.synthetic.generators import generate_task_episodes
from versatil.inference.synthetic_rollout import evaluate_rollouts, run_rollouts
from versatil.metrics.synthetic_metrics import compute_mode_endpoints

CKPT_ROOT = "/data/horse/ws/qizh093f-versatil/checkpoints/synthetic"
NUM_MODES = 2
IMAGE_SIZE = 64
NUM_ROLLOUTS = 10
TASK = SyntheticTaskName.CONDITIONAL_CIRCLE.value
THRESHOLDS = [0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15]
# The login node is heavily oversubscribed; on a GPU node this runs ~20x faster.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ckpt_dir(cell) -> str | None:
    subdir = method_config(cell.data.task, cell.method).split("/")[-1]
    path = os.path.join(CKPT_ROOT, subdir, cell.name)
    return path if os.path.isdir(path) else None


def final_ckpt(cell_path: str) -> str | None:
    """Highest-epoch checkpoint of the most recent run in this directory.

    A cell retrained under the same name writes beside the earlier run rather
    than replacing it: Lightning appends "-v1" instead of overwriting, so the
    directory ends up holding both runs and the un-suffixed files are the OLD
    ones. Matching only "latest-<epoch>.ckpt" therefore silently loads stale
    weights. Rank by epoch first and modification time second so the newest
    run wins, and say so when a directory holds more than one run.
    """
    candidates = []
    for path in glob.glob(f"{cell_path}/latest-*.ckpt"):
        m = re.search(r"latest-(\d+)(?:-v\d+)?\.ckpt$", os.path.basename(path))
        if m:
            candidates.append((int(m.group(1)), os.path.getmtime(path), path))
    if not candidates:
        fallback = f"{cell_path}/last.ckpt"
        return fallback if os.path.exists(fallback) else None
    days = {time.strftime("%m-%d", time.localtime(t)) for _, t, _ in candidates}
    if len(days) > 1:
        print(f"    NOTE mixed runs in {os.path.basename(cell_path)}: {sorted(days)}")
    return max(candidates)[2]


def clean_mode_endpoints(trajectory_length: int) -> np.ndarray:
    episodes = generate_task_episodes(
        task_name=TASK,
        num_episodes=200,
        seed=42,
        image_size=IMAGE_SIZE,
        num_modes=NUM_MODES,
        trajectory_length=trajectory_length,
        noise_std=0.0,
    )
    trajectories = np.array([ep["position"] for ep in episodes])
    mode_ids = np.array([int(ep["mode_id"][0, 0]) for ep in episodes])
    return compute_mode_endpoints(trajectories, mode_ids, NUM_MODES)


def scan_cell(cell, endpoints_cache):
    cell_path = ckpt_dir(cell)
    if cell_path is None:
        print(f"  MISS dir  {cell.name}")
        return None
    ckpt = final_ckpt(cell_path)
    if ckpt is None:
        print(f"  MISS ckpt {cell.name}")
        return None

    tl = cell.data.trajectory_length
    if tl not in endpoints_cache:
        endpoints_cache[tl] = clean_mode_endpoints(tl)
    mode_endpoints = endpoints_cache[tl]

    loader = FloatCheckpointLoader(
        device=DEVICE,
        checkpoint_path=cell_path,
        checkpoint_name=os.path.basename(ckpt),
    )
    policy = loader.policy
    policy.eval()

    per_mode = []
    with torch.no_grad():
        for mode in range(NUM_MODES):
            per_mode.append(
                run_rollouts(
                    policy=policy,
                    task_name=TASK,
                    num_rollouts=NUM_ROLLOUTS,
                    image_size=IMAGE_SIZE,
                    context_mode=mode,
                    temporal_aggregation=False,
                )
            )
    trajectories = np.concatenate(per_mode, axis=0)
    expected = np.concatenate(
        [np.full(b.shape[0], m, dtype=np.int64) for m, b in enumerate(per_mode)]
    )
    ends = trajectories[:, -1, :]
    errors = np.linalg.norm(ends - mode_endpoints[expected], axis=-1)

    sweep = {}
    for thr in THRESHOLDS:
        res = evaluate_rollouts(
            rollout_trajectories=trajectories,
            task_name=TASK,
            image_size=IMAGE_SIZE,
            num_modes=NUM_MODES,
            trajectory_length=tl,
            noise_std=0.0,
            expected_mode_ids=expected,
            endpoint_reach_threshold=thr,
        )
        sweep[thr] = float(res["conditional_success_rate"])

    tag = f"{cell.method:6s} T{tl:<3d} sig-{cell.data.sigma_multiplier:g}"
    print(
        f"  {tag}  err[min/med/max]="
        f"{errors.min():.4f}/{np.median(errors):.4f}/{errors.max():.4f}  "
        f"succ@.025={sweep[0.025]:.2f} succ@.05={sweep[0.05]:.2f}"
    )
    return {"cell": cell, "errors": errors, "expected": expected, "sweep": sweep}


def main() -> None:
    stage = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    cells = stage_cells(stage)
    print(f"stage={stage}: {len(cells)} cells  device={DEVICE}")

    endpoints_cache: dict[int, np.ndarray] = {}
    rows_err, rows_sweep = [], []
    for cell in cells:
        r = scan_cell(cell, endpoints_cache)
        if r is None:
            continue
        c = r["cell"]
        for err, exp in zip(r["errors"], r["expected"], strict=True):
            rows_err.append(
                {
                    "method": c.method,
                    "trajectory_length": c.data.trajectory_length,
                    "sigma": c.data.sigma_multiplier,
                    "noise_model": c.data.noise_model,
                    "requested_mode": int(exp),
                    "endpoint_error": float(err),
                }
            )
        row = {
            "method": c.method,
            "trajectory_length": c.data.trajectory_length,
            "sigma": c.data.sigma_multiplier,
            "noise_model": c.data.noise_model,
        }
        row.update({f"succ@{t}": r["sweep"][t] for t in THRESHOLDS})
        rows_sweep.append(row)

    err_fields = [
        "method",
        "trajectory_length",
        "sigma",
        "noise_model",
        "requested_mode",
        "endpoint_error",
    ]
    with open(
        os.path.join(out_dir, f"{stage}_endpoint_errors.csv"), "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=err_fields)
        w.writeheader()
        w.writerows(rows_err)
    sweep_fields = ["method", "trajectory_length", "sigma", "noise_model"] + [
        f"succ@{t}" for t in THRESHOLDS
    ]
    with open(
        os.path.join(out_dir, f"{stage}_threshold_sweep.csv"), "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=sweep_fields)
        w.writeheader()
        w.writerows(rows_sweep)
    print(f"Wrote {out_dir}/{stage}_endpoint_errors.csv and _threshold_sweep.csv")


if __name__ == "__main__":
    main()
