"""Recover each cell's final rollout success rate from its sbatch log.

The training callback logs ``Synthetic rollout: epoch N, success=...`` every
``val_every`` epochs, and the workspace logs
``Workspace initialized for experiment: <config>/<cell>`` once at startup.
Reading those two lines back gives the secondary metric without importing
versatil, so it stays cheap on the login node. When a cell was rerun the newest
log wins.
"""

import re
from pathlib import Path

EXPERIMENT_PATTERN = re.compile(r"Workspace initialized for experiment: (\S+)")
ROLLOUT_PATTERN = re.compile(r"Synthetic rollout: epoch \d+, success=([0-9.]+)")
LOG_GLOB = "tip2_train_*.log"


def final_success_by_cell(log_dir: Path) -> dict[str, float]:
    """Map each cell name to the success rate of its last logged rollout.

    Args:
        log_dir: Directory holding the ``tip2_train_*.log`` sbatch outputs.

    Returns:
        Cell name to final success rate; cells without a rollout line are absent.
    """
    newest: dict[str, tuple[float, float]] = {}
    for log_path in log_dir.glob(LOG_GLOB):
        text = log_path.read_text(errors="replace")
        experiment_match = EXPERIMENT_PATTERN.search(text)
        successes = ROLLOUT_PATTERN.findall(text)
        if experiment_match is None or not successes:
            continue
        cell_name = experiment_match.group(1).rsplit("/", 1)[-1]
        modified = log_path.stat().st_mtime
        if cell_name not in newest or modified > newest[cell_name][0]:
            newest[cell_name] = (modified, float(successes[-1]))
    return {cell_name: success for cell_name, (_, success) in newest.items()}
