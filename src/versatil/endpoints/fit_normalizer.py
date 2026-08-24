"""Hydra endpoint that fits an action normalizer once and saves its state.

Sweeps that perturb the action data itself must not refit normalization per
run. With min-max statistics, adding noise widens the fitted action range, so
the underlying signal occupies a smaller fraction of the normalized ``[-1, 1]``
support. A discretizer with fixed bin edges then resolves that signal more
coarsely at higher noise, which confounds the noise effect with a change in
effective resolution -- asymmetrically, because continuous action heads have no
comparable term.

Fit once on the unperturbed configuration with this endpoint, then point every
run in the sweep at the result via ``task.dataloader.normalizer_state_path``.

Example:
    python -m versatil.endpoints.fit_normalizer \\
        --config-name end_to_end_training_runs/synthetic/gpt_transformer \\
        task/dataset_schema=synthetic/sequential \\
        +normalizer_output_path=/path/to/sequential_clean_normalizer.pt
"""

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from versatil.common.logging import override_log_format
from versatil.configs.paths import get_hydra_configs_dir
from versatil.data.dataloader import get_dataloaders

EXPERIMENTS_DIR = get_hydra_configs_dir()
OUTPUT_PATH_KEY = "normalizer_output_path"

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(EXPERIMENTS_DIR),
    config_name="end_to_end_training_runs/synthetic/gpt_transformer",
)
def main(config: DictConfig) -> None:
    """Fit the normalizer for one data configuration and save its state dict.

    Args:
        config: Hydra config describing the task and dataloader to fit on. It
            must additionally define ``normalizer_output_path``, the file the
            fitted state is written to.

    Raises:
        ValueError: If ``normalizer_output_path`` is missing from the config.
    """
    override_log_format()
    output_path = config.get(OUTPUT_PATH_KEY)
    if not output_path:
        raise ValueError(
            f"'{OUTPUT_PATH_KEY}' must be set, e.g. "
            f"+{OUTPUT_PATH_KEY}=/path/to/normalizer.pt"
        )
    if config.task.dataloader.get("normalizer_state_path"):
        # Loading a stored state and then re-saving it would silently copy a
        # normalizer rather than fit one, defeating the point of this endpoint.
        raise ValueError(
            "task.dataloader.normalizer_state_path must be unset when fitting "
            "a normalizer; it overrides the statistics this endpoint measures."
        )

    _, _, normalizer, _, _ = get_dataloaders(config=config)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(normalizer.state_dict(), destination)
    logger.info("Saved fitted normalizer state to %s", destination)


if __name__ == "__main__":
    main()
