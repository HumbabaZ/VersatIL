"""Fixed default-noise synthetic stores for the Tip 2 tokenization sweep.

Tip 2 sweeps tokenization granularity (FAST ``scale``/``vocab_size``, binning
``num_bins``) on a single fixed store per task. Noise is Tip 1's variable, not
Tip 2's, so each store uses its task's own default generation parameters
(``noise_std`` 0.012 for sequential, 0.008 for conditional_circle) with the
native ``position`` injection and no temporal smoothing -- i.e. plain default
synthetic data, not the Tip 1 noise grid.

Two things differ from the shipped ``*.yaml`` defaults, both recorded here so
the sweep and the calibration read the same store:

* ``num_episodes`` is raised to 2000. Each episode yields exactly one
  full-length chunk (on-the-fly diff actions drop the terminal row, so
  ``trajectory_length - 1 = 59`` usable rows fill one ``horizon = 59`` window),
  and evaluation runs on full-length chunks only. At the default 1000 episodes
  the 5% validation split leaves ~50 full-length samples, too few for the
  granularity signal to clear the mode-mismatch floor; 2000 doubles it.
* The store lives at a Tip 2 path, unique per generation parameter combination,
  so it can never be resolved by an ordinary experiment and is never silently
  reused with the wrong parameters (the zarr cache does not hash generation
  parameters).
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from versatil.analysis.tip1_noise.sweep import TASK_SCHEMA_GROUP, method_config
from versatil.configs.paths import get_hydra_configs_dir
from versatil.data.preprocessing.create_zarr_from_synthetic import (
    create_replay_buffer_from_synthetic,
)
from versatil.data.preprocessing.replay_buffer import ReplayBuffer
from versatil.data.synthetic.constants import NoiseInjection

logger = logging.getLogger(__name__)

STORE_ROOT = Path("/data/horse/ws/qizh093f-versatil/zarr/tip2_synthetic")
DATA_SEED = 42
NUM_EPISODES = 2000


@dataclass(frozen=True)
class Tip2Store:
    """One fixed default-noise store shared by calibration and the sweep.

    Attributes:
        task: Key into ``TASK_SCHEMA_GROUP`` (``"sequential"`` /
            ``"conditional"``), selecting the dataset schema group.
        noise_std: The task's default action-label noise level.
        num_episodes: Episode count (2000; see module docstring).
        seed: Generation seed, also the ``data_seed`` the sweep fixes across
            replicates.
    """

    task: str
    noise_std: float
    num_episodes: int = NUM_EPISODES
    seed: int = DATA_SEED

    @property
    def name(self) -> str:
        """Store identifier encoding every generation parameter."""
        noise_tag = f"{self.noise_std:g}".replace(".", "p")
        return (
            f"{self.task}__noise-{noise_tag}__ep-{self.num_episodes}__seed-{self.seed}"
        )

    @property
    def zarr_path(self) -> str:
        """Absolute store path, unique per generation parameter combination."""
        return str(STORE_ROOT / f"{self.name}.zarr")

    def data_overrides(self) -> list[str]:
        """Hydra overrides pinning the generated dataset.

        Every field that shapes the stored episodes is set explicitly, including
        the ones that merely restate a default (``noise_injection=position``,
        ``noise_smoothing_sigma=0``), so the recorded command fully determines
        the data. Evaluation-time knobs (``eval_reference_noise_std``,
        ``num_rollouts``) are not here; they belong to the training command.
        """
        return [
            f"task/dataset_schema={TASK_SCHEMA_GROUP[self.task]}",
            f"task.dataset_schema.zarr_path={self.zarr_path}",
            f"task.dataset_schema.noise_std={self.noise_std:g}",
            f"task.dataset_schema.noise_injection={NoiseInjection.POSITION.value}",
            "task.dataset_schema.noise_smoothing_sigma=0",
            f"task.dataset_schema.num_episodes={self.num_episodes}",
            f"task.dataset_schema.seed={self.seed}",
        ]


SEQUENTIAL = Tip2Store(task="sequential", noise_std=0.012)
CONDITIONAL_CIRCLE = Tip2Store(task="conditional", noise_std=0.008)

STORES = {"sequential": SEQUENTIAL, "conditional": CONDITIONAL_CIRCLE}


def generate(store: Tip2Store, force: bool = False) -> str:
    """Generate ``store``'s zarr from default synthetic parameters.

    Args:
        store: The store specification.
        force: Delete and regenerate if the path already exists. Without it an
            existing store raises rather than being reused, because the zarr
            cache does not hash generation parameters and a stale store would be
            consumed silently.

    Returns:
        The generated store's path.

    Raises:
        FileExistsError: If the store exists and ``force`` is false.
    """
    path = Path(store.zarr_path)
    if path.exists():
        if not force:
            raise FileExistsError(
                f"{path} already exists. Pass force=True to regenerate; the zarr "
                "cache does not hash generation parameters, so reuse is silent."
            )
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(
        config_dir=str(get_hydra_configs_dir()), version_base=None
    ):
        config = compose(
            config_name=method_config(task=store.task, method="fast"),
            overrides=store.data_overrides(),
        )
    schema = instantiate(config.task.dataset_schema)
    create_replay_buffer_from_synthetic(schema=schema)

    buffer = ReplayBuffer.create_from_path(store.zarr_path)
    episode_ends = np.asarray(buffer.episode_ends[:])
    logger.info(
        "Generated %s: %d episodes, %d timesteps",
        store.name,
        episode_ends.shape[0],
        int(episode_ends[-1]),
    )
    return store.zarr_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Tip 2 fixed store.")
    parser.add_argument("task", choices=sorted(STORES))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    path = generate(STORES[args.task], force=args.force)
    print(path)


if __name__ == "__main__":
    _main()
