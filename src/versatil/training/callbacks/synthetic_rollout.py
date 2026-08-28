"""Callback for evaluating synthetic benchmark policies during training."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import Callback

from versatil.data.constants import ProprioKey, SyntheticObsKey
from versatil.data.preprocessing.replay_buffer import ReplayBuffer
from versatil.data.synthetic.task_layout import get_task_layout
from versatil.data.synthetic.visualization import plot_trajectories_2d
from versatil.inference.synthetic_rollout import evaluate_rollouts, run_rollouts
from versatil.models.policy import Policy
from versatil.training.callbacks.wandb_figure import figure_to_wandb_image
from versatil.training.constants import PrecisionType


class SyntheticRolloutCallback(Callback):
    """Run rollouts and log mode coverage metrics at the end of each training epoch.

    Puts the policy in eval mode, generates trajectories via closed-loop
    rollout, computes mode coverage and goal success against regenerated
    expert demonstrations, and logs metrics + trajectory plots to wandb. The
    training-data plot is read from the policy's own store so it shows the
    actual injected noise rather than a clean regenerated reference.

    Args:
        task_name: SyntheticTaskName.value string.
        num_modes: Number of behavioral modes to generate for expert
            reference. Must match the training dataset.
        num_styles: Number of sinusoidal styles per corridor gap. Ignored
            by tasks that do not use styles.
        trajectory_length: Length of generated expert and rollout
            trajectories.
        noise_std: Standard deviation of the noise used for the expert
            reference the rollouts are scored against.
        zarr_path: Store the policy trains on, read once to plot the actual
            training demonstrations rather than a regenerated clean reference.
        num_rollouts: Number of rollout trajectories per evaluation.
        image_size: Side length for rendered observation images.
        log_every_n_epochs: Evaluate every N epochs.
    """

    def __init__(
        self,
        task_name: str,
        num_modes: int,
        num_styles: int,
        trajectory_length: int,
        noise_std: float,
        zarr_path: str,
        num_rollouts: int = 50,
        image_size: int = 64,
        log_every_n_epochs: int = 1,
    ):
        """Initialize rollout generation and logging parameters."""
        super().__init__()
        self.task_name = task_name
        self.num_modes = num_modes
        self.num_styles = num_styles
        self.trajectory_length = trajectory_length
        self.noise_std = noise_std
        self.zarr_path = zarr_path
        self.num_rollouts = num_rollouts
        self.image_size = image_size
        self.log_every_n_epochs = log_every_n_epochs
        self._training_data_logged = False

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Run rollouts, compute metrics, log to wandb and console."""
        if trainer.logger is not None and not self._training_data_logged:
            self._log_training_data(trainer=trainer)
            self._training_data_logged = True

        is_log_epoch = trainer.current_epoch % self.log_every_n_epochs == 0
        is_last_epoch = trainer.current_epoch == trainer.max_epochs - 1
        if not is_log_epoch and not is_last_epoch:
            return

        policy: Policy = pl_module.policy
        was_training = policy.training
        policy.eval()

        # Rollout evaluation is diagnostic only. A failure here (e.g. a
        # tokenized decoder emitting an undecodable action sequence) must not
        # abort training, so any error is downgraded to a warning.
        try:
            context_modes = self._resolve_context_modes(policy=policy)
            precision_type = PrecisionType(str(trainer.precision))
            with (
                torch.no_grad(),
                precision_type.autocast(device_type=pl_module.device.type),
            ):
                per_mode_trajectories = [
                    run_rollouts(
                        policy=policy,
                        task_name=self.task_name,
                        num_rollouts=self.num_rollouts,
                        image_size=self.image_size,
                        context_mode=mode,
                        temporal_aggregation=False,  # open-loop
                    )
                    for mode in context_modes
                ]
            trajectories = (
                per_mode_trajectories[0]
                if len(per_mode_trajectories) == 1
                else np.concatenate(per_mode_trajectories, axis=0)
            )
            # Concatenation drops which context each batch was rolled out
            # with, so rebuild that label here: without it success cannot tell
            # a policy that follows the context from one that ignores it.
            expected_mode_ids = (
                None
                if context_modes == [None]
                else np.concatenate(
                    [
                        np.full(batch.shape[0], mode, dtype=np.int64)
                        for mode, batch in zip(
                            context_modes, per_mode_trajectories, strict=True
                        )
                    ]
                )
            )

            results = evaluate_rollouts(
                rollout_trajectories=trajectories,
                task_name=self.task_name,
                image_size=self.image_size,
                num_modes=self.num_modes,
                num_styles=self.num_styles,
                trajectory_length=self.trajectory_length,
                noise_std=self.noise_std,
                expected_mode_ids=expected_mode_ids,
            )
        except Exception:
            logging.warning(
                f"Synthetic rollout evaluation failed at epoch "
                f"{trainer.current_epoch}, skipping.",
                exc_info=True,
            )
            if was_training:
                policy.train()
            return

        epoch = trainer.current_epoch
        mode_coverage = results["mode_coverage"]
        entropy_ratio = results["mode_entropy_ratio"]
        valid_mode_coverage = results["valid_mode_coverage"]
        valid_entropy_ratio = results["valid_mode_entropy_ratio"]
        per_mode = results["per_mode_count"]
        success_rate = results["success_rate"]
        collision_rate = results["collision_rate"]
        endpoint_reach_rate = results["endpoint_reach_rate"]
        path_length_rate = results["path_length_rate"]

        log_parts = [
            f"epoch {epoch}",
            f"success={success_rate:.2f}",
            f"collision={collision_rate:.2f}",
            f"endpoint_reach={endpoint_reach_rate:.2f}",
            f"path_length={path_length_rate:.2f}",
            f"valid_mode_coverage={valid_mode_coverage:.2f}",
            f"valid_entropy={valid_entropy_ratio:.2f}",
            f"raw_mode_coverage={mode_coverage:.2f}",
            f"raw_entropy={entropy_ratio:.2f}",
            f"per_mode={per_mode}",
        ]
        has_context_metrics = "context_accuracy" in results
        if has_context_metrics:
            log_parts += [
                f"context_accuracy={results['context_accuracy']:.2f}",
                f"conditional_success={results['conditional_success_rate']:.2f}",
            ]
        logging.info(f"Synthetic rollout: {', '.join(log_parts)}")

        if trainer.logger is not None:
            metrics: dict[str, float | wandb.Image] = {
                "synthetic/success_rate": success_rate,
                "synthetic/collision_rate": collision_rate,
                "synthetic/endpoint_reach_rate": endpoint_reach_rate,
                "synthetic/path_length_rate": path_length_rate,
                "synthetic/valid_mode_coverage": valid_mode_coverage,
                "synthetic/valid_mode_entropy_ratio": valid_entropy_ratio,
                "synthetic/mode_coverage": mode_coverage,
                "synthetic/mode_entropy_ratio": entropy_ratio,
            }
            if has_context_metrics:
                metrics["synthetic/context_accuracy"] = results["context_accuracy"]
                metrics["synthetic/conditional_success_rate"] = results[
                    "conditional_success_rate"
                ]
            for mode_index, count in per_mode.items():
                metrics[f"synthetic/mode_{mode_index}_count"] = count

            rollout_figure = plot_trajectories_2d(
                trajectories=trajectories,
                task_name=self.task_name,
                num_modes=self.num_modes,
                num_styles=self.num_styles,
                noise_std=self.noise_std,
            )
            metrics["synthetic/rollout_trajectories"] = figure_to_wandb_image(
                rollout_figure, dpi=150
            )
            plt.close(rollout_figure)

            trainer.logger.log_metrics(metrics, step=epoch)

        if was_training:
            policy.train()

    def _resolve_context_modes(self, policy: Policy) -> list[int | None]:
        """Determine which context modes to roll out.

        Returns [None] for non-conditional policies. For policies that consume
        the CONTEXT observation, returns one entry per layout mode so every
        mode gets its own rollout batch.
        """
        has_context = (
            SyntheticObsKey.CONTEXT.value
            in policy.observation_space.observations_metadata
        )
        if not has_context:
            return [None]
        layout = get_task_layout(
            task_name=self.task_name,
            num_modes=self.num_modes,
            num_styles=self.num_styles,
            noise_std=self.noise_std,
        )
        return list(range(layout.num_modes))

    def _log_training_data(self, trainer: pl.Trainer) -> None:
        """Log the actual training demonstrations to wandb on the first epoch.

        Reads the policy's own store so the plot shows the noise the model
        trains on. Regenerating a reference here instead would draw a clean
        trajectory whenever the reference noise is zero, hiding label- or
        position-space corruption that is fully present in the training data.
        """
        buffer = ReplayBuffer.create_from_path(zarr_path=self.zarr_path)
        num_to_plot = min(100, buffer.n_episodes)
        trajectories = []
        mode_ids = []
        for episode_index in range(num_to_plot):
            episode = buffer.get_episode(episode_index)
            trajectories.append(episode[ProprioKey.SYNTHETIC_POSITION.value])
            mode_ids.append(int(episode[SyntheticObsKey.MODE_ID.value][0, 0]))
        figure = plot_trajectories_2d(
            trajectories=np.stack(trajectories),
            task_name=self.task_name,
            mode_ids=np.array(mode_ids),
            title="Training Data",
            num_modes=self.num_modes,
            num_styles=self.num_styles,
            noise_std=self.noise_std,
        )
        trainer.logger.log_metrics(
            {"synthetic/training_data": figure_to_wandb_image(figure, dpi=150)},
            step=0,
        )
        plt.close(figure)
