"""Capture rollout action tokens from a synthetic-benchmark checkpoint.

Loads the policy with its tokenizer and normalizer via the production float
checkpoint loader, runs in-process synthetic rollouts with token capture
enabled, and writes one combined JSONL across all context modes. Run as
``python -m versatil.analysis.token_usage.collect_synthetic_rollout``.
"""

import argparse
import logging

import torch

from versatil.analysis.token_usage.rollout_sink import RolloutTokenSink
from versatil.checkpoint_loading.float_policy import FloatCheckpointLoader
from versatil.common.logging import override_log_format
from versatil.data.constants import SyntheticObsKey
from versatil.data.synthetic.constants import DEFAULT_IMAGE_SIZE
from versatil.inference.synthetic_rollout import run_rollouts
from versatil.training.constants import CheckpointFilename

UNCONDITIONAL_MODE_TAG = -1


def collect_synthetic_rollout_tokens(
    checkpoint_path: str,
    task_name: str,
    output_path: str,
    num_rollouts: int = 20,
    num_modes: int = 2,
    image_size: int = DEFAULT_IMAGE_SIZE,
    checkpoint_name: str = CheckpointFilename.DEFAULT_CHECKPOINT.value,
    device: str = "cuda",
) -> str:
    """Run synthetic rollouts and capture predicted action tokens.

    Args:
        checkpoint_path: Directory with config, tokenizer, and the checkpoint.
        task_name: SyntheticTaskName value, e.g. "circle".
        output_path: JSONL destination for captured tokens.
        num_rollouts: Rollouts per context mode.
        num_modes: Behavioral-mode count for conditional tasks.
        image_size: Rendered observation side length; match training.
        checkpoint_name: Checkpoint filename inside checkpoint_path.
        device: Torch device string.

    Returns:
        The output path written.
    """
    loader = FloatCheckpointLoader(
        device=torch.device(device),
        checkpoint_path=checkpoint_path,
        checkpoint_name=checkpoint_name,
    )
    policy = loader.policy
    policy.eval()

    has_context = (
        SyntheticObsKey.CONTEXT.value in policy.observation_space.observations_metadata
    )
    context_modes: list[int | None] = list(range(num_modes)) if has_context else [None]

    sink = RolloutTokenSink(output_path=output_path)
    policy.set_token_usage_sink(sink=sink)
    for context_mode in context_modes:
        sink.set_context(
            context={
                "context_mode": context_mode
                if context_mode is not None
                else UNCONDITIONAL_MODE_TAG,
                "task_name": task_name,
            }
        )
        run_rollouts(
            policy=policy,
            task_name=task_name,
            num_rollouts=num_rollouts,
            image_size=image_size,
            context_mode=context_mode,
            temporal_aggregation=False,
        )
    flushed_path = sink.flush()
    policy.set_token_usage_sink(sink=None)
    logging.info(f"Captured synthetic rollout tokens to {flushed_path}")
    return str(flushed_path)


def main() -> None:
    """Run the synthetic rollout token capture from the command line."""
    override_log_format()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-rollouts", type=int, default=20)
    parser.add_argument("--num-modes", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--checkpoint-name",
        default=CheckpointFilename.DEFAULT_CHECKPOINT.value,
    )
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    collect_synthetic_rollout_tokens(
        checkpoint_path=arguments.checkpoint_path,
        task_name=arguments.task_name,
        output_path=arguments.output,
        num_rollouts=arguments.num_rollouts,
        num_modes=arguments.num_modes,
        image_size=arguments.image_size,
        checkpoint_name=arguments.checkpoint_name,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
