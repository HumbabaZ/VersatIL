"""Load LIBERO action chunks and the fitted normalizer for the FAST sweep.

Chunks are produced exactly as training produces tokenizer inputs: precomputed
actions are read from the zarr replay buffer, continuous components are
normalized with the training-fitted normalizer, the binary gripper is left raw,
and the components are concatenated in metadata order and windowed per episode
over the training split.
"""

from dataclasses import dataclass

import numpy as np
import torch

from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.data.episodic_dataset import EpisodicDataset
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.task import ActionSpace, ObservationSpace


@dataclass(frozen=True)
class ActionComponentLayout:
    """Placement of one action component inside a concatenated chunk.

    Attributes:
        key: Action metadata key (also the zarr storage key).
        start: Inclusive start column in the concatenated action vector.
        end: Exclusive end column in the concatenated action vector.
        needs_normalization: Whether this component was normalized (continuous
            position/orientation) or left raw (binary gripper).
    """

    key: str
    start: int
    end: int
    needs_normalization: bool


@dataclass
class ActionChunkData:
    """Normalized LIBERO action chunks plus the metadata to de-normalize them.

    Attributes:
        chunks_normalized: Array with shape (num_chunks, time_horizon,
            action_dim); the exact tensor FAST is fitted and evaluated on.
        normalizer: Training-fitted normalizer, used to map reconstruction error
            back into the original action space.
        layout: Per-component placement inside ``action_dim``.
    """

    chunks_normalized: np.ndarray
    normalizer: LinearNormalizer
    layout: list[ActionComponentLayout]

    @property
    def time_horizon(self) -> int:
        """Chunk time horizon."""
        return int(self.chunks_normalized.shape[1])

    @property
    def action_dim(self) -> int:
        """Concatenated action dimension."""
        return int(self.chunks_normalized.shape[2])


@dataclass
class PerStepContext:
    """Horizon-independent inputs for windowing action chunks at any horizon.

    The per-step normalized actions, the fitted normalizer, the component layout,
    and the episode boundaries do not depend on the chunk horizon, so a horizon
    sweep computes them once and re-windows chunks per horizon cheaply.

    Attributes:
        per_step_actions: Concatenated normalized per-step actions, shape
            (num_steps, action_dim).
        normalizer: Training-fitted normalizer.
        layout: Per-component placement inside ``action_dim``.
        episode_ends: Cumulative episode end indices.
        episode_selection_mask: Boolean mask of training-split episodes.
    """

    per_step_actions: np.ndarray
    normalizer: LinearNormalizer
    layout: list[ActionComponentLayout]
    episode_ends: np.ndarray
    episode_selection_mask: np.ndarray


def load_per_step_context(
    dataset_schema,
    action_space: ActionSpace,
    observation_space: ObservationSpace,
    dataloader_config: DataLoaderConfig,
    observation_horizon: int,
    seed: int,
    base_horizon: int = 10,
) -> PerStepContext:
    """Build the horizon-independent per-step actions, normalizer, and boundaries.

    Args:
        dataset_schema: Instantiated dataset schema pointing at the zarr store.
        action_space: Instantiated action space (LIBERO precomputed actions).
        observation_space: Instantiated observation space.
        dataloader_config: Data-loading settings; supplies normalization type and
            winsorization used at training time.
        observation_horizon: History length required by the dataset.
        seed: Split seed, matching the training split.
        base_horizon: Chunk horizon used only to construct the dataset's sampler;
            it does not affect the per-step actions, normalizer, or boundaries.

    Returns:
        The reusable per-step context.
    """
    dataset = EpisodicDataset(
        zarr_path=dataset_schema.zarr_path,
        action_space=action_space,
        observation_space=observation_space,
        dataloader_config=dataloader_config,
        pred_horizon=base_horizon,
        obs_horizon=observation_horizon,
        train=True,
        seed=seed,
        augment_images=False,
    )
    normalizer, _ = dataset.get_normalizer_and_tokenizer(
        device=torch.device("cpu"),
        winsorize_depth=dataloader_config.winsorize_depth,
        depth_winsorize_quantiles=dataloader_config.depth_winsorize_quantiles,
        winsorize_kinematics=dataloader_config.winsorize_kinematics,
        kinematics_winsorize_quantiles=dataloader_config.kinematics_winsorize_quantiles,
        tokenization_config=None,
        clamp_kinematics_range=dataloader_config.clamp_kinematics_range,
        min_kinematics_std=dataloader_config.min_kinematics_std,
        min_kinematics_range=dataloader_config.min_kinematics_range,
        action_sample_size=0,
    )
    per_step_actions, layout = _compute_normalized_per_step_actions(
        dataset=dataset, normalizer=normalizer
    )
    return PerStepContext(
        per_step_actions=per_step_actions,
        normalizer=normalizer,
        layout=layout,
        episode_ends=np.asarray(dataset.episode_ends),
        episode_selection_mask=np.asarray(dataset.episode_selection_mask),
    )


def chunk_data_at_horizon(
    context: PerStepContext,
    horizon: int,
    max_chunks: int = 0,
    seed: int = 42,
) -> ActionChunkData:
    """Window the per-step context into fixed-length chunks at one horizon."""
    chunks = _window_selected_episodes(
        per_step_actions=context.per_step_actions,
        episode_ends=context.episode_ends,
        episode_selection_mask=context.episode_selection_mask,
        prediction_horizon=horizon,
    )
    chunks = _subsample_chunks(chunks=chunks, max_chunks=max_chunks, seed=seed)
    return ActionChunkData(
        chunks_normalized=chunks,
        normalizer=context.normalizer,
        layout=context.layout,
    )


def load_libero_action_chunks(
    dataset_schema,
    action_space: ActionSpace,
    observation_space: ObservationSpace,
    dataloader_config: DataLoaderConfig,
    prediction_horizon: int,
    observation_horizon: int,
    seed: int,
    max_chunks: int = 0,
) -> ActionChunkData:
    """Build normalized action chunks and the fitted normalizer at one horizon.

    Thin wrapper over :func:`load_per_step_context` + :func:`chunk_data_at_horizon`
    kept for the single-horizon reconstruction endpoint.

    Raises:
        ValueError: If no episode is long enough to form a single chunk.
    """
    context = load_per_step_context(
        dataset_schema=dataset_schema,
        action_space=action_space,
        observation_space=observation_space,
        dataloader_config=dataloader_config,
        observation_horizon=observation_horizon,
        seed=seed,
        base_horizon=prediction_horizon,
    )
    return chunk_data_at_horizon(
        context=context,
        horizon=prediction_horizon,
        max_chunks=max_chunks,
        seed=seed,
    )


def _compute_normalized_per_step_actions(
    dataset: EpisodicDataset,
    normalizer: LinearNormalizer,
) -> tuple[np.ndarray, list[ActionComponentLayout]]:
    """Compute the concatenated per-step normalized action array and its layout.

    Mirrors the tokenizer-input construction in
    ``TransformBuilder._create_tokenizer``: only numerical action keys with a
    prediction head are included, continuous keys are normalized, and the binary
    gripper is passed through raw. Components follow action-metadata order.
    """
    action_processor = dataset.action_processor
    action_keys = dataset.action_space.get_required_zarr_keys()
    source_data = {
        key: dataset.replay_buffer[key][:]
        for key in action_keys
        if key in dataset.replay_buffer
    }
    action_data, action_meta = action_processor.compute_sample_actions(
        padded_data=source_data,
        action_slice_start=0,
        action_slice_end=dataset.replay_buffer.n_steps - 1,
    )
    ordered_keys = [
        key
        for key, meta in action_meta.items()
        if meta.is_numerical and meta.requires_prediction_head
    ]
    columns = []
    layout: list[ActionComponentLayout] = []
    cursor = 0
    for key in ordered_keys:
        meta = action_meta[key]
        values = action_data[key].astype(np.float32)
        if meta.needs_normalization:
            normalized = normalizer[key].normalize(values)
            values = _to_numpy(normalized).astype(np.float32)
        width = values.shape[-1]
        layout.append(
            ActionComponentLayout(
                key=key,
                start=cursor,
                end=cursor + width,
                needs_normalization=meta.needs_normalization,
            )
        )
        cursor += width
        columns.append(values)
    return np.concatenate(columns, axis=-1), layout


def _window_selected_episodes(
    per_step_actions: np.ndarray,
    episode_ends: np.ndarray,
    episode_selection_mask: np.ndarray,
    prediction_horizon: int,
) -> np.ndarray:
    """Slice fixed-length chunks inside each selected (training) episode.

    LIBERO uses precomputed actions, so every step of an episode is a valid chunk
    row and no terminal step is dropped.
    """
    row_count = per_step_actions.shape[0]
    chunks = []
    episode_start = 0
    for episode_index in range(len(episode_ends)):
        episode_end = min(int(episode_ends[episode_index]), row_count)
        length = episode_end - episode_start
        if bool(episode_selection_mask[episode_index]) and length >= prediction_horizon:
            episode_actions = per_step_actions[episode_start:episode_end]
            for offset in range(length - prediction_horizon + 1):
                chunks.append(episode_actions[offset : offset + prediction_horizon])
        episode_start = episode_end
    if not chunks:
        raise ValueError(
            "No training episode is long enough to build a single action chunk of "
            f"prediction_horizon={prediction_horizon}."
        )
    return np.stack(chunks, axis=0)


def _subsample_chunks(chunks: np.ndarray, max_chunks: int, seed: int) -> np.ndarray:
    """Randomly subsample chunks to ``max_chunks`` for a bounded runtime."""
    if max_chunks <= 0 or chunks.shape[0] <= max_chunks:
        return chunks
    generator = np.random.default_rng(seed)
    indices = generator.choice(chunks.shape[0], size=max_chunks, replace=False)
    return chunks[np.sort(indices)]


def _to_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return a NumPy view of a tensor or array."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return values
