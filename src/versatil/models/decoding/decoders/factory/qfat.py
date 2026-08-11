"""Quantization-Free Action Transformer (QFAT) for continuous multi-modal action prediction.

Autoregressive GPT decoder that predicts Gaussian Mixture Model parameters per action
timestep, following the GIVT approach: no discrete tokenization, continuous action space.

Reference: https://github.com/ziyadsheeba/qfat
"""

import torch
import torch.nn as nn

from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.action_heads.gaussian import GaussianHead
from versatil.models.decoding.action_masking import make_attention_mask
from versatil.models.decoding.constants import (
    DecoderOutputKey,
    GMMInitStrategy,
    MixtureSamplingMode,
)
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
from versatil.models.decoding.decoders.factory.mode_act import (
    MixtureOfDensitiesActionTransformer,
)
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
    SinusoidalPositionalEncoding2D,
)
from versatil.models.layers.transformer.autoregressive_decoder import GPTDecoder


class QFATActionTransformer(ActionDecoder):
    """Quantization-Free Action Transformer for continuous multi-modal action prediction.

    Combines a GPT-style causal transformer backbone with per-timestep Gaussian Mixture
    Model outputs. Unlike MoDE-ACT (bidirectional, K separate heads), QFAT uses causal
    self-attention and a single joint linear projection per action key.

    Training: teacher-forcing over the prediction horizon.
    Inference: autoregressive sampling from the GMM with KV caching.
    """

    def __init__(
        self,
        action_heads: dict[str, ActionHead],
        input_keys: list[str],
        action_space: ActionSpace,
        observation_space: ObservationSpace,
        observation_horizon: int,
        prediction_horizon: int,
        device: str,
        embedding_dimension: int = 512,
        number_of_heads: int = 8,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        number_of_layers: int = 6,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.0,
        positional_encoding_type: str | None = PositionalEncodingType.ROPE.value,
        num_mixture_components: int = 8,
        gmm_init_strategy: str = GMMInitStrategy.KMEANS_PLUS_PLUS.value,
        inference_sampling_mode: str = MixtureSamplingMode.STOCHASTIC_MEAN.value,
        max_seq_len: int = 512,
        min_logvar: float = -10.0,
        max_logvar: float = 4.0,
    ):
        """Initialize QFATActionTransformer.

        Args:
            action_heads: Action heads per action key (GaussianHead for continuous, ActionHead for gripper).
            input_keys: Feature keys from the encoding pipeline.
            action_space: Action space configuration.
            observation_space: Observation space configuration.
            observation_horizon: Number of observation timesteps.
            prediction_horizon: Number of action timesteps to predict autoregressively.
            device: Device to run model on.
            embedding_dimension: Transformer hidden dimension.
            number_of_heads: Number of query attention heads.
            number_of_key_value_heads: KV heads for GQA (None = same as query heads).
            feedforward_dimension: FFN hidden size (default: 4 * embedding_dimension).
            number_of_layers: Number of transformer layers.
            activation: Activation function.
            normalization_type: Normalization type.
            attention_type: Attention type.
            dropout_rate: Dropout probability.
            attention_dropout: Attention dropout probability.
            positional_encoding_type: Positional encoding (rope, sinusoidal, or None).
            num_mixture_components: Number of GMM components K.
            gmm_init_strategy: Strategy for initializing GMM means (kmeans_plus_plus or uniform).
            inference_sampling_mode: How to sample from GMM at inference time.
            max_seq_len: Maximum sequence length for the GPT decoder.
            min_logvar: Minimum logvar for GMM components.
            max_logvar: Maximum logvar for GMM components.
        """
        if gmm_init_strategy not in [s.value for s in GMMInitStrategy]:
            raise ValueError(
                f"Unknown gmm_init_strategy: {gmm_init_strategy}. "
                f"Expected one of {[s.value for s in GMMInitStrategy]}"
            )
        if inference_sampling_mode not in [m.value for m in MixtureSamplingMode]:
            raise ValueError(
                f"Unknown inference_sampling_mode: {inference_sampling_mode}. "
                f"Expected one of {[m.value for m in MixtureSamplingMode]}"
            )

        decoder_input = DecoderInput(keys=input_keys, requires_actions=True)
        super().__init__(
            decoder_input=decoder_input,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        self.embedding_dimension = embedding_dimension
        self.num_mixture_components = num_mixture_components
        self.gmm_init_strategy = gmm_init_strategy
        self.inference_sampling_mode = inference_sampling_mode
        self.max_seq_len = max_seq_len
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        # Ordered key list and per-key action dims (preserved from action_heads insertion order)
        self._action_key_order = list(self.action_heads.keys())
        self._action_dims = {
            key: meta.prediction_dimension
            for key, meta in action_space.actions_metadata.items()
            if meta.requires_prediction_head
        }
        self._total_action_dim = sum(
            self._action_dims[k] for k in self._action_key_order
        )

        self._build_transformer_components(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            number_of_layers=number_of_layers,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            positional_encoding_type=positional_encoding_type,
            max_seq_len=max_seq_len,
        )
        self._build_gmm_projections()
        self.to(self.device)

    def _build_transformer_components(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        number_of_key_value_heads: int | None,
        feedforward_dimension: int | None,
        number_of_layers: int,
        activation: str,
        normalization_type: str,
        attention_type: str,
        dropout_rate: float,
        attention_dropout: float,
        positional_encoding_type: str | None,
        max_seq_len: int,
    ) -> None:
        """Build observation encoder, action input projection, BOS token, and GPT decoder."""
        image_pe = SinusoidalPositionalEncoding2D(
            embedding_dimension=embedding_dimension, normalize=True
        )
        temporal_pe = (
            LearnedPositionalEncoding1D(embedding_dimension=embedding_dimension)
            if self.observation_horizon > 1
            else None
        )
        # Every feature crossing the pipeline boundary carries a canonical (B, T, ...)
        # layout, so the input builder infers the feature kind from rank alone. The
        # temporal positional encoding stays None for single-frame observations, where
        # the leading time axis has length one.
        self.input_sequence_builder = TransformerInputBuilder(
            embedding_dimension=embedding_dimension,
            spatial_positional_encoding_layer=image_pe,
            flat_positional_encoding_layer=SinusoidalPositionalEncoding1D(
                embedding_dimension=embedding_dimension
            ),
            temporal_positional_encoding_layer=temporal_pe,
        )

        self.action_input_proj = nn.Linear(self._total_action_dim, embedding_dimension)

        self.gpt_decoder = GPTDecoder(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout_rate,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            use_cross_attention=False,
            positional_encoding_type=positional_encoding_type,
            maximum_sequence_length=max_seq_len,
        )

        self.action_bos_embedding = nn.Parameter(torch.empty(1, 1, embedding_dimension))
        nn.init.normal_(
            self.action_bos_embedding, mean=0.0, std=self.gpt_decoder.initializer_range
        )

    def _build_gmm_projections(self) -> None:
        """Build per-key GMM mean/logvar projections and a shared routing projection.

        Keys backed by GaussianHead get mean + logvar projections (output: {key}_mean,
        {key}_logvar). Other keys (e.g. binary gripper) get a direct projection
        (output: {key} as stacked expert logits/means).
        """
        self._gaussian_keys: list[str] = []
        self._direct_keys: list[str] = []

        mean_projs: dict[str, nn.Module] = {}
        logvar_projs: dict[str, nn.Module] = {}
        direct_projs: dict[str, nn.Module] = {}

        K = self.num_mixture_components
        for key in self._action_key_order:
            head = self.action_heads[key]
            action_dim = self._action_dims[key]
            if isinstance(head, GaussianHead):
                self._gaussian_keys.append(key)
                # mean
                mean_projs[key] = nn.Linear(self.embedding_dimension, K * action_dim)
                # variance
                logvar_projs[key] = nn.Linear(self.embedding_dimension, K * action_dim)
            else:
                self._direct_keys.append(key)
                direct_projs[key] = nn.Linear(self.embedding_dimension, K * action_dim)

        self.mean_projs = nn.ModuleDict(mean_projs)
        self.logvar_projs = nn.ModuleDict(logvar_projs)
        self.direct_projs = nn.ModuleDict(direct_projs)
        self.routing_proj = nn.Linear(
            self.embedding_dimension, self.num_mixture_components
        )

    def get_auxiliary_output_keys(self) -> set[str]:
        """QFAT produces per-timestep routing weights."""
        keys = super().get_auxiliary_output_keys()
        keys.add(DecoderOutputKey.ROUTING_WEIGHTS.value)
        return keys

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Initialize GMM projections from normalizer statistics using k-means++."""
        super().set_normalizer(normalizer)
        self._initialize_routing_uniform()
        for key in self._gaussian_keys:
            if key not in normalizer.params_dict:
                continue
            field_normalizer = normalizer[key]
            output_stats = field_normalizer.get_output_stats()
            chunks = field_normalizer.get_output_sample_chunks()
            candidate_sample = (
                None if chunks is None else chunks.reshape(-1, chunks.shape[-1])
            )
            self._initialize_gmm_projection(
                key=key,
                output_stats=output_stats,
                candidate_sample=candidate_sample,
            )

    def _initialize_routing_uniform(self) -> None:
        """Zero-init the routing projection so all components start equally weighted."""
        nn.init.zeros_(self.routing_proj.weight)
        nn.init.zeros_(self.routing_proj.bias)

    def _initialize_gmm_projection(
        self,
        key: str,
        output_stats: dict[str, torch.Tensor],
        candidate_sample: torch.Tensor | None = None,
    ) -> None:
        """Initialize mean and logvar projections for one action key.

        Uses the same k-means++ / uniform spread logic as MoDE-ACT and
        QFAT-style fixed variance (sigma = half-range).

        Args:
            key: Action key to initialize.
            output_stats: Dict with "min", "max", "std" from the normalizer.
            candidate_sample: Optional (N, action_dim) action samples for k-means++ seeding.
        """
        data_min = output_stats["min"]
        data_max = output_stats["max"]
        K = self.num_mixture_components
        action_dim = self._action_dims[key]

        if self.gmm_init_strategy == GMMInitStrategy.KMEANS_PLUS_PLUS.value:
            if candidate_sample is not None and candidate_sample.numel() > 0:
                candidates = candidate_sample.to(
                    dtype=data_min.dtype, device=data_min.device
                )
            else:
                candidates = (
                    MixtureOfDensitiesActionTransformer._sample_uniform_candidates(
                        data_min=data_min,
                        data_max=data_max,
                        num_candidates=1000,
                        out_dim=action_dim,
                    )
                )
            centers = (
                MixtureOfDensitiesActionTransformer._compute_kmeans_plus_plus_centers(
                    candidate_points=candidates,
                    number_of_mixture_components=K,
                )
            )
        else:
            centers = MixtureOfDensitiesActionTransformer._compute_uniform_centers(
                data_min=data_min,
                data_max=data_max,
                number_of_mixture_components=K,
            )

        # QFAT-style fixed variance: sigma = half data range
        data_range = data_max - data_min
        expert_sigma = (data_range / 2.0).clamp(min=1e-6)
        expert_logvar = (2 * torch.log(expert_sigma)).clamp(
            self.min_logvar, self.max_logvar
        )

        with torch.no_grad():
            # mean_proj bias: K centers stacked → (K * action_dim,)
            nn.init.zeros_(self.mean_projs[key].weight)
            self.mean_projs[key].bias.copy_(centers.reshape(-1))

            # logvar_proj bias: same logvar for all K components → (K * action_dim,)
            nn.init.zeros_(self.logvar_projs[key].weight)
            self.logvar_projs[key].bias.copy_(
                expert_logvar.unsqueeze(0).expand(K, -1).reshape(-1)
            )

    def _select_input_features(
        self, features: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Select decoder-required features and their padding masks."""
        missing = [k for k in self.decoder_input.keys if k not in features]
        if missing:
            raise ValueError(
                f"QFATActionTransformer missing required features {missing}. "
                f"Available: {sorted(features.keys())}."
            )
        selected: dict[str, torch.Tensor] = {}
        for key in self.decoder_input.keys:
            selected[key] = features[key]
            mask_key = f"{key}_{EncoderOutputKeys.PADDING_MASK.value}"
            if mask_key in features:
                selected[mask_key] = features[mask_key]
        return selected

    def _concat_actions(self, actions: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate action tensors in action_key_order.

        Args:
            actions: Per-key tensors of shape (B, T, D_k).

        Returns:
            (B, T, total_action_dim) concatenated float tensor.
        """
        return torch.cat([actions[k].float() for k in self._action_key_order], dim=-1)

    def _compute_gmm_params(
        self, action_hidden: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Project hidden states to per-timestep GMM parameters.

        Args:
            action_hidden: (B, T, embedding_dimension)

        Returns:
            Dict containing:
                routing_weights: (B, T, K)
                {key}_mean: (B, T, K, D) for Gaussian keys
                {key}_logvar: (B, T, K, D) for Gaussian keys
                {key}: (B, T, K, D) for direct (non-Gaussian) keys
        """
        B, T, _ = action_hidden.shape
        K = self.num_mixture_components

        routing_logits = self.routing_proj(action_hidden)  # (B, T, K)
        routing_weights = torch.softmax(routing_logits, dim=-1)

        out: dict[str, torch.Tensor] = {
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights
        }

        for key in self._gaussian_keys:
            D = self._action_dims[key]
            mean = self.mean_projs[key](action_hidden).view(B, T, K, D)
            logvar = self.logvar_projs[key](action_hidden).view(B, T, K, D)
            logvar = logvar.clamp(self.min_logvar, self.max_logvar)
            out[f"{key}_{DecoderOutputKey.MEAN.value}"] = mean
            out[f"{key}_{DecoderOutputKey.LOGVAR.value}"] = logvar

        for key in self._direct_keys:
            D = self._action_dims[key]
            out[key] = self.direct_projs[key](action_hidden).view(B, T, K, D)

        return out

    def _sample_from_gmm(
        self, gmm_params: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Sample one action from GMM params.

        Args:
            gmm_params: Output of _compute_gmm_params with T=1 shapes.

        Returns:
            Per-key action tensors of shape (B, 1, D).
        """
        routing_weights = gmm_params[
            DecoderOutputKey.ROUTING_WEIGHTS.value
        ]  # (B, 1, K)
        B, T, K = routing_weights.shape
        device = routing_weights.device

        if self.inference_sampling_mode == MixtureSamplingMode.DETERMINISTIC.value:
            component_idx = routing_weights.argmax(dim=-1)  # (B, T)
        else:
            flat_weights = routing_weights.reshape(B * T, K)
            component_idx = torch.multinomial(flat_weights, num_samples=1).squeeze(-1)
            component_idx = component_idx.view(B, T)  # (B, T)

        batch_idx = torch.arange(B, device=device).unsqueeze(1)  # (B, 1)
        time_idx = torch.arange(T, device=device).unsqueeze(0)  # (1, T)

        result: dict[str, torch.Tensor] = {}

        for key in self._gaussian_keys:
            mean = gmm_params[f"{key}_{DecoderOutputKey.MEAN.value}"]  # (B, T, K, D)
            selected_mean = mean[batch_idx, time_idx, component_idx, :]  # (B, T, D)
            if (
                self.inference_sampling_mode
                == MixtureSamplingMode.STOCHASTIC_SAMPLE.value
            ):
                logvar = gmm_params[f"{key}_{DecoderOutputKey.LOGVAR.value}"]
                selected_logvar = logvar[batch_idx, time_idx, component_idx, :]
                std = torch.exp(0.5 * selected_logvar)
                result[key] = selected_mean + std * torch.randn_like(selected_mean)
            else:
                result[key] = selected_mean

        for key in self._direct_keys:
            stacked = gmm_params[key]  # (B, T, K, D)
            result[key] = stacked[batch_idx, time_idx, component_idx, :]  # (B, T, D)

        return result

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: Encoded observation features from the pipeline.
            actions: Ground-truth actions for teacher-forcing (training) or None (inference).

        Returns:
            Training: per-key GMM parameters + routing weights.
            Inference: sampled per-key actions (B, T, D).
        """
        decoder_features = self._select_input_features(features)
        obs_tokens, pos_enc, obs_padding_mask = self.input_sequence_builder(
            decoder_features
        )
        if pos_enc is not None:
            obs_tokens = obs_tokens + pos_enc

        if actions is not None:
            return self._forward_training(obs_tokens, obs_padding_mask, actions)
        return self._forward_inference(obs_tokens, obs_padding_mask)

    def _forward_training(
        self,
        obs_tokens: torch.Tensor,
        obs_padding_mask: torch.Tensor | None,
        actions: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Teacher-forcing forward pass.

        Input action tokens are shifted by one: [BOS, a_0, ..., a_{T-2}] predicts
        GMM params for [a_0, a_1, ..., a_{T-1}].

        Args:
            obs_tokens: (B, N_obs, D)
            obs_padding_mask: (B, N_obs) or None
            actions: Per-key ground-truth tensors (B, T, D_k)

        Returns:
            GMM parameter dict with per-timestep routing_weights (B, T, K).
        """
        B = obs_tokens.shape[0]

        raw_actions = self._concat_actions(actions)  # (B, T, total_action_dim)
        action_embeds = self.action_input_proj(raw_actions)  # (B, T, D)

        bos = self.action_bos_embedding.to(dtype=action_embeds.dtype).expand(B, 1, -1)
        # Shifted input: [BOS, a_0_embed, ..., a_{T-2}_embed]
        action_input = torch.cat([bos, action_embeds[:, :-1, :]], dim=1)  # (B, T, D)

        full_attn_mask, _ = make_attention_mask(
            feature_tokens=obs_tokens,
            action_tokens=action_input,
            feature_token_mask=obs_padding_mask,
            causal_actions=True,
        )

        full_seq = torch.cat([obs_tokens, action_input], dim=1)  # (B, N_obs+T, D)
        hidden_states, _ = self.gpt_decoder(
            hidden_states=full_seq,
            self_attention_mask=full_attn_mask,
        )

        N_obs = obs_tokens.shape[1]
        action_hidden = hidden_states[:, N_obs:, :]  # (B, T, D)
        return self._compute_gmm_params(action_hidden)

    def _forward_inference(
        self,
        obs_tokens: torch.Tensor,
        obs_padding_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Autoregressive inference with KV caching.

        Samples prediction_horizon actions step by step, feeding each sampled
        action back as the next input token.

        Args:
            obs_tokens: (B, N_obs, D)
            obs_padding_mask: (B, N_obs) or None

        Returns:
            Per-key sampled actions of shape (B, prediction_horizon, D_k).
        """
        B = obs_tokens.shape[0]
        T = self.prediction_horizon
        device = obs_tokens.device
        dtype = obs_tokens.dtype

        gen_cache = self.gpt_decoder.create_empty_generation_cache(
            batch_size=B, device=device, dtype=dtype
        )

        # Process the observation prefix with bidirectional attention
        N_obs = obs_tokens.shape[1]
        prefix_attn_mask = torch.zeros(
            B, 1, N_obs, N_obs, dtype=torch.bool, device=device
        )
        _, gen_cache = self.gpt_decoder(
            hidden_states=obs_tokens,
            self_attention_mask=prefix_attn_mask,
            key_padding_mask=obs_padding_mask,
            generation_cache=gen_cache,
        )

        sampled: dict[str, list[torch.Tensor]] = {k: [] for k in self._action_key_order}
        next_input = self.action_bos_embedding.to(dtype=dtype).expand(B, 1, -1)

        for _ in range(T):
            hidden_out, gen_cache = self.gpt_decoder(
                hidden_states=next_input,
                generation_cache=gen_cache,
            )  # (B, 1, D)

            gmm_params = self._compute_gmm_params(hidden_out)
            action_step = self._sample_from_gmm(gmm_params)  # {key: (B, 1, D)}

            for key in self._action_key_order:
                sampled[key].append(action_step[key])

            # Project sampled action as input for the next step
            action_vec = torch.cat(
                [action_step[k].float() for k in self._action_key_order], dim=-1
            )  # (B, 1, total_action_dim)
            next_input = self.action_input_proj(action_vec)  # (B, 1, D)

        return {key: torch.cat(steps, dim=1) for key, steps in sampled.items()}
