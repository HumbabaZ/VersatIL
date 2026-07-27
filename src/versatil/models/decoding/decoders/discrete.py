"""Base decoder for tokenized action prediction."""

import torch
import torch.nn as nn

from versatil.data.constants import SampleKey
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization import ActionTokenizer, Tokenizer
from versatil.data.tokenization.action_discretizer import BinnedActionDiscretizer
from versatil.models.decoding.action_heads.base import BaseActionHead
from versatil.models.decoding.constants import ActionHeadLayout, DecoderOutputKey
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput


class DiscreteDecoder(ActionDecoder):
    """Base class for decoders trained on tokenized action targets.

    Shape notation:
        B: batch size, A: target action-token length, D: token embedding
        dimension, V: action-token vocabulary size.
    """

    requires_tokenized_actions: bool = True

    def __init__(
        self,
        decoder_input: DecoderInput,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        action_heads: dict[str, BaseActionHead],
        device: str,
        observation_horizon: int,
        prediction_horizon: int,
        temperature: float,
        learnable_temperature: bool,
        deterministic: bool,
    ) -> None:
        """Initialize common discrete-action decoder state."""
        super().__init__(
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
        )
        self.deterministic: bool = deterministic
        self.temperature: nn.Parameter = nn.Parameter(
            torch.tensor(temperature, dtype=torch.float32),
            requires_grad=learnable_temperature,
        )
        self.token_embedding: nn.Module | None = None
        self.vocab_size: int | None = None
        self.eos_token_id: int | None = None
        self.valid_generation_token_ids: torch.Tensor | None = None

    def _init_action_bos_embedding(
        self,
        embedding_dimension: int,
        initializer_range: float,
    ) -> None:
        """Create the learned BOS embedding used before action tokens."""
        self.action_bos_embedding = nn.Parameter(torch.empty(1, 1, embedding_dimension))
        nn.init.normal_(
            self.action_bos_embedding,
            mean=0.0,
            std=initializer_range,
        )

    def _action_token_initializer_range(self) -> float:
        """Return the normal initializer std used for token embeddings."""
        raise NotImplementedError

    def _action_token_embedding_dimension(self) -> int:
        """Return the embedding dimension consumed by token embeddings."""
        raise NotImplementedError(
            f"{type(self).__name__} must define the action-token embedding dimension."
        )

    def set_tokenizer(self, tokenizer: Tokenizer | None = None) -> None:
        """Set tokenizer and bind a vocabulary action head when configured."""
        action_tokenizer = self._require_action_tokenizer(tokenizer=tokenizer)
        self.tokenizer = action_tokenizer
        if self.action_head_layout == ActionHeadLayout.VOCABULARY:
            self._bind_vocabulary_action_tokenizer(action_tokenizer=action_tokenizer)
            self.eos_token_id = int(action_tokenizer.eos_token_id)
            self.valid_generation_token_ids = self._build_valid_generation_token_ids(
                action_tokenizer=action_tokenizer,
                tokenizer_vocab_size=int(action_tokenizer.vocab_size),
                eos_token_id=self.eos_token_id,
            )

    def _require_action_tokenizer(
        self,
        tokenizer: Tokenizer | None,
    ) -> ActionTokenizer:
        """Return the action tokenizer required by discrete decoders."""
        if tokenizer is None or tokenizer.action_tokenizer is None:
            raise ValueError(
                f"{type(self).__name__} requires a tokenizer for tokenized action prediction."
            )
        return tokenizer.action_tokenizer

    def _bind_vocabulary_action_tokenizer(
        self,
        action_tokenizer: ActionTokenizer,
    ) -> None:
        """Tie the local vocabulary head to newly created token embeddings."""
        device = self.temperature.device
        self.vocab_size = action_tokenizer.vocab_size
        embedding_dimension = self._action_token_embedding_dimension()
        output_block_in_features = self.action_heads[
            DecoderOutputKey.ACTION_LOGITS.value
        ].output_proj.in_features
        initializer_range = self._action_token_initializer_range()

        if output_block_in_features != embedding_dimension:
            token_input_embedding = nn.Embedding(
                self.vocab_size,
                output_block_in_features,
            ).to(device)
            token_projection = nn.Linear(
                output_block_in_features,
                embedding_dimension,
            ).to(device)
            self.token_embedding = nn.Sequential(
                token_input_embedding,
                token_projection,
            ).to(device)
            nn.init.normal_(
                token_projection.weight,
                mean=0.0,
                std=initializer_range,
            )
        else:
            token_input_embedding = nn.Embedding(
                self.vocab_size,
                embedding_dimension,
            ).to(device)
            self.token_embedding = token_input_embedding

        nn.init.normal_(
            token_input_embedding.weight,
            mean=0.0,
            std=initializer_range,
        )
        lm_head = nn.Linear(
            output_block_in_features,
            self.vocab_size,
            bias=False,
            device=device,
        )
        lm_head.weight = token_input_embedding.weight
        self.action_heads[
            DecoderOutputKey.ACTION_LOGITS.value
        ].output_dim = self.vocab_size
        self.action_heads[DecoderOutputKey.ACTION_LOGITS.value].output_proj = lm_head

    def _validate_action_tokenizer_is_set(self) -> None:
        """Ensure tokenizer-dependent action-token modules are initialized."""
        if (
            self.token_embedding is None
            or self.tokenizer is None
            or self.vocab_size is None
        ):
            raise ValueError(
                f"{type(self).__name__} requires set_tokenizer() to be called before forward."
            )

    @staticmethod
    def _uses_fixed_length_action_generation(
        action_tokenizer: ActionTokenizer,
    ) -> bool:
        """Return whether inference should generate a known action-token count."""
        return isinstance(
            action_tokenizer.action_discretizer, BinnedActionDiscretizer
        )

    def _get_action_payload_token_count(self) -> int | None:
        """Return the required action-token count before any optional EOS.

        Fixed-length discretizers (binning) map each chunk to exactly
        ``time_horizon * action_dim`` tokens; variable-length discretizers
        (FAST) return None.
        """
        if self.tokenizer is None:
            return None
        if not self._uses_fixed_length_action_generation(
            action_tokenizer=self.tokenizer
        ):
            return None
        time_horizon = self.tokenizer.action_discretizer.time_horizon
        action_dim = self.tokenizer.action_discretizer.action_dim
        if time_horizon is None or action_dim is None:
            raise ValueError(
                f"{type(self).__name__} fixed-length generation requires the "
                "action discretizer to know time_horizon and action_dim."
            )
        return int(time_horizon * action_dim)

    def _build_valid_generation_token_ids(
        self,
        action_tokenizer: ActionTokenizer,
        tokenizer_vocab_size: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        """Return action-token IDs that inference is allowed to sample.

        Fixed-length generation (binned discretizers) excludes EOS: the
        detokenizer requires exactly ``time_horizon * action_dim`` payload
        tokens, so an early EOS sample would truncate the sequence and fail
        decoding. The generation loop is already capped at the payload count.
        Variable-length generation keeps EOS so the sequence can terminate.
        """
        token_count = action_tokenizer.action_discretizer.token_count
        local_token_ids = list(range(int(token_count)))
        action_token_ids = action_tokenizer.token_id_mapping.encode(local_token_ids)
        valid_token_ids = torch.as_tensor(
            action_token_ids,
            dtype=torch.long,
            device=self.device,
        )
        if not self._uses_fixed_length_action_generation(
            action_tokenizer=action_tokenizer
        ):
            eos_token = torch.tensor(
                [eos_token_id], dtype=torch.long, device=self.device
            )
            valid_token_ids = torch.cat([valid_token_ids, eos_token], dim=0)
        if (
            valid_token_ids.min().item() < 0
            or valid_token_ids.max().item() >= tokenizer_vocab_size
        ):
            raise ValueError(
                f"{type(self).__name__} valid action-token IDs must lie inside "
                f"the action tokenizer vocabulary [0, {tokenizer_vocab_size})."
            )
        return valid_token_ids

    def _get_max_generation_steps(self, available_context_steps: int) -> int:
        """Return the action-token generation cap within context capacity.

        Fixed-length discretizers generate exactly their payload token count so
        the deterministic chunk length is reproduced; variable-length
        discretizers generate up to ``max_token_len`` and stop at EOS.
        """
        if self.tokenizer is None:
            raise ValueError(
                f"{type(self).__name__} requires set_tokenizer() to be called before inference."
            )
        if available_context_steps < 1:
            raise ValueError(
                f"{type(self).__name__} has no context capacity left for action-token generation."
            )
        action_payload_token_count = self._get_action_payload_token_count()
        if action_payload_token_count is None:
            return min(int(self.tokenizer.max_token_len), available_context_steps)
        if available_context_steps < action_payload_token_count:
            raise ValueError(
                f"{type(self).__name__} needs {action_payload_token_count} context "
                "slots for fixed-length action-token generation, but only "
                f"{available_context_steps} are available."
            )
        return action_payload_token_count

    def _get_target_token_ids(
        self,
        actions: dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """Read teacher-forcing target token IDs from the action dictionary."""
        if SampleKey.TOKENIZED_ACTIONS.value not in actions:
            raise ValueError(
                f"{type(self).__name__} training requires "
                f"'{SampleKey.TOKENIZED_ACTIONS.value}' in actions."
            )
        target_token_ids = actions[SampleKey.TOKENIZED_ACTIONS.value]
        if target_token_ids.ndim != 2:
            raise ValueError(
                f"'{SampleKey.TOKENIZED_ACTIONS.value}' must have shape "
                f"(B, token_length), got {target_token_ids.shape}."
            )
        if target_token_ids.shape[0] != batch_size:
            raise ValueError(
                f"'{SampleKey.TOKENIZED_ACTIONS.value}' batch size must match "
                f"feature batch size {batch_size}, got {target_token_ids.shape[0]}."
            )
        return target_token_ids

    def _expand_action_bos_embedding(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Expand the learned BOS embedding to the current batch."""
        return self.action_bos_embedding.to(device=device, dtype=dtype).expand(
            batch_size,
            -1,
            -1,
        )

    def _sample_next_action_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample the next action token from the valid action-token subset.

        Restricting to ``valid_generation_token_ids`` keeps the model inside the
        action-token vocabulary; for fixed-length discretizers this subset also
        excludes EOS so generation cannot terminate before the payload count.

        Args:
            logits: Token logits of shape ``(B, 1, V)``.

        Returns:
            Token ids of shape ``(B, 1)`` for both sampling modes.
        """
        if self.valid_generation_token_ids is None:
            raise ValueError(
                f"{type(self).__name__} valid action-token IDs are not initialized."
            )
        valid_token_ids = self.valid_generation_token_ids.to(device=logits.device)
        valid_logits = logits.index_select(dim=-1, index=valid_token_ids)
        if self.deterministic:
            selected_indices = torch.argmax(valid_logits, dim=-1)  # (B, 1)
        else:
            scaled_logits = valid_logits / self.temperature.clamp(min=0.01)
            probabilities = torch.softmax(scaled_logits, dim=-1)
            selected_indices = torch.multinomial(
                probabilities.squeeze(1),
                num_samples=1,
            )
        return valid_token_ids[selected_indices]
