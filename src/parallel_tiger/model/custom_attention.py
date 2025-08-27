from transformers.models.t5.modeling_t5 import *
from typing import Optional
import logging
logger = logging.getLogger(__name__)


# TODO: Describe the different types of bias in function of the attention module (encoder, decoder self-attention and decoder cross-attention)


def generate_item_position_ids(total_length, n, device=None):
    """
    :param total_length: seq_length
    :param n: n_query within each item
    :return: PyTorch tensor
    """
    vector = torch.arange(total_length, dtype=torch.long, device=device)
    # output vector, let the position id be the same within the items
    pos_vector = torch.div(vector, n, rounding_mode="floor")  # (total_length,)
    return pos_vector


class BiasComputer:
    """Helper class to compute different types of relative position biases."""
    
    def __init__(self, attention_module):
        self.attention = attention_module
        self.n_query = attention_module.n_query
        self.n_heads = attention_module.n_heads
    
    def _get_device(self) -> torch.device:
        """Get device from attention module parameters."""
        if hasattr(self.attention, "relative_item_bias"):
            return self.attention.relative_item_bias.weight.device
        elif hasattr(self.attention, "codebook_relative_bias_table"):
            return self.attention.codebook_relative_bias_table.device
        else:
            raise ValueError(
                "Cannot determine device: neither relative_item_bias nor "
                "codebook_relative_bias_table are defined in the attention module."
            )
    
    def _generate_codebook_positions(
        self, length: int, device: torch.device
    ) -> torch.Tensor:
        """Generate codebook-level position IDs."""
        return torch.arange(self.n_query, device=device).repeat(
            length // self.n_query + 1
        )[:length]
    
    def _compute_codebook_relative_bias(
        self, query_length: int, key_length: int, device: torch.device
    ) -> torch.Tensor:
        """Compute codebook relative position bias (shared across all attention types)."""
        context_cb_pos = self._generate_codebook_positions(query_length, device)[:, None]
        memory_cb_pos = self._generate_codebook_positions(key_length, device)[None, :]
        
        relative_cb_pos = memory_cb_pos - context_cb_pos  # (query_length, key_length)
        
        # Ensure indices are in valid range [0, 2 * (n_query - 1)]
        cb_pos_idx = (relative_cb_pos + self.n_query - 1).clamp(
            0, 2 * self.n_query - 2  # Fixed: was 2 * (self.n_query - 1)
        )
        
        return self.attention.codebook_relative_bias_table[cb_pos_idx]
    
    def _compute_relative_position_bucket(
        self, relative_position: torch.Tensor
    ) -> torch.Tensor:
        """Compute relative position buckets using the attention module's bucketing function."""
        return self.attention._relative_position_bucket(
            relative_position,
            bidirectional=(not self.attention.is_decoder),
            num_buckets=self.attention.relative_attention_num_buckets,
            max_distance=self.attention.relative_attention_max_distance,
        )
    
    def compute_encoder_self_attention_bias(
        self, query_length: int, key_length: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Compute bias for encoder self-attention."""
        if device is None:
            device = self._get_device()
        
        bias = torch.zeros(query_length, key_length, self.n_heads, device=device)
        
        # Item relative position bias
        if self.attention.has_relative_item_bias:
            context_position = generate_item_position_ids(
                query_length, n=self.n_query, device=device
            )[:, None]
            memory_position = generate_item_position_ids(
                key_length, n=self.n_query, device=device
            )[None, :]
            relative_position = memory_position - context_position
            
            relative_position_bucket = self._compute_relative_position_bucket(relative_position)
            bias += self.attention.relative_item_bias(relative_position_bucket)
        
        # Codebook relative position bias
        if self.attention.has_relative_codebook_bias:
            bias += self._compute_codebook_relative_bias(query_length, key_length, device)
        
        return bias.permute([2, 0, 1]).unsqueeze(0)  # (1, n_heads, query_length, key_length)
    
    def compute_decoder_self_attention_bias(
        self, query_length: int, key_length: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Compute bias for decoder self-attention."""
        if device is None:
            device = self._get_device()
        
        bias = torch.zeros(query_length, key_length, self.n_heads, device=device)
        
        # Item relative position bias (all zeros for same item)
        if self.attention.has_relative_item_bias:
            relative_position = torch.zeros(
                query_length, key_length, device=device, dtype=torch.long
            )
            relative_position_bucket = self._compute_relative_position_bucket(relative_position)
            bias += self.attention.relative_item_bias(relative_position_bucket)
        
        # Codebook relative position bias
        if self.attention.has_relative_codebook_bias:
            bias += self._compute_codebook_relative_bias(query_length, key_length, device)
        
        return bias.permute([2, 0, 1]).unsqueeze(0)
    
    def compute_decoder_cross_attention_bias(
        self, query_length: int, key_length: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Compute bias for decoder cross-attention."""
        if device is None:
            device = self._get_device()
        
        bias = torch.zeros(query_length, key_length, self.n_heads, device=device)
        
        # Item relative position bias
        if self.attention.has_relative_item_bias:  # Fixed: was has_relative_attention_bias
            encoder_item_position = generate_item_position_ids(
                key_length, n=self.n_query, device=device
            )
            relative_item_position = encoder_item_position - key_length // self.n_query
            relative_position = relative_item_position.unsqueeze(0).expand(query_length, -1)
            
            relative_position_bucket = self._compute_relative_position_bucket(relative_position)
            bias += self.attention.relative_item_bias(relative_position_bucket)
        
        # Codebook relative position bias
        if self.attention.has_relative_codebook_bias:
            bias += self._compute_codebook_relative_bias(query_length, key_length, device)
        
        return bias.permute([2, 0, 1]).unsqueeze(0)



class CustomT5Attention(T5Attention):
    """
    Custom T5Attention module with item-level and codebook-level relative position bias.

    We initialize the T5Attention module with `has_relative_attention_bias=False` to avoid creating
    the default T5 relative attention bias table, since we replace it with our own bias
    mechanism. Instead, we optionally create relative item bias and/or relative codebook
    bias embedding tables when required.

    Because of this customization, we override T5Attention's `forward()` method. The only
    functional change is in the section where the `position_bias` is initialized:
    in the original T5, the check is:
        `if not self.has_relative_attention_bias:`
    Here, we replace it with:
        `if not self.has_relative_item_bias and not self.has_relative_codebook_bias:`

    Original T5 logic for context:
        ```
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device
                )
        ```

    """
    def __init__(
        self,
        config: T5Config,
        has_relative_item_bias=False,
        has_relative_codebook_bias=False,
    ):
        super().__init__(
            config, has_relative_attention_bias=False
        )  # no bias as we will define it ourselves
        self.has_relative_item_bias = has_relative_item_bias
        self.has_relative_codebook_bias = has_relative_codebook_bias
        self.pruned_heads = set()
        self.gradient_checkpointing = False
        self.has_relative_codebook_bias = has_relative_codebook_bias

        if has_relative_item_bias:
            self.relative_item_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )

        if has_relative_codebook_bias:
            # Custom codebook relative position bias table
            self.codebook_relative_bias_table = nn.Parameter(
                torch.empty(2 * config.n_query - 1, self.n_heads)
            )

        # Initialize bias computer
        self.n_query = config.n_query
        self.bias_computer = BiasComputer(self)


    def compute_bias(self, query_length, key_length, device=None):
        raise NotImplementedError(
            "CustomT5Attention.compute_bias() should be overwritten in subclasses to compute the position bias."
        )

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        key_length = (
            real_seq_length if key_value_states is None else key_value_states.shape[1]
        )

        def shape(states):
            """projection"""
            return states.view(
                batch_size, -1, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return (
                states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
            )

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_item_bias and not self.has_relative_codebook_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device
                )

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = (
                    position_bias + mask
                )  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(
            torch.matmul(attn_weights, value_states)
        )  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (
            (key_states, value_states) if (self.is_decoder and use_cache) else None
        )
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


### ENCODER (SELF-) ATTENTION ###
class CustomEncoderSelfAttention(CustomT5Attention):
    """Overridden Encoder Self-Attention with custom position bias."""
    def __init__(
        self,
        config: T5Config,
        has_relative_item_bias=False,
        has_relative_codebook_bias=False,
    ):
        super().__init__(
            config,
            has_relative_item_bias=has_relative_item_bias,
            has_relative_codebook_bias=has_relative_codebook_bias,
        )
        # self.n_query = config.n_query

    def compute_bias(self, query_length, key_length, device=None):
        return self.bias_computer.compute_encoder_self_attention_bias(
            query_length, key_length, device
        )


class CustomEncoderT5LayerSelfAttention(T5LayerSelfAttention):
    """Overridden Encoder T5 Layer Self-Attention with custom position bias."""
    def __init__(
        self, config, has_relative_item_bias=False, has_relative_codebook_bias=False
    ):
        super().__init__(config, has_relative_item_bias)
        self.SelfAttention = CustomEncoderSelfAttention(
            config,
            has_relative_item_bias=has_relative_item_bias,
            has_relative_codebook_bias=has_relative_codebook_bias,
        )
        self.SelfAttention.n_query = config.n_query


class CustomEncoderT5Block(T5Block):
    """Overridden Encoder T5 Block with custom position bias."""
    def __init__(
        self, config, has_relative_item_bias=False, has_relative_codebook_bias=False
    ):
        super().__init__(
            config, has_relative_attention_bias=False
        )  # no bias as we will define it ourselves
        self.layer[0] = CustomEncoderT5LayerSelfAttention(
            config,
            has_relative_item_bias=has_relative_item_bias,
            has_relative_codebook_bias=has_relative_codebook_bias,
        )


class CustomBiasT5Stack(T5Stack):
    """Overridden T5 Stack with custom position bias."""
    def __init__(
        self,
        config,
        embed_tokens=None,
        has_relative_item_bias=True,
        has_relative_codebook_bias=False,
    ):
        super().__init__(config, embed_tokens)

        self.block = nn.ModuleList(
            [
                CustomEncoderT5Block(
                    config,
                    has_relative_item_bias=bool(i == 0) and has_relative_item_bias,
                    has_relative_codebook_bias=bool(i == 0)
                    and has_relative_codebook_bias,
                )
                for i in range(config.num_layers)
            ]
        )


### DECODER SELF-ATTENTION ###
class CustomDecoderSelfAttention(CustomT5Attention):
    """Overridden Decoder Self-Attention with custom position bias."""
    def __init__(
        self,
        config: T5Config,
        has_relative_item_bias=False,
        has_relative_codebook_bias=False,
    ):
        super().__init__(
            config,
            has_relative_item_bias=has_relative_item_bias,
            has_relative_codebook_bias=has_relative_codebook_bias,
        )
        # self.n_query = config.n_query

    def compute_bias(self, query_length, key_length, device=None):
        return self.bias_computer.compute_decoder_self_attention_bias(
            query_length, key_length, device
        )


class CustomDecoderT5LayerSelfAttention(T5LayerSelfAttention):
    """Overridden Decoder T5 Layer Self-Attention with custom position bias."""
    def __init__(
        self, config, has_relative_item_bias=False, has_relative_codebook_bias=False
    ):
        super().__init__(
            config, has_relative_attention_bias=False
        )  # no bias as we will define it ourselves
        self.SelfAttention = CustomDecoderSelfAttention(
            config,
            has_relative_item_bias=has_relative_item_bias,
            has_relative_codebook_bias=has_relative_codebook_bias,
        )
        self.SelfAttention.n_query = config.n_query


### DECODER CROSS-ATTENTION
class CustomDecoderCrossAttention(CustomT5Attention):
    """Overridden Decoder Cross-Attention with custom position bias."""
    def __init__(
        self, config, has_relative_item_bias=False, has_relative_codebook_bias=False
    ):
        super().__init__(
            config,
            has_relative_item_bias=has_relative_item_bias,
            has_relative_codebook_bias=has_relative_codebook_bias,
        )
        # self.n_query = config.n_query

    def compute_bias(self, query_length, key_length, device=None):
        return self.bias_computer.compute_decoder_cross_attention_bias(
            query_length, key_length, device
        )


class CustomDecoderT5LayerCrossAttention(T5LayerCrossAttention):
    """Overridden Decoder T5 Layer Cross-Attention with custom position bias."""
    def __init__(
        self, config, has_relative_item_bias=False, has_relative_codebook_bias=False
    ):
        super().__init__(config)
        self.EncDecAttention = CustomDecoderCrossAttention(
            config,
            has_relative_item_bias=has_relative_item_bias,
            has_relative_codebook_bias=has_relative_codebook_bias,
        )
        self.EncDecAttention.n_query = config.n_query


class QDecoderT5Block(T5Block):
    """Overridden Decoder T5 Block with custom position bias."""
    def __init__(
        self,
        config,
        has_relative_item_bias_sa=False,
        has_relative_codebook_bias_sa=False,
        has_relative_item_bias_ca=False,
        has_relative_codebook_bias_ca=False,
    ):
        super().__init__(
            config, has_relative_attention_bias=False
        )  # no bias as we will define it ourselves

        self.layer[0] = CustomDecoderT5LayerSelfAttention(
            config,
            has_relative_item_bias=has_relative_item_bias_sa,
            has_relative_codebook_bias=has_relative_codebook_bias_sa,
        )
        self.layer[1] = CustomDecoderT5LayerCrossAttention(
            config,
            has_relative_item_bias=has_relative_item_bias_ca,
            has_relative_codebook_bias=has_relative_codebook_bias_ca,
        )