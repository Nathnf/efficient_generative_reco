import torch
import torch.nn as nn
import copy
from transformers import T5ForConditionalGeneration

from transformers.models.t5.modeling_t5 import (
    T5Stack,
    T5LayerNorm,
    BaseModelOutputWithPastAndCrossAttentions,
    add_start_docstrings_to_model_forward,
    T5_INPUTS_DOCSTRING,
    replace_return_docstrings,
    Seq2SeqLMOutput,
    warnings,
    __HEAD_MASK_WARNING_MSG,
    BaseModelOutput,
    CrossEntropyLoss
)
from typing import Optional, Tuple, Union

from parallel_tiger.model.custom_attention import QDecoderT5Block, CustomBiasT5Stack

import logging
logger = logging.getLogger(__name__)



"""
SETRec-inspired T5 modifications for query-based decoding and custom attention.

This code is adapted from SETRec (https://arxiv.org/pdf/2502.10833), which overwrites
`T5ForConditionalGeneration` and `T5Stack` to support a query-based decoder mechanism.

Modifications in this version:

1. **Attention mask**: We use a full attention mask instead of the original sparse mask.
   (Optionally, this could be parameterized to allow both behaviors.)

2. **Custom attention modules**: All standard T5 attention modules are replaced with
   our custom modules from `custom_attention.py`, which implement item-level and
   codebook-level relative position biases.

Notes:

- The SETRec implementation appears to have copied the `forward()` method from
  `T5ForConditionalGeneration` without changes. We retain their forward method here
  for safety, although it seems functionally identical to the original T5.
"""


class Qdecoder(T5Stack):
    def __init__(
        self,
        config,
        embed_tokens,
        num_query=4,
        has_relative_item_bias_sa=True,
        has_relative_codebook_bias_sa=True,
        has_relative_item_bias_ca=True,
        has_relative_codebook_bias_ca=True,
    ):
        super().__init__(config, embed_tokens)

        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [
                QDecoderT5Block(
                    config,
                    has_relative_item_bias_sa=bool(i == 0)
                    and has_relative_item_bias_sa,
                    has_relative_codebook_bias_sa=bool(i == 0)
                    and has_relative_codebook_bias_sa,
                    has_relative_item_bias_ca=bool(i == 0)
                    and has_relative_item_bias_ca,
                    has_relative_codebook_bias_ca=bool(i == 0)
                    and has_relative_codebook_bias_ca,
                )
                for i in range(config.num_layers)
            ]
        )

        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # add query vector for each token position
        self.n_query = num_query
        self.d_query = self.config.hidden_size
        self.query_emb = nn.Embedding(self.n_query, self.d_query)
        self.query_input_ids = torch.LongTensor([i for i in range(self.n_query)])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # logger.debug(f"Qdecoder.forward: input_ids={input_ids}, attention_mask={attention_mask}, encoder_hidden_states={encoder_hidden_states}, encoder_attention_mask={encoder_attention_mask}, inputs_embeds={inputs_embeds}")
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.query_emb = self.query_emb.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (
            past_key_values[0][0].shape[2] + seq_length
            if past_key_values is not None
            else seq_length
        )

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(
                    f"`use_cache` can only be set to `True` if {self} is used as a decoder"
                )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, mask_seq_length, device=inputs_embeds.device
            )

        # Use a full attention mask for Qdecoder (instead of sparse like in SETRec)
        attention_mask = torch.ones(
            batch_size, mask_seq_length, device=inputs_embeds.device
        )
        extended_attention_mask = attention_mask[:, None, None, :].to(
            dtype=self.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
            self.dtype
        ).min

        # logger.debug("Self Attention Mask:")
        # logger.debug(f"Qdecoder.forward: extended_attention_mask shape={extended_attention_mask.shape}, extended_attention_mask={extended_attention_mask}")
        # # logger.debug(f"Qdecoder.forward: extended_attention_mask={extended_attention_mask}, extended_attention_mask shape={extended_attention_mask.shape}")

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # logger.debug("Cross Attention Mask:")
        # logger.debug(f"Qdecoder.forward: encoder_attention_mask shape={encoder_attention_mask.shape}, encoder_attention_mask={encoder_attention_mask}")
        # logger.debug(f"Qdecoder.forward: encoder_extended_attention_mask shape={encoder_extended_attention_mask.shape}, encoder_extended_attention_mask={encoder_extended_attention_mask}")

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(
            cross_attn_head_mask, self.config.num_layers
        )
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(
            zip(self.block, past_key_values)
        ):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(
                        hidden_states.device
                    )
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = (
                        encoder_extended_attention_mask.to(hidden_states.device)
                    )
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(
                        hidden_states.device
                    )
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(
                        hidden_states.device
                    )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    4 if output_attentions else 3
                ]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (
                    present_key_value_state,
                )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=bias),
            nn.ReLU(),                         # or GELU
            nn.Linear(hidden_dim, out_dim, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class QT5(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        # Don't ignore lm_head.weight - we need it for loading, then replace
        r"lm_heads\.\d+\.weight",  # These will be created during replacement
        # r"decoder\.query_emb\.weight",
        r"encoder\.block\.0\.layer\.0\.SelfAttention\.relative_attention_bias",
        r"decoder\.block\.0\.layer\.0\.SelfAttention\.relative_attention_bias",
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
        # Don't ignore lm_head.weight - we expect it during loading
    ]

    def __init__(
            self, 
            config, 
            n_query, 
            code_num=256,
            is_inference=False,
            is_aggregate_tokens=False,
            use_multi_head=False,
            has_relative_encoder_item_bias=True,
            has_relative_encoder_codebook_bias=True,
            has_relative_decoder_item_bias_sa=True,
            has_relative_decoder_codebook_bias_sa=False,
            has_relative_decoder_item_bias_ca=False,
            has_relative_decoder_codebook_bias_ca=False
        ):
        super(T5ForConditionalGeneration, self).__init__(config)

        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # encoder should be the same as T5
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        encoder_config.n_query = n_query

        if is_aggregate_tokens: 
            self.encoder = T5Stack(encoder_config, self.shared)
        else:
            self.encoder = CustomBiasT5Stack(
                encoder_config,
                self.shared,
                has_relative_item_bias=has_relative_encoder_item_bias,
                has_relative_codebook_bias=has_relative_encoder_codebook_bias,
            )

        # decoder would be different
        self.n_query = n_query
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        decoder_config.n_query = n_query

        self.decoder = Qdecoder(
            decoder_config,
            self.shared,
            self.n_query,
            has_relative_item_bias_sa=has_relative_decoder_item_bias_sa,
            has_relative_codebook_bias_sa=has_relative_decoder_codebook_bias_sa,
            has_relative_item_bias_ca=has_relative_decoder_item_bias_ca,
            has_relative_codebook_bias_ca=has_relative_decoder_codebook_bias_ca,
        )

        # Store parameters for potential head replacement
        self.n_query = n_query
        self.code_num = code_num
        
        # Create standard lm_head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Custom multi projection heads
        if use_multi_head and is_inference:
            self._create_multi_heads()

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        logger.debug("self.config.tie_word_embeddings: %s", self.config.tie_word_embeddings)
        self.config.tie_word_embeddings = False
        logger.debug("self.config.tie_word_embeddings (after): %s", self.config.tie_word_embeddings) 

    def _create_multi_heads(self):
        # self.lm_heads = nn.ModuleList([
        #     nn.Linear(self.config.d_model, self.code_num, bias=False)
        #     for _ in range(self.n_query)
        # ])
        self.lm_heads = nn.ModuleList([
            ProjectionMLP(
                in_dim=self.config.d_model,
                hidden_dim=self.config.d_model,   # or smaller, e.g. 256
                out_dim=self.code_num,
                bias=False
            )
            for _ in range(self.n_query)
        ])

    def replace_projection_head(self):
        original_head = self.lm_head

        self._create_multi_heads()

        with torch.no_grad():
            std = original_head.weight.data.std().item()
            for head in self.lm_heads:
                for layer in head.net:
                    if isinstance(layer, nn.Linear):
                        layer.weight.data.normal_(mean=0.0, std=std)
                        if layer.bias is not None:
                            layer.bias.zero_()

        del self.lm_head
        logger.debug(f"Replaced single lm_head with {self.n_query} heads of size {self.code_num}")

    def get_output_embeddings(self):
        """
        Returns the output embedding modules (multiple heads in our case).
        For compatibility, return the ModuleList containing all heads.
        """
        if hasattr(self, 'lm_heads'):
            return None
        return super().get_output_embeddings()


    _CONFIG_FOR_DOC = "T5Config"

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            logger.warning("[WARNING - SHOULDN'T BE HERE?] - Shifting labels to create decoder input IDs")
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0] # (batch_size, seq_length=n_query, d_model)
        # logger.debug("sequence_output shape: %s", sequence_output.shape)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            if hasattr(self, 'lm_heads'):
                self.lm_heads = self.lm_heads.to(self.encoder.first_device)
                device = next(self.lm_heads[0].parameters()).device
                sequence_output = sequence_output.to(device)
            else:
                self.lm_head = self.lm_head.to(self.encoder.first_device)
                sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        # Compute logits based on whether we have multi-heads or single head
        if hasattr(self, 'lm_heads'):
            # Multi-head case (after replacement)
            lm_logits = torch.stack([
                self.lm_heads[i](sequence_output[:, i, :]) 
                for i in range(self.n_query)
            ], dim=1)  # Shape: (batch_size, n_query, code_num)
        else:
            # Single head case (default)
            lm_logits = self.lm_head(sequence_output)  # (batch_size, seq_length, vocab_size)
        # logger.debug("lm_logits shape: %s", lm_logits.shape)

        loss = None
        if labels is not None:
            logger.debug("SHOULDN'T BE HERE - computing QT5 loss...")
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )