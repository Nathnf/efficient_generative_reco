import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import T5Config
from einops import reduce

from parallel_tiger.model.config import EncoderAggregation, TrainingMode, TrainingConfig, ModelConfig # type: ignore[reportMissingImports]
from parallel_tiger.model.Q_t5 import QT5                                                             # type: ignore[reportMissingImports]
from parallel_tiger.generation.trie import Trie                                                       # type: ignore[reportMissingImports]

import time as time
from typing import Union, Dict, Tuple, Optional, cast

import logging

logger = logging.getLogger(__name__)


class WeightInitializer:
    """Handles weight initialization for query embeddings and selected positional encodings."""

    def __init__(self, model: QT5, config: ModelConfig):
        self.model = model
        self.config = config

    def initialize_all_weights(self) -> None:
        """Initialize all model weights."""
        logger.info("Initializing weights with normal distribution...")

        self._initialize_query_embeddings()
        if not self.config.is_aggregate_tokens:
            self._initialize_encoder_biases()
        self._initialize_decoder_biases()

    def _initialize_query_embeddings(self) -> None:
        """Initialize query embeddings."""
        nn.init.normal_(self.model.decoder.query_emb.weight.data, mean=0, std=0.02)

    def _initialize_encoder_biases(self) -> None:
        """Initialize encoder attention biases."""
        bias_config = self.config.bias_config
        assert (
            bias_config is not None
        ), "Bias configuration must be provided."  # to pass pylance checks
        encoder_attention = self.model.encoder.block[0].layer[0].SelfAttention

        if bias_config.has_relative_encoder_item_bias:
            nn.init.normal_(
                encoder_attention.relative_item_bias.weight, mean=0, std=0.02
            )

        if bias_config.has_relative_encoder_codebook_bias:
            nn.init.normal_(
                encoder_attention.codebook_relative_bias_table, mean=0, std=0.02
            )

    def _initialize_decoder_biases(self) -> None:
        """Initialize decoder attention biases."""
        bias_config = self.config.bias_config
        assert (
            bias_config is not None
        ), "Bias configuration must be provided."  # to pass pylance checks
        decoder_self_attention = self.model.decoder.block[0].layer[0].SelfAttention
        decoder_cross_attention = self.model.decoder.block[0].layer[1].EncDecAttention

        # Self-attention biases
        if bias_config.has_relative_decoder_item_bias_sa:
            nn.init.normal_(
                decoder_self_attention.relative_item_bias.weight, mean=0, std=0.02
            )

        if bias_config.has_relative_decoder_codebook_bias_sa:
            nn.init.normal_(
                decoder_self_attention.codebook_relative_bias_table, mean=0, std=0.02
            )

        # Cross-attention biases
        if bias_config.has_relative_decoder_item_bias_ca:
            nn.init.normal_(
                decoder_cross_attention.relative_item_bias.weight, mean=0, std=0.02
            )

        if bias_config.has_relative_decoder_codebook_bias_ca:
            nn.init.normal_(
                decoder_cross_attention.codebook_relative_bias_table, mean=0, std=0.02
            )


class ItemTokenAggregator:
    """
    Aggregates item token embeddings before the encoder.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    def _aggregate_mask_per_item(self, mask: torch.Tensor, bs: int) -> torch.Tensor:
        mask = mask.view(bs, -1, self.config.n_query)  # (bs, n_items, n_query)
        mask = mask.all(dim=-1)  # strict version (could also do the mean)
        return mask

    def aggregate(self, item_embeddings: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # item_embeddings: (bs, n_items * n_query, d_model)
        bs, _, emb_dim = item_embeddings.shape
        item_embeddings = item_embeddings.view(bs, -1, self.config.n_query, emb_dim)

        if self.config.encoder_aggregation == EncoderAggregation.SUM.value:
            item_embeddings = reduce(item_embeddings, 'b i t d -> b i d', 'sum')
            mask = self._aggregate_mask_per_item(mask, bs)

        elif self.config.encoder_aggregation == EncoderAggregation.MEAN.value:
            item_embeddings = reduce(item_embeddings, 'b i t d -> b i d', 'mean')
            mask = self._aggregate_mask_per_item(mask, bs)

        elif self.config.encoder_aggregation == EncoderAggregation.CONCAT.value:
            raise ValueError("Concatenation not supported yet.")
            # NOTE: will need to modify the model dimension OR use a projeciton layer.
            # return item_embeddings.view(item_embeddings.size(0), -1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.config.encoder_aggregation}")

        return item_embeddings, mask


class LossComputer:
    """
    Handles different loss computation strategies:
    1. Standard (parallel - predict all n_query tokens)
    1.a) Unnormalized (all codebook tokens contribute equally)
    1.b) Depthwise (codebook tokens contribute according to the weights defined in config)
    2. Masked (only certain needs to be predicted, the rest is given as input of the decoder)
    3. Autoregressive (predict one token at a time, using the previous tokens as input)
    """

    def __init__(self, config: TrainingConfig, device: torch.device, use_multi_head: bool = False):
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        self.weights_per_codebook = (
            torch.tensor(self.config.weights_per_codebook, device=device)
            if config.training_mode == TrainingMode.DEPTHWISE.value
            else None
        )
        self.use_multi_head = use_multi_head
        self.offset = torch.arange(self.config.n_query) * self.config.code_num # (n_query,)

        if config.training_mode == TrainingMode.STANDARD.value:
            self._loss_fn = self._compute_standard_loss
        elif config.training_mode == TrainingMode.DEPTHWISE.value:
            self._loss_fn = self._compute_depthwise_loss
        elif config.training_mode == TrainingMode.MASKED.value:
            if config.normalization_method == "per_sample":
                self._loss_fn = self._compute_masked_loss_per_sample
            elif config.normalization_method == "per_batch":
                self._loss_fn = self._compute_masked_loss_per_batch
            elif config.normalization_method == "unnormalized":
                self._loss_fn = self._compute_masked_loss_unnormalized
            else:
                raise ValueError(
                    "Unknown normalization method: %s", config.normalization_method
                )
        else:
            raise ValueError("Unknown training mode: %s", config.training_mode)

    def compute_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        bs: int,
        decoder_input_use_query_vectors_mask: Optional[torch.Tensor] = None,
        mask_token_no: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Compute the loss based on the configured training strategy.
        """
        if self.use_multi_head:
            # logger.debug("min gt: %s, max gt: %s", gt.min(), gt.max())
            gt = gt - self.offset.repeat(bs).to(gt.device) - self.config.special_tokenizer_tokens_num
            # logger.debug("min gt (remapped): %s, max gt (remapped): %s", gt.min(), gt.max())

        return self._loss_fn(
            pred=pred,
            gt=gt,
            bs=bs,
            decoder_input_use_query_vectors_mask=decoder_input_use_query_vectors_mask,
            mask_token_no=mask_token_no,
        )

    def _compute_standard_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        bs: int,
        decoder_input_use_query_vectors_mask: Optional[torch.Tensor] = None,
        mask_token_no: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Compute loss for standard training.
        """
        loss = self.criterion(pred, gt)  # (bs * n_query,) - loss for each token
        loss_per_codebook = loss.view(bs, self.config.n_query).mean(
            dim=0
        )  # (n_query,) - mean loss for each codebook
        # logger.debug(f"T54Rec.forward: loss_per_codebook={loss_per_codebook} (on {bs} samples)")

        return loss.mean(), loss_per_codebook

    def _compute_depthwise_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        bs: int,
        decoder_input_use_query_vectors_mask: Optional[torch.Tensor] = None,
        mask_token_no: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, None]:
        """
        Compute loss for depthwise training.
        """
        assert self.weights_per_codebook is not None
        loss = self.criterion(pred, gt)  # (bs * n_query,) - loss for each token
        loss_weights = self.weights_per_codebook.repeat(
            bs
        )  # (bs * n_query,) - weights for each token
        loss = loss * loss_weights  # (bs * n_query,) - weighted loss
        loss = loss.mean()  # Mean loss over all tokens

        return loss, None

    def _prepare_masked_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        decoder_input_use_query_vectors_mask: Optional[torch.Tensor],
        mask_token_no: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Common preprocessing for masked loss computation."""
        assert decoder_input_use_query_vectors_mask is not None
        assert mask_token_no is not None

        loss_mask = decoder_input_use_query_vectors_mask.view(-1).to(
            pred.device
        )  # (bs * n_query,) - True where the masking happens (i.e. where tokens need to be predicted i.e. where query vectors are used as decoder input)
        # logger.debug(f"loss_mask={loss_mask}")
        # logger.debug(f"loss_mask sum={loss_mask.sum()}")
        pred_masked = pred[loss_mask]  # (num_valid_tokens, vocab_size)
        gt_masked = gt[loss_mask]  # (num_valid_tokens,)
        # logger.debug("T54Rec.forward (mask): pred shape=%s, gt shape=%s", pred.shape, gt.shape)
        # logger.debug("T54Rec.forward (mask): pred=%s, gt=%s", pred, gt)
        # logger.debug("T54Rec.forward (mask): pred_masked shape=%s, gt_masked shape=%s", pred_masked.shape, gt_masked.shape)
        # logger.debug("T54Rec.forward (mask): pred_masked=%s, gt_masked=%s", pred_masked, gt_masked)

        return (
            loss_mask,
            pred_masked,
            gt_masked,
            decoder_input_use_query_vectors_mask,
            mask_token_no,
        )  # adding `decoder_input_use_query_vectors_mask` and `mask_token_no` to pass pylance checks

    def _compute_masked_loss_per_sample(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        bs: int,
        decoder_input_use_query_vectors_mask: Optional[torch.Tensor] = None,
        mask_token_no: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, None]:
        """Compute loss for masked training with per-sample normalization."""
        # NOTE: ...[loss_mask] probably not needed because we could just put the weights to 0 for the unmasked tokens
        _, _, _, decoder_input_use_query_vectors_mask, mask_token_no = (
            self._prepare_masked_loss(
                pred, gt, decoder_input_use_query_vectors_mask, mask_token_no
            )
        )

        # NOTE: By construction, there is always at least one token to predict, so `mask_token_no` is always > 0
        loss_weights_per_sample = torch.where(
            decoder_input_use_query_vectors_mask,
            1 / mask_token_no.unsqueeze(1).float(),
            0.0,
        ).to(
            pred.device
        )  # (bs, n_query) - weights for each sample
        # logger.debug(f"loss_weights_per_sample={loss_weights_per_sample}")
        loss_weights_per_sample = loss_weights_per_sample.view(
            -1
        )  # (bs * n_query,) - weights for each sample

        # loss_weights_per_sample = loss_weights_per_sample[
        #     loss_mask
        # ]  # (bs * num_valid_tokens,) - weights for valid tokens
        # # logger.debug(f"loss_weights_per_sample (valid tokens)={loss_weights_per_sample}")
        # loss = self.criterion(pred_masked, gt_masked)
        # loss = loss * loss_weights_per_sample  # (bs * num_valid_tokens,) - weighted loss

        loss = self.criterion(pred, gt)
        # logger.debug(f"loss (before masking)={loss}")
        loss = loss * loss_weights_per_sample  # (bs * n_query,) - weighted loss
        # logger.debug(f"loss (after masking)={loss}")

        loss_per_codebook = loss.view(bs, self.config.n_query).mean(dim=0)

        return loss.mean(), loss_per_codebook

    def _compute_masked_loss_per_batch(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        bs: int,
        decoder_input_use_query_vectors_mask: Optional[torch.Tensor] = None,
        mask_token_no: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, None]:
        """Compute loss for masked training with per-batch normalization."""
        loss_mask, _, _, decoder_input_use_query_vectors_mask, mask_token_no = (
            self._prepare_masked_loss(
                pred, gt, decoder_input_use_query_vectors_mask, mask_token_no
            )
        )
        loss_fn = nn.CrossEntropyLoss(
            reduction="sum", ignore_index=-100
        )  # Use 'sum' to compute loss over the batch
        loss = loss_fn(pred, gt)
        loss /= loss_mask.sum()  # Normalize by the number of valid tokens
        return loss, None

    def _compute_masked_loss_unnormalized(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        bs: int,
        decoder_input_use_query_vectors_mask: Optional[torch.Tensor] = None,
        mask_token_no: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, None]:
        """Compute loss for masked training without normalization."""
        _, pred_masked, gt_masked, decoder_input_use_query_vectors_mask, mask_token_no = (
            self._prepare_masked_loss(
                pred, gt, decoder_input_use_query_vectors_mask, mask_token_no
            )
        )
        loss = self.criterion(pred_masked, gt_masked)
        return loss.mean(), None

    def _compute_autoregressive_loss(
        self, pred: torch.FloatTensor, gt: torch.Tensor
    ) -> Tuple[torch.FloatTensor, None]:
        """
        Compute loss for autoregressive training.

        # TODO: MODIFY THE COLLATOR SO IT'S EASIER --> we could use the masking loss compute!
        """
        return torch.FloatTensor(), None


def _get_valid_mask(
    t: int,
    beam_tokens: torch.Tensor,
    transition_masks: Dict,
    transition_mask_t3_full: Optional[torch.Tensor],
    prefix_to_uidx_t3: Optional[torch.Tensor],
    uidx_to_next_tokens_t3: Optional[torch.Tensor],
    trie: Optional[Trie],
    bs: int,
    k: int,
    codebook_num: int,
    no_special_tokenizer_tokens: int,
    device: torch.device
) -> torch.Tensor:
    """Get valid token mask for current timestep."""
    # NOTE: SHOULD I ALSO INCLUDE THE FIRST TRANSITION? (i.e. which first tokens are valid?)
    if t in transition_masks and transition_masks[t] is not None:
        mask = transition_masks[t]
        if t == 1:
            # mask: (codebook_num, codebook_num)                # NOTE: PUT IT INSIDE DOCSTRING?
            # beam_tokens[:, :, 0] -> (bs, k)                   # NOTE: PUT IT INSIDE DOCSTRING?
            # logger.debug(f"t: {t}, mask shape: {mask.shape}, beam_tokens[:, :, 0] shape: {beam_tokens[:, :, 0].shape}")
            # logger.debug(f"mask[beam_tokens[:, :, 0]] shape: {mask[beam_tokens[:, :, 0]].shape}")
            return mask[beam_tokens[:, :, 0]] # (bs, k, codebook_num)
        elif t == 2:
            # mask: (codebook_num, codebook_num, codebook_num) # NOTE: PUT IT INSIDE DOCSTRING?
            # beam_tokens[:, :, :2] -> (bs, k, 2)              # NOTE: PUT IT INSIDE DOCSTRING?
            return mask[beam_tokens[:, :, 0], beam_tokens[:, :, 1]] # (bs, k, codebook_num)

    # NOTE: Takes longer but correct
    elif t == 3 and transition_mask_t3_full is not None:
        return transition_mask_t3_full[beam_tokens[:, :, 0], beam_tokens[:, :, 1], beam_tokens[:, :, 2]] # (bs, k, codebook_num)

    elif (t == 3 and prefix_to_uidx_t3 is not None and
          uidx_to_next_tokens_t3 is not None):
        # prefix_to_uidx_t3: (codebook_num, codebook_num, codebook_num) - transition mask for t=3
        # uidx_to_next_tokens_t3: (|U|, code_num) - valid next tokens for each unique prefix of length 3
        # beam_tokens[:, :, :3] -> (bs, k, 3)
        uidx = prefix_to_uidx_t3[
            beam_tokens[:, :, 0], beam_tokens[:, :, 1], beam_tokens[:, :, 2]
        ]
        return uidx_to_next_tokens_t3[uidx]
    
    elif trie is not None:
        valid_mask = torch.zeros(
            (bs, k, codebook_num), dtype=torch.bool, device=device
        )
        for b in range(bs):
            for prev_beam in range(k):
                # get path up to current timestep. shape: (t,)
                path = beam_tokens[b, prev_beam].tolist()
                # add offset to account for codebook position
                path = [
                    p + no_special_tokenizer_tokens + step * codebook_num
                    for step, p in enumerate(path)
                ]
                valid_tokens = trie.get(path)
                if valid_tokens is not None:
                    # reverse offset
                    valid_tokens = [
                        v - no_special_tokenizer_tokens - t * codebook_num
                        for v in valid_tokens
                    ]
                    valid_mask[b, prev_beam, valid_tokens] = True
        return valid_mask

    logger.info("Careful, no transition masks found for step {}. Resorting to unconstrained search.".format(t))
    return torch.ones((bs, k, codebook_num), dtype=torch.bool, device=device)


def _update_beams(
    current_log_probs_extended: torch.Tensor,
    valid_mask: torch.Tensor,
    beam_tokens: torch.Tensor,
    beam_scores: torch.Tensor,
    k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update beam tokens and scores."""
    bs = current_log_probs_extended.shape[0]
    device = current_log_probs_extended.device

    # logger.debug(f"current_log_probs_extended shape: {current_log_probs_extended.shape}")
    # logger.debug(f"valid_mask shape: {valid_mask.shape}")
    valid_logits = current_log_probs_extended.masked_fill(~valid_mask, float("-inf"))
    valid_topk = valid_logits.topk(k, dim=-1)
    
    joint_scores = beam_scores.unsqueeze(-1) + valid_topk.values
    flat_scores = joint_scores.flatten(1)
    topk_joint = flat_scores.topk(k, dim=-1)
    
    beam_idx = topk_joint.indices // k
    token_idx = topk_joint.indices % k
    
    beam_tokens = beam_tokens.gather(
        1, beam_idx.unsqueeze(-1).expand(-1, -1, beam_tokens.size(-1))
    )
    
    b_idx = torch.arange(bs, device=device).unsqueeze(1).expand(-1, k)
    new_token = valid_topk.indices[b_idx, beam_idx, token_idx]
    beam_tokens = torch.cat([beam_tokens, new_token.unsqueeze(-1)], dim=-1)
    
    beam_scores = topk_joint.values
    
    return beam_tokens, beam_scores




class CustomGeneration:
    def __init__(self, model: 'T54Rec', use_multi_head: bool, mask_token_id: int):
        self.model = model
        self.use_multi_head = use_multi_head
        self.mask_token_id = mask_token_id

    def _get_model_properties(self, logits: torch.Tensor) -> Tuple[int, int, torch.device, int, int]:
        """Extract common model properties."""
        bs, n_query, _ = logits.shape
        device = logits.device
        code_num = self.model.cfg.code_num
        no_special_tokenizer_tokens = self.model.cfg.special_tokenizer_tokens_num
        return bs, n_query, device, code_num, no_special_tokenizer_tokens

    def _get_constraints(self, use_constraints: bool, device: torch.device, code_num: int) -> Tuple[torch.Tensor, dict[int, torch.Tensor] | dict[int, None], torch.Tensor | None]:
        """Get first token and transition constraints."""
        if use_constraints:
            assert self.model.first_token_constraints_fast is not None, \
                "First token constraints not set. Use set_first_token_constraints_fast."
            assert self.model.transition_constraints_fast is not None, \
                "Transition constraints not set. Use set_transition_constraints_fast."
            
            first_token_constraints = self.model.first_token_constraints_fast
            transition_masks = self.model.transition_constraints_fast
            transition_masks = {t: mask.to_dense() for t, mask in transition_masks.items() if mask is not None}
            transition_mask_t3_full = self.model.transition_mask_t3_full.to(device)
        else:
            first_token_constraints = torch.ones(code_num, dtype=torch.bool, device=device)
            transition_masks = {1: None, 2: None}
            transition_mask_t3_full = None

        return first_token_constraints, transition_masks, transition_mask_t3_full

    def _initialize_beams(self, log_probs: torch.Tensor, first_token_constraints: torch.Tensor, 
                         no_special_tokenizer_tokens: int, code_num: int, topK: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize beam search with first token."""
        log_probs0 = log_probs[:,0,:] if self.use_multi_head else log_probs[:, 0, no_special_tokenizer_tokens : code_num + no_special_tokenizer_tokens]
        # log_probs0: (bs, code_num)
        log_probs0 = log_probs0.masked_fill(~first_token_constraints.to_dense(), float("-inf"))

        topk = log_probs0.topk(topK, dim=-1)
        beam_tokens = topk.indices.unsqueeze(-1)  # (bs, k, 1)
        beam_scores = topk.values  # (bs, k)
        
        return beam_tokens, beam_scores

    def _initialize_generation(
        self, log_probs: torch.Tensor, use_constraints: bool, topK: int
    ) -> Tuple[int, int, torch.device, int, int, dict[int, torch.Tensor] | dict[int, None], torch.Tensor | None, torch.Tensor, torch.Tensor]:
        # Get model properties
        bs, n_query, device, code_num, no_special_tokenizer_tokens = self._get_model_properties(log_probs)
        
        # Prepare transition masks
        first_token_constraints, transition_masks, transition_mask_t3_full = self._get_constraints(use_constraints, device, code_num)

        # First token
        beam_tokens, beam_scores = self._initialize_beams(
            log_probs, first_token_constraints, no_special_tokenizer_tokens, code_num, topK
        )
        return (bs, n_query, device, code_num, no_special_tokenizer_tokens, 
                transition_masks, transition_mask_t3_full, beam_tokens, beam_scores)
    
    def _add_token_offsets(self, beam_tokens: torch.Tensor, n_query: int, 
                          no_special_tokenizer_tokens: int, code_num: int):
        """Add offset for special tokens and codebook tokens."""
        for t in range(n_query):
            beam_tokens[:, :, t] += no_special_tokenizer_tokens + t * code_num
        return beam_tokens

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, topK: int, use_constraints: bool) -> torch.Tensor:
        raise NotImplementedError("This method should be implemented in subclasses.")
    

class ParallelBeamSearchGenerator(CustomGeneration):
    def __init__(self, model: 'T54Rec', use_multi_head: bool=False, mask_token_id: int=3):
        super().__init__(model, use_multi_head, mask_token_id)

    def generate(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        topK: int = 5, 
        use_constraints: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Fast constrained beam search generation.
        
        Args:
            input_ids: Input token IDs (bs, seq_len)
            attention_mask: Attention mask (bs, seq_len)
            topK: Number of beams for beam search
            use_constraints: Whether to use generation constraints
            
        Returns:
            Dictionary with 'sequences' and 'sequences_scores' keys
        """
        # Get model predictions
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        _, logits = self.model.predict(input_ids, attention_mask)
        log_probs = F.log_softmax(logits, dim=-1)  # (bs, n_query, vocab_size)
        # logger.debug(f"log_probs shape: {log_probs.shape}")
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        logger.debug(f"predict time: {start.elapsed_time(end)} ms")

        start = torch.cuda.Event(enable_timing=True)
        start.record()
        # Initialize generation (first step)
        (bs, n_query, device, code_num, no_special_tokenizer_tokens,
         transition_masks, transition_mask_t3_full, beam_tokens, beam_scores) = self._initialize_generation(
            log_probs, use_constraints, topK
        )

        # Iterate through remaining positions
        for t in range(1, n_query):
            current_log_probs = log_probs[:, t, :] if self.use_multi_head else log_probs[
                :,
                t,
                no_special_tokenizer_tokens
                + t * code_num : no_special_tokenizer_tokens
                + (t + 1) * code_num,
            ]  # (bs, code_num)
            # logger.debug(f"t: {t}, current_log_probs shape: {current_log_probs.shape}")

            # Get valid token mask
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            valid_mask = _get_valid_mask(
                t=t,
                beam_tokens=beam_tokens,
                transition_masks=transition_masks,
                transition_mask_t3_full=transition_mask_t3_full if use_constraints else None,
                prefix_to_uidx_t3=self.model.prefix_to_uidx_t3 if use_constraints else None,
                uidx_to_next_tokens_t3=self.model.uidx_to_next_tokens_t3 if use_constraints else None,
                trie=self.model.candidate_trie if use_constraints else None,
                bs=bs,
                k=topK,
                codebook_num=code_num,
                no_special_tokenizer_tokens=no_special_tokenizer_tokens,
                device=device
            )
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            logger.debug(f"t: {t}, valid_mask time: {start.elapsed_time(end)} ms")

            # Update beams
            # logger.debug(f"t: {t}, valid_mask shape: {valid_mask.shape}")
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            beam_tokens, beam_scores = _update_beams(
                current_log_probs_extended=current_log_probs[:, None, :],
                valid_mask=valid_mask,
                beam_tokens=beam_tokens,
                beam_scores=beam_scores,
                k=topK
            )
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            logger.debug(f"t: {t}, update_beams time: {start.elapsed_time(end)} ms")

        # Add offset for special tokens and codebook tokens
        beam_tokens = self._add_token_offsets(
            beam_tokens, n_query, no_special_tokenizer_tokens, code_num
        )
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        logger.debug(f"beam search time: {start.elapsed_time(end)} ms")
        
        return {"sequences": beam_tokens, "sequences_scores": beam_scores}
    

class AutoregressiveGenerator(CustomGeneration):
    def __init__(self, model: 'T54Rec', use_multi_head: bool = False, mask_token_id: int = 3):
        super().__init__(model, use_multi_head, mask_token_id)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        topK: int = 5,
        use_constraints: bool = True,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Autoregressive generation with beam search using helper functions.
        
        Args:
            input_ids: Input token IDs (bs, seq_len) 
            attention_mask: Attention mask (bs, seq_len)
            topK: Number of beams for beam search
            use_constraints: Whether to use generation constraints
            temperature: Temperature for sampling
            
        Returns:
            Dictionary with 'sequences' and 'sequences_scores' keys
        """
        # Get model predictions
        _, logits = self.model.predict(input_ids, attention_mask)
        log_probs = F.log_softmax(logits, dim=-1)  # (bs, n_query, vocab_size)

        # Initialize generation (first step)
        (bs, n_query, device, code_num, no_special_tokenizer_tokens,
         transition_masks, transition_mask_t3_full, beam_tokens, beam_scores) = self._initialize_generation(
            log_probs, use_constraints, topK
        )

        for t in range(1, n_query):  
            all_current_log_probs = []          
            for k_idx in range(topK):
                # Prepare decoder input tokens for current candidate
                decoder_input_tokens = beam_tokens[:, k_idx, :]  # (bs, t)
                
                # Pad to n_query length with mask tokens
                decoder_input_tokens_full = torch.ones((bs, n_query), dtype=torch.long, device=device) * self.mask_token_id
                decoder_input_tokens_full[:, :t] = decoder_input_tokens
                
                # Create masks
                decoder_input_mask = torch.ones(
                    (bs, n_query), dtype=torch.bool, device=device
                ) # True: use query vectors
                decoder_input_mask[:, :t] = False # use previous tokens instead of query vectors for already generated tokens
                
                # Get predictions for current beam
                _, current_logits = self.model.predict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_tokens=decoder_input_tokens_full,
                    decoder_input_mask=decoder_input_mask,
                )

                current_logits_t = current_logits[:, t, :] if self.use_multi_head else current_logits[
                    :,
                    t,
                    no_special_tokenizer_tokens + t * code_num:no_special_tokenizer_tokens + (t + 1) * code_num,
                ]
                # (bs, code_num)
                current_log_probs = F.log_softmax(current_logits_t, dim=-1)  # (bs, code_num)
                all_current_log_probs.append(current_log_probs)

            # Stack log probs for all beams: (bs, k, code_num)
            current_log_probs_stacked = torch.stack(all_current_log_probs, dim=1)

            valid_mask = _get_valid_mask(
                t=t,
                beam_tokens=beam_tokens,
                transition_masks=transition_masks,
                transition_mask_t3_full=transition_mask_t3_full,
                prefix_to_uidx_t3=self.model.prefix_to_uidx_t3 if use_constraints else None,
                uidx_to_next_tokens_t3=self.model.uidx_to_next_tokens_t3 if use_constraints else None,
                trie=self.model.candidate_trie if use_constraints else None,
                bs=bs,
                k=topK,
                codebook_num=code_num,
                no_special_tokenizer_tokens=no_special_tokenizer_tokens,
                device=device
            )
            
            # Update beams using helper function
            beam_tokens, beam_scores = _update_beams(
                current_log_probs_extended=current_log_probs_stacked,
                valid_mask=valid_mask,
                beam_tokens=beam_tokens,
                beam_scores=beam_scores,
                k=topK
            )

        # Add offset for special tokens and codebook tokens
        beam_tokens = self._add_token_offsets(
            beam_tokens, n_query, no_special_tokenizer_tokens, code_num
        )

        return {"sequences": beam_tokens, "sequences_scores": beam_scores}
    


class T54Rec(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(T54Rec, self).__init__()
        self.cfg = cfg  # `cfg` and not `config` because the latter is used in the transformers library

        logger.info(f"Initializing language decoder ...")
        # Initialize item aggregator
        self._setup_input_components()
        # Initialize core model
        self.t5_model = self._create_t5_model()
        self.t5_model.config.use_cache = False

        # Initialize components
        self._setup_loss_computer()
        if cfg.is_inference:
            self._setup_generation_components()

        # Initialize weights when training the model
        if not cfg.is_inference:
            logger.info("Initializing weights with normal distribution ...")
            weight_initializer = WeightInitializer(self.t5_model, cfg)
            weight_initializer.initialize_all_weights()

        logger.info("Language decoder initialized.")

    def _setup_input_components(self) -> None:
        """Initialize input components."""
        self.encoder_aggregator = ItemTokenAggregator(self.cfg)
        if self.cfg.is_aggregate_tokens:
            assert not (self.cfg.bias_config.has_relative_encoder_item_bias or self.cfg.bias_config.has_relative_encoder_codebook_bias)

    def _create_t5_model(self) -> QT5:
        """
        Create the T5 model with the specified configuration.
        TODO: Explain that for multi-head projection, we need to load a T5 model and then replace the projection head.
        At inference, we simply load the QT5 model (with multi-head projection already in place).
        """
        if self.cfg.is_pretrained_model:
            # For inference or finetuning with SETRec config
            logger.info("Loading pretrained T5 model...")
            model = QT5.from_pretrained(
                self.cfg.base_model,
                local_files_only=True,
                cache_dir=self.cfg.cache_dir,
                device_map=self.cfg.device_map,
                cfg=self.cfg,
            )
            model = cast(QT5, model)

        else:
            # To train from scratch
            logger.info("Initializing T5Config...")
            config_path = os.path.join(self.cfg.cache_dir, "config.json")
            t5_config = T5Config.from_json_file(config_path)
            model = QT5(
                t5_config=t5_config,
                cfg=self.cfg,
            )

        # Replace single head with multiple heads after successful loading
        if self.cfg.use_multi_head and not self.cfg.is_inference:
            logger.info("Replacing single projection head with multiple heads.")
            model.replace_projection_head()

        logger.info("T5 model created successfully.")
        return model

    def _setup_loss_computer(self) -> None:
        """Initialize loss computation components."""
        device = next(self.t5_model.parameters()).device
        training_config = self.cfg.training_config
        assert training_config is not None, "Training configuration must be provided."
        self.loss_computer = LossComputer(training_config, device, use_multi_head=self.cfg.use_multi_head)

    def _setup_generation_components(self) -> None:
        """Initialize generation-specific components."""
        generation_mode = getattr(self.cfg, 'generation_mode', 'parallel_beam_search')
        logger.info(f"Using generation mode: {generation_mode}")

        if generation_mode == 'parallel_beam_search':
            self.generator = ParallelBeamSearchGenerator(self, use_multi_head=self.cfg.use_multi_head)
        elif generation_mode == 'autoregressive_beam_search':
            mask_token_id = getattr(self.cfg, 'mask_token_id', 3)
            self.generator = AutoregressiveGenerator(self, use_multi_head=self.cfg.use_multi_head, mask_token_id=mask_token_id)
        else:
            raise ValueError(f"Unknown generation mode: {generation_mode}")
        
        # Initialize constraint storage
        self.first_token_constraints_fast: Optional[torch.Tensor] = None
        self.transition_constraints_fast: Optional[Dict[int, torch.Tensor]] = None
        self.prefix_to_uidx_t3: Optional[torch.Tensor] = None
        self.uidx_to_next_tokens_t3: Optional[torch.Tensor] = None
        self.candidate_trie: Optional[Trie] = None

    def set_first_token_constraints_fast(
        self, first_token_constraints: torch.Tensor
    ) -> None:
        """
        Set the first token constraints for the model.
        :param first_token_constraints: (vocab_size,) boolean tensor indicating valid first tokens
        """
        self.first_token_constraints_fast = first_token_constraints.to(
            dtype=torch.bool, device=self.t5_model.device
        )

    def set_transition_constraints_fast(
        self, transition_mask_t1: torch.Tensor, transition_mask_t2: torch.Tensor
    ) -> None:
        """
        Set the transition constraints for the model.
        :param transition_mask_t1: (vocab_size, vocab_size) boolean tensor indicating valid transitions for the first token
        :param transition_mask_t2: (vocab_size, vocab_size) boolean tensor indicating valid transitions for the second token
        """
        self.transition_constraints_fast = {
            1: transition_mask_t1.to(dtype=torch.bool, device=self.t5_model.device),
            2: transition_mask_t2.to(dtype=torch.bool, device=self.t5_model.device),
        }
    
    def set_transition_constraints_t3_full(
        self, transition_mask_t3: torch.Tensor
    ) -> None:
        """
        Set the transition constraints for the model.
        # :param transition_mask_t1: (vocab_size, vocab_size) boolean tensor indicating valid transitions for the first token
        # :param transition_mask_t2: (vocab_size, vocab_size) boolean tensor indicating valid transitions for the second token
        """
        self.transition_mask_t3_full = transition_mask_t3

    def set_transition_constraints_fast_t3(
        self, prefix_to_uidx_t3: torch.Tensor, uidx_to_next_tokens_t3: torch.Tensor
    ) -> None:
        """
        Set the transition constraints for the model for T3.
        :param prefix_to_uidx_t3: (vocab_size,) tensor mapping prefixes to unique indices
        :param uidx_to_next_tokens_t3: (unique_indices, vocab_size) tensor mapping unique indices to next tokens
        """
        self.prefix_to_uidx_t3 = prefix_to_uidx_t3.to(
            dtype=torch.long, device=self.t5_model.device
        )
        self.uidx_to_next_tokens_t3 = uidx_to_next_tokens_t3.to(
            dtype=torch.bool, device=self.t5_model.device
        )
        # NOTE: return a single object?

    def set_candidate_trie(self, candidate_trie: Trie) -> None:
        """
        Set the candidate trie for the model.
        :param candidate_trie: Trie object containing the candidates
        """
        self.candidate_trie = candidate_trie

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_input_tokens: Optional[torch.Tensor] = None,
        decoder_input_mask: Optional[torch.Tensor] = None,
    ):
        # input_ids: (bs, seq_len)
        # attention_mask: (bs, seq_len)
        # decoder_input_tokens: (bs, n_query) - optional, used for inference with specific tokens
        # decoder_input_mask: (bs, n_query) - optional, used for inference with specific tokens
        #           mask where True means "use query vectors", and False means "use GT token embeddings".

        # 1. query embeddings
        bs = input_ids.shape[0]
        device = input_ids.device
        query_embeds = self.t5_model.decoder.query_emb(
            self.t5_model.decoder.query_input_ids.to(device)
        ).expand(
            bs, -1, -1
        )  # (bs, n_query, emb_dim)

        # 2. encoder inputs
        inputs = self.t5_model.shared(input_ids.to(device))  # (bs, seq_len, emb_dim), seq_len = n_items * n_query
        inputs, attention_mask = self.encoder_aggregator.aggregate(item_embeddings=inputs, mask=attention_mask)

        # 3. decoder inputs
        if decoder_input_tokens is not None and decoder_input_mask is not None:
            token_embeds = self.t5_model.shared(
                decoder_input_tokens.to(device)
            )  # (bs, n_query, emb_dim)
            # logger.debug(f"decoder_input_tokens={decoder_input_tokens}")
            # logger.debug(f"decoder_input_mask={decoder_input_mask}")
            # logger.debug(f"token_embeds={token_embeds}")
            # logger.debug(f"query_embeds={query_embeds}")
            decoder_inputs = torch.where(
                decoder_input_mask.unsqueeze(-1),  # (bs, n_query, 1)
                query_embeds,  # (bs, n_query, emb_dim)
                token_embeds,  # (bs, n_query, emb_dim)
            )
            # logger.debug(f"decoder_inputs={decoder_inputs}")
        else:
            # If no decoder input tokens are provided, use query embeddings as decoder inputs
            decoder_inputs = query_embeds

        # 4. query-guided simultaneous decoding (t5 forward)
        assert (
            attention_mask.size()[0] == inputs.size()[0]
            and attention_mask.size()[1] == inputs.size()[1]
        )
        outputs = self.t5_model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_inputs_embeds=decoder_inputs,
            return_dict=True,
            output_hidden_states=True,
        )
        logits = outputs.logits  # (bs, n_query, vocab_size)

        return outputs, logits

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        decoder_input_tokens: Union[torch.Tensor, None] = None,
        decoder_input_use_query_vectors_mask: Union[torch.Tensor, None] = None,
        mask_token_no: Union[torch.Tensor, None] = None,
    ):
        loss = None
        bs = input_ids.shape[0]

        outputs, logits = self.predict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_tokens=decoder_input_tokens,
            decoder_input_mask=decoder_input_use_query_vectors_mask,
        )

        if labels is not None:
            labels = labels.to(logits.device)  # (bs, n_query)
            pred = logits.view(-1, logits.size(-1))  # (bs * n_query, vocab_size)
            gt = labels.view(-1)  # (bs * n_query)

            loss, loss_per_codebook = self.loss_computer.compute_loss(
                pred=pred,
                gt=gt,
                bs=bs,
                decoder_input_use_query_vectors_mask=decoder_input_use_query_vectors_mask,
                mask_token_no=mask_token_no,
            )

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=(loss_per_codebook,) if loss_per_codebook is not None else None,
        )

    # def generate(self, input_ids, input_mask, topk=5, use_constraints=True, t1=0.0, t2=0.0):
    def generate(self, input_ids, input_mask, topK=5, use_constraints=True):
        """
        Generate sequences using beam search.
        input_ids: (bs, seq_len)
        input_mask: (bs, seq_len)
        topK: Number of beams for beam search
        use_constraints: Whether to use generation constraints
        """
        return self.generator.generate(
            input_ids, input_mask, topK, use_constraints
        )