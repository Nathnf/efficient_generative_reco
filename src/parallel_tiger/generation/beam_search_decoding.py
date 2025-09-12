import torch
import torch.nn.functional as F
from parallel_tiger.generation.trie import Trie                                                       # type: ignore[reportMissingImports]

import time as time
from typing import List, Dict, Tuple, Optional

import logging

logger = logging.getLogger(__name__)



def _get_valid_mask(
    t: int,
    beam_tokens: torch.Tensor,
    transition_masks: Dict,
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
            # mask: (codebook_num, codebook_num)
            # beam_tokens[:, :, 0] -> (bs, k)
            # logger.debug(f"t: {t}, mask shape: {mask.shape}, beam_tokens[:, :, 0] shape: {beam_tokens[:, :, 0].shape}")
            # logger.debug(f"mask[beam_tokens[:, :, 0]] shape: {mask[beam_tokens[:, :, 0]].shape}")
            return mask[beam_tokens[:, :, 0]] # (bs, k, codebook_num)
        elif t == 2:
            # mask: (codebook_num, codebook_num, codebook_num)
            # beam_tokens[:, :, :2] -> (bs, k, 2)
            return mask[beam_tokens[:, :, 0], beam_tokens[:, :, 1]] # (bs, k, codebook_num)

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





def _update_beams_greedy(
    current_valid_logits_extended: torch.Tensor,
    beam_tokens: torch.Tensor,
    beam_scores: torch.Tensor,
    k: int,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update beam tokens and scores."""
    bs = current_valid_logits_extended.shape[0]
    device = current_valid_logits_extended.device

    # logger.debug(f"current_valid_logits_extended shape: {current_valid_logits_extended.shape}")
    valid_log_probs = F.log_softmax(current_valid_logits_extended, dim=-1)
    valid_topk = valid_log_probs.topk(k, dim=-1)

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



def _update_beams_stochastic(
    current_valid_logits_extended: torch.Tensor,
    beam_tokens: torch.Tensor,
    beam_scores: torch.Tensor,
    k: int,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update beam tokens and scores."""
    bs = current_valid_logits_extended.shape[0]
    device = current_valid_logits_extended.device

    # logger.debug(f"current_valid_logits_extended shape: {current_valid_logits_extended.shape}")
    probs = torch.softmax(current_valid_logits_extended / temperature, dim=-1) # (bs, k, code_num)

    # Flatten over (beam, vocab) for each batch
    flat_probs = (probs * torch.exp(beam_scores).unsqueeze(-1)).flatten(1) # (bs, k * code_num)
    flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)  # re-normalize

    # Sample k new beam indices without replacement
    try:
        topk_sampled = torch.multinomial(flat_probs, num_samples=k, replacement=False)  # (bs, k)
    except:
        logger.info("Sampling without replacement not possible. Sampling with replacement.")
        topk_sampled = torch.multinomial(flat_probs, num_samples=k, replacement=True)  # (bs, k)
    beam_idx = topk_sampled // probs.size(-1)  # (bs, k)
    token_idx = topk_sampled % probs.size(-1)  # (bs, k)

    beam_tokens = beam_tokens.gather(
        1, beam_idx.unsqueeze(-1).expand(-1, -1, beam_tokens.size(-1))
    )

    b_idx = torch.arange(bs, device=device).unsqueeze(1).expand(-1, k)
    new_token = token_idx  # (bs, k)
    beam_tokens = torch.cat([beam_tokens, new_token.unsqueeze(-1)], dim=-1)

    beam_scores = (beam_scores.gather(1, beam_idx) + torch.log(probs[b_idx, beam_idx, token_idx]))
    
    return beam_tokens, beam_scores



class CustomGeneration:
    def __init__(
        self, 
        model: 'T54Rec', # type: ignore[reportMissingTypeStubs]
        use_multi_head: bool, 
        mask_token_id: int, 
        stochastic: bool,
        temperatures: Optional[List[float]] = None,
    ) -> None:
        self.model = model
        self.use_multi_head = use_multi_head
        self.mask_token_id = mask_token_id
        self.stochastic = stochastic
        self.temperatures = temperatures if temperatures else [1.0] * model.cfg.n_query
        self._initialize_beams_fn = self._initialize_beams_stochastic if stochastic else self._initialize_beams_greedy

    def _get_model_properties(self, logits: torch.Tensor) -> Tuple[int, int, torch.device, int, int]:
        """Extract common model properties."""
        bs, n_query, _ = logits.shape
        device = logits.device
        code_num = self.model.cfg.code_num
        no_special_tokenizer_tokens = self.model.cfg.special_tokenizer_tokens_num
        return bs, n_query, device, code_num, no_special_tokenizer_tokens

    def _get_constraints(self, use_constraints: bool, device: torch.device, code_num: int) -> Tuple[torch.Tensor, dict[int, torch.Tensor] | dict[int, None]]:
        """Get first token and transition constraints."""
        if use_constraints:
            assert self.model.first_token_constraints_fast is not None, \
                "First token constraints not set. Use set_first_token_constraints_fast."
            assert self.model.transition_constraints_fast is not None, \
                "Transition constraints not set. Use set_transition_constraints_fast."
            
            first_token_constraints = self.model.first_token_constraints_fast
            transition_masks = self.model.transition_constraints_fast
            transition_masks = {t: mask.to_dense() for t, mask in transition_masks.items() if mask is not None}
        else:
            first_token_constraints = torch.ones(code_num, dtype=torch.bool, device=device)
            transition_masks = {1: None, 2: None}
        
        return first_token_constraints, transition_masks

    def _initialize_beams_greedy(self, valid_logits0: torch.Tensor, topK: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize beam search with first token."""
        log_probs0 = F.log_softmax(valid_logits0, dim=-1)  # (bs, code_num)

        topk = log_probs0.topk(topK, dim=-1)
        beam_tokens = topk.indices.unsqueeze(-1)  # (bs, k, 1)
        beam_scores = topk.values  # (bs, k)
        
        return beam_tokens, beam_scores

    def _initialize_beams_stochastic(self, valid_logits0: torch.Tensor, topK: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize beam search with first token."""
        temperature = self.temperatures[0] if self.temperatures is not None else 1.0
        probs0 = torch.softmax(valid_logits0 / temperature, dim=-1)  # (bs, code_num)

        topk_sampled = torch.multinomial(probs0, num_samples=topK, replacement=True) # should we use False here (with try/except)?
        beam_tokens = topk_sampled.unsqueeze(-1)  # (bs, k, 1)
        beam_scores = torch.log(probs0.gather(-1, topk_sampled)).squeeze(-1)  # (bs, k)

        return beam_tokens, beam_scores

    def _initialize_generation(
        self, logits: torch.Tensor, use_constraints: bool, topK: int
    ) -> Tuple[int, int, torch.device, int, int, dict[int, torch.Tensor] | dict[int, None], torch.Tensor, torch.Tensor]:
        # Get model properties
        bs, n_query, device, code_num, no_special_tokenizer_tokens = self._get_model_properties(logits)

        # Prepare transition masks
        first_token_constraints, transition_masks = self._get_constraints(use_constraints, device, code_num)

        # First token
        logits0 = logits[:,0,:] if self.use_multi_head else logits[:, 0, no_special_tokenizer_tokens : code_num + no_special_tokenizer_tokens]
        # logits0: (bs, code_num)
        valid_logits0 = logits0.masked_fill(~first_token_constraints.to_dense(), float("-inf"))
        beam_tokens, beam_scores = self._initialize_beams_fn(
            valid_logits0, topK
        )
        return (bs, n_query, device, code_num, no_special_tokenizer_tokens, 
                transition_masks, beam_tokens, beam_scores)
    
    def _add_token_offsets(self, beam_tokens: torch.Tensor, n_query: int, 
                          no_special_tokenizer_tokens: int, code_num: int):
        """Add offset for special tokens and codebook tokens."""
        for t in range(n_query):
            beam_tokens[:, :, t] += no_special_tokenizer_tokens + t * code_num
        return beam_tokens

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, topK: int, use_constraints: bool) -> torch.Tensor:
        raise NotImplementedError("This method should be implemented in subclasses.")
    

class ParallelBeamSearchGenerator(CustomGeneration):
    def __init__(
        self, 
        model: 'T54Rec', # type: ignore[reportMissingTypeStubs]
        use_multi_head: bool=False, 
        mask_token_id: int=3, 
        stochastic: bool=False, 
        temperatures: Optional[List[float]]=None
    ) -> None:
        super().__init__(model, use_multi_head, mask_token_id, stochastic, temperatures)
        self._update_beams_fn = _update_beams_stochastic if stochastic else _update_beams_greedy

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
        _, logits = self.model.predict(input_ids, attention_mask)  # (bs, n_query, vocab_size)
        # logger.debug(f"logits shape: {logits.shape}")

        # Initialize generation (first step)
        (bs, n_query, device, code_num, no_special_tokenizer_tokens,
         transition_masks, beam_tokens, beam_scores) = self._initialize_generation(
            logits, use_constraints, topK
        )

        # Iterate through remaining positions
        for t in range(1, n_query):
            current_logits = logits[:, t, :] if self.use_multi_head else logits[
                :,
                t,
                no_special_tokenizer_tokens
                + t * code_num : no_special_tokenizer_tokens
                + (t + 1) * code_num,
            ]  # (bs, code_num)
            # logger.debug(f"t: {t}, current_log_probs shape: {current_log_probs.shape}")

            # Get valid token mask
            valid_mask = _get_valid_mask(
                t=t,
                beam_tokens=beam_tokens,
                transition_masks=transition_masks,
                prefix_to_uidx_t3=self.model.prefix_to_uidx_t3 if use_constraints else None,
                uidx_to_next_tokens_t3=self.model.uidx_to_next_tokens_t3 if use_constraints else None,
                trie=self.model.candidate_trie if use_constraints else None,
                bs=bs,
                k=topK,
                codebook_num=code_num,
                no_special_tokenizer_tokens=no_special_tokenizer_tokens,
                device=device
            )
            
            # Update beams
            # logger.debug(f"t: {t}, valid_mask shape: {valid_mask.shape}")
            valid_logits = current_logits[:, None, :].masked_fill(~valid_mask, float("-inf"))
            beam_tokens, beam_scores = self._update_beams_fn(
                current_valid_logits_extended=valid_logits,
                beam_tokens=beam_tokens,
                beam_scores=beam_scores,
                k=topK,
                temperature=self.temperatures[t],
            )
        
        # Add offset for special tokens and codebook tokens
        beam_tokens = self._add_token_offsets(
            beam_tokens, n_query, no_special_tokenizer_tokens, code_num
        )
        
        return {"sequences": beam_tokens, "sequences_scores": beam_scores}





# class AutoregressiveGenerator(CustomGeneration):
#     def __init__(self, model: 'T54Rec', use_multi_head: bool = False, mask_token_id: int = 3):
#         super().__init__(model, use_multi_head, mask_token_id)

#     def generate(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: torch.Tensor,
#         topK: int = 5,
#         use_constraints: bool = True,
#         temperature: float = 1.0,
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Autoregressive generation with beam search using helper functions.
        
#         Args:
#             input_ids: Input token IDs (bs, seq_len) 
#             attention_mask: Attention mask (bs, seq_len)
#             topK: Number of beams for beam search
#             use_constraints: Whether to use generation constraints
#             temperature: Temperature for sampling
            
#         Returns:
#             Dictionary with 'sequences' and 'sequences_scores' keys
#         """
#         # Get model predictions
#         _, logits = self.model.predict(input_ids, attention_mask)
#         log_probs = F.log_softmax(logits, dim=-1)  # (bs, n_query, vocab_size)

#         # Initialize generation (first step)
#         (bs, n_query, device, code_num, no_special_tokenizer_tokens,
#          transition_masks, beam_tokens, beam_scores) = self._initialize_generation(
#             log_probs, use_constraints, topK
#         )

#         for t in range(1, n_query):  
#             all_current_log_probs = []          
#             for k_idx in range(topK):
#                 # Prepare decoder input tokens for current candidate
#                 decoder_input_tokens = beam_tokens[:, k_idx, :]  # (bs, t)
                
#                 # Pad to n_query length with mask tokens
#                 decoder_input_tokens_full = torch.ones((bs, n_query), dtype=torch.long, device=device) * self.mask_token_id
#                 decoder_input_tokens_full[:, :t] = decoder_input_tokens
                
#                 # Create masks
#                 decoder_input_mask = torch.ones(
#                     (bs, n_query), dtype=torch.bool, device=device
#                 ) # True: use query vectors
#                 decoder_input_mask[:, :t] = False # use previous tokens instead of query vectors for already generated tokens
                
#                 # Get predictions for current beam
#                 _, current_logits = self.model.predict(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     decoder_input_tokens=decoder_input_tokens_full,
#                     decoder_input_mask=decoder_input_mask,
#                 )

#                 current_logits_t = current_logits[:, t, :] if self.use_multi_head else current_logits[
#                     :,
#                     t,
#                     no_special_tokenizer_tokens + t * code_num:no_special_tokenizer_tokens + (t + 1) * code_num,
#                 ]
#                 # (bs, code_num)
#                 current_log_probs = F.log_softmax(current_logits_t, dim=-1)  # (bs, code_num)
#                 all_current_log_probs.append(current_log_probs)

#             # Stack log probs for all beams: (bs, k, code_num)
#             current_log_probs_stacked = torch.stack(all_current_log_probs, dim=1)

#             valid_mask = _get_valid_mask(
#                 t=t,
#                 beam_tokens=beam_tokens,
#                 transition_masks=transition_masks,
#                 prefix_to_uidx_t3=self.model.prefix_to_uidx_t3 if use_constraints else None,
#                 uidx_to_next_tokens_t3=self.model.uidx_to_next_tokens_t3 if use_constraints else None,
#                 trie=self.model.candidate_trie if use_constraints else None,
#                 bs=bs,
#                 k=topK,
#                 codebook_num=code_num,
#                 no_special_tokenizer_tokens=no_special_tokenizer_tokens,
#                 device=device
#             )
            
#             # Update beams using helper function
#             beam_tokens, beam_scores = _update_beams(
#                 current_log_probs_extended=current_log_probs_stacked,
#                 valid_mask=valid_mask,
#                 beam_tokens=beam_tokens,
#                 beam_scores=beam_scores,
#                 k=topK
#             )

#         # Add offset for special tokens and codebook tokens
#         beam_tokens = self._add_token_offsets(
#             beam_tokens, n_query, no_special_tokenizer_tokens, code_num
#         )

#         return {"sequences": beam_tokens, "sequences_scores": beam_scores}