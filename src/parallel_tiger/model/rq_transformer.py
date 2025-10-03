from parallel_tiger.model.base_rq_transformer import *
from parallel_tiger.generation.beam_search_decoding_rq import ParallelBeamSearchGenerator

class RQTransformer(BaseRQTransformer):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_spatial_seq_len,
        depth_seq_len,
        spatial_layers,
        depth_layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        pad_id = 0,
        attention_type = None,
        num_special_tokens = 4
    ):
        super().__init__(
            num_tokens=num_tokens,
            dim=dim,
            max_spatial_seq_len=max_spatial_seq_len,
            depth_seq_len=depth_seq_len,
            spatial_layers=spatial_layers,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_mult=ff_mult,
            ff_dropout=ff_dropout,
            pad_id=pad_id,
            num_special_tokens=num_special_tokens
        )

        self.depth_transformer = DecoderOnlyTransformer(
            dim = dim,
            layers = depth_layers,
            attn_cls=CausalSelfAttention,
            ff_cls=FeedForward,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

    def _get_logits_from_spatial_token(self, ids, attention_mask, spatial_index=-1):
        tokens_with_depth_pos, spatial_tokens, b, _ = self._spatial_forward(ids, attention_mask) # (b, s+1, f)

        # --- IMPORTANT: pad tokens_with_depth_pos exactly like the training path ---
        tokens_with_depth_pos = F.pad(tokens_with_depth_pos, (0, 0, 0, 0, 0, 1), value=0.)  # -> (b, s+1, d, f)
        selected_spatial_token = spatial_tokens[:, spatial_index, :]            # (b, f)  # start token
        selected_token_with_depth_pos = tokens_with_depth_pos[:, spatial_index, :, :]  # (b, d, f) matched block - only keep the tokens with depth pos corresponding to the selected spatial token
        depth_tokens = torch.cat((selected_spatial_token[:, None, :], selected_token_with_depth_pos), dim=1)  # (b, d+1, f)

        depth_tokens = self.depth_transformer(depth_tokens) # (b, d+1, f)

        logits = torch.stack([layer(depth_tokens[:, i, :]) for i, layer in enumerate(self.to_logits)], dim=1) # (b, d, num_tokens)

        return logits, b

    def forward(self, ids, attention_mask, *args):
        tokens_with_depth_pos, spatial_tokens, b, _ = self._spatial_forward(ids, attention_mask) # (b, s+1, f), int, int

        spatial_tokens = rearrange(spatial_tokens, 'b s f -> b s 1 f')

        # spatial tokens become the start tokens of the depth dimension
        tokens_with_depth_pos = F.pad(tokens_with_depth_pos, (0, 0, 0, 0, 0, 1), value = 0.)

        depth_tokens = torch.cat((spatial_tokens, tokens_with_depth_pos), dim = -2)

        depth_tokens = rearrange(depth_tokens, '... n d -> (...) n d')

        depth_tokens = self.depth_transformer(depth_tokens) # (b * (s+1), d+1, f)

        depth_tokens = rearrange(depth_tokens, '(b s) d f -> b s d f', b = b)

        logits = torch.stack([layer(depth_tokens[:, :, i, :]) for i, layer in enumerate(self.to_logits)], dim=2) # (b, s+1, d, num_tokens)
        logits = logits[:, :-1, :, :] # remove logits corresponding to last item (no ground truth)
        preds = rearrange(logits, 'b s d f -> (b s d) f')

        labels = self._adapt_labels_to_multi_head(ids)
        loss, loss_per_codebook = self._compute_loss_with_mask(
            preds,
            labels,
            None,  # RQTransformer always uses all query vectors, so no masking
            self.depth_seq_len,
        )
        return loss, loss_per_codebook

    # Teacher forcing version - like training
    def forward_validation(self, ids, attention_mask, labels, *args):
        ids = torch.cat((ids, labels[:,None,:]), dim=1)
        # # print("attention_mask shape before:", attention_mask.shape)
        # # print("ids shape after:", ids.shape)
        # attention mask": (b, s, d) -> (b, s+1, d)
        attention_mask = F.pad(attention_mask, (0, 0, 0, 1), value=1)
        # # print("attention_mask shape after:", attention_mask.shape)
        logits, _ = self._get_logits_from_spatial_token(ids, attention_mask, spatial_index=-2) # (b, d, num_tokens)
        preds = rearrange(logits, 'b d f -> (b d) f') # (b * d, num_tokens)
        
        labels = self._adapt_labels_to_multi_head(labels)
        loss, loss_per_codebook = self._compute_loss_with_mask(
            preds,
            labels,
            None,  # RQTransformer always uses all query vectors, so no masking
            self.depth_seq_len,
        )
        return loss, loss_per_codebook, logits

    def generate(self, ids, attention_mask, topK=20, use_constraints=True):
        """
        Beam search generation from the last spatial token.
        NOTE: We have to be careful about the global vs local indexing.
        - Global indexing (tokenizer and embedding layer): all codebook tokens + special tokens
        - Local indexing (output of a single projection head): only the associated codebook tokens (no special tokens)

        NOTE 2: Probably easier to have a single projection matrix (of size dim x (num_tokens * depth_seq_len + num_special_tokens)?)
        """
        assert (self.candidate_trie is not None) or not use_constraints, "You must set a candidate trie for constrained decoding if use_constraints is True"

        tokens_with_depth_pos, spatial_tokens, b, _ = self._spatial_forward(ids, attention_mask) # (b, s, d, f), (b, s+1, f), int, int
        tokens_with_depth_pos = F.pad(tokens_with_depth_pos, (0, 0, 0, 0, 0, 1), value = 0.) # (b, s+1, d, f)
        last_spatial_token = spatial_tokens[:, -1, :] # (b, f)
        
        # init depth transformer input: [start_token, depth_pos_0..n]
        depth_tokens = [last_spatial_token[:, None, :]]  # start

        # step 0
        depth_out = self.depth_transformer(depth_tokens[0])  # (b, 1, f)
        logits0 = self.to_logits[0](depth_out[:, -1, :])  # (b, num_tokens)
        # logits0 = logits0 / temperature # if we decide to use sampling - for now we do greedy decoding

        valid_tokens_mask = self._get_valid_mask(
            step=0,
            logits=logits0,
            flattened_beams=None,
            use_constraints=use_constraints
        )
        logits0 = logits0 + valid_tokens_mask
        log_probs0 = safe_log_softmax(logits0, dim=-1)  # (b, num_tokens)
        topk_vals, topk_idx = log_probs0.topk(topK, dim=-1)  # (b, topK)
        topk_idx = topk_idx + self.num_special_tokens  # from local to global ids
    
        # Initialize beam state
        beam_scores = topk_vals  # (b, topK)
        beam_tokens = topk_idx.unsqueeze(-1)  # (b, topK, 1) -> prefixes

        for step in range(1, self.depth_seq_len):
            flattened_beams = beam_tokens.reshape(b * topK, step)  # (b*topK, step)
            flat_scores = beam_scores.reshape(b * topK)        # (b*topK,)

            depth_tokens = [last_spatial_token.repeat_interleave(topK, dim=0)[:, None, :]]  # (b*topK, 1, f)
            for d in range(step):
                # NOTE: flattened_beams need to be global IDs
                token_emb = self.token_emb(flattened_beams[:, d])  # (b*topK, f)
                pos_emb = self.depth_pos_emb(torch.tensor(d, device=token_emb.device))  # (f,)
                depth_tokens.append((token_emb + pos_emb)[..., None, :])  # (b*topK, 1, f)
            
            depth_input = torch.cat(depth_tokens, dim=1)  # (b*topK, step+1, f)
            depth_out = self.depth_transformer(depth_input)  # (b*topK, step+1, f)
            logits = self.to_logits[step](depth_out[:, -1, :])  # (b*topK, num_tokens)

            valid_tokens_mask = self._get_valid_mask(
                step=step,
                logits=logits,
                flattened_beams=flattened_beams,
                use_constraints=use_constraints
            )
            logits = logits + valid_tokens_mask
            log_probs = safe_log_softmax(logits, dim=-1)  # (b*topK, num_tokens)

            # expand beams
            next_vals, next_idx = log_probs.topk(topK, dim=-1)  # (b*topK, topK)
            next_idx = next_idx + self.num_special_tokens + (step * self.num_tokens)  # from local to global ids

            # add scores
            cand_scores = flat_scores[:, None] + next_vals  # (b*topK, topK)
            cand_beams = torch.cat((flattened_beams[:, None, :].repeat(1, topK, 1), next_idx[..., None]), dim=-1)  # (b*topK, topK, step+1)

            # reshape to (b, topK*topK)
            cand_scores = cand_scores.view(b, topK * topK)  # (b, topK*topK)
            cand_beams = cand_beams.view(b, topK * topK, -1)  # (b, topK*topK, step+1)

            # prune to topK
            beam_scores, beam_idx = cand_scores.topk(topK, dim=-1)  # (b, topK)
            batch_idx = torch.arange(b, device=beam_tokens.device)[:, None]
            beam_tokens = cand_beams[batch_idx, beam_idx]  # (b, topK, step+1)

        return {"sequences": beam_tokens, "sequences_scores": beam_scores}
    
    def _get_valid_mask(self, step, logits, flattened_beams=None, use_constraints=True):
        """Return an additive logits mask (0 for allowed, -inf for disallowed) at a decoding step.
        Uses specialized precomputed masks if available, otherwise falls back to the trie."""

        if not use_constraints:
            return torch.zeros_like(logits)

        device = logits.device

        if step == 0:
            if self.first_token_constraint_mask is not None:
                first_token_constraint_mask = self.first_token_constraint_mask.to(device=device)
                valid_tokens_mask = torch.zeros_like(logits).masked_fill(~first_token_constraint_mask, float("-inf"))
                return valid_tokens_mask
            else:
                assert self.candidate_trie is not None, "Trie needs to be set"
                valid_tokens_mask = torch.full_like(logits, float('-inf'))
                valid_next_tokens = self.candidate_trie.get([]) # type: ignore[OptionalMemberAccess]
                local_valid_next_tokens = [self._global_to_local_id(t, 0) for t in valid_next_tokens]
                valid_tokens_mask[:, local_valid_next_tokens] = 0.
                return valid_tokens_mask
            
        assert flattened_beams is not None
        # flat_tokens: (b*topK, step)

        def _global_to_local_beam_ids(global_ids, step, device):
            depth_idx = torch.arange(step, device=device)[None, :]  # (1, step)
            return global_ids - (self.num_special_tokens + depth_idx * self.num_tokens)

        if step in (1, 2) and self.transition_constraint_masks[step] is not None:
            mask = self.transition_constraint_masks[step].to(device=device) # (num_tokens, num_tokens) or (num_tokens, num_tokens, num_tokens)
            flattened_beams_loc = _global_to_local_beam_ids(flattened_beams, step, device)
            valid_mask = mask[flattened_beams_loc[:, 0]] if step == 1 else mask[flattened_beams_loc[:, 0], flattened_beams_loc[:, 1]]
            return torch.zeros_like(logits).masked_fill(~valid_mask, float("-inf"))

        elif step == 3 and self.prefix_to_uidx_t3 is not None and self.uidx_to_next_tokens_t3 is not None:
                # prefix_to_uidx_t3: (codebook_num, codebook_num, codebook_num) - transition mask for t=3
                # uidx_to_next_tokens_t3: (|U|, code_num) - valid next tokens for each unique prefix of length 3
                prefix_to_uidx_t3 = self.prefix_to_uidx_t3.to(device=device)
                uidx_to_next_tokens_t3 = self.uidx_to_next_tokens_t3.to(device=device)
                flattened_beams_loc = _global_to_local_beam_ids(flattened_beams, step, device)
                uidx = prefix_to_uidx_t3[flattened_beams_loc[:, 0], flattened_beams_loc[:, 1], flattened_beams_loc[:, 2]]
                valid_mask = uidx_to_next_tokens_t3[uidx]
                valid_tokens_mask = torch.zeros_like(logits).masked_fill(~valid_mask, float("-inf"))
                return valid_tokens_mask
        
        # Fallback: Trie
        assert self.candidate_trie is not None, "Trie needs to be set"
        valid_tokens_mask = torch.full_like(logits, float('-inf'))
        for beam_id, prefix in enumerate(flattened_beams.tolist()):
            valid_next_tokens = self.candidate_trie.get(prefix) # type: ignore[OptionalMemberAccess]
            # reverse offset
            valid_next_tokens = [self._global_to_local_id(t, step) for t in valid_next_tokens]
            valid_tokens_mask[beam_id, valid_next_tokens] = 0.

        return valid_tokens_mask

    def generate_teacher_forcing(self, ids, attention_mask, topK=20, use_constraints=True):
        """
        Generate topK candidates using teacher forcing (step-by-step with ground truth context).
        This should provide logits computed from the same context as free-running generation.
        """
        # Compute logits without concatenating ground truth labels
        logits, _ = self._get_logits_from_spatial_token(ids, attention_mask, spatial_index=-1)
        
        generator = ParallelBeamSearchGenerator(
            model=self,
            use_multi_head=True,
            stochastic=False,  # LATER: NOT HARDCODE IT
            temperatures=None, # IDEM
        )
        return generator.generate(logits, topK, use_constraints)