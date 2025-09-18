from parallel_tiger.model.base_rq_transformer import *

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

    def forward(self, ids, attention_mask, *args):
        tokens_with_depth_pos, spatial_tokens, b, spatial_seq_len = self._spatial_forward(ids, attention_mask) # (b, s+1, f), int, int

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
        loss = F.cross_entropy(preds, labels, ignore_index = -100)
        return loss, None

    def _get_logits_from_last_spatial_token(self, ids, attention_mask):
        tokens_with_depth_pos, spatial_tokens, b, _ = self._spatial_forward(ids, attention_mask) # (b, s+1, f)

        # --- IMPORTANT: pad tokens_with_depth_pos exactly like the training path ---
        tokens_with_depth_pos = F.pad(tokens_with_depth_pos, (0, 0, 0, 0, 0, 1), value=0.)  # -> (b, s+1, d, f)

        # Now the spatial_tokens and tokens_with_depth_pos have the same spatial length
        last_spatial_token = spatial_tokens[:, -1, :]            # (b, f)  # start token
        last_token_with_depth_pos = tokens_with_depth_pos[:, -1, :, :]  # (b, d, f) matched block - only keep the tokens with depth pos corresponding to the last spatial token
        depth_tokens = torch.cat((last_spatial_token[:, None, :], last_token_with_depth_pos), dim=1)  # (b, 1+d, f)
        
        depth_tokens = self.depth_transformer(depth_tokens) # (b, 1+d, f)

        logits = torch.stack([layer(depth_tokens[:, i, :]) for i, layer in enumerate(self.to_logits)], dim=1) # (b, d, num_tokens)

        return logits, b
    
    def generate(self, ids, attention_mask, topK=20, use_constraints=True):
        # TODO: HANDLE OFFSET FOR MULTI-HEAD PROJECTION LAYER LATER
        # TODO 2: If use constraints thing
        assert (self.candidate_trie is not None) or not use_constraints, "You must set a candidate trie for constrained decoding if use_constraints is True"

        # let's do step by step without a loop - then add a loop
        tokens_with_depth_pos, spatial_tokens, b, _ = self._spatial_forward(ids, attention_mask) # (b, s, d, f), (b, s+1, f), int, int
        tokens_with_depth_pos = F.pad(tokens_with_depth_pos, (0, 0, 0, 0, 0, 1), value = 0.) # (b, s+1, d, f)
        last_spatial_token = spatial_tokens[:, -1, :] # (b, f)
        
        # init depth transformer input: [start_token, depth_pos_0..n]
        depth_tokens = [last_spatial_token[:, None, :]]  # start

        # step 0
        depth_out = self.depth_transformer(depth_tokens[0])  # (b, 1, f)
        logits0 = self.to_logits[0](depth_out[:, -1, :])  # (b, num_tokens)
        # logits0 = logits0 / temperature # if we decide to use sampling - for now we do greedy decoding
        if not use_constraints:
            valid_tokens = torch.zeros_like(logits0)
        else:
            valid_tokens = torch.full_like(logits0, float('-inf'))
            for i in range(b):
                valid_next_tokens = self.candidate_trie.get([]) # type: ignore[OptionalMemberAccess]
                valid_tokens[i, valid_next_tokens] = 0.
        logits0 = logits0 + valid_tokens
        log_probs0 = safe_log_softmax(logits0, dim=-1)  # (b, num_tokens)
        topk_vals, topk_idx = log_probs0.topk(topK, dim=-1)  # (b, topK)
    
        # Initialize beam state
        beam_scores = topk_vals  # (b, topK)
        beam_tokens = topk_idx.unsqueeze(-1)  # (b, topK, 1) -> prefixes

        for step in range(1, self.depth_seq_len):
            flat_tokens = beam_tokens.reshape(b * topK, step)  # (b*topK, step)
            flat_scores = beam_scores.reshape(b * topK)        # (b*topK,)

            depth_tokens = [last_spatial_token.repeat_interleave(topK, dim=0)[:, None, :]]  # (b*topK, 1, f)
            for d in range(step):
                token_emb = self.token_emb(flat_tokens[:, d])  # (b*topK, f)
                pos_emb = self.depth_pos_emb(torch.tensor(d, device=token_emb.device))  # (f,)
                depth_tokens.append((token_emb + pos_emb)[..., None, :])  # (b*topK, 1, f)
            
            depth_input = torch.cat(depth_tokens, dim=1)  # (b*topK, step+1, f)
            depth_out = self.depth_transformer(depth_input)  # (b*topK, step+1, f)
            logits = self.to_logits[step](depth_out[:, -1, :])  # (b*topK, num_tokens)

            # mask logits based on candidate trie
            if not use_constraints:
                valid_mask = torch.zeros_like(logits)
            else:
                valid_mask = torch.full_like(logits, float('-inf'))
                for beam_id in range(b * topK):
                    prefix = flat_tokens[beam_id].tolist()
                    prefix_w_offset = [p + self.num_special_tokens + (step * self.num_tokens) for step, p in enumerate(prefix)]
                    valid_next_tokens = self.candidate_trie.get(prefix_w_offset) # type: ignore[OptionalMemberAccess]
                    # reverse offset
                    valid_next_tokens = [t % self.num_tokens - self.num_special_tokens for t in valid_next_tokens] # or t - step * self.num_tokens - self.num_special_tokens
                    valid_mask[beam_id, valid_next_tokens] = 0.
            logits = logits + valid_mask
            log_probs = safe_log_softmax(logits, dim=-1)  # (b*topK, num_tokens)

            # expand beams
            next_vals, next_idx = log_probs.topk(topK, dim=-1)  # (b*topK, topK)

            # add scores
            cand_scores = flat_scores[:, None] + next_vals  # (b*topK, topK)
            cand_tokens = torch.cat((flat_tokens[:, None, :].repeat(1, topK, 1), next_idx[..., None]), dim=-1)  # (b*topK, topK, step+1)

            # reshape to (b, topK*topK)
            cand_scores = cand_scores.view(b, topK * topK)  # (b, topK*topK)
            cand_tokens = cand_tokens.view(b, topK * topK, -1)  # (b, topK*topK, step+1)

            # prune to topK
            beam_scores, beam_idx = cand_scores.topk(topK, dim=-1)  # (b, topK)
            batch_idx = torch.arange(b, device=beam_tokens.device)[:, None]
            beam_tokens = cand_tokens[batch_idx, beam_idx]  # (b, topK, step+1)
            
            # # or?
            # beam_tokens = cand_tokens.gather(1, beam_idx.unsqueeze(-1).repeat(1, 1, cand_tokens.shape[-1]))  # (b, topK, step+1)

        # add token offset back
        offset = torch.arange(self.depth_seq_len, device=beam_tokens.device) * self.num_tokens
        beam_tokens = beam_tokens + offset[None, None, :] + self.num_special_tokens

        return {"sequences": beam_tokens, "sequences_scores": beam_scores}

    def forward_validation(self, ids, attention_mask, labels, *args):
        logits, b = self._get_logits_from_last_spatial_token(ids, attention_mask) # (b, d, num_tokens)
        preds = rearrange(logits, 'b d f -> (b d) f') # (b * d, num_tokens)
        
        labels = self._adapt_labels_to_multi_head(labels)

        # TODO: ADD CUSTOM LOSS COMPUTER (cf. T54Rec)
        loss = F.cross_entropy(preds, labels, ignore_index = -100) # NB: there shouldn't be any padding in validation
        return loss, None

    # def forward_validation(self, ids, attention_mask, labels):
    #     tokens_with_depth_pos, spatial_tokens, b, spatial_seq_len = self._spatial_forward(ids, attention_mask) # (b, s+1, f), int, int

    #     spatial_tokens = rearrange(spatial_tokens, 'b s f -> b s 1 f')

    #     # spatial tokens become the start tokens of the depth dimension
    #     tokens_with_depth_pos = F.pad(tokens_with_depth_pos, (0, 0, 0, 0, 0, 1), value = 0.)

    #     depth_tokens = torch.cat((spatial_tokens, tokens_with_depth_pos), dim = -2)

    #     depth_tokens = rearrange(depth_tokens, '... n d -> (...) n d')

    #     depth_tokens = self.depth_transformer(depth_tokens) # (b * (s+1), d+1, f)

    #     depth_tokens = rearrange(depth_tokens, '(b s) d f -> b s d f', b = b)

    #     # keep only the depth tokens corresponding to the last spatial token
    #     depth_tokens = depth_tokens[:, -1, :, :] # (b, d, f)
    #     print(f"forward_validation: depth_tokens shape after keeping only last spatial token: {depth_tokens.shape}")

    #     # NOTE: CHANGING THE DIM HERE IS IMPORTANT
    #     logits = torch.stack([layer(depth_tokens[:, i, :]) for i, layer in enumerate(self.to_logits)], dim=1) # (b, d, num_tokens)
    #     print(f"forward_validation: logits shape: {logits.shape}")
    #     preds = rearrange(logits, 'b d f -> (b d) f')
        
    #     labels = labels.view(-1) # (b * d,)

    #     # adapt labels to multi-head projection layer
    #     offset = torch.arange(self.depth_seq_len, device=labels.device) * self.num_tokens
    #     labels = labels - offset.repeat(b).to(labels.device) - self.num_special_tokens

    #     print(f"forward_validation: preds.shape: {preds.shape}")
    #     print(f"forward_validation: labels.shape: {labels.shape}")

    #     loss = F.cross_entropy(preds, labels, ignore_index = -100)
    #     return loss