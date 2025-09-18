import torch
import torch.nn.functional as F
from torch import nn, einsum
from typing import Optional

from einops import rearrange, reduce, repeat
from parallel_tiger.generation.beam_search_decoding_rq import ParallelBeamSearchGenerator
from parallel_tiger.generation.trie import Trie   

import logging
logger = logging.getLogger(__name__)

# helper functions 

def safe_log_softmax(logits, dim=-1):
    row_invalid = (logits == float('-inf')).all(dim=dim)
    log_probs = F.log_softmax(logits, dim=dim)
    log_probs[row_invalid] = -float("inf")
    return log_probs

# helper classes

def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class BaseSelfAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def _apply_mask(self, sim, mask=None):
        raise NotImplementedError

    def forward(self, x, mask=None):
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = self._apply_mask(sim, mask)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CausalSelfAttention(BaseSelfAttention):
    def _apply_mask(self, sim, mask=None):
        i, j = sim.shape[-2:]
        device = sim.device
        causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
        if mask is not None: # padding mask
            causal_mask = causal_mask | ~mask[:, None, None, :]
        mask_value = -torch.finfo(sim.dtype).max
        return sim.masked_fill(causal_mask, mask_value)

class QFullSelfAttention(BaseSelfAttention):
    def _apply_mask(self, sim, mask=None):
        return sim

class QSparseSelfAttention(BaseSelfAttention):
    def _apply_mask(self, sim, mask=None):
        n = sim.shape[-1]
        mask_value = -torch.finfo(sim.dtype).max
        mask = torch.eye(n, dtype=torch.bool, device=sim.device)
        return sim.masked_fill(mask, mask_value)

class DecoderOnlyBlock(nn.Module):
    def __init__(self, attn, ff):
        super().__init__()
        self.attn = attn
        self.ff = ff

    def forward(self, x, mask=None):
        x = self.attn(x, mask=mask) + x
        x = self.ff(x) + x
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, dim, layers, attn_cls, ff_cls, dim_head, heads, attn_dropout=0., ff_dropout=0., ff_mult=4, attention_type=None):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderOnlyBlock(attn_cls(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout), ff_cls(dim=dim))
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)

# main class

class RQTransformer(nn.Module):
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
        num_special_tokens = 4,
        attention_type = None
    ):
        super().__init__()
        self.dim = dim
        self.max_spatial_seq_len = max_spatial_seq_len
        self.depth_seq_len = depth_seq_len
        self.num_tokens = num_tokens
        self.num_special_tokens = num_special_tokens

        # self.token_emb = nn.Embedding(num_tokens, dim)
        self.token_emb = nn.Embedding(num_tokens * depth_seq_len + num_special_tokens, dim)
        self.spatial_start_token = nn.Parameter(torch.randn(dim))

        self.spatial_pos_emb = nn.Embedding(max_spatial_seq_len + 1, dim) # account for a boundary case
        self.depth_pos_emb = nn.Embedding(depth_seq_len, dim)

        self.spatial_transformer = DecoderOnlyTransformer(
            dim = dim,
            layers = spatial_layers,
            attn_cls=CausalSelfAttention,
            ff_cls=FeedForward,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
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

        # self.to_logits = nn.Linear(dim, num_tokens)
        self.to_logits = nn.ModuleList(
            nn.Linear(dim, num_tokens, bias=False) for _ in range(depth_seq_len)
        ) # NOTE: THIS DOES A SEPARATE PROJECTION TO LOGITS FOR ALL DEPTH TOKENS
        self.pad_id = pad_id
        self.candidate_trie: Optional[Trie] = None

    def set_candidate_trie(self, candidate_trie: Trie):
        self.candidate_trie = candidate_trie

    def set_first_token_constraint_mask(self, first_token_constraint_mask):
        # placeholder
        pass

    def set_transition_constraint_masks(self, transition_mask_t1, transition_mask_t2):
        # placeholder
        pass

    def set_transition_constraints_fast_t3(self, prefix_to_uidx_t3, uidx_to_next_tokens_t3):
        # placeholder
        pass

    def _insert_start_token(self, spatial_tokens, attention_mask, start_token):
        # spatial_tokens: (b, s, f)
        # attention_mask: (b, s) - True for non-padding tokens
        # start_token: (f,)
        b, s, f = spatial_tokens.shape
        device = spatial_tokens.device
        is_padding = ~attention_mask
        last_padding_index = is_padding.sum(dim=-1)
        spatial_tokens_extended = torch.zeros((b, s+1, f), dtype=spatial_tokens.dtype, device=device)
        batch_dim = torch.arange(b, dtype=torch.long, device=device)
        spatial_tokens_extended[batch_dim, last_padding_index] = start_token.expand(b, f)
        is_padding_extended = torch.cat([is_padding, torch.zeros((b,1), dtype=torch.bool, device=device)], dim=1)
        spatial_tokens_r_extended = torch.cat([spatial_tokens, torch.zeros((b,1,f), device=device)], dim=1)
        spatial_tokens_extended = torch.where(
            is_padding_extended.unsqueeze(-1),
            spatial_tokens_r_extended,
            spatial_tokens_extended
        )
        is_not_padding_extended = torch.cat([torch.zeros((b,1), dtype=torch.bool, device=device), attention_mask], dim=1)
        spatial_tokens_l_extended = torch.cat([torch.zeros((b,1,f), device=device), spatial_tokens], dim=1)
        spatial_tokens_extended = torch.where(
            is_not_padding_extended.unsqueeze(-1),
            spatial_tokens_l_extended,
            spatial_tokens_extended
        )
        attention_mask_extended = torch.cat([attention_mask, torch.ones((b,1), dtype=torch.bool, device=device)], dim=1)
        return spatial_tokens_extended, attention_mask_extended

    def _spatial_forward(self, ids, attention_mask):
        # require flattened input for compability with MQL4GRec data collator
        assert ids.ndim == 3 # ids: (b, s, d)
        assert attention_mask.ndim == 3 # attention_mask: (b, s, d)

        b, spatial_seq_len, depth, device = *ids.shape, ids.device
        assert spatial_seq_len <= (self.max_spatial_seq_len + 1), f'spatial dimension ({spatial_seq_len}) is greater than the max_spatial_seq_len set ({self.max_spatial_seq_len + 1})'
        assert depth == self.depth_seq_len, 'depth dimension must be equal to depth_seq_len'

        # get token embeddings
        tokens = self.token_emb(ids) # (b, spatial_seq_len, d, f)

        spatial_pos = self.spatial_pos_emb(torch.arange(spatial_seq_len, device = device))
        depth_pos = self.depth_pos_emb(torch.arange(depth, device = device))

        tokens_with_depth_pos = tokens + depth_pos

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions
        spatial_tokens = reduce(tokens_with_depth_pos, 'b s d f -> b s f', 'sum') + spatial_pos 
        # s: spatial dim (seq len)
        # d: depth dim (depth_seq_len)
        # f: feature dim (embedding size)

        spatial_attention_mask = attention_mask.any(dim = -1)

        # Insert start token at the position of the last padding token for each batch element - also adapt attention mask
        spatial_tokens, spatial_attention_mask = self._insert_start_token(spatial_tokens, spatial_attention_mask, self.spatial_start_token) # (b, s+1, f), (b, s+1)

        spatial_tokens = self.spatial_transformer(spatial_tokens, spatial_attention_mask) # (b, s+1, f)

        return tokens_with_depth_pos, spatial_tokens, b, spatial_seq_len

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

        labels = ids.flatten() # (b * s * d,)

        # adapt labels to multi-head projection layer
        offset = torch.arange(self.depth_seq_len, device=labels.device) * self.num_tokens
        labels = torch.where(
            labels==self.pad_id,
            -100,
            labels - offset.repeat(b*spatial_seq_len).to(labels.device) - self.num_special_tokens
        )

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
        labels = labels.view(-1) # (b * d,)

        # adapt labels to multi-head projection layer
        offset = torch.arange(self.depth_seq_len, device=labels.device) * self.num_tokens
        labels = labels - offset.repeat(b).to(labels.device) - self.num_special_tokens

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


class RQQTransformer(nn.Module):
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
        attention_type = 'full',
        num_special_tokens = 4
    ):
        assert attention_type in {'full', 'sparse'}
        super().__init__()
        self.dim = dim
        self.max_spatial_seq_len = max_spatial_seq_len
        self.depth_seq_len = depth_seq_len
        self.num_tokens = num_tokens
        self.num_special_tokens = num_special_tokens

        # self.token_emb = nn.Embedding(num_tokens, dim)
        self.token_emb = nn.Embedding(num_tokens * depth_seq_len + num_special_tokens, dim)
        self.spatial_start_token = nn.Parameter(torch.randn(dim))

        self.spatial_pos_emb = nn.Embedding(max_spatial_seq_len + 1, dim) # account for a boundary case
        self.depth_pos_emb = nn.Embedding(depth_seq_len, dim)

        self.depth_queries = nn.Parameter(torch.randn(depth_seq_len, dim)) # learnable depth queries # depth_seq_len = number of queries

        self.spatial_transformer = DecoderOnlyTransformer(
            dim = dim,
            layers = spatial_layers,
            attn_cls=CausalSelfAttention,
            ff_cls=FeedForward,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

        self.depth_transformer = DecoderOnlyTransformer(
            dim = dim,
            layers = depth_layers,
            attn_cls=QFullSelfAttention if attention_type == 'full' else QSparseSelfAttention,
            ff_cls=FeedForward,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult,
        )

        # self.to_logits = nn.Linear(dim, num_tokens) # NOTE: THIS DOES A SHARED PROJECTION TO LOGITS FOR ALL DEPTH TOKENS
        self.to_logits = nn.ModuleList(
            nn.Linear(dim, num_tokens, bias=False) for _ in range(depth_seq_len)
        ) # NOTE: THIS DOES A SEPARATE PROJECTION TO LOGITS FOR ALL DEPTH TOKENS
        self.pad_id = pad_id
        self._setup_generation_components()
        self._check_insert_start_token()

    def set_first_token_constraint_mask(self, first_token_constraint_mask):
        self.first_token_constraint_mask = first_token_constraint_mask.to(dtype=torch.bool)

    def set_transition_constraint_masks(self, transition_mask_t1, transition_mask_t2):
        self.transition_constraint_masks = {
            1: transition_mask_t1.to(dtype=torch.bool),
            2: transition_mask_t2.to(dtype=torch.bool),
        }

    def set_transition_constraints_fast_t3(self, prefix_to_uidx_t3, uidx_to_next_tokens_t3):
        self.prefix_to_uidx_t3 = prefix_to_uidx_t3.to(dtype=torch.long)
        self.uidx_to_next_tokens_t3 = uidx_to_next_tokens_t3.to(dtype=torch.bool)

    def set_candidate_trie(self, candidate_trie):
        self.candidate_trie = candidate_trie

    def _setup_generation_components(self):
        self.generator = ParallelBeamSearchGenerator(
            model=self,
            use_multi_head=True,
            stochastic=False,   # LATER: NOT HARDCODE IT
            temperatures=None,  # IDEM
        )

    def _insert_start_token(self, spatial_tokens, attention_mask, start_token):
        # spatial_tokens: (b, s, f)
        # attention_mask: (b, s) - True for non-padding tokens
        # start_token: (f,)
        b, s, f = spatial_tokens.shape
        device = spatial_tokens.device
        is_padding = ~attention_mask
        last_padding_index = is_padding.sum(dim=-1)
        spatial_tokens_extended = torch.zeros((b, s+1, f), dtype=spatial_tokens.dtype, device=device)
        batch_dim = torch.arange(b, dtype=torch.long, device=device)
        spatial_tokens_extended[batch_dim, last_padding_index] = start_token.expand(b, f)
        is_padding_extended = torch.cat([is_padding, torch.zeros((b,1), dtype=torch.bool, device=device)], dim=1)
        spatial_tokens_r_extended = torch.cat([spatial_tokens, torch.zeros((b,1,f), device=device)], dim=1)
        spatial_tokens_extended = torch.where(
            is_padding_extended.unsqueeze(-1),
            spatial_tokens_r_extended,
            spatial_tokens_extended
        )
        is_not_padding_extended = torch.cat([torch.zeros((b,1), dtype=torch.bool, device=device), attention_mask], dim=1)
        spatial_tokens_l_extended = torch.cat([torch.zeros((b,1,f), device=device), spatial_tokens], dim=1)
        spatial_tokens_extended = torch.where(
            is_not_padding_extended.unsqueeze(-1),
            spatial_tokens_l_extended,
            spatial_tokens_extended
        )
        attention_mask_extended = torch.cat([attention_mask, torch.ones((b,1), dtype=torch.bool, device=device)], dim=1)
        return spatial_tokens_extended, attention_mask_extended
    
    def _insert_start_token_unvectorized(self, spatial_tokens, attention_mask, start_token):
        new_spatial_tokens = []
        b, s, _ = spatial_tokens.shape
        is_padding = ~attention_mask
        attention_mask_extended = torch.zeros((b, s + 1), dtype=torch.bool, device=spatial_tokens.device)
        for i in range(b):
            # get last padding index
            last_padding_index = is_padding[i].sum().item()
            new_spatial_tokens.append(
                torch.cat([
                    spatial_tokens[i, :last_padding_index],
                    start_token.unsqueeze(0),
                    spatial_tokens[i, last_padding_index:],
                ], dim=0)
            )
            attention_mask_extended[i, :last_padding_index] = attention_mask[i, :last_padding_index]
            attention_mask_extended[i, last_padding_index] = True  # start token position
            attention_mask_extended[i, last_padding_index + 1:] = attention_mask[i, last_padding_index:]
        new_spatial_tokens = torch.stack(new_spatial_tokens, dim=0)
        
        return new_spatial_tokens, attention_mask_extended

    def _check_insert_start_token(self):
        b, s, f = 256, 20, 128
        spatial_tokens = torch.randn(b, s, f)
        start_token = torch.zeros(f, device=spatial_tokens.device)
        import random
        attention_mask = torch.zeros((b, s), dtype=torch.bool, device=spatial_tokens.device)
        for i in range(b):
            num_paddings = random.randint(0, s-1)   # s-1 because we want at least one non-padding token
            attention_mask[i, -num_paddings:] = 1
        spatial_tokens_extended, attention_mask_extended = self._insert_start_token(spatial_tokens, attention_mask, start_token)
        spatial_tokens_extended_unvector, attention_mask_extended_unvector = self._insert_start_token_unvectorized(spatial_tokens, attention_mask, start_token)
        assert torch.equal(spatial_tokens_extended, spatial_tokens_extended_unvector), "_spatial_forward: _insert_start_token and _insert_start_token_unvectorized do not give the same result for spatial_tokens"
        assert torch.equal(attention_mask_extended, attention_mask_extended_unvector), "_spatial_forward: _insert_start_token and _insert_start_token_unvectorized do not give the same result for attention_mask"

    def _compute_loss_with_mask(
        self,
        preds,
        labels,
        use_query_vectors_mask,
        depth_seq_len,
        # spatial_seq_len=None,
    ):
        """
        Compute the cross-entropy loss with optional masking of certain query vectors.
        Adapted to both training case (b, s, d) and validation case (b, d).

        preds: (N, num_tokens) where N = b*d or b*s*d
        labels: (N,)
        use_query_vectors_mask: None, or (b, d) / (b, s, d)
        depth_seq_len: int
        spatial_seq_len: int or None (if present, use (b, s, d) mode)
        """

        if use_query_vectors_mask is not None:
            use_query_vectors_mask_flat = use_query_vectors_mask.flatten()
            labels = torch.where(
                use_query_vectors_mask_flat,
                labels,
                -100
            )

            # compute loss weights normalized per sample
            num_masked_per_sample = use_query_vectors_mask.sum(dim=-1)  # (b,) or (b, s)
            loss_weights = torch.where(
                use_query_vectors_mask,
                1.0 / num_masked_per_sample.unsqueeze(-1).float(),  # safe by construction
                0.0
            ).flatten()
        else:
            loss_weights = torch.ones_like(labels, dtype=torch.float32)

        loss = F.cross_entropy(
            preds, labels, ignore_index=-100, reduction="none"
        ) * loss_weights

        # if spatial_seq_len is None:
        #     # case (b, d)
        #     b = labels.shape[0] // depth_seq_len
        #     loss_per_codebook = loss.view(b, depth_seq_len).sum(dim=0)  # (d,)
        # else:
        #     # case (b, s, d)
        #     b = labels.shape[0] // (spatial_seq_len * depth_seq_len)
        #     loss_per_codebook = loss.view(b * spatial_seq_len, depth_seq_len).sum(dim=0)  # (d,)
        loss_per_codebook = loss.view(-1, depth_seq_len).sum(dim=0)  # (d,)
        norm_factor = preds.size(0) if use_query_vectors_mask is None else use_query_vectors_mask.sum().item()
        loss_per_codebook = loss_per_codebook / norm_factor

        return loss.mean(), loss_per_codebook

    def _spatial_forward(self, ids, attention_mask):
        # require flattened input for compability with MQL4GRec data collator
        assert ids.ndim == 3 # ids: (b, s, d)
        assert attention_mask.ndim == 3 # attention_mask: (b, s, d)

        b, spatial_seq_len, depth, device = *ids.shape, ids.device
        assert spatial_seq_len <= (self.max_spatial_seq_len + 1), f'spatial dimension ({spatial_seq_len}) is greater than the max_spatial_seq_len set ({self.max_spatial_seq_len + 1})'
        assert depth == self.depth_seq_len, 'depth dimension must be equal to depth_seq_len'

        # get token embeddings
        tokens = self.token_emb(ids) # (b, spatial_seq_len, d, f)

        spatial_pos = self.spatial_pos_emb(torch.arange(spatial_seq_len, device = device))
        depth_pos = self.depth_pos_emb(torch.arange(depth, device = device))

        tokens_with_depth_pos = tokens + depth_pos

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions
        spatial_tokens = reduce(tokens_with_depth_pos, 'b s d f -> b s f', 'sum') + spatial_pos 
        # s: spatial dim (seq len)
        # d: depth dim (depth_seq_len)
        # f: feature dim (embedding size)

        spatial_attention_mask = attention_mask.any(dim = -1)

        # Insert start token at the position of the last padding token for each batch element - also adapt attention mask
        spatial_tokens, spatial_attention_mask = self._insert_start_token(spatial_tokens, spatial_attention_mask, self.spatial_start_token) # (b, s+1, f), (b, s+1)

        spatial_tokens = self.spatial_transformer(spatial_tokens, spatial_attention_mask) # (b, s+1, f)

        return tokens_with_depth_pos, spatial_tokens, b, spatial_seq_len

    def forward(self, ids, attention_mask, use_query_vectors_mask=None):
        # if use_query_vectors_mask is not None:
        # --> boolean mask of shape (b, s, d), with:
        #   - True: masked --> model predicts this token and uses the query vector
        #   - False: unmasked --> model is given the ground truth token and does not compute a loss on this token

        assert ids.numel() > 0, "Input ids cannot be empty"

        tokens_with_depth_pos, spatial_tokens, b, spatial_seq_len = self._spatial_forward(ids, attention_mask) # (b, s, d, f), (b, s+1, f), int, int

        spatial_tokens = rearrange(spatial_tokens, 'b s f -> b s 1 f')

        depth_queries = repeat(self.depth_queries, 'd f -> b s d f', b = b, s=spatial_tokens.shape[1])

        if use_query_vectors_mask is not None:
            assert use_query_vectors_mask.shape == ids.shape, "use_query_vectors_mask must have the same shape as ids"
            # replace the tokens in ids where use_query_vectors_mask is False to ground truth tokens
            mask = use_query_vectors_mask.unsqueeze(-1)
            depth_queries_items = depth_queries[:, :-1, :, :] # (b, s, d, f) # remove last item (no ground truth)
            depth_queries_items = torch.where(mask, depth_queries_items, tokens_with_depth_pos)
            depth_queries = torch.cat((depth_queries_items, depth_queries[:, -1:, :, :]), dim=1) # (b, s+1, d, f) # add back last item (will be discarded later)
            # NOTE: Should we already discard the last item here (we don't have its ground truth) ? Or is it better to keep it for the depth transformer (more context) and discard it later ?

        depth_tokens = torch.cat((spatial_tokens, depth_queries), dim=2) # (b, s+1, 1+d, f)

        depth_tokens = rearrange(depth_tokens, '... n d -> (...) n d')

        depth_tokens = self.depth_transformer(depth_tokens) # (b*(s+1), 1+d, f)

        queries_out = depth_tokens[:, 1:, :] # (b*(s+1), d, f) 

        queries_out = rearrange(queries_out, '(b s) d f -> b s d f', b = b)

        logits = torch.stack([layer(queries_out[:,:,i,:]) for i, layer in enumerate(self.to_logits)], dim=2) # (b, s+1, d, num_tokens)

        logits = logits[:, :spatial_seq_len, :, :] # remove logits corresponding to last item (no ground truth)

        logits = rearrange(logits, 'b ... f -> b (...) f')

        # preds = logits.view(-1, logits.size(-1)) # (b * seq_len, num_tokens) # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        preds = logits.reshape(-1, logits.size(-1)) # (b * seq_len, num_tokens)
        labels = ids.flatten() # (b * seq_len,)

        # adapt labels to multi-head projection layer
        offset = torch.arange(self.depth_seq_len, device=labels.device) * self.num_tokens
        labels = torch.where(
            labels==self.pad_id,
            -100,
            labels - offset.repeat(b*spatial_seq_len).to(labels.device) - self.num_special_tokens
        )

        loss, loss_per_codebook = self._compute_loss_with_mask(
            preds,
            labels,
            use_query_vectors_mask,
            self.depth_seq_len,
        )
        return loss, loss_per_codebook

    def _get_logits_from_last_spatial_token(self, ids, attention_mask, use_query_vectors_mask=None):
        tokens_with_depth_pos, spatial_tokens, b, _ = self._spatial_forward(ids, attention_mask) # (b, s, d, f), (b, s+1, f), int, int

        last_spatial_token = spatial_tokens[:, -1, :] # (b, f) # only keep the last spatial token

        last_spatial_token = last_spatial_token[:, None, None, :]
        depth_queries = repeat(self.depth_queries, 'd f -> b 1 d f', b = b)

        if use_query_vectors_mask is not None:
            # replace the tokens in ids where use_query_vectors_mask is False to ground truth tokens
            mask = use_query_vectors_mask[:, None, :, None] # (b, 1, d, 1)
            gt_embeddings = tokens_with_depth_pos[:, -1, :, :][:, None, :, :] # (b, 1, d, f) - ground truth embeddings corresponding to the last spatial token
            depth_queries = torch.where(mask, depth_queries, gt_embeddings)

        depth_tokens = torch.cat((last_spatial_token, depth_queries), dim=2) # (b, 1, 1+d, f)

        depth_tokens = depth_tokens.squeeze(1) # (b, 1+d, f)

        depth_tokens = self.depth_transformer(depth_tokens) # (b, d, f)

        queries_out = depth_tokens[:, 1:, :] # (b, d, f)

        logits = torch.stack([layer(queries_out[:,i,:]) for i, layer in enumerate(self.to_logits)], dim=1) # (b, d, num_tokens)

        return logits, b
    
    def forward_inference(self, ids, attention_mask):
        logits, _ = self._get_logits_from_last_spatial_token(ids, attention_mask) # (b, d, num_tokens)
        return logits

    def forward_validation(self, ids, attention_mask, labels, use_query_vectors_mask=None):
        if use_query_vectors_mask is not None:
            assert use_query_vectors_mask.shape == labels.shape, "use_query_vectors_mask must have the same shape as labels = (b, d)"

        logits, b = self._get_logits_from_last_spatial_token(ids, attention_mask, use_query_vectors_mask) # (b, d, num_tokens)
        preds = logits.view(-1, logits.size(-1)) # (b * d, num_tokens)
        labels = labels.view(-1) # (b * d,)

        # adapt labels to multi-head projection layer
        offset = torch.arange(self.depth_seq_len, device=labels.device) * self.num_tokens
        labels = labels - offset.repeat(b).to(labels.device) - self.num_special_tokens

        loss, loss_per_codebook = self._compute_loss_with_mask(
            preds,
            labels,
            use_query_vectors_mask,
            self.depth_seq_len,
        )
        return loss, loss_per_codebook

    def generate(self, ids, attention_mask, topK=20, use_constraints=True):
        logits = self.forward_inference(ids, attention_mask)
        return self.generator.generate(logits, topK, use_constraints)



import pytorch_lightning as pl
import transformers

class LitRQQTransformer(pl.LightningModule):
    def __init__(
        self, 
        model, 
        lr=1e-3, 
        weight_decay=1e-2, 
        lr_scheduler_type='linear', 
        warmup_steps=100, 
        distributed=True,
        topK=20,
        use_constraints=True,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_steps = warmup_steps
        self.distributed = distributed
        self.topK = topK
        self.use_constraints = use_constraints
        # self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler = transformers.get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=int(self.trainer.estimated_stepping_batches)
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def grad_norm(self, norm_type=2):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        return total_norm ** (1. / norm_type)

    def training_step(self, batch, batch_idx):
        ids, attention_mask, use_query_vectors_mask = batch["input_ids"], batch["attention_mask"], batch["use_query_vectors_mask"]
        loss, loss_per_codebook = self.model(ids, attention_mask, use_query_vectors_mask)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=self.distributed, batch_size=ids.size(0))
        # for i, l in enumerate(loss_per_codebook): # creates a graph per codebook on ClearML...
        #     self.log(f"train_loss_codebook_{i+1}", l, prog_bar=False, on_step=False, on_epoch=True, sync_dist=self.distributed)
        self.loss_per_codebook = loss_per_codebook # to be fetched by a custom ClearML callback
        return loss

    def validation_step(self, batch, batch_idx):
        ids, attention_mask, labels, use_query_vectors_mask = batch["input_ids"], batch["attention_mask"], batch["labels"], batch["use_query_vectors_mask"]
        loss, loss_per_codebook = self.model.forward_validation(ids, attention_mask, labels, use_query_vectors_mask)
        self.log("eval_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=self.distributed, batch_size=ids.size(0))
        # for i, l in enumerate(loss_per_codebook): # idem
        #     self.log(f"eval_loss_codebook_{i+1}", l, prog_bar=False, on_step=False, on_epoch=True, sync_dist=self.distributed)
        self.loss_per_codebook = loss_per_codebook # idem
        return loss

    def predict_step(self, batch, batch_idx):
        inputs, targets, users = batch
        ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

        output = self.model.generate(
            ids,
            attention_mask,
            self.topK,
            self.use_constraints
        ) # {"sequences": ..., "sequences_scores": ...}

        # Attach metadata for evaluation
        return {
            "preds": output["sequences"],
            "scores": output["sequences_scores"],
            "targets": targets,
            "users": users
        }

