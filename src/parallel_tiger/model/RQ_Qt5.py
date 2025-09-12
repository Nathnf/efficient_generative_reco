import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce, repeat
from parallel_tiger.generation.beam_search_decoding_rq import ParallelBeamSearchGenerator

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# helper classes

def FeedForward(*, dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        h, device = self.heads, x.device

        x = self.norm(x) # (b, n, inner_dim)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        mask_value = -torch.finfo(sim.dtype).max
        mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(mask, mask_value)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

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
        pad_id = 0
    ):
        super().__init__()
        self.dim = dim
        self.max_spatial_seq_len = max_spatial_seq_len
        self.depth_seq_len = depth_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.spatial_start_token = nn.Parameter(torch.randn(dim))

        self.spatial_pos_emb = nn.Embedding(max_spatial_seq_len + 1, dim) # account for a boundary case
        self.depth_pos_emb = nn.Embedding(depth_seq_len, dim)

        self.spatial_transformer = Transformer(
            dim = dim,
            layers = spatial_layers,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

        self.depth_transformer = Transformer(
            dim = dim,
            layers = depth_layers,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

        self.to_logits = nn.Linear(dim, num_tokens)
        self.pad_id = pad_id

    def generate(self, prime = None, filter_thres = 0.9, temperature = 1., default_batch_size = 1):
        total_seq_len = self.depth_seq_len * self.max_spatial_seq_len
        device = next(self.parameters()).device

        if not exists(prime):
            prime = torch.empty((default_batch_size, 0), dtype = torch.long, device = device)

        seq = prime

        for _ in range(total_seq_len - seq.shape[-1]):
            logits = self.forward(seq)[:, -1]
            logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(logits, dim = -1, temperature = temperature)
            seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim = -1)

        return rearrange(seq, 'b (s d) -> b s d', d = self.depth_seq_len)

    def forward_empty(self, batch_size):
        # take care of special case
        # where you sample from input of 0 (start token only)

        spatial_tokens = repeat(self.spatial_start_token, 'd -> b 1 d', b = batch_size)
        depth_tokens = self.spatial_transformer(spatial_tokens)
        depth_tokens = self.depth_transformer(depth_tokens)
        return self.to_logits(depth_tokens)

    def forward(self, ids, return_loss = False):
        assert ids.ndim in {2, 3}
        flattened_dim = ids.ndim == 2
        ids_orig_ndim = ids.ndim

        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0])

        if flattened_dim:
            # allow for ids to be given in the shape of (batch, seq)
            # in which case it will be auto-padded to the next nearest multiple of depth seq len
            seq_len = ids.shape[-1]
            padding = remainder_to_mult(seq_len, self.depth_seq_len)
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = rearrange(ids, 'b (s d) -> b s d', d = self.depth_seq_len)
        else:
            seq_len = ids.shape[1] * ids.shape[2]

        b, space, depth, device = *ids.shape, ids.device
        assert space <= (self.max_spatial_seq_len + 1), 'spatial dimension is greater than the max_spatial_seq_len set'
        assert depth == self.depth_seq_len, 'depth dimension must be equal to depth_seq_len'

        # get token embeddings

        tokens = self.token_emb(ids)

        spatial_pos = self.spatial_pos_emb(torch.arange(space, device = device))
        depth_pos = self.depth_pos_emb(torch.arange(depth, device = device))

        tokens_with_depth_pos = tokens + depth_pos

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions

        spatial_tokens = reduce(tokens_with_depth_pos, 'b s d f -> b s f', 'sum') + spatial_pos

        spatial_tokens = torch.cat((
            repeat(self.spatial_start_token, 'f -> b 1 f', b = b),
            spatial_tokens
        ), dim = -2)        

        spatial_tokens = self.spatial_transformer(spatial_tokens)

        spatial_tokens = rearrange(spatial_tokens, 'b s f -> b s 1 f')

        # spatial tokens become the start tokens of the depth dimension

        tokens_with_depth_pos = F.pad(tokens_with_depth_pos, (0, 0, 0, 0, 0, 1), value = 0.)

        depth_tokens = torch.cat((spatial_tokens, tokens_with_depth_pos), dim = -2)

        depth_tokens = rearrange(depth_tokens, '... n d -> (...) n d')

        depth_tokens = self.depth_transformer(depth_tokens)

        depth_tokens = rearrange(depth_tokens, '(b s) d f -> b s d f', b = b)

        logits = self.to_logits(depth_tokens)
        logits = rearrange(logits, 'b ... f -> b (...) f')
        logits = logits[:, :(seq_len + 1)]

        if not return_loss:
            logits = logits[:, 1:]

            if flattened_dim:
                return rearrange(logits, 'b ... n -> b (...) n')

            return logits

        logits = logits[:, :-1]
        
        preds = rearrange(logits, 'b ... c -> b c (...)')
        labels = rearrange(ids, 'b s d -> b (s d)')

        loss = F.cross_entropy(preds, labels, ignore_index = self.pad_id)
        return loss
    



import logging
logger = logging.getLogger(__name__)

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

    def _spatial_forward(self, ids, attention_mask):
        # require flattened input for compability with MQL4GRec data collator
        assert ids.ndim == 2 # ids: (b, spatial_seq_len * d)
        assert attention_mask.ndim == 2 # attention_mask: (b, spatial_seq_len * d)

        ids = rearrange(ids, 'b (s d) -> b s d', d = self.depth_seq_len) # (b, spatial_seq_len, d)
        attention_mask = rearrange(attention_mask, 'b (s d) -> b s d', d = self.depth_seq_len) # (b, spatial_seq_len, d)

        b, spatial_seq_len, depth, device = *ids.shape, ids.device
        assert spatial_seq_len <= (self.max_spatial_seq_len + 1), f'spatial dimension ({spatial_seq_len}) is greater than the max_spatial_seq_len set ({self.max_spatial_seq_len + 1})'
        assert depth == self.depth_seq_len, 'depth dimension must be equal to depth_seq_len'
        # # print(f"_spatial_forward: ids shape: {ids.shape}")
        # # print(f"_spatial_forward: ids: \n{ids}\n")

        # get token embeddings
        tokens = self.token_emb(ids) # (b, spatial_seq_len, d, f)
        # # print(f"_spatial_forward: tokens shape: {tokens.shape}")
        # # print(f"_spatial_forward: tokens: \n{tokens}\n")

        spatial_pos = self.spatial_pos_emb(torch.arange(spatial_seq_len, device = device))
        depth_pos = self.depth_pos_emb(torch.arange(depth, device = device))
        # # print(f"_spatial_forward: spatial_pos shape: {spatial_pos.shape}")
        # # print(f"_spatial_forward: spatial_pos: \n{spatial_pos}\n")
        # # print(f"_spatial_forward: depth_pos shape: {depth_pos.shape}")
        # # print(f"_spatial_forward: depth_pos: \n{depth_pos}\n")

        tokens_with_depth_pos = tokens + depth_pos
        # # print(f"_spatial_forward: tokens_with_depth_pos: \n{tokens_with_depth_pos}\n")
        #### logger.info(f"_spatial_forward: passed tokens + depth_pos")

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions
        spatial_tokens = reduce(tokens_with_depth_pos, 'b s d f -> b s f', 'sum') + spatial_pos 
        # s: spatial dim (seq len)
        # d: depth dim (depth_seq_len)
        # f: feature dim (embedding size)
        # # print(f"_spatial_forward: spatial_tokens shape before adding start token: {spatial_tokens.shape}")
        # # print(f"_spatial_forward: spatial_tokens before adding start token: \n{spatial_tokens}\n")

        spatial_attention_mask = attention_mask.any(dim = -1)
        # # print(f"_spatial_forward: spatial_attention_mask shape: {spatial_attention_mask.shape}")
        # # print(f"_spatial_forward: spatial_attention_mask: \n{spatial_attention_mask}\n")

        # replace every spatial pad token by the start token
        # spatial_tokens[~spatial_attention_mask] = self.spatial_start_token # NB: can't simply do that because we remove the shifting --> model not autoregressive anymore
        # Insert start token at the position of the last padding token for each batch element - also adapt attention mask
        spatial_tokens, spatial_attention_mask = self._insert_start_token(spatial_tokens, spatial_attention_mask, self.spatial_start_token) # (b, s+1, f), (b, s+1)
        # # print(f"_spatial_forward: spatial_tokens after inserting start token: \n{spatial_tokens}\n")
        # # print(f"_spatial_forward: spatial_attention_mask after inserting start token: \n{spatial_attention_mask}\n")
        #### logger.info(f"_spatial_forward: passed _insert_start_token")

        spatial_tokens = self.spatial_transformer(spatial_tokens, spatial_attention_mask) # (b, s+1, f)
        # # print(f"_spatial_forward: spatial_tokens shape after transformer: {spatial_tokens.shape}")
        # # print(f"_spatial_forward: spatial_tokens after transformer: \n{spatial_tokens}\n")
        #### logger.info(f"_spatial_forward: passed spatial_transformer")

        return spatial_tokens, b, spatial_seq_len

    def forward(self, ids, attention_mask):
        assert ids.numel() > 0, "Input ids cannot be empty"
        # # print(f"forward: ids device: {ids.device}")
        # # print(f"forward: attention_mask device: {attention_mask.device}")

        spatial_tokens, b, spatial_seq_len = self._spatial_forward(ids, attention_mask) # (b, s+1, f), int, int

        spatial_tokens = rearrange(spatial_tokens, 'b s f -> b s 1 f')

        depth_queries = repeat(self.depth_queries, 'd f -> b s d f', b = b, s=spatial_tokens.shape[1])

        depth_tokens = torch.cat((spatial_tokens, depth_queries), dim=2) # (b, s+1, 1+d, f)
        # # print(f"forward: depth_tokens shape before depth transformer: {depth_tokens.shape}")
        # # print(f"forward: depth_tokens before depth transformer: \n{depth_tokens}\n")

        depth_tokens = rearrange(depth_tokens, '... n d -> (...) n d')
        # # print(f"forward: depth_tokens shape before depth transformer (flattened): {depth_tokens.shape}")

        depth_tokens = self.depth_transformer(depth_tokens) # (b*(s+1), 1+d, f)
        # # print(f"forward: depth_tokens shape after depth transformer: {depth_tokens.shape}")
        # # print(f"forward: depth_tokens after depth transformer: \n{depth_tokens}\n")

        queries_out = depth_tokens[:, 1:, :] # (b*(s+1), d, f) 
        # # print(f"forward: queries_out shape after removing spatial token: {queries_out.shape}")
        # # print(f"forward: queries_out after removing spatial token: \n{queries_out}\n")

        queries_out = rearrange(queries_out, '(b s) d f -> b s d f', b = b)
        # # print(f"forward: queries_out shape after unflattening: {queries_out.shape}")

        logits = torch.stack([layer(queries_out[:,:,i,:]) for i, layer in enumerate(self.to_logits)], dim=2) # (b, s+1, d, num_tokens)
        # # print(f"forward: logits shape before rearranging: {logits.shape}")
        # # print(f"forward: logits before rearranging: \n{logits}\n")

        logits = logits[:, :spatial_seq_len, :, :] # remove logits corresponding to last item (no ground truth)
        # # print(f"forward: logits shape after removing last item: {logits.shape}")
        # # print(f"forward: logits after removing last item: \n{logits}\n")

        logits = rearrange(logits, 'b ... f -> b (...) f')
        # # print(f"forward: logits shape after rearranging: {logits.shape}")

        # preds = logits.view(-1, logits.size(-1)) # (b * seq_len, num_tokens) # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        preds = logits.reshape(-1, logits.size(-1)) # (b * seq_len, num_tokens)
        labels = ids.flatten() # (b * seq_len,)
        # # print(f"forward: preds shape: {preds.shape}")
        # # print(f"forward: labels shape: {labels.shape}")
        # # print(f"forward: preds: \n{preds}\n")
        # # print(f"forward: labels: \n{labels}\n")

        # adapt labels to multi-head projection layer
        offset = torch.arange(self.depth_seq_len, device=labels.device) * self.num_tokens
        labels = torch.where(
            labels==self.pad_id,
            -100,
            labels - offset.repeat(b*spatial_seq_len).to(labels.device) - self.num_special_tokens
        )
        # # print(f"forward: labels after offset correction: \n{labels}\n")

        # TODO: ADD CUSTOM LOSS COMPUTER (cf. T54Rec)
        loss = F.cross_entropy(preds, labels, ignore_index = -100)
        return loss

    def _get_logits_from_last_spatial_token(self, ids, attention_mask):
        spatial_tokens, b, _ = self._spatial_forward(ids, attention_mask) # (b, s+1, f)
        #### logger.info("_get_logits_from_last_spatial_token: passed _spatial_forward")

        last_spatial_token = spatial_tokens[:, -1, :] # (b, f) # only keep the last spatial token
        # # print(f"_get_logits_from_last_spatial_token: last_spatial_token shape: {last_spatial_token.shape}")
        # # print(f"_get_logits_from_last_spatial_token: last_spatial_token: \n{last_spatial_token}\n")
        #### logger.info("_get_logits_from_last_spatial_token: passed spatial_tokens[:, -1, :]")

        last_spatial_token = last_spatial_token[:, None, None, :]
        depth_queries = repeat(self.depth_queries, 'd f -> b 1 d f', b = b)
        depth_tokens = torch.cat((last_spatial_token, depth_queries), dim=2) # (b, 1, 1+d, f)
        # # print(f"_get_logits_from_last_spatial_token: depth_tokens shape before squeezing: {depth_tokens.shape}")
        # # print(f"_get_logits_from_last_spatial_token: depth_tokens before squeezing: \n{depth_tokens}\n")
        #### logger.info("_get_logits_from_last_spatial_token: passed depth_tokens cat")

        depth_tokens = depth_tokens.squeeze(1) # (b, 1+d, f)

        depth_tokens = self.depth_transformer(depth_tokens) # (b, d, f)
        # # print(f"_get_logits_from_last_spatial_token: depth_tokens shape after depth transformer: {depth_tokens.shape}")
        # # print(f"_get_logits_from_last_spatial_token: depth_tokens after depth transformer: \n{depth_tokens}\n")
        #### logger.info("_get_logits_from_last_spatial_token: passed depth_transformer")

        queries_out = depth_tokens[:, 1:, :] # (b, d, f)
        # # print(f"_get_logits_from_last_spatial_token: queries_out shape after removing spatial token: {queries_out.shape}")
        # # print(f"_get_logits_from_last_spatial_token: queries_out after removing spatial token: \n{queries_out}\n")
        #### logger.info("_get_logits_from_last_spatial_token: passed removing spatial token")

        logits = torch.stack([layer(queries_out[:,i,:]) for i, layer in enumerate(self.to_logits)], dim=1) # (b, d, num_tokens)
        # # print(f"_get_logits_from_last_spatial_token: logits shape before returning: {logits.shape}")
        # # print(f"_get_logits_from_last_spatial_token: logits before returning: \n{logits}\n")
        #### logger.info("_get_logits_from_last_spatial_token: passed logits computation")

        return logits, b
    
    def forward_inference(self, ids, attention_mask):
        logits, _ = self._get_logits_from_last_spatial_token(ids, attention_mask) # (b, d, num_tokens)
        return logits

    def forward_validation(self, ids, attention_mask, labels):
        # # print(f"forward: ids device: {ids.device}")
        # # print(f"forward: attention_mask device: {attention_mask.device}")
        # # print(f"forward: labels device: {labels.device}")
        logits, b = self._get_logits_from_last_spatial_token(ids, attention_mask) # (b, d, num_tokens)
        preds = logits.view(-1, logits.size(-1)) # (b * d, num_tokens)
        labels = labels.view(-1) # (b * d,)
        # # print(f"forward_validation: preds shape: {preds.shape}")
        # # print(f"forward_validation: labels shape: {labels.shape}")
        # # print(f"forward_validation: preds: \n{preds}\n")
        # # print(f"forward_validation: labels: \n{labels}\n")

        # adapt labels to multi-head projection layer
        offset = torch.arange(self.depth_seq_len, device=labels.device) * self.num_tokens
        labels = labels - offset.repeat(b).to(labels.device) - self.num_special_tokens
        # # print(f"forward_validation: labels after offset correction: \n{labels}\n")

        # TODO: ADD CUSTOM LOSS COMPUTER (cf. T54Rec)
        loss = F.cross_entropy(preds, labels, ignore_index = -100) # NB: there shouldn't be any padding in validation
        return loss

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
        use_constraints=True
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

    def training_step(self, batch, batch_idx):
        ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        loss = self.model(ids, attention_mask)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=self.distributed)
        return loss

    def validation_step(self, batch, batch_idx):
        ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
        loss = self.model.forward_validation(ids, attention_mask, labels)
        self.log("eval_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=self.distributed)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Example: linear warmup
        scheduler = transformers.get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=int(self.trainer.estimated_stepping_batches)
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def predict_step(self, batch, batch_idx):
        inputs, targets, users = batch
        ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

        #### logger.info(f"predict_step: ids device: {ids.device}")

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

