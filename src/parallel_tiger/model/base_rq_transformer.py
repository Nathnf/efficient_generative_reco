import torch
import torch.nn.functional as F
from torch import nn, einsum
from typing import Optional

from einops import rearrange, reduce

from parallel_tiger.generation.trie import Trie   

import pytorch_lightning as pl
import transformers

import logging
logger = logging.getLogger(__name__)

# helper functions 

def safe_log_softmax(logits, dim=-1):
    # prevent NaNs when all logits are -inf
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

class BaseRQTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_spatial_seq_len,
        depth_seq_len,
        spatial_layers,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        pad_id = 0,
        num_special_tokens = 4
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

        self.to_logits = nn.ModuleList(
            nn.Linear(dim, num_tokens, bias=False) for _ in range(depth_seq_len)
        ) # NOTE: THIS DOES A SEPARATE PROJECTION TO LOGITS FOR ALL DEPTH TOKENS

        self.pad_id = pad_id
        self.candidate_trie: Optional[Trie] = None

        self._check_insert_start_token()

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

    def _adapt_labels_to_multi_head(self, labels):
        labels = labels.view(-1)  # (b * s * d,) if training or (b * d,) if validation/inference
        outer_dim = labels.shape[0] // self.depth_seq_len  # b*s if training, b if validation/inference
        offset = torch.arange(self.depth_seq_len, device=labels.device) * self.num_tokens
        labels = torch.where(
            labels == self.pad_id,
            -100,
            labels - offset.repeat(outer_dim).to(labels.device) - self.num_special_tokens,
        )
        return labels


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

