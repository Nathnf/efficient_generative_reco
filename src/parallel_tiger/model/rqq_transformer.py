from parallel_tiger.model.base_rq_transformer import *
from parallel_tiger.generation.beam_search_decoding_rq import ParallelBeamSearchGenerator
from einops import repeat

class RQQTransformer(BaseRQTransformer):
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

        self.depth_queries = nn.Parameter(torch.randn(depth_seq_len, dim)) # learnable depth queries # depth_seq_len = number of queries

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

        self._setup_generation_components()

    def _setup_generation_components(self):
        self.generator = ParallelBeamSearchGenerator(
            model=self,
            use_multi_head=True,
            stochastic=False,   # LATER: NOT HARDCODE IT
            temperatures=None,  # IDEM
        )

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
        
        labels = self._adapt_labels_to_multi_head(ids)

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

        logits, _ = self._get_logits_from_last_spatial_token(ids, attention_mask, use_query_vectors_mask) # (b, d, num_tokens)
        preds = logits.view(-1, logits.size(-1)) # (b * d, num_tokens)
        
        labels = self._adapt_labels_to_multi_head(labels)

        loss, loss_per_codebook = self._compute_loss_with_mask(
            preds,
            labels,
            use_query_vectors_mask,
            self.depth_seq_len,
        )
        return loss, loss_per_codebook, logits

    def generate(self, ids, attention_mask, topK=20, use_constraints=True):
        logits = self.forward_inference(ids, attention_mask)
        return self.generator.generate(logits, topK, use_constraints)