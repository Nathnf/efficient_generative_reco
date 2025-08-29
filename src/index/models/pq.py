import torch
import torch.nn as nn
from .vq import VectorQuantizer


class ProductVectorQuantizer(nn.Module):
    """Product Quantization: split the input vector into L sub-vectors,
    quantize each sub-vector independently with its own codebook,
    and concatenate the results.
    """

    def __init__(
        self, 
        n_e_list, 
        e_dim, 
        beta,
        kmeans_init, 
        kmeans_iters, 
        sk_epsilons, 
        sk_iters,
        use_linear
        ):
        """
        Args:
            n_e_list: list of codebook sizes, one per subvector
            e_dim: total embedding dimension (will be split)
        """
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)

        # Each sub-quantizer handles a sub-dimension
        assert e_dim % self.num_quantizers == 0, "e_dim must be divisible by num_quantizers"
        self.sub_dim = e_dim // self.num_quantizers

        self.vq_layers = nn.ModuleList([
            VectorQuantizer(
                n_e, 
                self.sub_dim, 
                beta=beta,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                sk_epsilon=sk_epsilon,
                sk_iters=sk_iters,
                use_linear=use_linear
            )
            for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)
        ])

    def forward(self, x, use_sk=True):
        # x: (B, e_dim)
        B, D = x.shape
        assert D == self.e_dim

        all_losses = []
        all_indices = []
        all_distances = []
        all_quantized = []

        # Split along embedding dimension
        x_split = x.view(B, self.num_quantizers, self.sub_dim)

        for i, quantizer in enumerate(self.vq_layers):
            x_sub = x_split[:, i, :]                   # (B, sub_dim)
            x_q, loss, indices, dist = quantizer(x_sub, use_sk=use_sk)
            all_losses.append(loss)
            all_indices.append(indices)
            all_distances.append(dist)
            all_quantized.append(x_q)

        # Concatenate quantized subvectors back
        x_q = torch.cat(all_quantized, dim=-1)

        mean_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=1)  # (B, num_quantizers)
        all_distances = torch.stack(all_distances, dim=1)

        return x_q, mean_loss, all_indices, all_distances






# # More vectorized version - but not integrated into the RQ-VAE infrastructure
# import torch.nn.functional as F
# from sklearn.cluster import KMeans
# from einops import rearrange


# class ProductVectorQuantizer(torch.nn.Module):
#     def __init__(
#         self,
#         K,  # Number of centroids per codebook
#         L,  # Number of codebooks
#         e_dim,
#         beta=0.25,
#         kmeans_init=True,
#         kmeans_iters=100
#     ):
#         super().__init__()
#         assert e_dim % L == 0, "Embedding dimension must be divisible by the number of codebooks."
        
#         self.K = K
#         self.L = L
#         self.e_dim = e_dim
#         self.Ld = e_dim // L  # Dimension per codebook

#         self.beta = beta  # Commitment loss weight

#         self.kmeans_init = kmeans_init
#         self.kmeans_iters = kmeans_iters

#         self.codebooks = torch.nn.Parameter(
#             torch.zeros(self.L, self.K, self.Ld), requires_grad=True
#         )
#         self.initted = False

#     @property
#     def device(self):
#         return self.codebooks.device

#     def init_embs(self, data): 
#         # Data : B x L x Ld
#         print("Initializing codebooks with KMeans.")
#         for l in range(self.L): 
#             _d = data[:, l, :].cpu().detach().float().numpy()
#             cluster = KMeans(n_clusters=self.K, max_iter=self.kmeans_iters).fit(_d)
#             centers = cluster.cluster_centers_
#             tensor_centers = torch.from_numpy(centers).to(self.device)

#             self.codebooks[l].data.copy_(tensor_centers)
#         self.initted = True

#     def forward(self, x): 
#         # x : B x D
#         B, D = x.shape
#         x = rearrange(x, "b (l d) -> b l d", l=self.L, d=self.Ld)

#         if not self.initted and self.training: 
#             self.init_embs(x)

#         # Parallel forward through the codebooks
#         distances = (
#             torch.sum(x ** 2, dim=-1, keepdim=True) # B x L x 1
#             + torch.sum(self.codebooks ** 2, dim=-1)[None, :, :] # 1 x L x K
#             - 2 * torch.einsum("bld,lkd->blk", x, self.codebooks)
#         )  # B x L x K
#         indices = torch.argmin(distances, dim=-1)  # B x L

#         xq = self.codebooks[torch.arange(self.L, device=self.device).view(1, self.L), indices]  # B x L x Ld

#         commitment_loss = F.mse_loss(xq.detach(), x)
#         codebook_loss = F.mse_loss(xq, x.detach())
#         pq_loss = codebook_loss + self.beta * commitment_loss

#         xq = x + (xq - x).detach() # Straight-Through Estimator

#         xq = rearrange(xq, "b l d -> b (l d)", l=self.L, d=self.Ld)

#         return xq, pq_loss, indices, distances