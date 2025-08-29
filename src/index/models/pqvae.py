from .abstract_vq import AbstractVQVAE
from .pq import ProductVectorQuantizer


class PQVAE(AbstractVQVAE):
    """Product Vector Quantized Variational Autoencoder."""
    
    def _build_quantizer(self):
        """Build the Product Vector Quantizer."""
        return ProductVectorQuantizer(
            n_e_list=self.num_emb_list,
            e_dim=self.e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
            use_linear=self.use_linear
        )
    
    def _get_quantizer_name(self) -> str:
        """Get quantizer name for this implementation."""
        return 'pq'