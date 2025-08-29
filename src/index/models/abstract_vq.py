"""
Abstract base class for Vector Quantized Variational Autoencoders.

This module provides a common interface and shared functionality for different
types of VQ-VAE implementations (Residual VQ, Product VQ, etc.).
"""

import torch
from torch import nn
from torch.nn import functional as F
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from .layers import MLPLayers


class AbstractVQVAE(nn.Module, ABC):
    """
    Abstract base class for Vector Quantized Variational Autoencoders.
    
    This class provides common functionality for encoder-decoder architectures
    with vector quantization in the latent space. Subclasses must implement
    the specific quantization method.
    """
    
    def __init__(self,
                 in_dim: int = 768,
                 num_emb_list: List[int] = [256, 256, 256, 256],
                 e_dim: int = 64,
                 beta: float = 0.25,
                 quant_loss_weight: float = 1.0,
                 layers: List[int] = [2048,1024,512,256,128,64], # or [512, 256, 128]?!
                 dropout_prob: float = 0.0,
                 bn: bool = False,
                 loss_type: str = "mse",
                 kmeans_init: bool = False,
                 kmeans_iters: int = 100,
                 sk_epsilons: List[float] = [0, 0, 0.003, 0.01],
                 sk_iters: int = 100,
                 use_linear: int = 0):
        """
        Initialize the Abstract VQ-VAE.
        
        Args:
            in_dim: Input dimension
            num_emb_list: List of embedding sizes for each quantizer
            e_dim: Embedding dimension
            layers: Hidden layer dimensions for encoder/decoder
            dropout_prob: Dropout probability
            bn: Whether to use batch normalization
            loss_type: Type of reconstruction loss ('mse' or 'l1')
            quant_loss_weight: Weight for quantization loss
            kmeans_init: Whether to use k-means initialization
            kmeans_iters: Number of k-means iterations
            sk_epsilons: Sinkhorn-Knopp epsilons for each quantizer
            sk_iters: Number of Sinkhorn-Knopp iterations
            use_linear: Whether to use linear layers in quantizer
        """
        super(AbstractVQVAE, self).__init__()
        
        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.beta = beta
        self.quant_loss_weight = quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.use_linear = use_linear
        
        self._build_encoder_decoder()
        
        self.quantizer = self._build_quantizer()
    
    def _build_encoder_decoder(self) -> None:
        """Build the encoder and decoder networks."""
        # encoder: input_dim -> hidden_layers -> embedding_dim
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(
            layers=self.encode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn
        )
        
        # decoder: embedding_dim -> hidden_layers -> input_dim (reverse)
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(
            layers=self.decode_layer_dims,
            dropout=self.dropout_prob,
            bn=self.bn
        )
    
    @abstractmethod
    def _build_quantizer(self) -> nn.Module:
        """
        Build the vector quantizer.
        
        This method must be implemented by subclasses to create the specific
        type of quantizer (e.g., ResidualVectorQuantizer, ProductVectorQuantizer).
        
        Returns:
            The quantizer module
        """
        pass
    
    @abstractmethod
    def _get_quantizer_name(self) -> str:
        """
        Get the name of the quantizer for variable naming.
        
        Returns:
            String name of the quantizer (e.g., 'rq', 'pq')
        """
        pass

    def forward(self, x: torch.Tensor, use_sk: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VQ-VAE.
        
        Args:
            x: Input tensor
            use_sk: Whether to use Sinkhorn-Knopp regularization
            
        Returns:
            Tuple of (reconstructed_output, quantization_loss, indices)
        """
        # encode
        x_encoded = self.encoder(x)
        # quantize
        x_quantized, quant_loss, indices, distances = self.quantizer(x_encoded, use_sk=use_sk)
        # decode
        reconstructed = self.decoder(x_quantized)
        
        return reconstructed, quant_loss, indices
    
    @torch.no_grad()
    def get_indices(self, xs: torch.Tensor, use_sk: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get quantization indices without gradients.
        
        Args:
            xs: Input tensor
            use_sk: Whether to use Sinkhorn-Knopp regularization
            
        Returns:
            Tuple of (indices, distances)
        """
        x_encoded = self.encoder(xs)
        _, _, indices, distances = self.quantizer(x_encoded, use_sk=use_sk)
        return indices, distances
    
    def compute_loss(self, 
                    reconstructed: torch.Tensor, 
                    quant_loss: torch.Tensor, 
                    target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the total loss (reconstruction + quantization).
        
        Args:
            reconstructed: Reconstructed output from decoder
            quant_loss: Quantization loss from the quantizer
            target: Target tensor for reconstruction loss (if None, uses original input)
            
        Returns:
            Tuple of (total_loss, reconstruction_loss)
        """
        if target is None:
            raise ValueError("Target tensor must be provided for loss computation")
        
        # Compute reconstruction loss
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(reconstructed, target, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(reconstructed, target, reduction='mean')
        else:
            raise ValueError(f'Incompatible loss type: {self.loss_type}')
        
        # Total loss
        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        
        return loss_total, loss_recon
    
    # def encode(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Encode input to latent space.
        
    #     Args:
    #         x: Input tensor
            
    #     Returns:
    #         Encoded tensor
    #     """
    #     return self.encoder(x)
    
    # def decode(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Decode from latent space.
        
    #     Args:
    #         x: Latent tensor
            
    #     Returns:
    #         Decoded tensor
    #     """
    #     return self.decoder(x)
    
    # def quantize(self, x: torch.Tensor, use_sk: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Quantize encoded representations.
        
    #     Args:
    #         x: Encoded tensor
    #         use_sk: Whether to use Sinkhorn-Knopp regularization
            
    #     Returns:
    #         Tuple of (quantized_tensor, loss, indices, distances)
    #     """
    #     return self.quantizer(x, use_sk=use_sk)
    
    def get_quantizer_parameters(self) -> dict:
        """
        Get quantizer-specific parameters for serialization/configuration.
        
        Returns:
            Dictionary of quantizer parameters
        """
        return {
            'num_emb_list': self.num_emb_list,
            'e_dim': self.e_dim,
            'beta': self.beta,
            'quant_loss_weight': self.quant_loss_weight,
            'kmeans_init': self.kmeans_init,
            'kmeans_iters': self.kmeans_iters,
            'sk_epsilons': self.sk_epsilons,
            'sk_iters': self.sk_iters,
            'use_linear': self.use_linear
        }
    
    def get_model_config(self) -> dict:
        """
        Get complete model configuration.
        
        Returns:
            Dictionary of all model parameters
        """
        return {
            'in_dim': self.in_dim,
            'num_emb_list': self.num_emb_list,
            'e_dim': self.e_dim,
            'beta': self.beta,
            'quant_loss_weight': self.quant_loss_weight,
            'layers': self.layers,
            'dropout_prob': self.dropout_prob,
            'bn': self.bn,
            'loss_type': self.loss_type,
            'kmeans_init': self.kmeans_init,
            'kmeans_iters': self.kmeans_iters,
            'sk_epsilons': self.sk_epsilons,
            'sk_iters': self.sk_iters,
            'use_linear': self.use_linear,
            'quantizer_type': self._get_quantizer_name()
        }
