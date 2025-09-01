"""
Configuration classes for the T5-based recommendation model.

This module contains all configuration dataclasses used throughout the model,
providing a centralized and type-safe way to manage model parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union
from enum import Enum


def validate_enum(name: str, value: str, enum_cls: type[Enum]) -> None:
    allowed = [m.value for m in enum_cls]
    if value not in allowed:
        raise ValueError(
            f"Invalid {name}={value}. Must be one of {allowed}"
        )


@dataclass
class BiasConfig:
    """Configuration for different types of attention biases."""
    has_relative_encoder_item_bias: bool = True
    has_relative_encoder_codebook_bias: bool = True
    has_relative_decoder_item_bias_sa: bool = True
    has_relative_decoder_codebook_bias_sa: bool = False
    has_relative_decoder_item_bias_ca: bool = False
    has_relative_decoder_codebook_bias_ca: bool = False


class TrainingMode(Enum):
    STANDARD = "standard"             # 1a: Unnormalized parallel training
    DEPTHWISE = "standard_depthwise"  # 1b: Depthwise weighted parallel training  
    MASKED = "masked"                 # 2: Masked/partial training
    AUTOREGRESSIVE = "autoregressive" # 3: Sequential autoregressive training


class NormalizationMethod(Enum):
    """Enumeration for loss normalization methods (masked training)."""
    UNNORMALIZED = "unnormalized"
    PER_SAMPLE = "per_sample"
    PER_BATCH = "per_batch"


@dataclass
class TrainingConfig:
    """Configuration for training behavior."""
    n_query: int = 4
    code_num: int = 256
    special_tokenizer_tokens_num: int = 4
    training_mode: str = TrainingMode.STANDARD.value
    normalization_method: str = NormalizationMethod.PER_SAMPLE.value  # Only relevant for masked mode
    weights_per_codebook: Optional[List[float]] = None  # Only used in depthwise mode
    
    def __post_init__(self):
        validate_enum("training_mode", self.training_mode, TrainingMode)
        validate_enum("normalization_method", self.normalization_method, NormalizationMethod)
        if self.training_mode == TrainingMode.DEPTHWISE.value and self.weights_per_codebook is None:
            self.weights_per_codebook = [1.0, 0.33154722, 0.19349867, 0.11130733]


class GenerationMode(Enum):
    """Configuration for generation behavior."""
    PARALLEL_BEAM_SEARCH = "parallel_beam_search"
    AUTOREGRESSIVE_BEAM_SEARCH = "autoregressive_beam_search"


class EncoderAggregation(Enum):
    FLATTEN = "flatten"         # the encoder sees every item token embedding
    SUM = "sum"                 # the encoder sees the sum of all item token embeddings
    MEAN = "mean"               # the encoder sees the mean of all item token embeddings
    CONCAT = "concat"           # the encoder sees the concatenation of all item token embeddings


@dataclass
class ModelConfig:
    """Main configuration for the T5 recommendation model."""
    n_query: int = 4
    code_num: int = 256
    special_tokenizer_tokens_num: int = 4
    is_pretrained_model: bool = True
    is_inference: bool = False
    base_model: str = ""
    cache_dir: str = ""
    device_map: Union[dict[str, int], str] = 'auto'
    bias_config: Optional[BiasConfig] = None
    training_config: Optional[TrainingConfig] = None
    generation_mode: str = GenerationMode.PARALLEL_BEAM_SEARCH.value
    mask_token_id: int = 3
    use_multi_head: bool = False
    encoder_aggregation: str = EncoderAggregation.SUM.value
    is_aggregate_tokens: bool = field(init=False)

    def __post_init__(self):
        if self.is_inference:
            validate_enum("generation_mode", self.generation_mode, GenerationMode)
        validate_enum("encoder_aggregation", self.encoder_aggregation, EncoderAggregation)
        if self.bias_config is None:
            self.bias_config = BiasConfig()
        if self.training_config is None:
            self.training_config = TrainingConfig()
        self.is_aggregate_tokens = self.encoder_aggregation != EncoderAggregation.FLATTEN.value

    def to_dict(self):
        return {
            "n_query": self.n_query,
            "code_num": self.code_num,
            "special_tokenizer_tokens_num": self.special_tokenizer_tokens_num,
            "is_pretrained_model": self.is_pretrained_model,
            "is_inference": self.is_inference,
            "base_model": self.base_model,
            "cache_dir": self.cache_dir,
            "device_map": self.device_map,
            "bias_config": self.bias_config,
            "training_config": self.training_config,
            "generation_mode": self.generation_mode,
            "mask_token_id": self.mask_token_id,
            "use_multi_head": self.use_multi_head,
            "encoder_aggregation": self.encoder_aggregation,
            "is_aggregate_tokens": self.is_aggregate_tokens,
        }


def create_config_from_hydra_cfg(
    cfg, 
    is_inference: bool = False, 
    is_pretrained_model: bool = True,
    device_map: Union[dict[str, int], str] = 'auto',
    tokenizer_special_tokens_num: int = 4,
) -> ModelConfig:
    """
    Create ModelConfig from Hydra configuration object.
    
    This is a convenience function to bridge between Hydra configs and the new configuration system.
    
    Args:
        cfg: Hydra configuration object
        is_inference: Whether this is for inference (affects base_model selection)
        is_pretrained_model: Whether to use a pretrained model. Use False to train a model from scratch.
        device_map: Device map for model parallelism
        tokenizer_special_tokens_num: Number of special tokens in tokenizer
        
    Returns:
        ModelConfig instance
    """
    # Create bias configuration
    bias_config = BiasConfig(
        has_relative_encoder_item_bias=cfg.has_relative_encoder_item_bias,
        has_relative_encoder_codebook_bias=cfg.has_relative_encoder_codebook_bias,
        has_relative_decoder_item_bias_sa=cfg.has_relative_decoder_item_bias_sa,
        has_relative_decoder_codebook_bias_sa=cfg.has_relative_decoder_codebook_bias_sa,
        has_relative_decoder_item_bias_ca=cfg.has_relative_decoder_item_bias_ca,
        has_relative_decoder_codebook_bias_ca=cfg.has_relative_decoder_codebook_bias_ca,
    )

    # Create training configuration
    if hasattr(cfg, 'train'):
        training_config = TrainingConfig(
            n_query=cfg.n_query,
            code_num=cfg.code_num,
            special_tokenizer_tokens_num=tokenizer_special_tokens_num,
            training_mode=getattr(cfg.train, 'training_mode', TrainingMode.STANDARD.value),
            normalization_method=getattr(cfg.train, 'normalization_method', NormalizationMethod.PER_SAMPLE.value),
            weights_per_codebook=getattr(cfg.train, 'weights_per_codebook', None)
        )
    else:
        training_config = TrainingConfig()
    
    # Create main configuration
    model_config = ModelConfig(
        n_query=cfg.n_query,
        code_num=cfg.code_num,
        special_tokenizer_tokens_num=tokenizer_special_tokens_num,
        is_pretrained_model=is_pretrained_model,
        is_inference=is_inference,
        base_model=cfg.base_model if not is_inference else cfg.infer.ckpt_dir,
        cache_dir=cfg.cache_dir,
        device_map=device_map,
        bias_config=bias_config,
        training_config=training_config,
        generation_mode=cfg.infer.generation_mode if is_inference else '',
        use_multi_head=cfg.use_multi_head,
        encoder_aggregation=cfg.encoder_aggregation,
    )

    return model_config