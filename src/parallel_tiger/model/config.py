"""
Configuration classes for the T5-based recommendation model.

This module contains all configuration dataclasses used throughout the model,
providing a centralized and type-safe way to manage model parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union
from enum import Enum
import logging
import os
import pickle

logger = logging.getLogger(__name__)


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
class T5ModelConfig:
    """Configuration specific to T5 model architecture."""
    vocab_size: int = 1028
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_heads: int = 8
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    tie_word_embeddings: bool = False

    def __post_init__(self):
        self._validate_config()

    def _check_pos_int(self, value: int, name: str) -> None:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")

    def _check_non_neg_float(self, value: float, name: str) -> None:
        if not isinstance(value, float) or value < 0:
            raise ValueError(f"{name} must be a non-negative float, got {value}")

    def _validate_config(self):
        self._check_pos_int(self.vocab_size, "vocab_size")
        self._check_pos_int(self.d_model, "d_model")
        self._check_pos_int(self.d_kv, "d_kv")
        self._check_pos_int(self.d_ff, "d_ff")
        self._check_pos_int(self.num_layers, "num_layers")
        self._check_pos_int(self.num_heads, "num_heads")
        self._check_pos_int(self.relative_attention_num_buckets, "relative_attention_num_buckets")
        self._check_pos_int(self.relative_attention_max_distance, "relative_attention_max_distance")
        self._check_non_neg_float(self.dropout_rate, "dropout_rate")
        self._check_non_neg_float(self.layer_norm_epsilon, "layer_norm_epsilon")

        # NOTE: T5 doesn't need it (see low-level code, where they use `self.inner_dim = self.num_heads * self.d_kv` and then project from/to `d_model`)
        # if self.d_model != self.d_kv * self.num_heads:
        #     raise ValueError(
        #         f"d_model ({self.d_model}) must be equal to d_kv ({self.d_kv}) * num_heads ({self.num_heads})"
        #     )
        
        if not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {self.dropout_rate}")

        logger.info("✅ T5 config validation passed")


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
    t5_model_config: Optional[T5ModelConfig] = None
    stochastic_decoding: bool = False
    temperatures: Optional[List[float]] = None

    def __post_init__(self):
        if self.is_inference:
            validate_enum("generation_mode", self.generation_mode, GenerationMode)
        validate_enum("encoder_aggregation", self.encoder_aggregation, EncoderAggregation)
        if self.bias_config is None:
            self.bias_config = BiasConfig()
        if self.training_config is None:
            self.training_config = TrainingConfig()
        self.is_aggregate_tokens = self.encoder_aggregation != EncoderAggregation.FLATTEN.value
        if self.t5_model_config is None:
            self.t5_model_config = T5ModelConfig()

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
            "t5_model_config": self.t5_model_config,
            "stochastic_decoding": self.stochastic_decoding,
            "temperatures": self.temperatures,
        }

    def update_config_to_inference_mode(self, cfg_infer, device_map):
        self.is_inference = True
        self.is_pretrained_model = True
        # self.training_config = None
        self.base_model = cfg_infer.ckpt_dir
        self.generation_mode = cfg_infer.generation_mode
        self.device_map = device_map
        self.stochastic_decoding = cfg_infer.stochastic_decoding
        self.temperatures = cfg_infer.temperatures

    def save(self, folder_path: str):
        path = os.path.join(folder_path, "model_config.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, folder_path: str):
        path = os.path.join(folder_path, "model_config.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"ModelConfig file not found at {path}")
        
        with open(path, "rb") as f:
            return pickle.load(f)


# def create_config_from_hydra_cfg(
#     cfg, 
#     is_inference: bool = False, 
#     is_pretrained_model: bool = True,
#     device_map: Union[dict[str, int], str] = 'auto',
#     tokenizer_special_tokens_num: int = 4,
# ) -> ModelConfig:
#     """
#     Create ModelConfig from Hydra configuration object.
    
#     This is a convenience function to bridge between Hydra configs and the new configuration system.
    
#     Args:
#         cfg: Hydra configuration object
#         is_inference: Whether this is for inference (affects base_model selection)
#         is_pretrained_model: Whether to use a pretrained model. Use False to train a model from scratch.
#         device_map: Device map for model parallelism
#         tokenizer_special_tokens_num: Number of special tokens in tokenizer
        
#     Returns:
#         ModelConfig instance
#     """
#     # Create bias configuration
#     bias_config = BiasConfig(
#         has_relative_encoder_item_bias=cfg.model.has_relative_encoder_item_bias,
#         has_relative_encoder_codebook_bias=cfg.model.has_relative_encoder_codebook_bias,
#         has_relative_decoder_item_bias_sa=cfg.model.has_relative_decoder_item_bias_sa,
#         has_relative_decoder_codebook_bias_sa=cfg.model.has_relative_decoder_codebook_bias_sa,
#         has_relative_decoder_item_bias_ca=cfg.model.has_relative_decoder_item_bias_ca,
#         has_relative_decoder_codebook_bias_ca=cfg.model.has_relative_decoder_codebook_bias_ca,
#     )

#     # Create training configuration
#     if hasattr(cfg, 'train'):
#         training_config = TrainingConfig(
#             n_query=cfg.n_query,
#             code_num=cfg.code_num,
#             special_tokenizer_tokens_num=tokenizer_special_tokens_num,
#             training_mode=getattr(cfg.train, 'training_mode', TrainingMode.STANDARD.value),
#             normalization_method=getattr(cfg.train, 'normalization_method', NormalizationMethod.PER_SAMPLE.value),
#             weights_per_codebook=getattr(cfg.train, 'weights_per_codebook', None)
#         )
#     else:
#         training_config = TrainingConfig()

#     # Create T5 model configuration
#     t5_cfg = T5ModelConfig(
#         vocab_size=cfg.n_query*cfg.code_num + tokenizer_special_tokens_num,
#         d_model=cfg.model.d_model,
#         d_kv=cfg.model.d_kv,
#         d_ff=cfg.model.d_ff,
#         num_layers=cfg.model.num_layers,
#         num_heads=cfg.model.num_heads,
#         relative_attention_num_buckets=cfg.model.relative_attention_num_buckets,
#         relative_attention_max_distance=cfg.model.relative_attention_max_distance,
#         dropout_rate=cfg.model.dropout_rate,
#         layer_norm_epsilon=cfg.model.layer_norm_epsilon,
#         tie_word_embeddings=cfg.model.tie_word_embeddings and not cfg.model.use_multi_head,
#     )
#     # NOTE: tie_word_embeddings is critical here
#     # If True, the input and output embeddings share weights.
#     # This is incompatible with multiple projection heads, so we disable it in that case.
#     # For single head models:
#     # With tied embeddings, save_pretrained() does not store lm_head in model.safetensors
#     # (it is reconstructed from the embedding layer on load).
#     # If untied weights are desired, tie_word_embeddings must be set to False;
#     # otherwise lm_head will be missing when reloading from safetensors and will be randomly initialized.
#     logger.debug("tie_word_embeddings: %s", cfg.model.tie_word_embeddings and not cfg.model.use_multi_head)

#     # Create main configuration
#     model_config = ModelConfig(
#         n_query=cfg.n_query,
#         code_num=cfg.code_num,
#         special_tokenizer_tokens_num=tokenizer_special_tokens_num,
#         is_pretrained_model=is_pretrained_model,
#         is_inference=is_inference,
#         base_model=cfg.base_model if not is_inference else cfg.infer.ckpt_dir,
#         cache_dir=cfg.cache_dir,
#         device_map=device_map,
#         bias_config=bias_config,
#         training_config=training_config,
#         generation_mode=cfg.infer.generation_mode if is_inference else '',
#         use_multi_head=cfg.model.use_multi_head,
#         encoder_aggregation=cfg.model.encoder_aggregation,
#         t5_model_config=t5_cfg,
#     )

#     return model_config





def create_train_config_from_hydra_cfg(
    cfg, 
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
        has_relative_encoder_item_bias=cfg.model.has_relative_encoder_item_bias,
        has_relative_encoder_codebook_bias=cfg.model.has_relative_encoder_codebook_bias,
        has_relative_decoder_item_bias_sa=cfg.model.has_relative_decoder_item_bias_sa,
        has_relative_decoder_codebook_bias_sa=cfg.model.has_relative_decoder_codebook_bias_sa,
        has_relative_decoder_item_bias_ca=cfg.model.has_relative_decoder_item_bias_ca,
        has_relative_decoder_codebook_bias_ca=cfg.model.has_relative_decoder_codebook_bias_ca,
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

    # Create T5 model configuration
    t5_cfg = T5ModelConfig(
        vocab_size=cfg.n_query*cfg.code_num + tokenizer_special_tokens_num,
        d_model=cfg.model.d_model,
        d_kv=cfg.model.d_kv,
        d_ff=cfg.model.d_ff,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        relative_attention_num_buckets=cfg.model.relative_attention_num_buckets,
        relative_attention_max_distance=cfg.model.relative_attention_max_distance,
        dropout_rate=cfg.model.dropout_rate,
        layer_norm_epsilon=cfg.model.layer_norm_epsilon,
        tie_word_embeddings=cfg.model.tie_word_embeddings and not cfg.model.use_multi_head,
    )
    # NOTE: tie_word_embeddings is critical here
    # If True, the input and output embeddings share weights.
    # This is incompatible with multiple projection heads, so we disable it in that case.
    # For single head models:
    # With tied embeddings, save_pretrained() does not store lm_head in model.safetensors
    # (it is reconstructed from the embedding layer on load).
    # If untied weights are desired, tie_word_embeddings must be set to False;
    # otherwise lm_head will be missing when reloading from safetensors and will be randomly initialized.
    logger.debug("tie_word_embeddings: %s", cfg.model.tie_word_embeddings and not cfg.model.use_multi_head)

    # Create main configuration
    model_config = ModelConfig(
        n_query=cfg.n_query,
        code_num=cfg.code_num,
        special_tokenizer_tokens_num=tokenizer_special_tokens_num,
        is_pretrained_model=is_pretrained_model,
        is_inference=False,
        base_model=cfg.base_model,
        cache_dir=cfg.cache_dir,
        device_map=device_map,
        bias_config=bias_config,
        training_config=training_config,
        use_multi_head=cfg.model.use_multi_head,
        encoder_aggregation=cfg.model.encoder_aggregation,
        t5_model_config=t5_cfg,
    )

    return model_config