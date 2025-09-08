import datetime
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")
    return cur


def log_trainable_parameters(model):
    """
    Logs the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    logger.info(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


def log_embedding_tables(cfg, model, just_head_layer=False):
    """
    Logs the embedding tables of the model.
    Useful to verify that the initialization is correct (no infinite values).
    """
    if not just_head_layer:
        logger.debug(
            "model.t5_model.decoder.query_emb.weight.data (shape: {}): {}".format(
                model.t5_model.decoder.query_emb.weight.data.shape,
                model.t5_model.decoder.query_emb.weight.data,
            )
        )
        if cfg.model.has_relative_encoder_item_bias:
            logger.debug(
                "encoder item bias table (shape: {}): {}".format(
                    model.t5_model.encoder.block[0]
                    .layer[0]
                    .SelfAttention.relative_item_bias.weight.data.shape,
                    model.t5_model.encoder.block[0]
                    .layer[0]
                    .SelfAttention.relative_item_bias.weight.data,
                )
            )
        if cfg.model.has_relative_encoder_codebook_bias:
            logger.debug(
                "encoder codebook bias table (shape: {}): {}".format(
                    model.t5_model.encoder.block[0]
                    .layer[0]
                    .SelfAttention.codebook_relative_bias_table.shape,
                    model.t5_model.encoder.block[0]
                    .layer[0]
                    .SelfAttention.codebook_relative_bias_table,
                )
            )
        if cfg.model.has_relative_decoder_item_bias_sa:
            logger.debug(
                "decoder item bias table (self-attention) (shape: {}): {}".format(
                    model.t5_model.decoder.block[0]
                    .layer[0]
                    .SelfAttention.relative_item_bias.weight.data.shape,
                    model.t5_model.decoder.block[0]
                    .layer[0]
                    .SelfAttention.relative_item_bias.weight.data,
                )
            )
        if cfg.model.has_relative_decoder_codebook_bias_sa:
            logger.debug(
                "decoder codebook bias table (self-attention) (shape: {}): {}".format(
                    model.t5_model.decoder.block[0]
                    .layer[0]
                    .SelfAttention.codebook_relative_bias_table.shape,
                    model.t5_model.decoder.block[0]
                    .layer[0]
                    .SelfAttention.codebook_relative_bias_table,
                )
            )
        if cfg.model.has_relative_decoder_item_bias_ca:
            logger.debug(
                "decoder item bias table (cross-attention) (shape: {}): {}".format(
                    model.t5_model.decoder.block[0]
                    .layer[1]
                    .EncDecAttention.relative_item_bias.weight.data.shape,
                    model.t5_model.decoder.block[0]
                    .layer[1]
                    .EncDecAttention.relative_item_bias.weight.data,
                )
            )
        if cfg.model.has_relative_decoder_codebook_bias_ca:
            logger.debug(
                "decoder codebook bias table (cross-attention) (shape: {}): {}".format(
                    model.t5_model.decoder.block[0]
                    .layer[1]
                    .EncDecAttention.codebook_relative_bias_table.shape,
                    model.t5_model.decoder.block[0]
                    .layer[1]
                    .EncDecAttention.codebook_relative_bias_table,
                )
            )

    if hasattr(model.t5_model, "lm_heads"):
        for i, head in enumerate(model.t5_model.lm_heads):
            for j, layer in enumerate(head.net):
                if isinstance(layer, nn.Linear):
                    logger.debug(
                        "decoder lm head {} layer {} (shape: {}): {}".format(
                            i,
                            j,
                            layer.weight.data.shape,
                            layer.weight.data,
                        )
                    )
    else:
        logger.debug(
            "model.t5_model.lm_head.weight.data (shape: {}): {}".format(
                model.t5_model.lm_head.weight.data.shape,
                model.t5_model.lm_head.weight.data,
            )
        )
