import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # For detailed debugging information
import sys
from typing import List
import numpy as np
import torch
import transformers

from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback

import hydra
from omegaconf import DictConfig, OmegaConf

import logging
from clearml import Task
from transformers.integrations.integration_utils import ClearMLCallback

from parallel_tiger.model.model_t5 import T54Rec
from parallel_tiger.model.config import (
    create_config_from_hydra_cfg
)
from parallel_tiger.utils.misc import (
    set_seed,
)
from parallel_tiger.utils.io import (
    ensure_dir,
)
from parallel_tiger.utils.data_loading import (
    load_datasets,
)
from parallel_tiger.utils.logging_utils import (
    log_trainable_parameters,
    log_embedding_tables,
)
from parallel_tiger.data.collator import Collator
from parallel_tiger.tokenizer.custom_tokenizer import (
    save_custom_vocab,
    load_custom_tokenizer,
)


logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Assuming model returns codebook losses in hidden_states
    # Extract per-codebook losses from predictions (actually model outputs)
    if isinstance(predictions, tuple):
        _, codebook_losses = predictions
    else:
        _ = predictions
        codebook_losses = None  # Handle safely in case

    metrics = {}
    if codebook_losses is not None:
        codebook_losses_array = np.array(codebook_losses)
        codebook_losses = torch.from_numpy(codebook_losses_array).float().cpu()

        n_query = 4  # number of codebooks
        # Reshape: each row is one sample, each column is a codebook
        codebook_losses = codebook_losses.view(-1, n_query)

        # Compute mean per codebook
        means = codebook_losses.mean(dim=0)

        for i in range(n_query):
            metrics[f"codebook_loss_{i+1}"] = means[i].item()

    return metrics



def train(cfg: DictConfig):

    task = None
    if cfg.train.enable_clearml and hasattr(cfg, 'project_name') and hasattr(cfg.train, 'experiment_name'):
        task = Task.init(
            project_name=cfg.project_name, 
            task_name=cfg.train.experiment_name,
            reuse_last_task_id=False,
        )
        logger.info("ClearML task initialized.")
        logger.info(f"Task ID: {task.id}")
        task.connect(OmegaConf.to_container(cfg))
    else:
        sys.modules["clearml"] = None

    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    if ddp:
        device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)

    logger.info(f"device_map: {device_map}")

    if not os.path.exists(os.path.join(cfg.output_dir, "custom_vocab.json")):
        logger.info("Creating and saving custom vocab...")
        save_custom_vocab(
            code_num=cfg.code_num,
            filename=os.path.join(cfg.output_dir, "custom_vocab.json"),
        )

    tokenizer = load_custom_tokenizer(
        filename=os.path.join(cfg.output_dir, "custom_vocab.json")
    )
    tokenizer.model_max_length = 512
    tokenizer.padding_side = "left"

    cfg.base_model = (
        cfg.load_model_name if cfg.load_model_name is not None else cfg.base_model
    )

    model_config = create_config_from_hydra_cfg(
        cfg,
        is_inference=False,
        is_pretrained_model=True,
        device_map=device_map,
        tokenizer_special_tokens_num=len(tokenizer.special_tokens_map),
    )
    model = T54Rec(model_config)

    if local_rank == 0:
        log_embedding_tables(cfg, model)

    train_data, valid_data = load_datasets(cfg)

    add_num = 0
    for dataset in train_data.datasets:
        add_num += tokenizer.add_tokens(dataset.get_new_tokens()) # type: ignore[attr-defined]

    collator = Collator(
        cfg,
        tokenizer
    )

    model.t5_model.resize_token_embeddings(len(tokenizer))
    model.t5_model.config.vocab_size = len(tokenizer)

    if local_rank == 0:
        log_embedding_tables(cfg, model, just_head_layer=True)

    if local_rank == 0:
        logger.info("add {} new token.".format(add_num))
        logger.info("data num: {}".format(len(train_data)))
        logger.info("Model Embedding shape: {}".format(model.t5_model.shared.weight.shape))
        logger.info("Tokenizer vocab map: {}".format(tokenizer.get_vocab()))
        tokenizer.save_pretrained(cfg.output_dir)
        model.t5_model.config.save_pretrained(cfg.output_dir)
        logger.info("train sequence")
        for dataset in train_data.datasets:
            logger.info("{}".format(dataset[100]))
        logger.info("val sequence")
        logger.info("{}".format(valid_data[100]))
        logger.info("{}".format(model))
        log_trainable_parameters(model)

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    early_stop = EarlyStoppingCallback(early_stopping_patience=cfg.train.patient)
    callbacks: List[TrainerCallback] = [early_stop]
    if task is not None:
        callbacks.append(ClearMLCallback())

    gradient_accumulation_steps = cfg.train.batch_size // cfg.train.micro_batch_size

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=cfg.seed,
            per_device_train_batch_size=cfg.train.micro_batch_size,
            per_device_eval_batch_size=cfg.train.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=cfg.train.warmup_ratio,
            num_train_epochs=cfg.train.num_epochs,
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            lr_scheduler_type=cfg.train.lr_scheduler,
            fp16=cfg.train.fp16,
            bf16=cfg.train.bf16,
            optim=cfg.train.optim,
            gradient_checkpointing=cfg.train.gradient_checkpointing,
            eval_strategy=cfg.train.save_and_eval_strategy if cfg.train.val_set_size > 0 else "no",
            logging_strategy=cfg.train.save_and_eval_strategy,
            save_strategy=cfg.train.save_and_eval_strategy,
            logging_steps=cfg.train.logging_step, # will be used only if `cfg.train.save_and_eval_strategy == "steps"`
            eval_steps=cfg.train.save_and_eval_steps, # idem
            save_steps=cfg.train.save_and_eval_steps, # idem
            output_dir=cfg.output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            # deepspeed=cfg.train.deepspeed,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=cfg.train.group_by_length,
            save_safetensors=False,
            report_to=None,
            eval_delay=1 if cfg.train.save_and_eval_strategy == "epoch" else 2*cfg.train.logging_step,
        ),
        compute_metrics=compute_metrics,
        data_collator=collator,
        callbacks=callbacks,
    )
    model.t5_model.config.use_cache = False

    trainer.train(
        resume_from_checkpoint=cfg.train.resume_from_checkpoint,
    )
    model.t5_model.save_pretrained(cfg.output_dir)

    if local_rank == 0:
        log_embedding_tables(cfg, model, suffix='after')


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="train_config.yaml",
)
def main(cfg: DictConfig):
    logger.info("Current configuration:\n")
    logger.info(OmegaConf.to_yaml(cfg))
    train(cfg)

if __name__ == "__main__":
    main()