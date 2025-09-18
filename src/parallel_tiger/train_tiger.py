import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # For detailed debugging information
import sys
import time
from typing import List
import torch
import transformers

from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback

import hydra
from omegaconf import DictConfig, OmegaConf

import logging
from clearml import Task

from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from parallel_tiger.model.config import (
    create_train_config_from_hydra_cfg
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
)
from parallel_tiger.data.collator import Collator
from parallel_tiger.tokenizer.custom_tokenizer import (
    save_custom_vocab,
    load_custom_tokenizer,
)


logger = logging.getLogger(__name__)


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
        sys.modules["clearml"] = None # type: ignore[reportArgumentType]

    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)

    if ddp:
        device_map = {"": local_rank}

    logger.info(f"device_map: {device_map}")

    if cfg.custom_tokenizer:
        if not os.path.exists(os.path.join(cfg.output_dir, "custom_vocab.json")):
            logger.info("Creating and saving custom vocab...")
            save_custom_vocab(
                code_num=cfg.code_num,
                filename=os.path.join(cfg.output_dir, "custom_vocab.json"),
            )

        tokenizer = load_custom_tokenizer(
            filename=os.path.join(cfg.output_dir, "custom_vocab.json")
        )
    else:
        tokenizer = T5Tokenizer.from_pretrained(cfg.base_model) # t5-small

    tokenizer.model_max_length = 512
    tokenizer.padding_side = "left"

    model_config = create_train_config_from_hydra_cfg(
        cfg,
        is_pretrained_model=False,
        device_map=device_map,
        tokenizer_special_tokens_num=len(tokenizer.special_tokens_map),
    )
    t5_config = T5Config(**model_config.t5_model_config.__dict__)
    model = T5ForConditionalGeneration(t5_config)

    train_data, valid_data = load_datasets(cfg)

    add_num = 0
    for dataset in train_data.datasets:
        add_num += tokenizer.add_tokens(dataset.get_new_tokens()) # type: ignore[attr-defined]

    collator = Collator(
        cfg,
        tokenizer
    )

    if not cfg.custom_tokenizer:
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)

    if local_rank == 0:
        logger.info("add {} new token.".format(add_num))
        logger.info("data num: {}".format(len(train_data)))
        logger.info("Model Embedding shape: {}".format(model.shared.weight.shape))
        logger.info("Tokenizer vocab map: {}".format(tokenizer.get_vocab()))
        tokenizer.save_pretrained(cfg.output_dir)
        model_config.save(cfg.output_dir)
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
    # # transformers does it automatically
    # if local_rank==0 and task is not None:
    #     callbacks.append(ClearMLCallback())

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
            max_steps=cfg.train.max_steps,
            warmup_steps=cfg.train.warmup_steps,
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
        data_collator=collator,
        callbacks=callbacks,
    )
    model.config.use_cache = False

    start_time = time.time()
    trainer.train(
        resume_from_checkpoint=cfg.train.resume_from_checkpoint,
    )
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Training time: {training_time} seconds")
    if task is not None and local_rank == 0:
        task.get_logger().report_single_value('training_time', training_time)

    model.save_pretrained(
        cfg.output_dir, 
        is_main_process=(local_rank == 0),
        safe_serialization=False, # if encountering problem with loading `lm_head` after training, disable safe_serialization. However, the issue should be fixed inside Q_t5 init method.
    )



@hydra.main(
    version_base=None,
    config_path="../../conf/tiger",
    config_name="train_config.yaml",
)
def main(cfg: DictConfig):
    logger.info("Current configuration:\n")
    logger.info(OmegaConf.to_yaml(cfg))
    train(cfg)

if __name__ == "__main__":
    main()