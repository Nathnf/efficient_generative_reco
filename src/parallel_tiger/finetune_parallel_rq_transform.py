import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # For detailed debugging information
import sys
import time
from typing import List
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping,
    ModelSummary,
    TQDMProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger

import hydra
from omegaconf import DictConfig, OmegaConf

import logging
from clearml import Task

from parallel_tiger.model.RQ_Qt5 import RQQTransformer, LitRQQTransformer
from parallel_tiger.utils.misc import (
    set_seed,
)
from parallel_tiger.utils.io import (
    ensure_dir,
)
from parallel_tiger.utils.data_loading_rq import (
    load_datasets,
)
from parallel_tiger.utils.logging_utils import (
    log_trainable_parameters,
)
from parallel_tiger.data.rq_collator import TrainCollator, ValidationCollator
from parallel_tiger.tokenizer.custom_tokenizer import (
    save_custom_vocab,
    load_custom_tokenizer,
)


logging.getLogger("fsspec").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# TODO: make it compatible with RQ model [LATER]
# def compute_metrics(eval_pred):
#     predictions, _ = eval_pred
#     # Assuming model returns codebook losses in hidden_states
#     # Extract per-codebook losses from predictions (actually model outputs)
#     if isinstance(predictions, tuple):
#         _, codebook_losses = predictions
#     else:
#         _ = predictions
#         codebook_losses = None  # Handle safely in case

#     metrics = {}
#     if codebook_losses is not None:
#         codebook_losses_array = np.array(codebook_losses)
#         codebook_losses = torch.from_numpy(codebook_losses_array).float().cpu()

#         n_query = 4  # number of codebooks
#         # Reshape: each row is one sample, each column is a codebook
#         codebook_losses = codebook_losses.view(-1, n_query)

#         # Compute mean per codebook
#         means = codebook_losses.mean(dim=0)

#         for i in range(n_query):
#             metrics[f"codebook_loss_{i+1}"] = means[i].item()

#     return metrics



def train(cfg: DictConfig):

    task, tb_logger = None, None
    if cfg.train.enable_clearml and hasattr(cfg, 'project_name') and hasattr(cfg.train, 'experiment_name'):
        task = Task.init(
            project_name=cfg.project_name, 
            task_name=cfg.train.experiment_name,
            reuse_last_task_id=False,
        )
        logger.info("ClearML task initialized.")
        logger.info(f"Task ID: {task.id}")
        task.connect(OmegaConf.to_container(cfg))

        tb_logger = TensorBoardLogger(
            save_dir=cfg.output_dir,
            name="lightning_logs"
        )
    else:
        sys.modules["clearml"] = None

    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)
    ddp = torch.cuda.device_count() > 1

    # local_rank = int(os.environ.get("LOCAL_RANK") or 0)

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

    model = RQQTransformer(
        num_tokens=cfg.code_num,
        dim=cfg.model.dim,
        max_spatial_seq_len=cfg.model.max_spatial_seq_len,
        depth_seq_len=cfg.n_query,
        spatial_layers=cfg.model.spatial_layers,
        depth_layers=cfg.model.depth_layers,
        dim_head=cfg.model.dim_head,
        heads=cfg.model.heads,
        attn_dropout=cfg.model.attn_dropout,
        ff_mult=cfg.model.ff_mult,
        ff_dropout=cfg.model.ff_dropout,
        pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        attention_type=cfg.model.attention_type,
        num_special_tokens=len(tokenizer.special_tokens_map)
    )

    for name, param in model.state_dict().items():
        try:
            if param.max() > 1e3:
                print(f"Layer: {name}, Parameters: {param}")
        except:
            print(f"Could not compute max for layer: {name}")

    train_data, valid_data = load_datasets(cfg)

    tokenizer.add_tokens(train_data.get_new_tokens()) # type: ignore[attr-defined]

    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        collate_fn=TrainCollator(cfg, tokenizer),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataloader.num_workers,
        # pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_data,
        shuffle=False,
        collate_fn=ValidationCollator(cfg, tokenizer),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataloader.num_workers,
        # pin_memory=True,
    )    

    # if local_rank == 0:
    logger.info("data num: {}".format(len(train_data)))
    logger.info("Tokenizer number of tokens: {}".format(len(tokenizer.get_vocab())))
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info("train sequence")
    logger.info("{}".format(train_data[100]))
    logger.info("val sequence")
    logger.info("{}".format(valid_data[100]))
    logger.info("{}".format(model))
    log_trainable_parameters(model)

    pl_module = LitRQQTransformer(
        model=model,
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        lr_scheduler_type=cfg.train.lr_scheduler,
        warmup_steps=cfg.train.warmup_steps,
        distributed=ddp
    )

    early_stopping = EarlyStopping(
        monitor="eval_loss",
        mode="min",
        patience=cfg.train.early_stopping_patience,
        verbose=False,
    )
    model_summary = ModelSummary(max_depth=4)
    checkpoint = ModelCheckpoint(
        save_top_k=1, 
        monitor="eval_loss",
        mode="min", 
        save_weights_only=True,
        dirpath=cfg.output_dir,
        filename="best-checkpoint"
    )
    progress_bar = TQDMProgressBar(refresh_rate=100)
    callbacks = [early_stopping, model_summary, checkpoint, progress_bar]
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="ddp" if ddp else "auto",
        max_epochs=cfg.train.max_epochs,
        accumulate_grad_batches=cfg.train.gradient_accumulation_steps,
        precision="16-mixed" if cfg.train.fp16 else "bf16-mixed" if cfg.train.bf16 else 32,
        logger=tb_logger,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        log_every_n_steps=cfg.train.logging_step,
        default_root_dir=cfg.output_dir,
        limit_val_batches=cfg.train.limit_val_batches,
        callbacks=callbacks,
        enable_checkpointing=True,
    )

    print("Total number of steps: ", trainer.estimated_stepping_batches)

    start_time = time.time()
    trainer.fit(
        pl_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    training_time = time.time() - start_time
    print('training_time', training_time)

    pl_module.load_state_dict(torch.load(checkpoint.best_model_path)["state_dict"])
    # Or?
    # ckpt_path = checkpoint.best_model_path
    # pl_module = LitRQQTransformer.load_from_checkpoint(ckpt_path, model=model, ...)

    if task is not None:
        task.get_logger().report_single_value('training_time', training_time)
        task.close()

    return trainer, pl_module


@hydra.main(
    version_base=None,
    config_path="../../conf/parallel_rq",
    config_name="train_config.yaml",
)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)
    logger.info("Current configuration:\n")
    logger.info(OmegaConf.to_yaml(cfg))
    _, _ = train(cfg)

if __name__ == "__main__":
    main()