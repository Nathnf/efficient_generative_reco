import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
from typing import Tuple, Optional
import numpy as np
import torch
from torch.distributed import is_initialized, get_rank
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping,
    ModelSummary,
    TQDMProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only # type: ignore[ReportPrivateImportUsage]

import hydra
from omegaconf import DictConfig, OmegaConf

import logging
from clearml import Task
from tqdm import tqdm

from parallel_tiger.model.RQ_Qt5 import RQQTransformer, LitRQQTransformer
from parallel_tiger.utils.misc import (
    set_seed,
)
from parallel_tiger.utils.io import (
    ensure_dir,
)
from parallel_tiger.utils.data_loading_rq import (
    load_datasets,
    load_test_dataset
)
from parallel_tiger.utils.logging_utils import (
    log_trainable_parameters,
)
from parallel_tiger.data.rq_collator import TrainCollator, ValidationCollator, TestCollator
from parallel_tiger.tokenizer.custom_tokenizer import (
    save_custom_vocab,
    load_custom_tokenizer,
    CustomT5Tokenizer
)
from parallel_tiger.generation.vectorized_constraints import compute_or_load_transition_constraints_codebook_fast
from parallel_tiger.generation.trie import Trie
from parallel_tiger.evaluation.metrics import get_metrics_results, get_topk_results


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


@rank_zero_only
def initialize_logging_task(cfg: DictConfig) -> Optional[Tuple[Task, TensorBoardLogger]]:
    if cfg.enable_clearml and hasattr(cfg, 'project_name') and hasattr(cfg, 'exp_name'):

        # # get job number from hydra config
        # hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        # print("hydra_cfg.job: ", hydra_cfg.job)
        # job_num = getattr(hydra_cfg.job, 'num', None)
        # if job_num is None:
        #     import random
        #     job_num = random.randint(1000, 9999)
        #     logger.warning("Hydra job number not found. Using random job number %s.", job_num)

        for version in range(100):
            task_name = f"{cfg.exp_name}_v{version}"
            existing_tasks = Task.get_tasks(project_name=cfg.project_name, task_name=f"^{task_name}$") # exact match
            if len(existing_tasks) == 0:
                job_num = version
                logger.info(f"Using version {job_num} for task {cfg.exp_name}")
                break
        else:
            import random
            job_num = random.randint(1000, 9999)
            logger.info(f"All versions 0-99 taken. Using random job number {job_num}.")

        task = Task.init(
            project_name=cfg.project_name,
            task_name=f"{cfg.exp_name}_v{job_num}",
            reuse_last_task_id=False,
        )
        logger.info("ClearML task initialized.")
        logger.info(f"Task ID: {task.id}")
        task.connect(OmegaConf.to_container(cfg))

        tb_logger = TensorBoardLogger(
            save_dir=cfg.output_dir,
            name="lightning_logs"
        )
        return task, tb_logger
    else:
        sys.modules["clearml"] = None # type: ignore[reportArgumentType]
        return None


def train(cfg: DictConfig):
    result = initialize_logging_task(cfg)
    task, tb_logger = result if result is not None else (None, None)

    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)
    ddp = torch.cuda.device_count() > 1

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
                logger.info(f"Layer: {name}, Parameters: {param}")
        except:
            logger.warning(f"Could not compute max for layer: {name}")

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
        batch_size=cfg.train.batch_size_eval,
        num_workers=cfg.dataloader.num_workers,
        # pin_memory=True,
    )    

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if local_rank == 0:
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
        distributed=ddp,
        topK=cfg.infer.num_beams, # IF NEEDED, OVERRIDE DURING INFERENCE
        use_constraints=cfg.infer.use_constraints, # IDEM
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
        inference_mode=False,
    )

    logger.info("Total number of steps: {}".format(trainer.estimated_stepping_batches))

    start_time = time.time()
    trainer.fit(
        pl_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    training_time = time.time() - start_time
    logger.info('training_time: {}'.format(training_time))

    pl_module.load_state_dict(torch.load(checkpoint.best_model_path)["state_dict"])
    # TODO: save model as well (or already done by checkpoint?)

    if task is not None:
        task.get_logger().report_single_value('training_time', training_time)

    return trainer, pl_module, tokenizer, task


def predict(
    cfg: DictConfig, 
    trainer: pl.Trainer, 
    pl_module: pl.LightningModule,
    tokenizer: CustomT5Tokenizer,
    task: Optional[Task]=None,
):
    test_data = load_test_dataset(cfg)
    logger.info(f"Test dataset size: {len(test_data)} sequences")
    all_items = test_data.get_all_items()
    logger.info("test sequence")
    logger.info("{}".format(test_data[100]))

    test_dataloader = DataLoader(
        test_data,
        shuffle=False,
        collate_fn=TestCollator(cfg, tokenizer),
        batch_size=cfg.infer.batch_size,
        num_workers=cfg.dataloader.num_workers,
        # pin_memory=True,
    )    

    (
        first_token_constraints_fast,
        transition_mask_t1,
        transition_mask_t2,
        prefix_to_uidx_t3,
        uidx_to_next_tokens_t3,
    ) = compute_or_load_transition_constraints_codebook_fast(
        cfg=cfg,
        tokenizer=tokenizer,
        all_items=all_items,
        first_token_constraints_path=cfg.infer.first_token_constraints_path,
        transition_constraints_t1_path=cfg.infer.transition_constraints_t1_path,
        transition_constraints_t2_path=cfg.infer.transition_constraints_t2_path,
        prefix_to_uidx_t3_path=cfg.infer.prefix_to_uidx_t3_path,
        uidx_to_next_tokens_t3_path=cfg.infer.uidx_to_next_tokens_t3_path,
        num_special_tokenizer_tokens=len(tokenizer.special_tokens_map),
    )
    pl_module.model.set_first_token_constraint_mask(first_token_constraints_fast)
    pl_module.model.set_transition_constraint_masks(transition_mask_t1, transition_mask_t2)
    pl_module.model.set_transition_constraints_fast_t3(prefix_to_uidx_t3, uidx_to_next_tokens_t3)

    candidate_trie = Trie([tokenizer.encode(candidate) for candidate in all_items])
    pl_module.model.set_candidate_trie(candidate_trie)

    start_time = time.time()
    local_preds = trainer.predict(pl_module, dataloaders=test_dataloader)
    print("device: ", pl_module.device, "len(preds): ", len(local_preds) if local_preds is not None else 0)
    inference_time = time.time() - start_time
    logger.info('exact inference time: %s (only trainer.predict)', inference_time)

    if task:
        task.get_logger().report_single_value('inference_time', inference_time)

    # gather predictions from all GPUs to rank 0
    gathered_preds = gather_predictions(local_preds)

    return gathered_preds, all_items


def gather_predictions(preds):
    """Gather predictions from all GPUs to rank 0."""
    if is_initialized():
        all_preds = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(all_preds, preds)
        # flatten list of lists
        all_preds = [p for sublist in all_preds for p in sublist]
    else:
        all_preds = preds
    return all_preds


def evaluate_predictions_gathered(predictions, tokenizer, all_items, pl_module, cfg):
    """
    Evaluate predictions returned by trainer.predict.
    Args:
        predictions: list of dicts, each from predict_step:
            {
              "preds": (bs, num_beams, seq_len),
              "scores": (bs, num_beams),
              "targets": list[str] or list[int],
              "users": list[str] or list[int],
            }
        tokenizer: custom tokenizer
        all_items: full item vocabulary
        cfg: config
    Returns:
        metrics_results: dict with aggregated results per metric
    """
    metrics = cfg.infer.metrics.split(",")
    metrics_results = {}
    total = 0

    for step, batch in enumerate(tqdm(predictions, desc="Evaluating", unit="batch")):
        output_ids = batch["preds"]           # (bs, num_beams, seq_len)
        scores = batch["scores"]       # (bs, num_beams)
        targets = batch["targets"]
        users = batch["users"]

        # Flatten beam dimension
        output_ids = output_ids.view(-1, output_ids.shape[-1])  # (bs*num_beams, seq_len)
        scores = scores.view(-1)                               # (bs*num_beams,)

        # Decode sequences -> list[str], move to CPU first
        output_ids = output_ids.cpu()
        decoded_outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        # Now compute top-k results & metrics
        topk_res = get_topk_results(
            predictions=decoded_outputs,
            scores=scores.cpu().tolist(),
            targets=targets,
            k=pl_module.topK,
            all_items=all_items if cfg.infer.filter_items else None,
        )

        batch_metrics_res = get_metrics_results(topk_res, metrics)

        # Accumulate results across batches
        for m, res in batch_metrics_res.items():
            metrics_results[m] = metrics_results.get(m, 0) + res

        total += len(targets)

        # Intermediate performance logging for large datasets
        if (step + 1) % 50 == 0:
            temp = {}
            for m in metrics_results:
                temp[m] = metrics_results[m] / total
            logger.info("Metrics results: {}".format(temp))

    # Normalize by number of targets
    logger.info(f"Total number of sequences (evaluation time): {total}")
    for m in metrics_results:
        metrics_results[m] /= total

    return metrics_results



@hydra.main(
    version_base=None,
    config_path="../../conf/parallel_rq",
    config_name="main.yaml",
)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)
    logger.info("Current configuration:\n")
    logger.info(OmegaConf.to_yaml(cfg))
    trainer, pl_module, tokenizer, task = train(cfg)
    preds, all_items = predict(cfg, trainer, pl_module, tokenizer, task)
    if get_rank() == 0:
        metrics = evaluate_predictions_gathered(preds, tokenizer, all_items, pl_module, cfg)

    if task:
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
            task.get_logger().report_single_value(key, value)
        task.close()
        logger.info("ClearML task closed.")

if __name__ == "__main__":
    main()


# NB: REFLECHIR A COMMENT JE PEUX TESTER PLUSIEURS CHOSES AVEC LE MEME ENTRAINEMENT (ex. greedy or stochastic sampling, different temperatures, etc.)
# IDEE QUI ME VIENT: 
# BOUCLE SUR LE NOMBRE DE PARAMETRES A TESTER / CONFIG D'INFERENCE A TESTER? --> AUTANT DE PREDICTIONS QUE NECESSAIRES