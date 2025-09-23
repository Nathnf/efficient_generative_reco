import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
from typing import Tuple, Optional
import torch
from torch.distributed import is_initialized, get_rank
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping,
    ModelSummary,
    TQDMProgressBar,
    LearningRateMonitor,
    Callback
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only # type: ignore[ReportPrivateImportUsage]
from pytorch_lightning.tuner import Tuner

import hydra
from omegaconf import DictConfig, OmegaConf

import logging
from clearml import Task
from tqdm import tqdm

from parallel_tiger.model.base_rq_transformer import LitRQQTransformer
from parallel_tiger.model.rq_transformer import RQTransformer
from parallel_tiger.model.rqq_transformer import RQQTransformer
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



@rank_zero_only
def initialize_logging_task(cfg: DictConfig) -> Optional[Tuple[Task, TensorBoardLogger]]:
    if cfg.enable_clearml and hasattr(cfg, 'project_name') and hasattr(cfg, 'exp_name'):

        for version in range(1000):
            task_name = f"{cfg.exp_name}_v{version}"
            existing_tasks = Task.get_tasks(project_name=cfg.project_name, task_name=f"^{task_name}$") # exact match
            if len(existing_tasks) == 0:
                job_num = version
                logger.info(f"Using version {job_num} for task {cfg.exp_name}")
                break
        else:
            job_num = int(time.time()) % 10000 # ensures variability even when everything is seeded the same
            logger.info(f"All versions 0-999 taken. Using random job number {job_num}.")

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

class GradNormLogger(Callback):
    def on_after_backward(self, trainer, pl_module):
        assert trainer.logger is not None
        total_norm = pl_module.grad_norm(2)  # L2 norm
        trainer.logger.log_metrics({"grad_norm": total_norm}, step=trainer.global_step)


class ClearMLCodebookLogger(Callback):
    """
    Logs per-codebook losses to a single ClearML graph.
    Works for training and validation.
    Collects `loss_per_codebook` from the module at epoch end.
    """

    def __init__(self, title="codebook_loss", mode="train"):
        """
        Args:
            title (str): ClearML graph title
            mode (str): "train" or "val" for prefixing series
        """
        super().__init__()
        self.title = title
        self.mode = mode

    def on_train_epoch_end(self, trainer, pl_module):
        if self.mode != "train":
            return
        self._report_codebook_losses(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.mode != "val":
            return
        self._report_codebook_losses(trainer, pl_module)

    def _report_codebook_losses(self, trainer, pl_module):
        # Ensure ClearML task exists
        task = Task.current_task()
        if task is None:
            return

        if self.mode == "train":
            loss_per_codebook = getattr(pl_module, "train_codebook_epoch_loss", None)
        else:
            loss_per_codebook = getattr(pl_module, "val_codebook_epoch_loss", None)

        if loss_per_codebook is None:
            return
        
        # gather losses from all GPUs if distributed
        if isinstance(loss_per_codebook, list):
            # stack into (num_devices, num_codebooks)
            loss_per_codebook = torch.stack(loss_per_codebook, dim=0).mean(dim=0)

        if isinstance(loss_per_codebook, torch.Tensor):
            loss_per_codebook = loss_per_codebook.detach().cpu().numpy()

        for i, val in enumerate(loss_per_codebook):
            task.get_logger().report_scalar(
                title=f"{self.title}_per_epoch",
                series=f"{self.mode}_codebook_{i+1}",
                value=float(val),
                iteration=trainer.current_epoch,
            )

class CurriculumCallback(pl.Callback):
    def __init__(self, collator, n_query, start_epoch=0, end_epoch=20, task=None):
        self.collator = collator
        self.n_query = n_query
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.task_logger = task.get_logger() if task is not None else None

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # linear schedule: gradually increase from 1 to n_query
        res = min(1 + ((epoch - self.start_epoch) * self.n_query) // (self.end_epoch - self.start_epoch), self.n_query)
        current_mask_num = res if res > 0 else None
        self.collator.set_current_mask_num(current_mask_num)
        # logger.debug(f"[Curriculum] Epoch {epoch}: setting mask_num = {current_mask_num}")
        
        if self.task_logger is not None:
            value_to_report = current_mask_num if current_mask_num is not None else -1
            self.task_logger.report_scalar(
                title="Curriculum Mask", 
                series="mask_num",
                value=value_to_report, 
                iteration=epoch
            )


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

    model_cls = RQQTransformer if cfg.get("model_type", "rqqt").lower() == "rqqt" else RQTransformer
    logger.info(f"Using model class: {model_cls.__name__}")
    model = model_cls(
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
        logger.info("{}".format(train_data[min(100, len(train_data) - 1)]))
        logger.info("val sequence")
        logger.info("{}".format(valid_data[min(100, len(valid_data) - 1)]))
        logger.info("{}".format(model))
        log_trainable_parameters(model)
        # logger.debug("Model state dict at initialization: \n{}".format(model.state_dict()))

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
    lr_monitor = LearningRateMonitor(logging_interval='step')
    grad_norm_logger = GradNormLogger()
    codebook_logger_train = ClearMLCodebookLogger(title="train_codebook_loss", mode="train")
    codebook_logger_val = ClearMLCodebookLogger(title="val_codebook_loss", mode="val")
    callbacks = [early_stopping, model_summary, checkpoint, progress_bar, lr_monitor, grad_norm_logger, codebook_logger_train, codebook_logger_val]

    if cfg.train.get("enable_curriculum", False):
        assert cfg.train.training_mode == "masked", "Curriculum learning only makes sense with masked training"
        assert cfg.train.curriculum_end_epoch <= cfg.train.max_epochs, "Curriculum end epoch must be <= max epochs"
        logger.info("Curriculum learning enabled")
        # For now, curriculum starts at epoch 0
        curriculum_callback = CurriculumCallback(
            collator=train_dataloader.collate_fn, 
            n_query=cfg.n_query, 
            start_epoch=cfg.train.curriculum_start_epoch,
            end_epoch=cfg.train.curriculum_end_epoch,
            task=task,
        )
        callbacks.append(curriculum_callback)

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

    # logger.info("Total number of steps: {}".format(trainer.estimated_stepping_batches))

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        pl_module, 
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        min_lr=1e-6, 
        max_lr=1.0, 
        num_training=100,
    )
    fig = lr_finder.plot(suggest=True)
    fig.show()
    try:
        fig.savefig(os.path.join(cfg.output_dir, "lr_finder_plot.png"))
    except:
        pass
    new_lr = lr_finder.suggestion()
    logger.info("Suggested LR: %s", new_lr)
    pl_module.lr = new_lr

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

    # if local_rank == 0:
    #     logger.debug("Model state dict after training: \n{}".format(model.state_dict()))

    if task is not None:
        task.get_logger().report_single_value('learning_rate', new_lr if new_lr is not None else cfg.train.learning_rate)
        task.get_logger().report_single_value('best_eval_loss', checkpoint.best_model_score.item())
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
    logger.info("{}".format(test_data[min(100, len(test_data) - 1)]))

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
        all_preds = [p for sublist in all_preds for p in sublist] # type: ignore[reportGeneralTypeIssues]
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