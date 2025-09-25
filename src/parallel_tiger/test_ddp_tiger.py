import json
import os
import time as t
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CLEARML_LOG_LEVEL"] = "DEBUG"

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from torch.utils.data import DataLoader
from tqdm import tqdm
from clearml import Task

from transformers import T5ForConditionalGeneration, T5Tokenizer
from parallel_tiger.tokenizer.custom_tokenizer import CustomT5Tokenizer
from parallel_tiger.model.config import ModelConfig
from parallel_tiger.utils.io import ensure_dir
from parallel_tiger.utils.misc import set_seed
from parallel_tiger.utils.data_loading import (
    load_test_dataset,
)
from parallel_tiger.data.collator import TestCollator
from parallel_tiger.evaluation.metrics import get_topk_results, get_metrics_results
from parallel_tiger.generation.trie import Trie, prefix_allowed_tokens_fn

import hydra
from omegaconf import DictConfig, OmegaConf

import logging
logger = logging.getLogger(__name__)


def gather_list(target, world_size):

    target_gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(obj=target, object_list=target_gather_list)

    all_device_target = []
    for target_list in target_gather_list:
        all_device_target += target_list # type: ignore[reportOperatorIssue]

    return all_device_target


def test_ddp(cfg: DictConfig):

    set_seed(cfg.seed)
    ensure_dir(cfg.infer.output_dir)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)

    device_map = {"": local_rank}
    device = torch.device("cuda", local_rank)
    logger.debug(f"local_rank: {local_rank}, device: {device}")
    logger.debug(f"device_map: {device_map}")

    task = None
    if cfg.infer.enable_clearml and local_rank == 0 and hasattr(cfg, 'project_name') and hasattr(cfg.infer, 'experiment_name'):
        # Try to fetch the associated training task
        train_tasks = []
        try:
            train_tasks = Task.get_tasks(
                project_name=cfg.project_name,
                task_name=f"^{cfg.infer.experiment_name}$", # exact matching
                task_filter={"status": ["completed"], "order_by": ["-last_update"]},
            )
        except Exception as e:
            logger.error(f"Error fetching training tasks: {e}")
        task = Task.init(
            project_name=cfg.project_name,
            task_name=cfg.infer.experiment_name,
            task_type=Task.TaskTypes.inference,
            reuse_last_task_id=False,
        )
        if train_tasks:
            task.set_parent(train_tasks[0])
        task.connect(OmegaConf.to_container(cfg))
    else:
        sys.modules["clearml"] = None # type: ignore[reportArgumentType]

    if cfg.custom_tokenizer:
        tokenizer = CustomT5Tokenizer.from_pretrained(
            cfg.infer.ckpt_dir,
        ) 
    else:
        tokenizer = T5Tokenizer.from_pretrained(
            cfg.infer.ckpt_dir,
        )
    tokenizer.pad_token_id = 0
    special_tokenizer_tokens_num = len(tokenizer.special_tokens_map)

    model_config = ModelConfig.load(cfg.output_dir)
    model_config.update_config_to_inference_mode(cfg.infer, device_map)
    logger.info(f"Model config: {model_config}")
    model = T5ForConditionalGeneration.from_pretrained(
        cfg.infer.ckpt_dir,
        device_map=device_map
    )

    if cfg.infer.debug:
        cfg.infer.test_batch_size = 3
        cfg.infer.num_beams = 5

    test_data = load_test_dataset(cfg)
    all_items = test_data.get_all_items()

    # # TODO: PUT THAT IN A FUNCTION (and call it elsewhere?)
    # all_items_tok_split = [parse_item(item) for item in all_items]
    # num_first_tokens = len(set(item[0] for item in all_items_tok_split))
    # logger.debug(f"Number of different 1st tokens: {num_first_tokens}")
    # num_1_2 = len(set((item[0], item[1]) for item in all_items_tok_split))
    # num_1_2_3 = len(set((item[0], item[1], item[2]) for item in all_items_tok_split))
    # num_1_2_3_4 = len(set((item[0], item[1], item[2], item[3]) for item in all_items_tok_split))
    # logger.debug(f"Mean number of 2nd tokens: {num_1_2 / num_first_tokens:.2f} ({num_1_2}/{num_first_tokens})")
    # logger.debug(f"Mean number of 3rd tokens: {num_1_2_3 / num_1_2:.2f} ({num_1_2_3}/{num_1_2})")
    # logger.debug(f"Mean number of 4th tokens: {num_1_2_3_4 / num_1_2_3:.2f} ({num_1_2_3_4}/{num_1_2_3})")

    collator = TestCollator(cfg, tokenizer)
    logger.info("len all items: {}".format(len(all_items)))
    logger.info("Number of special tokens in tokenizer: {}".format(special_tokenizer_tokens_num))

    ddp_sampler = DistributedSampler(
        test_data, num_replicas=world_size, rank=local_rank, drop_last=True
    )

    candidate_trie = Trie(
        [[0] + tokenizer.encode(candidate) for candidate in all_items]
    )
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    model = DistributedDataParallel(model, device_ids=[local_rank])

    prompt_ids = [0]
    logger.info("TASK: {}".format(cfg.infer.test_task))
    test_data = load_test_dataset(cfg)

    test_loader = DataLoader(
        test_data,
        batch_size=cfg.infer.test_batch_size,
        collate_fn=collator,
        sampler=ddp_sampler,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
    )

    model.eval()

    all_outputs = []
    all_scores = []
    all_targets = []
    all_users = []

    save_dict = {}

    metrics = cfg.infer.metrics.split(",")
    all_prompt_results = []
    inference_time = []

    with torch.no_grad():

        for prompt_id in prompt_ids:

            if local_rank == 0:
                logger.info("Start prompt: {}".format(prompt_id))

            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                users = batch[2]
                bs = len(targets)
                num_beams = cfg.infer.num_beams

                start = t.perf_counter()
                output = model.module.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=4,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                    do_sample=cfg.infer.do_sample,
                )
                torch.cuda.synchronize()
                end = t.perf_counter()
                inference_time.append(end - start)

                output_ids = output["sequences"]  # (bs, num_beams, seq_len)
                scores = output["sequences_scores"]  # (bs, num_beams)

                # # Flatten the first two dimensions
                # # to ensure compatibility with MQL4GRec's original implementation
                # output_ids = output_ids.view(
                #     -1, output_ids.shape[-1]
                # )  # (bs * num_beams, seq_len)
                # scores = scores.view(-1)  # (bs * num_beams,)

                output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                if cfg.infer.debug:
                    if (
                        local_rank == 0
                    ):
                        logger.info("scores b: {}".format(scores.reshape(bs, num_beams)))
                        logger.info("output b: {}".format(np.array(output).reshape(bs, num_beams)))
                        logger.info("targets b: {}".format(targets))

                    if step > 10:
                        break

                output = gather_list(output, world_size)
                scores = gather_list(scores.cpu().tolist(), world_size)
                targets = gather_list(targets, world_size)
                users = gather_list(users, world_size)

                all_outputs.extend(output)
                all_scores.extend(scores)
                all_targets.extend(targets)
                all_users.extend(users)

                save_dict["all_outputs"] = all_outputs
                save_dict["all_scores"] = all_scores
                save_dict["all_targets"] = all_targets
                save_dict["all_users"] = all_users

                if local_rank == 0:
                    topk_res = get_topk_results(
                        output,
                        scores,
                        targets,
                        num_beams,
                        all_items=all_items if cfg.infer.filter_items else None,
                    )
                    batch_metrics_res = get_metrics_results(topk_res, metrics)
                    for m, res in batch_metrics_res.items():
                        if m not in metrics_results:
                            metrics_results[m] = res
                        else:
                            metrics_results[m] += res

                    total += len(targets)
                    if (step + 1) % 50 == 0:
                        temp = {}
                        for m in metrics_results:
                            temp[m] = metrics_results[m] / total
                        logger.info("Metrics results: {}".format(temp))

                dist.barrier()

            if local_rank == 0 and not cfg.infer.debug:
                for m in metrics_results:
                    metrics_results[m] = metrics_results[m] / total

                all_prompt_results.append(metrics_results)
                logger.info("======================================================")
                logger.info("Prompt {} results: {}".format(prompt_id, metrics_results))
                logger.info("======================================================")
                logger.info("")

                # --- ClearML: log per-prompt metrics ---
                if task is not None:
                    for m, val in metrics_results.items():
                        task.get_logger().report_scalar(
                            title=f"Prompt_{prompt_id}",
                            series=m,
                            value=val,
                            iteration=0
                        )

                with open(cfg.infer.save_file, "w") as f:
                    json.dump(save_dict, f, indent=4)

            dist.barrier()

    dist.barrier()

    if local_rank == 0 and not cfg.infer.debug:
        mean_results = {}
        min_results = {}
        max_results = {}

        for m in metrics:
            all_res = [_[m] for _ in all_prompt_results]
            mean_results[m] = sum(all_res) / len(all_res)
            min_results[m] = min(all_res)
            max_results[m] = max(all_res)

        logger.info("======================================================")
        logger.info("Mean results: {}".format(mean_results))
        logger.info("Min results: {}".format(min_results))
        logger.info("Max results: {}".format(max_results))
        logger.info("======================================================")

        save_data = {}
        save_data["test_prompt_ids"] = cfg.infer.test_prompt_ids
        save_data["mean_results"] = mean_results
        save_data["min_results"] = min_results
        save_data["max_results"] = max_results
        save_data["all_prompt_results"] = all_prompt_results

        with open(cfg.infer.results_file, "w") as f:
            json.dump(save_data, f, indent=4)
        logger.info("Save file: {}".format(cfg.infer.results_file))

        if task is not None:
            total_infer_time = sum(inference_time)
            logger.info(f"Total inference time (s): {total_infer_time:.2f}")
            logger.info(f"Number of inference calls: {len(inference_time)}")
            logger.info(f"Mean inference time per call (s): {total_infer_time / len(inference_time):.4f}")
            # --- ClearML: log aggregated metrics ---
            for m in metrics:
                task.get_logger().report_scalar("Mean Results", m, mean_results[m], iteration=0)
                task.get_logger().report_single_value(f"Mean_{m}", mean_results[m])
                task.get_logger().report_scalar("Min Results", m, min_results[m], iteration=0)
                task.get_logger().report_scalar("Max Results", m, max_results[m], iteration=0)
            # task.upload_artifact("evaluation_results", save_data) # comment line because it creates a deadlock
            # TODO: solve issue. See https://github.com/clearml/clearml-agent/issues/73 

            task.get_logger().report_single_value("Mean Inference Time (s)", total_infer_time)
            task.get_logger().report_single_value("Mean Inference Time per call (s)", total_infer_time / len(inference_time))

    return task

@hydra.main(
    version_base=None, 
    config_path="../../conf/tiger", 
    config_name="infer_config.yaml")
def main(cfg: DictConfig):
    logger.info("Current configuration:\n")
    logger.info(OmegaConf.to_yaml(cfg))
    t0 = t.time()
    task = test_ddp(cfg)
    inference_n_eval_time = t.time() - t0
    logger.info("Time taken for inference: {}".format(inference_n_eval_time))
    if task:
        task.get_logger().report_single_value("inference_n_eval_time", inference_n_eval_time)
        task.close()
        logger.info("ClearML task closed.")

if __name__ == "__main__":
    main()