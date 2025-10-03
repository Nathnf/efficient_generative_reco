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


from parallel_tiger.model.base_rq_transformer import safe_log_softmax

class AutoregressiveGenerateWithConstraints:
    def __init__(self, model, num_tokens=256, num_query=4, num_special_tokens=4):
        self.model = model
        self.num_tokens = num_tokens
        self.num_query = num_query
        self.num_special_tokens = num_special_tokens
        self.candidate_trie = None
        self.first_token_constraint_mask = None
        self.transition_constraint_masks = {1: None, 2: None}
        self.prefix_to_uidx_t3 = None
        self.uidx_to_next_tokens_t3 = None
        
    def set_candidate_trie(self, candidate_trie: Trie):
        self.candidate_trie = candidate_trie

    def _set_first_token_constraint_mask(self, first_token_constraint_mask):
        self.first_token_constraint_mask = first_token_constraint_mask.to(dtype=torch.bool)

    def _set_transition_constraint_masks(self, transition_mask_t1, transition_mask_t2):
        self.transition_constraint_masks = {
            1: transition_mask_t1.to(dtype=torch.bool),
            2: transition_mask_t2.to(dtype=torch.bool),
        }

    def _set_transition_constraints_fast_t3(self, prefix_to_uidx_t3, uidx_to_next_tokens_t3):
        self.prefix_to_uidx_t3 = prefix_to_uidx_t3.to(dtype=torch.long)
        self.uidx_to_next_tokens_t3 = uidx_to_next_tokens_t3.to(dtype=torch.bool)

    def set_fast_constraints(
        self,
        first_token_constraint_mask,
        transition_mask_t1,
        transition_mask_t2,
        prefix_to_uidx_t3,
        uidx_to_next_tokens_t3,
    ):
        self._set_first_token_constraint_mask(first_token_constraint_mask)
        self._set_transition_constraint_masks(transition_mask_t1, transition_mask_t2)
        self._set_transition_constraints_fast_t3(prefix_to_uidx_t3, uidx_to_next_tokens_t3)

    def _get_valid_mask(self, step, logits, flattened_beams=None, use_constraints=True):
        """Return an additive logits mask (0 for allowed, -inf for disallowed) at a decoding step.
        Uses specialized precomputed masks if available, otherwise falls back to the trie."""

        if not use_constraints:
            return torch.zeros_like(logits)

        device = logits.device

        if step == 0:
            if self.first_token_constraint_mask is not None:
                first_token_constraint_mask = self.first_token_constraint_mask.to(device=device)
                valid_tokens_mask = torch.zeros_like(logits).masked_fill(~first_token_constraint_mask, float("-inf"))
                return valid_tokens_mask
            else:
                assert self.candidate_trie is not None, "Trie needs to be set"
                valid_tokens_mask = torch.full_like(logits, float('-inf'))
                valid_next_tokens = self.candidate_trie.get([]) # type: ignore[OptionalMemberAccess]
                local_valid_next_tokens = [self._global_to_local_id(t, 0) for t in valid_next_tokens]
                valid_tokens_mask[:, local_valid_next_tokens] = 0.
                return valid_tokens_mask
            
        assert flattened_beams is not None
        # flat_tokens: (b*topK, step)

        def _global_to_local_beam_ids(global_ids, step, device):
            depth_idx = torch.arange(step, device=device)[None, :]  # (1, step)
            return global_ids - (self.num_special_tokens + depth_idx * self.num_tokens)

        if step in (1, 2) and self.transition_constraint_masks[step] is not None:
            mask = self.transition_constraint_masks[step].to(device=device) # (num_tokens, num_tokens) or (num_tokens, num_tokens, num_tokens)
            flattened_beams_loc = _global_to_local_beam_ids(flattened_beams, step, device)
            valid_mask = mask[flattened_beams_loc[:, 0]] if step == 1 else mask[flattened_beams_loc[:, 0], flattened_beams_loc[:, 1]]
            # print("valid_mask.shape, logits.shape:", valid_mask.shape, logits.shape)
            return torch.zeros_like(logits).masked_fill(~valid_mask, float("-inf"))

        elif step == 3 and self.prefix_to_uidx_t3 is not None and self.uidx_to_next_tokens_t3 is not None:
                # prefix_to_uidx_t3: (codebook_num, codebook_num, codebook_num) - transition mask for t=3
                # uidx_to_next_tokens_t3: (|U|, code_num) - valid next tokens for each unique prefix of length 3
                prefix_to_uidx_t3 = self.prefix_to_uidx_t3.to(device=device)
                uidx_to_next_tokens_t3 = self.uidx_to_next_tokens_t3.to(device=device)
                flattened_beams_loc = _global_to_local_beam_ids(flattened_beams, step, device)
                uidx = prefix_to_uidx_t3[flattened_beams_loc[:, 0], flattened_beams_loc[:, 1], flattened_beams_loc[:, 2]]
                valid_mask = uidx_to_next_tokens_t3[uidx]
                valid_tokens_mask = torch.zeros_like(logits).masked_fill(~valid_mask, float("-inf"))
                return valid_tokens_mask
        
        # Fallback: Trie
        assert self.candidate_trie is not None, "Trie needs to be set"
        valid_tokens_mask = torch.full_like(logits, float('-inf'))
        for beam_id, prefix in enumerate(flattened_beams.tolist()):
            valid_next_tokens = self.candidate_trie.get(prefix) # type: ignore[OptionalMemberAccess]
            # reverse offset
            valid_next_tokens = [self._global_to_local_id(t, step) for t in valid_next_tokens]
            valid_tokens_mask[beam_id, valid_next_tokens] = 0.

        return valid_tokens_mask

    # def _local_to_global_id(self, local_id, depth_idx):
    #     return local_id + depth_idx * self.num_tokens + self.num_special_tokens

    def _global_to_local_id(self, global_id, depth_idx):
        # return (global_id - self.num_special_tokens) % self.num_tokens
        return global_id - (self.num_special_tokens + depth_idx * self.num_tokens)

    def generate(self, input_ids, attention_mask, num_beams=20, do_sample=False):
        batch_size = input_ids.size(0)
        decoder_input_ids = torch.zeros(
            (batch_size, 1),
            dtype=torch.long,
            device=input_ids.device,
        )
        logits0 = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        ).logits  # (bs, seq_len, vocab_size)
        # print("step 0 - logits0.shape (before slicing):", logits0.shape)
        next_token_logits = logits0[:, -1, :]  # (bs, vocab_size)
        next_token_logits = next_token_logits[:, self.num_special_tokens: self.num_special_tokens + self.num_tokens]  # (bs, num_tokens)
        # print("next_token_logits.shape:", next_token_logits.shape)
        valid_mask = self._get_valid_mask(step=0, logits=next_token_logits, use_constraints=True)
        next_token_logits = next_token_logits + valid_mask
        if do_sample:
            probs = torch.softmax(next_token_logits, dim=-1)  # (bs, num_tokens)
            next_tokens = torch.multinomial(probs, num_samples=num_beams)  # (bs, num_beams)
            next_token_scores = safe_log_softmax(next_token_logits, dim=-1).gather(1, next_tokens)  # (bs, num_beams)
        else:
            log_probs = torch.log_softmax(next_token_logits, dim=-1)  # (bs, num_tokens)
            next_token_scores, next_tokens = torch.topk(log_probs, k=num_beams, dim=-1)  # both (bs, num_beams)

        next_tokens = next_tokens + self.num_special_tokens  # adjust for offset
        beam_scores = next_token_scores  # (bs, num_beams)
        beam_tokens = next_tokens.unsqueeze(-1)  # (bs, num_beams, 1)

        for step in range(1, self.num_query):
            flattened_beams = beam_tokens.view(-1, beam_tokens.size(-1))  # (bs * num_beams, seq_len)
            flat_scores = beam_scores.view(-1)  # (bs * num_beams,)

            # Prepare the decoder inputs
            decoder_input_ids = torch.cat(
                [
                    torch.zeros(
                        (batch_size, num_beams, 1),
                        dtype=torch.long,
                        device=input_ids.device,
                    ),
                    beam_tokens,
                ],
                dim=-1,
            )  # (bs, num_beams, seq_len)
            decoder_input_ids = decoder_input_ids.view(-1, decoder_input_ids.size(-1))  # (bs * num_beams, seq_len)
            decoder_attention_mask = torch.ones_like(decoder_input_ids)  # (bs * num_beams, seq_len)

            logits = self.model(
                input_ids=input_ids.unsqueeze(1).expand(-1, num_beams, -1).reshape(-1, input_ids.size(-1)),  # (bs * num_beams, seq_len)
                attention_mask=attention_mask.unsqueeze(1).expand(-1, num_beams, -1).reshape(-1, attention_mask.size(-1)),  # (bs * num_beams, seq_len)
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            ).logits  # (bs * num_beams, seq_len, vocab_size)
            # print(f"step {step} - logits.shape:", logits.shape)
            next_token_logits = logits[:, -1, :]  # (bs * num_beams, vocab_size)
            # print("next_token_logits.shape (before slicing):", next_token_logits.shape)
            next_token_logits = next_token_logits[:, self.num_special_tokens + step*self.num_tokens: self.num_special_tokens + (step+1)*self.num_tokens]  # (bs * num_beams, num_tokens)
            # print("next_token_logits.shape:", next_token_logits.shape)

            valid_mask = self._get_valid_mask(step=step, logits=next_token_logits, flattened_beams=flattened_beams, use_constraints=True)
            next_token_logits = next_token_logits + valid_mask

            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)  # (bs * num_beams, vocab_size)
                next_tokens = torch.multinomial(probs, num_samples=num_beams).squeeze(1)  # (bs * num_beams, num_beams)
                next_token_scores = safe_log_softmax(next_token_logits, dim=-1).gather(1, next_tokens)  # (bs * num_beams, num_beams)
            else:
                log_probs = torch.log_softmax(next_token_logits, dim=-1)  # (bs * num_beams, vocab_size)
                next_token_scores, next_tokens = torch.topk(log_probs, k=num_beams, dim=-1)  # both (bs * num_beams, num_beams)

            next_tokens = next_tokens + self.num_special_tokens + step * self.num_tokens  # adjust for offset

            # Update beam scores and tokens
            cand_scores = flat_scores[:, None] + next_token_scores  # (bs * num_beams, num_beams)
            cand_beams = torch.cat((flattened_beams[:, None, :].repeat(1, num_beams, 1), next_tokens[..., None]), dim=-1)  # (bs * num_beams, num_beams, step+1)

            # reshape to (bs, num_beams * num_beams)
            cand_scores = cand_scores.view(batch_size, -1)  # (bs, num_beams * num_beams)
            cand_beams = cand_beams.view(batch_size, -1, cand_beams.size(-1))  # (bs, num_beams * num_beams, step+1)

            # prune to get the new beam scores and tokens
            beam_scores, beam_idx = cand_scores.topk(num_beams, dim=-1)  # (b, num_beams)
            batch_idx = torch.arange(batch_size, device=beam_tokens.device)[:, None]
            beam_tokens = cand_beams[batch_idx, beam_idx]  # (b, num_beams, step+1)

        # print({"sequences": beam_tokens, "sequences_scores": beam_scores})
        # flatten the first two dimensions to ensure compatibility with MQL4GRec's original implementation
        return {"sequences": beam_tokens.view(-1, beam_tokens.shape[-1]), "sequences_scores": beam_scores.view(-1)}


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
            task_name=cfg.infer.experiment_name+cfg.infer.suffix,
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

    # # Manually compute all_items from the index json file
    # path = os.path.join(
    #     cfg.dataset.data_path,
    #     cfg.dataset.name,
    #     cfg.dataset.name + cfg.dataset.index_file
    # )
    # with open(path, "r") as f:
    #     index_data = json.load(f)
    #     all_items = set()
    #     for index in index_data.values():
    #         all_items.add("".join(index))
    # # all_items = test_data.get_all_items()
    # print("-N-N-N-N-N-N-, len(all_items): {}".format(len(all_items)))

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
    model = DistributedDataParallel(model, device_ids=[local_rank])

    candidate_trie = Trie(
        [[0] + tokenizer.encode(candidate) for candidate in all_items]
    )

    fast_generation = cfg.infer.fast_generation and cfg.custom_tokenizer
    if fast_generation:
        logger.info("Using fast generation with constraints.")
        gen_module = AutoregressiveGenerateWithConstraints(
            model = model.module,
            num_tokens=cfg.code_num,
            num_query=cfg.n_query,
            num_special_tokens=special_tokenizer_tokens_num,
        )
        gen_module.set_candidate_trie(candidate_trie)
        from parallel_tiger.generation.vectorized_constraints import compute_or_load_transition_constraints_codebook_fast
        (
            first_token_constraint_mask,
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
            num_special_tokenizer_tokens=special_tokenizer_tokens_num,
        )
        gen_module.set_fast_constraints(
            first_token_constraint_mask,
            transition_mask_t1,
            transition_mask_t2,
            prefix_to_uidx_t3,
            uidx_to_next_tokens_t3,
        )
    else:
        logger.info("Using standard generation with constraints.")
        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

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
            correct_pred_no_total, incorrect_pred_no_total = 0, 0

            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                users = batch[2]
                bs = len(targets)
                num_beams = cfg.infer.num_beams

                start = t.perf_counter()
                if fast_generation:
                    output = gen_module.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        num_beams=num_beams
                    )
                else:
                    output = model.module.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=4,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens if cfg.infer.use_constraints else None,
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

                output_ids = output["sequences"]  # ??? (bs, num_beams, seq_len)
                scores = output["sequences_scores"]  # ??? (bs, num_beams)

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
                    topk_res, correct_pred_no, incorrect_pred_no = get_topk_results(
                        output,
                        scores,
                        targets,
                        num_beams,
                        all_items=all_items,
                        filter_invalid=cfg.infer.filter_items,
                        per_level_stats=cfg.infer.per_level_stats
                    )
                    correct_pred_no_total += correct_pred_no
                    incorrect_pred_no_total += incorrect_pred_no

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

                # Correct vs incorrect predictions
                logger.info(f"Total correct predictions: {correct_pred_no_total}, Total incorrect predictions: {incorrect_pred_no_total}")
                logger.info(f"Ratio of correct predictions: {correct_pred_no_total / (correct_pred_no_total + incorrect_pred_no_total):.4f}")

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

