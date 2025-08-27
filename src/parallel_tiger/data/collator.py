import torch
import copy
from transformers import LlamaTokenizer
from parallel_tiger.model.config import TrainingMode

import logging
logger = logging.getLogger(__name__)

class Collator(object):

    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.only_train_response = cfg.dataset.only_train_response
        self.tokenizer = tokenizer
        self.masked_training = cfg.train.training_mode==TrainingMode.MASKED.value
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # logger.debug(self.tokenizer.model_max_length)

    def __call__(self, batch):

        # logger.debug("batch:", batch)

        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]
        # logger.debug("input_texts:", input_texts)
        # logger.debug("label_texts:", label_texts)
        # logger.debug("max_length:", self.tokenizer.model_max_length)
        # logger.debug("pad_token_id:", self.tokenizer.pad_token_id)

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        labels = self.tokenizer(
            label_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        if not self.masked_training:
            inputs["labels"] = labels["input_ids"]
            inputs["labels"][inputs["labels"] == self.tokenizer.pad_token_id] = -100
            inputs["decoder_input_tokens"] = None
            inputs["decoder_input_use_query_vectors_mask"] = None

            # logger.debug(inputs.input_ids[0])
            # logger.debug(inputs.labels[0])

            return inputs

        else:
            # For each sample, choose a random number of tokens AND positions to mask
            # NOTE: Mask means that the model should predict these tokens
            bs, n_query = labels["input_ids"].shape
            device = labels["input_ids"].device
            decoder_input_use_query_vectors_mask = torch.zeros((bs, n_query), dtype=torch.bool).to(
                device
            )  # (bs, n_query)
            decoder_input_tokens = copy.deepcopy(labels["input_ids"])
            mask_token_no = torch.randint(1, n_query + 1, (bs,)).to(
                device
            )  # number of tokens to mask in each sample: (bs,)

            # Generate random noise and argsort to get permutations
            noise = torch.rand(bs, n_query)  # (bs, n_query)
            perms = torch.argsort(noise, dim=1)  # (bs, n_query)

            mask_indices = torch.zeros(
                (bs, n_query), dtype=torch.bool, device=device
            )  # (bs, n_query)

            # Create a tensor of shape (bs, n_query) with range [0..n_query-1] in each row
            range_matrix = (
                torch.arange(n_query, device=device).unsqueeze(0).expand(bs, -1)
            )  # (bs, n_query)

            # Compare with per-sample mask lengths (broadcasted); result is a boolean mask (bs, n_query)
            mask_positions = range_matrix < mask_token_no.unsqueeze(
                1
            )  # (bs, n_query), True where we want to mask

            # Gather the permuted indices to select for each sample
            selected_indices = torch.where(
                mask_positions, perms, torch.full_like(perms, fill_value=-1)
            )

            # Flatten and set selected indices to True (filtering -1s)
            row_indices = (
                torch.arange(bs, device=device).unsqueeze(1).expand(bs, n_query)
            )  # (bs, n_query)
            valid = selected_indices != -1

            mask_indices[row_indices[valid], selected_indices[valid]] = True
            decoder_input_tokens[mask_indices] = (
                self.tokenizer.mask_token_id
            )  # Set the masked tokens to mask token id
            decoder_input_use_query_vectors_mask[mask_indices] = True  # Set the mask to True

            inputs["labels"] = labels["input_ids"]
            inputs["labels"][inputs["labels"] == self.tokenizer.pad_token_id] = -100
            inputs["decoder_input_tokens"] = decoder_input_tokens
            inputs["decoder_input_use_query_vectors_mask"] = decoder_input_use_query_vectors_mask
            inputs["mask_token_no"] = mask_token_no

            return inputs


class TestCollator(object):

    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.prefix_token = getattr(cfg, "prefix_token", "")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

        if isinstance(self.tokenizer, LlamaTokenizer):
            # Allow batched inference
            self.tokenizer.padding_side = "left"

    def __call__(self, batch):

        input_texts = [d["input_ids"] + self.prefix_token for d in batch]
        targets = [d["labels"] for d in batch]
        users = [d["label"] for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        # logger.debug("self.prefix_token:", self.prefix_token)
        # logger.debug("inputs:", inputs)
        # logger.debug("targets:", targets)

        return (inputs, targets, users)
