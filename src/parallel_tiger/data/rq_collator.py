import random
import torch
from parallel_tiger.model.config import TrainingMode

import logging
logger = logging.getLogger(__name__)



class BaseCollator(object):
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.n_query = cfg.n_query
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.masked_training = cfg.train.training_mode==TrainingMode.MASKED.value
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # logger.debug(self.tokenizer.model_max_length)
        assert cfg.dataset.max_his_len * self.n_query <= self.tokenizer.model_max_length, f"max_his_len {cfg.dataset.max_his_len} * n_query {self.n_query} > model_max_length {self.tokenizer.model_max_length}"

    def __call__(self, batch):
        raise NotImplementedError


class TrainCollator(BaseCollator):

    def __init__(self, cfg, tokenizer):
        super().__init__(cfg, tokenizer)
        self.masked_mix_prob = float(getattr(cfg.train, 'masked_mix_prob', 1.0))
        self.current_mask_num = None

    def set_current_mask_num(self, current_mask_num):
        """Used by training loop to control curriculum learning of masking"""
        self.current_mask_num = current_mask_num

    def __call__(self, batch):
        # logger.debug("batch:", batch)

        input_texts = [d["input_ids"] for d in batch]
        # logger.debug("input_texts:", input_texts)
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
        bs, seq_len = inputs["input_ids"].shape
        assert seq_len % self.n_query == 0, f"Sequence length {seq_len} not divisible by n_query {self.n_query}"
        n_items = seq_len // self.n_query
        inputs["input_ids"] = inputs["input_ids"].view(bs, n_items, self.n_query)
        inputs["attention_mask"] = inputs["attention_mask"].view(bs, n_items, self.n_query)

        if not self.masked_training or (random.random() > self.masked_mix_prob):
            # NOTE: return plain dict, BatchEncoding may drop non-tensor keys like None during collation
            return {
                "input_ids": inputs["input_ids"].view(bs, n_items, self.n_query),
                "attention_mask": inputs["attention_mask"].view(bs, n_items, self.n_query),
                "use_query_vectors_mask": None,
            }

        else:
            # For each item, choose a random number of tokens AND positions to mask
            # NOTE: 
            # - Mask means that the model should predict these tokens (True)
            # - Unmask means that the model is given the ground truth token (False). No loss should be computed for these tokens.
            tokens = inputs["input_ids"]  # (bs, n_items, n_query)
            bs, n_items, n_query = tokens.shape
            device = tokens.device

            use_query_vectors_mask = torch.zeros_like(tokens, dtype=torch.bool)  # (bs, n_items, n_query)

            if self.current_mask_num is None:
                mask_token_no = torch.randint(1, n_query + 1, (bs, n_items), device=device)  # number of tokens to mask in each sample: (bs, n_items)
            else:
                mask_token_no = torch.full((bs, n_items), self.current_mask_num, device=device, dtype=torch.long)  # (bs, n_items)

            # Generate random noise and argsort to get permutations
            noise = torch.rand(bs, n_items, n_query, device=device)  # (bs, n_items, n_query)
            perms = torch.argsort(noise, dim=-1)  # (bs, n_items, n_query)

            # Create a tensor of shape (bs, n_items, n_query) with range [0..n_query-1] for each item
            range_matrix = (
                torch.arange(n_query, device=device).view(1, 1, n_query)
            )  # (1, 1, n_query)

            # Compare with per-sample mask lengths (broadcasted); result is a boolean mask (bs, n_items, n_query)
            mask_positions = range_matrix < mask_token_no.unsqueeze(-1)  # True where we want to mask

            # Gather the permuted indices to select for each sample
            selected_indices = torch.where(mask_positions, perms, -1) # (bs, n_items, n_query)

            mask_indices = torch.zeros_like(tokens, dtype=torch.bool)  # (bs, n_items, n_query)
            valid = selected_indices != -1
            row_idx = torch.arange(bs, device=device).view(bs, 1, 1).expand_as(selected_indices)  # (bs, n_items, n_query)
            item_idx = torch.arange(n_items, device=device).view(1, n_items, 1).expand_as(selected_indices)  # (bs, n_items, n_query)

            mask_indices[row_idx[valid], item_idx[valid], selected_indices[valid]] = True
            use_query_vectors_mask[mask_indices] = True

            inputs["use_query_vectors_mask"] = use_query_vectors_mask

            return inputs



class ValidationCollator(BaseCollator):

    def __init__(self, cfg, tokenizer):
        super().__init__(cfg, tokenizer)
        self.masked_mix_prob = float(getattr(cfg.train, 'masked_mix_prob', 1.0))

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

        bs, seq_len = inputs["input_ids"].shape
        assert seq_len % self.n_query == 0, f"Sequence length {seq_len} not divisible by n_query {self.n_query}"
        n_items = seq_len // self.n_query
        inputs["input_ids"] = inputs["input_ids"].view(bs, n_items, self.n_query)
        inputs["attention_mask"] = inputs["attention_mask"].view(bs, n_items, self.n_query)

        if not self.masked_training or (random.random() > self.masked_mix_prob):
            # NOTE: return plain dict, BatchEncoding may drop non-tensor keys like None during collation
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": torch.where(
                    labels["input_ids"] == self.tokenizer.pad_token_id,
                    torch.tensor(-100, dtype=labels["input_ids"].dtype),
                    labels["input_ids"],
                ),
                "use_query_vectors_mask": None,
            }

        else:
            # For each label item, choose a random number of tokens AND positions to mask
            # NOTE: 
            # - Mask means that the model should predict these tokens (True)
            # - Unmask means that the model is given the ground truth token (False). No loss should be computed for these tokens.
            bs, n_query = labels["input_ids"].shape
            device = labels["input_ids"].device
            use_query_vectors_mask = torch.zeros((bs, n_query), dtype=torch.bool).to(
                device
            )  # (bs, n_query)

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

            use_query_vectors_mask[mask_indices] = True  # Set the mask to True

            inputs["labels"] = labels["input_ids"]
            inputs["labels"][inputs["labels"] == self.tokenizer.pad_token_id] = -100
            inputs["use_query_vectors_mask"] = use_query_vectors_mask

            return inputs


class TestCollator(BaseCollator):

    def __init__(self, cfg, tokenizer):
        super().__init__(cfg, tokenizer)

    def __call__(self, batch):
        input_texts = [d["input_ids"] for d in batch]
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
        bs, seq_len = inputs["input_ids"].shape
        assert seq_len % self.n_query == 0, f"Sequence length {seq_len} not divisible by n_query {self.n_query}"
        n_items = seq_len // self.n_query
        inputs["input_ids"] = inputs["input_ids"].view(bs, n_items, self.n_query)
        inputs["attention_mask"] = inputs["attention_mask"].view(bs, n_items, self.n_query)

        return (inputs, targets, users)
