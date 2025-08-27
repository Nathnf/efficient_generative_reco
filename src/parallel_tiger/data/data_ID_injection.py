import random
import os
from torch.utils.data import Dataset
import json
import numpy as np
from collections import defaultdict
from typing import Union

import logging
logger = logging.getLogger(__name__)


def item_uid_to_token(item_uid: Union[str, int], n_tokens: int = 1) -> str:
    """
    Convert item unique ID to a token.
    E.g.: item_uid = "23" --> "<item_23>" (with n_tokens=1), or "<item_23><item_23><item_23><item_23>" (with n_tokens=4).
    """
    return f"<item_{item_uid}>" * n_tokens  # or f"#{item_uid}"


class BaseDataset(Dataset):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dataset = cfg.dataset.name
        self.data_path = os.path.join(cfg.dataset.data_path, self.dataset)

        self.max_his_len = cfg.dataset.max_his_len
        self.his_sep = cfg.dataset.his_sep
        self.index_file = cfg.dataset.index_file
        self.image_index_file = cfg.dataset.image_index_file
        self.add_prompt = cfg.dataset.add_prompt

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None

        self.indices = None
        self.image_indices = None
        self.cf_indices = None

        self.fg_image_indices = None

    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), "r") as f:
            self.inters = json.load(f)

        with open(
            os.path.join(self.data_path, self.dataset + self.index_file), "r"
        ) as f:
            self.indices = json.load(f)
        with open(
            os.path.join(self.data_path, self.dataset + self.image_index_file), "r"
        ) as f:
            image_indices = json.load(f)

        self.image_indices = {}
        for k, v in image_indices.items():
            if random.random() > 0.5:
                self.image_indices[k] = v

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        if self.indices is not None:
            for index in self.indices.values():
                for token in index:
                    self.new_tokens.add(token)

        if self.image_indices is not None:
            for index in self.image_indices.values():
                for token in index:
                    self.new_tokens.add(token)

        if self.fg_image_indices is not None:
            for index in self.fg_image_indices.values():
                for token in index:
                    self.new_tokens.add(token)

        # IF we use the ID indices, we need to add them as well
        if self.cf_indices is not None:
            for index in self.cf_indices.values():
                # Individual tokens
                self.new_tokens.add(index)

        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def get_all_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens

        if self.cfg.train.tasks == "seqrec":
            prefix_list = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>"]

        elif self.cfg.train.tasks == "seqimage":
            prefix_list = ["<A_{}>", "<B_{}>", "<C_{}>", "<D_{}>"]

        elif self.cfg.train.tasks == "idrec":
            prefix_list = []

        else:
            prefix_list = [
                "<a_{}>",
                "<b_{}>",
                "<c_{}>",
                "<d_{}>",
                "<A_{}>",
                "<B_{}>",
                "<C_{}>",
                "<D_{}>",
            ]

        new_tokens = set()

        for prefix in prefix_list:
            for i in range(self.cfg.code_num):
                token = prefix.format(i)
                new_tokens.add(token)

        # IF we use the ID indices, we need to add them as well
        # if self.cfg.train.tasks in ['idrec', 'item2id', 'id2item', 'image2id', 'id2image',
        #                       'seqitem2id', 'seqid2item', 'seqimage2id', 'seqid2image']:
        if "id" in self.cfg.train.tasks:
            try:
                n_items = len(self.indices)
            except:
                n_items = len(self.cf_indices)
            for n in range(n_items):
                token = item_uid_to_token(n, n_tokens=self.cfg.train.num_id_tokens)
                new_tokens.add(token)

        self.new_tokens = sorted(list(new_tokens))

        return self.new_tokens

    def get_all_items(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_prefix_allowed_tokens_fn(self, tokenizer):
        # TODO: something for the ID indices?

        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][1]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set(
                [tokenizer.eos_token_id]
            )
        sep = [0]

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i : i + len(sep)] == sep[::-1]:
                    # print(list(self.allowed_tokens[i]))
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def get_prefix_allowed_tokens_fn_new(self, tokenizer):
        # TODO: something for the ID indices?

        if self.allowed_tokens is None:
            self.allowed_tokens = defaultdict(set)
            first_ids = set()
            end_ids = set()
            for index in self.indices.values():
                token_id_list = [tokenizer(token)["input_ids"][1] for token in index]
                for i, token_id in enumerate(token_id_list):
                    if i == 0:
                        first_ids.add(token_id)
                    elif i == len(token_id_list) - 1:
                        end_ids.add(token_id)

                    if i < len(token_id_list) - 1:
                        self.allowed_tokens[token_id].add(token_id_list[i + 1])

            for ids in end_ids:
                self.allowed_tokens[ids] = first_ids

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()

            return list(self.allowed_tokens[sentence[-1]])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError


class SeqRecDataset(BaseDataset):

    def __init__(
        self,
        cfg,
        task="seqrec",
        mode="train",
        prompt_sample_num=1,
        prompt_id=0,
        sample_num=-1,
    ):
        super().__init__(cfg)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.train_data_mode = cfg.dataset.train_data_mode
        self.add_prompt = cfg.dataset.add_prompt
        self.task = task
        self.num_id_tokens = cfg.dataset.num_id_tokens
        # self.soft_prompt = cfg.soft_prompts[task] # SHOULD BE ADDED IN `finetune_parallel_tiger.py``

        # load data
        self._load_data()
        self._remap_items()

        # load data
        if self.mode == "train":
            self.inter_data = self._process_train_data()
        elif self.mode == "valid":
            self.inter_data = self._process_valid_data()
        elif self.mode == "test":
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

        logger.info("task: {}".format(self.task))
        logger.info("prompt_id: {} --- sample_num: {}".format(self.prompt_id, self.sample_num))
        logger.info("train_data_mode: {} --- add_prompt: {}".format(self.train_data_mode, self.add_prompt))
        logger.info("train_data[0]: {}".format(self.__getitem__(0)))
        logger.info("train_data[4]: {}".format(self.__getitem__(4)))
        # Without `add_prompt`:
        # >>> train_data[0]: {'input_ids': '<a_40><b_112><c_210><d_218>', 'labels': '<a_121><b_12><c_181><d_188>', 'label': 0}
        # >>> train_data[4]: {'input_ids': '<a_188><b_199><c_107><d_97><a_173><b_170><c_30><d_175><a_137><b_176><c_89><d_193>', 'labels': '<a_144><b_96><c_96><d_181>', 'label': 4}
        # With `add_prompt` (not handled by the MQL4GRec's original implementation):
        # >>> train_data[0]: {'input_ids': 'Given the following interaction history: <a_40><b_112><c_210><d_218>, predict the next item.', 'labels': '<a_121><b_12><c_181><d_188>', 'label': 0}
        # >>> train_data[4]: {'input_ids': 'Given the following interaction history: <a_188><b_199><c_107><d_97>, <a_173><b_170><c_30><d_175>, <a_137><b_176><c_89><d_193>, predict the next item.', 'labels': '<a_144><b_96><c_96><d_181>', 'label': 4}

    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), "r") as f:
            self.inters = json.load(f)
        if self.task == "seqrec":
            with open(
                os.path.join(self.data_path, self.dataset + self.index_file), "r"
            ) as f:
                self.indices = json.load(f)
        elif self.task == "seqimage":
            with open(
                os.path.join(self.data_path, self.dataset + self.cfg.dataset.image_index_file),
                "r",
            ) as f:
                self.indices = json.load(f)
        elif self.task == "idrec":
            with open(
                os.path.join(self.data_path, self.dataset + self.index_file), "r"
            ) as f:
                indices = json.load(f)
                self.indices = {
                    k: item_uid_to_token(k, n_tokens=self.num_id_tokens)
                    for k, _ in indices.items()
                }

    def _remap_items(self):

        if self.task == "seqrec" or self.task == "seqimage":
            self.remapped_inters = dict()
            for uid, items in self.inters.items():
                new_items = ["".join(self.indices[str(i)]) for i in items]
                self.remapped_inters[uid] = new_items

        elif self.task == "idrec":
            self.remapped_inters = dict()
            for uid, items in self.inters.items():
                # new_items = [item_uid_to_token(i, n_tokens=self.num_id_tokens) for i in items]
                new_items = [self.indices[str(i)] for i in items]
                self.remapped_inters[uid] = new_items

    def _process_train_data(self):

        inter_data = []

        if self.train_data_mode == 0:  ## origin, for encoder-decoder
            for uid in self.remapped_inters:
                items = self.remapped_inters[uid][:-2]
                for i in range(1, len(items)):
                    one_data = dict()
                    # one_data["user"] = uid
                    one_data["item"] = items[i]
                    history = items[:i]
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len :]

                    one_data["inters"] = history
                    inter_data.append(one_data)

        elif self.train_data_mode == 1:  ## all max length
            for uid in self.remapped_inters:
                items = self.remapped_inters[uid][:-2]

                if len(items) <= self.max_his_len + 1:
                    one_data = dict()
                    one_data["item"] = items[-1]
                    history = items[:-1]
                    one_data["inters"] = history
                    inter_data.append(one_data)
                else:
                    for i in range(self.max_his_len + 1, len(items)):
                        one_data = dict()
                        one_data["item"] = items[i]
                        history = items[:i]
                        history = history[-self.max_his_len :]
                        one_data["inters"] = history
                        inter_data.append(one_data)
        else:
            for uid in self.remapped_inters:
                items = self.remapped_inters[uid][:-2]
                inter_data.append(items)

        return inter_data

    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]

            one_data["inters"] = history
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len :]

            one_data["inters"] = history
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)

    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        # input = sft_prompt.format(instruction = instruction, response = "")
        # output = sft_prompt.format(instruction = instruction, response = response)

        # if self.mode == 'test':
        #     return input, response

        return instruction, response

    def __getitem__(self, index):

        if self.mode != "train" or self.train_data_mode <= 1:
            d = self.inter_data[index]

        elif self.train_data_mode == 2:  ## 随机选择最长序列
            d = dict()
            items = self.inter_data[index]

            if len(items) <= self.max_his_len + 1:
                d["item"] = items[-1]
                d["inters"] = items[:-1]
            else:
                end = random.randint(self.max_his_len + 1, len(items) - 1)
                start = end - self.max_his_len
                d["item"] = items[end]
                d["inters"] = items[start:end]

        elif self.train_data_mode == 3:  ### 先随机选择end，再最长序列
            d = dict()
            items = self.inter_data[index]

            end = random.randint(1, len(items) - 1)
            d["item"] = items[end]
            history = items[:end]
            history = history[-self.max_his_len :]
            d["inters"] = history

        else:  ### strat 和 end 都随机选择
            d = dict()
            items = self.inter_data[index]

            start = random.randint(0, len(items) - 2)
            end = random.randint(start + 1, len(items) - 1)

            d["item"] = items[end]
            history = items[start:end]
            history = history[-self.max_his_len :]
            d["inters"] = history

        if self.add_prompt:
            # if self.mode == 'train':
            #     prompt_id = random.randint(0, len(self.prompts) - 1)
            # else:
            #     prompt_id = self.prompt_id

            # prompt = self.prompts[prompt_id]
            prompt = {
                "instruction": "Given the following interaction history: {inters}, predict the next item.",
                "response": "{item}",
            }

            data = {}
            data["item"] = d["item"]
            data["inters"] = self.his_sep.join(d["inters"])

            input, output = self._get_text_data(data, prompt)

        else:
            input = "".join(d["inters"])
            output = d["item"]

        # print("Returning sample:", dict(input_ids=input, labels=output, label=index))
        return dict(input_ids=input, labels=output, label=index)
        # return dict(input_ids=input, labels=output)


class ItemImageDataset(BaseDataset):

    def __init__(self, cfg, task="item2image", prompt_sample_num=1, sample_num=-1):
        super().__init__(cfg)

        self.task = task.lower()
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.soft_prompt = cfg.soft_prompts[self.task]

        self.image_index_file = cfg.dataset.image_index_file

        # load data
        self._load_data()
        self.data_pair = self._process_data()

    def _load_data(self):

        with open(
            os.path.join(self.data_path, self.dataset + self.index_file), "r"
        ) as f:
            self.indices = json.load(f)
        with open(
            os.path.join(self.data_path, self.dataset + self.image_index_file), "r"
        ) as f:
            self.image_indices = json.load(f)

    def _process_data(self):

        data_pair = []
        for item_id, item_token in self.indices.items():
            item_index = "".join(self.indices[item_id])
            image_index = "".join(self.image_indices[item_id])

            if self.task == "item2image":
                data_pair.append([item_index, image_index])
            else:
                data_pair.append([image_index, item_index])

        return data_pair

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, index):

        d = self.data_pair[index]

        input = self.soft_prompt + d[0]
        output = d[1]

        return dict(input_ids=input, labels=output)


class ItemIDDataset(BaseDataset):

    def __init__(self, cfg, task="item2id", prompt_sample_num=1, sample_num=-1):
        super().__init__(cfg)

        self.task = task.lower()
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num
        self.num_id_tokens = cfg.dataset.num_id_tokens

        self.soft_prompt = cfg.soft_prompts[self.task]

        self.image_index_file = cfg.dataset.image_index_file

        # load data
        self._load_data()
        self.data_pair = self._process_data()

    def _load_data(self):

        with open(
            os.path.join(self.data_path, self.dataset + self.index_file), "r"
        ) as f:
            self.indices = json.load(f)
            self.cf_indices = {
                k: item_uid_to_token(k, n_tokens=self.num_id_tokens)
                for k, _ in self.indices.items()
            }

    def _process_data(self):

        data_pair = []
        for item_id, item_token in self.indices.items():
            item_index = "".join(self.indices[item_id])

            if self.task == "item2id":
                # data_pair.append([item_index, item_uid_to_token(item_id, n_tokens=self.num_id_tokens)])
                data_pair.append([item_index, self.cf_indices[item_id]])
            else:
                # data_pair.append([item_uid_to_token(item_id, n_tokens=self.num_id_tokens), item_index])
                data_pair.append([self.cf_indices[item_id], item_index])

        return data_pair

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, index):

        d = self.data_pair[index]

        input = self.soft_prompt + d[0]
        output = d[1]

        return dict(input_ids=input, labels=output)


class ImageIDDataset(BaseDataset):

    def __init__(self, cfg, task="image2id", prompt_sample_num=1, sample_num=-1):
        super().__init__(cfg)

        self.task = task.lower()
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num
        self.num_id_tokens = cfg.dataset.num_id_tokens

        self.soft_prompt = cfg.soft_prompts[self.task]

        self.image_index_file = cfg.dataset.image_index_file

        # load data
        self._load_data()
        self.data_pair = self._process_data()

    def _load_data(self):
        with open(
            os.path.join(self.data_path, self.dataset + self.image_index_file), "r"
        ) as f:
            self.image_indices = json.load(f)
            self.cf_indices = {
                k: item_uid_to_token(k, n_tokens=self.num_id_tokens)
                for k, _ in self.image_indices.items()
            }

    def _process_data(self):

        data_pair = []
        for item_id, item_token in self.image_indices.items():
            image_index = "".join(self.image_indices[item_id])

            if self.task == "image2id":
                # data_pair.append([image_index, item_uid_to_token(item_id, n_tokens=self.num_id_tokens)])
                data_pair.append([image_index, self.cf_indices[item_id]])
            else:
                # data_pair.append([item_uid_to_token(item_id, n_tokens=self.num_id_tokens), image_index])
                data_pair.append([self.cf_indices[item_id], image_index])

        return data_pair

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, index):

        d = self.data_pair[index]

        input = self.soft_prompt + d[0]
        output = d[1]

        return dict(input_ids=input, labels=output)


class FusionSeqRecDataset(BaseDataset):

    def __init__(
        self,
        cfg,
        task="seqitem2image",
        mode="train",
        prompt_sample_num=1,
        prompt_id=0,
        sample_num=-1,
    ):
        super().__init__(cfg)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.task = task

        self.image_index_file = cfg.dataset.image_index_file

        self.soft_prompt = cfg.soft_prompts[self.task]

        # load data
        self._load_data()
        # self._remap_items()

        # load data
        if self.mode == "train":
            self.inter_data = self._process_train_data()
        else:
            raise NotImplementedError

    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), "r") as f:
            self.inters = json.load(f)
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file), "r"
        ) as f:
            self.indices = json.load(f)
        with open(
            os.path.join(self.data_path, self.dataset + self.image_index_file), "r"
        ) as f:
            self.image_indices = json.load(f)

    def _process_train_data(self):

        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()

                item = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len :]

                if self.task == "seqitem2image":
                    history = ["".join(self.indices[str(i)]) for i in history]
                    one_data["inters"] = "".join(history)
                    one_data["item"] = "".join(self.image_indices[str(item)])

                elif self.task == "seqimage2item":
                    history = ["".join(self.image_indices[str(i)]) for i in history]
                    one_data["inters"] = "".join(history)
                    one_data["item"] = "".join(self.indices[str(item)])

                else:
                    one_data["inters"] = history
                    one_data["item"] = item

                inter_data.append(one_data)

        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == "train":
            return len(self.inter_data)
        else:
            raise NotImplementedError

    def __getitem__(self, index):

        d = self.inter_data[index]

        if self.task == "seqitem2image" or self.task == "seqimage2item":
            input = self.soft_prompt + d["inters"]
            output = d["item"]

        else:

            if random.random() > 0.5:
                history = ["".join(self.indices[str(i)]) for i in d["inters"]]
                input = self.soft_prompt + "".join(history)
                output = "".join(self.image_indices[str(d["item"])])
            else:
                history = ["".join(self.image_indices[str(i)]) for i in d["inters"]]
                input = self.soft_prompt + "".join(history)
                output = "".join(self.indices[str(d["item"])])

        return dict(input_ids=input, labels=output)


class FusionTextIDRecDataset(BaseDataset):

    def __init__(
        self,
        cfg,
        task="seqitem2id",
        mode="train",
        prompt_sample_num=1,
        prompt_id=0,
        sample_num=-1,
    ):
        super().__init__(cfg)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.task = task
        self.num_id_tokens = cfg.dataset.num_id_tokens

        self.image_index_file = cfg.dataset.image_index_file

        self.soft_prompt = cfg.soft_prompts[self.task]

        # load data
        self._load_data()
        # self._remap_items()

        # load data
        if self.mode == "train":
            self.inter_data = self._process_train_data()
        else:
            raise NotImplementedError

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), "r") as f:
            self.inters = json.load(f)
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file), "r"
        ) as f:
            self.indices = json.load(f)
        # with open(os.path.join(self.data_path, self.dataset + self.image_index_file), 'r') as f:
        #     self.image_indices = json.load(f)
        self.cf_indices = {
            k: item_uid_to_token(k, n_tokens=self.num_id_tokens)
            for k, _ in self.indices.items()
        }

    def _process_train_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()

                item = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len :]

                if self.task == "seqitem2id":
                    history = ["".join(self.indices[str(i)]) for i in history]
                    one_data["inters"] = "".join(history)
                    # one_data["item"] = item_uid_to_token(item, n_tokens=self.num_id_tokens)
                    one_data["item"] = self.cf_indices[str(item)]

                elif self.task == "seqid2item":
                    # history = [item_uid_to_token(i, n_tokens=self.num_id_tokens) for i in history]
                    history = [self.cf_indices[str(i)] for i in history]
                    one_data["inters"] = "".join(history)
                    one_data["item"] = "".join(self.indices[str(item)])

                else:
                    one_data["inters"] = history
                    one_data["item"] = item

                inter_data.append(one_data)

        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == "train":
            return len(self.inter_data)
        else:
            raise NotImplementedError

    def __getitem__(self, index):

        d = self.inter_data[index]

        if self.task == "seqitem2id" or self.task == "seqid2item":
            input = self.soft_prompt + d["inters"]
            output = d["item"]

        else:

            if random.random() > 0.5:
                history = ["".join(self.indices[str(i)]) for i in d["inters"]]
                input = self.soft_prompt + "".join(history)
                # output = item_uid_to_token(d['item'], n_tokens=self.num_id_tokens)
                output = self.cf_indices[str(d["item"])]
            else:
                # history = [item_uid_to_token(i, n_tokens=self.num_id_tokens) for i in d['inters']]
                history = [self.cf_indices[str(i)] for i in d["inters"]]
                input = self.soft_prompt + "".join(history)
                output = "".join(self.indices[str(d["item"])])

        return dict(input_ids=input, labels=output)


class FusionImageIDRecDataset(BaseDataset):

    def __init__(
        self,
        cfg,
        task="seqimage2id",
        mode="train",
        prompt_sample_num=1,
        prompt_id=0,
        sample_num=-1,
    ):
        super().__init__(cfg)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.task = task
        self.num_id_tokens = cfg.dataset.num_id_tokens

        self.image_index_file = cfg.dataset.image_index_file

        self.soft_prompt = cfg.soft_prompts[self.task]

        # load data
        self._load_data()
        # self._remap_items()

        # load data
        if self.mode == "train":
            self.inter_data = self._process_train_data()
        else:
            raise NotImplementedError

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), "r") as f:
            self.inters = json.load(f)
        # with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
        #     self.indices = json.load(f)
        with open(
            os.path.join(self.data_path, self.dataset + self.image_index_file), "r"
        ) as f:
            self.image_indices = json.load(f)
        self.cf_indices = {
            k: item_uid_to_token(k, n_tokens=self.num_id_tokens)
            for k, _ in self.image_indices.items()
        }

    def _process_train_data(self):
        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()

                item = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len :]

                if self.task == "seqimage2id":
                    history = ["".join(self.image_indices[str(i)]) for i in history]
                    one_data["inters"] = "".join(history)
                    # one_data["item"] = item_uid_to_token(item, n_tokens=self.num_id_tokens)
                    one_data["item"] = self.cf_indices[str(item)]

                elif self.task == "seqid2image":
                    # history = [item_uid_to_token(i, n_tokens=self.num_id_tokens) for i in history]
                    history = [self.cf_indices[str(i)] for i in history]
                    one_data["inters"] = "".join(history)
                    one_data["item"] = "".join(self.image_indices[str(item)])

                else:
                    one_data["inters"] = history
                    one_data["item"] = item

                inter_data.append(one_data)

        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == "train":
            return len(self.inter_data)
        else:
            raise NotImplementedError

    def __getitem__(self, index):

        d = self.inter_data[index]

        if self.task == "seqimage2id" or self.task == "seqid2image":
            input = self.soft_prompt + d["inters"]
            output = d["item"]

        else:

            if random.random() > 0.5:
                history = ["".join(self.image_indices[str(i)]) for i in d["inters"]]
                input = self.soft_prompt + "".join(history)
                # output = item_uid_to_token(d['item'], n_tokens=self.num_id_tokens)
                output = self.cf_indices[str(d["item"])]
            else:
                # history = [item_uid_to_token(i, n_tokens=self.num_id_tokens) for i in d['inters']]
                history = [self.cf_indices[str(i)] for i in d["inters"]]
                input = self.soft_prompt + "".join(history)
                output = "".join(self.image_indices[str(d["item"])])

        return dict(input_ids=input, labels=output)
