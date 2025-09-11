import os
import json
import numpy as np
from typing import Union
from parallel_tiger.data.data_ID_injection import BaseDataset

import logging
logger = logging.getLogger(__name__)


def item_uid_to_token(item_uid: Union[str, int], n_tokens: int = 1) -> str:
    """
    Convert item unique ID to a token.
    E.g.: item_uid = "23" --> "<item_23>" (with n_tokens=1), or "<item_23><item_23><item_23><item_23>" (with n_tokens=4).
    """
    return f"<item_{item_uid}>" * n_tokens  # or f"#{item_uid}"



class SeqRecDataset(BaseDataset):

    def __init__(
        self,
        cfg,
        task="seqrec",
        mode="train",
        prompt_id=0,
        sample_num=-1,
    ):
        super().__init__(cfg)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.train_data_mode = cfg.dataset.train_data_mode
        self.task = task
        self.num_id_tokens = cfg.dataset.num_id_tokens
        
        if self.mode == "train":
            assert self.train_data_mode == 0, \
                "Only train_data_mode=0 is supported in training mode."

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
        logger.info("train_data[0]: {}".format(self.__getitem__(0)))
        logger.info("train_data[4]: {}".format(self.__getitem__(4)))

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
        """
        Keep only the longest sequence per user (truncated to max_his_len if needed).
        NOTE: IMPORTANT --> No need to split history/target because the model is such that the training will happen on all causal subsequences in one forward pass.
        Returns a list of dicts with keys:
            - "item": target item (last in the sequence)
            - "inters": list of historical items (history)
        """

        inter_data = []

        for _, items in self.remapped_inters.items():
            history_items = items[:-2]

            if len(history_items) < 1:
                continue  # skip too short sequences

            if self.max_his_len > 0:
                history_items = history_items[-self.max_his_len :]
            
            inter_data.append({
                "item": None,
                "inters": history_items
            })

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

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        # if self.mode != "train" or self.train_data_mode <= 1:
        d = self.inter_data[index]
        input = "".join(d["inters"])
        output = d["item"]

        # print("Returning sample:", dict(input_ids=input, labels=output, label=index))
        return dict(input_ids=input, labels=output, label=index)