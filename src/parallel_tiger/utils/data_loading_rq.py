from typing import Tuple
from torch.utils.data import Dataset
from omegaconf import DictConfig

from parallel_tiger.data.data_rq_transformer import (
    SeqRecDataset,
)

import logging
logger = logging.getLogger(__name__)



def load_datasets(cfg: DictConfig) -> Tuple[SeqRecDataset, SeqRecDataset]:
    train_data = SeqRecDataset(
        cfg,
        task="seqrec",
        mode="train",
        sample_num=cfg.dataset.train_prompt_sample_num,
    )

    valid_data = SeqRecDataset(
        cfg,
        task="seqrec",
        mode="valid"
    )

    return train_data, valid_data


def load_test_dataset(cfg):

    if cfg.infer.test_task.lower() == "seqrec" or cfg.infer.test_task.lower() == "seqimage":
        test_data = SeqRecDataset(
            cfg, task=cfg.infer.test_task.lower(), mode="test", sample_num=cfg.infer.sample_num
        )
    else:
        raise NotImplementedError

    return test_data

