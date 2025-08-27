from typing import Tuple
from torch.utils.data import ConcatDataset
from omegaconf import DictConfig

from parallel_tiger.data.data_ID_injection import (
    SeqRecDataset,
    FusionSeqRecDataset,
    ItemImageDataset,
    ItemIDDataset,
    ImageIDDataset,
    FusionTextIDRecDataset,
    FusionImageIDRecDataset,
)

import logging
logger = logging.getLogger(__name__)



def load_datasets(cfg: DictConfig) -> Tuple[ConcatDataset, ConcatDataset]:
    tasks = cfg.train.tasks.split(",")

    # train_prompt_sample_num = [int(_) for _ in cfg.dataset.train_prompt_sample_num.split(",")]
    # assert len(tasks) == len(train_prompt_sample_num), "prompt sample number does not match task number"
    # train_data_sample_num = [int(_) for _ in cfg.dataset.train_data_sample_num.split(",")]
    # assert len(tasks) == len(train_data_sample_num), "data sample number does not match task number"

    train_prompt_sample_num = [1] * len(tasks)
    train_data_sample_num = [-1] * len(tasks)

    train_datasets = []
    for task, prompt_sample_num, data_sample_num in zip(
        tasks, train_prompt_sample_num, train_data_sample_num
    ):
        if task.lower() == "seqrec":
            dataset = SeqRecDataset(
                cfg,
                task=task.lower(),
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "seqimage":
            dataset = SeqRecDataset(
                cfg,
                task=task.lower(),
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "item2image" or task.lower() == "image2item":
            dataset = ItemImageDataset(
                cfg,
                task=task.lower(),
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif (
            task.lower() == "seqitem2image"
            or task.lower() == "seqimage2item"
            or task.lower() == "fusionseqrec"
        ):
            dataset = FusionSeqRecDataset(
                cfg,
                task=task.lower(),
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        ###
        # TODO: ADD THE NEW DATASETS I CREATED
        elif task.lower() == "idrec":
            dataset = SeqRecDataset(
                cfg,
                task=task.lower(),
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "item2id" or task.lower() == "id2item":
            dataset = ItemIDDataset(
                cfg,
                task=task.lower(),
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif task.lower() == "image2id" or task.lower() == "id2image":
            dataset = ImageIDDataset(
                cfg,
                task=task.lower(),
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif (
            task.lower() == "seqitem2id"
            or task.lower() == "seqid2item"
            or task.lower() == "fusionseqrecid"
        ):
            dataset = FusionTextIDRecDataset(
                cfg,
                task=task.lower(),
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )

        elif (
            task.lower() == "seqimage2id"
            or task.lower() == "seqid2image"
            or task.lower() == "fusionseqrecid"
        ):
            dataset = FusionImageIDRecDataset(
                cfg,
                task=task.lower(),
                mode="train",
                prompt_sample_num=prompt_sample_num,
                sample_num=data_sample_num,
            )
        ###

        else:
            raise NotImplementedError

        train_datasets.append(dataset)

    train_data = ConcatDataset(train_datasets)

    valid_datasets = []
    valid_tasks = cfg.train.valid_task.lower().split(",")
    for valid_task in valid_tasks:
        dataset = SeqRecDataset(cfg, task=valid_task, mode="valid")
        valid_datasets.append(dataset)
    valid_data = ConcatDataset(valid_datasets)

    # valid_data = SeqRecDataset(cfg, task=cfg.train.valid_task.lower(), mode="valid")

    return train_data, valid_data


def load_pretrain_datasets(args):

    tasks = args.tasks.split(",")

    train_prompt_sample_num = [1] * len(tasks)
    train_data_sample_num = [-1] * len(tasks)

    pretrain_datasets = args.pretrain_datasets.split(",")

    train_datasets = []
    valid_datasets = []
    for dataset in pretrain_datasets:
        args.dataset = dataset
        for task, prompt_sample_num, data_sample_num in zip(
            tasks, train_prompt_sample_num, train_data_sample_num
        ):
            if task.lower() == "seqrec":
                dataset = SeqRecDataset(
                    args,
                    task=task.lower(),
                    mode="train",
                    prompt_sample_num=prompt_sample_num,
                    sample_num=data_sample_num,
                )

            elif task.lower() == "seqimage":
                dataset = SeqRecDataset(
                    args,
                    task=task.lower(),
                    mode="train",
                    prompt_sample_num=prompt_sample_num,
                    sample_num=data_sample_num,
                )

            elif task.lower() == "item2image" or task.lower() == "image2item":
                dataset = ItemImageDataset(
                    args,
                    task=task.lower(),
                    prompt_sample_num=prompt_sample_num,
                    sample_num=data_sample_num,
                )

            elif (
                task.lower() == "seqitem2image"
                or task.lower() == "seqimage2item"
                or task.lower() == "fusionseqrec"
            ):
                dataset = FusionSeqRecDataset(
                    args,
                    task=task.lower(),
                    mode="train",
                    prompt_sample_num=prompt_sample_num,
                    sample_num=data_sample_num,
                )

            elif task.lower() == "idrec":
                dataset = SeqRecDataset(
                    args,
                    task=task.lower(),
                    mode="train",
                    prompt_sample_num=prompt_sample_num,
                    sample_num=data_sample_num,
                )

            else:
                raise NotImplementedError

            train_datasets.append(dataset)

        # valid_data = SeqRecDataset(args, mode="valid")
        valid_data = SeqRecDataset(args, task=args.valid_task.lower(), mode="valid")
        valid_datasets.append(valid_data)

    train_data = ConcatDataset(train_datasets)
    valid_data = ConcatDataset(valid_datasets)

    return train_data, valid_data


def load_test_dataset(cfg):

    if cfg.infer.test_task.lower() == "seqrec" or cfg.infer.test_task.lower() == "seqimage":
        test_data = SeqRecDataset(
            cfg, task=cfg.infer.test_task.lower(), mode="test", sample_num=cfg.infer.sample_num
        )
    else:
        raise NotImplementedError

    return test_data

