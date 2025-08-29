import argparse
import random
import torch
import numpy as np
from time import time
import logging
import os

from torch.utils.data import DataLoader, ConcatDataset

from datasets import EmbDataset, EmbDatasetAll
from models.rqvae import RQVAE
from models.pqvae import PQVAE
from trainer import  Trainer
import pprint

def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument("--data_root", type=str,
                        default="",
                        help="Input data path.")
    parser.add_argument('--datasets', type=str, default='Scientific')
    parser.add_argument('--embedding_file', type=str, default=".emb-llama-td.npy", help='')

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:1", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--beta', type=float, default=0.25, help='vq beta')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument("--ckpt_dir", type=str, default="", help="output directory for model")

    parser.add_argument("--quantizer_type", type=str, default="rqvae",
                        choices=["rqvae", "pqvae"],
                        help="model type, rqvae or pqvae")
    parser.add_argument("--use_linear", type=int, default=0, help="use linear layers or not (0 or 1)")
    return parser.parse_args()


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    pprint.pprint(vars(args))

    logging.basicConfig(level=logging.DEBUG)

    # """build dataset"""
    # train_datasets = []
    # for dataset in args.datasets:
    #     data_path = os.path.join(args.data_root, dataset, dataset + args.suffix)
    #     data_one = EmbDataset(data_path)
    #     train_datasets.append(data_one)
        
    data = EmbDatasetAll(args)
        
    # data = EmbDataset(args.data_path)
    if args.quantizer_type == "rqvae":
        mod = RQVAE
    elif args.quantizer_type == "pqvae":
        mod = PQVAE
    else:
        raise ValueError("model type must be rqvae or pqvae, but got {}".format(args.quantizer_type))

    model = mod(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        beta=args.beta,
        quant_loss_weight=args.quant_loss_weight,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        use_linear=args.use_linear,
    )

    print(model)
    data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)
    trainer = Trainer(args, model)
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision_rate)

