import collections
import json
import logging

import numpy as np
import torch
import copy
from tqdm import tqdm
import argparse
from collections import defaultdict

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
from models.pqvae import PQVAE

import os

def create_and_save_indices_dict(all_indices, prefix, output_dir, output_file):
    """
    Create indices dictionary with formatted codes and save to file.
    
    Args:
        all_indices: List of code assignments (list of lists)
        prefix: List of prefix templates for formatting codes
        output_dir: Directory to save the output file
        output_file: Full path to output file
    
    Returns:
        all_indices_dict: Dictionary mapping item indices to formatted codes
    """
    all_indices_dict = {}
    for item, indices in enumerate(all_indices):
        code = []
        for i, ind in enumerate(indices):
            code.append(prefix[i].format(int(ind)))
            
        all_indices_dict[item] = code
        
    print('Validating indices dictionary...')
    code2item = {}
    for item, code in all_indices_dict.items():
        code2item[str(code)] = item
        
    print('Validation check: ', len(code2item) == len(all_indices_dict))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_file, 'w') as fp:
        json.dump(all_indices_dict, fp, indent=4)
    
    return all_indices_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--quantizer_type', type=str, default="rqvae",
                        choices=["rqvae", "pqvae"],
                        help="model type, rqvae or pqvae")
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--content', type=str, default=None)
    parser.add_argument('--device', type=str, default="cuda:0")
    # parser.add_argument('--data_root', type=str, default=None)

    return parser.parse_args()

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

args = parse_args()

dataset = args.dataset
# data_root = args.data_root
ckpt_path = args.ckpt_path
output_dir = args.output_dir
output_file = args.output_file
# quantizer_type = args.quantizer_type
output_file = os.path.join(output_dir, output_file)
device = torch.device(args.device)

if args.content == 'image':
    prefix = ["<A_{}>","<B_{}>","<C_{}>","<D_{}>","<E_{}>"]
else:
    prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
args = ckpt["args"]
# print(2, args)
state_dict = ckpt["state_dict"]

emb_data_path = os.path.join(args.data_root, dataset, f'{dataset}{args.embedding_file}')
data = EmbDataset(emb_data_path)

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

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices = []
all_indices_str = []
all_distances = []
all_indices_str_set = set()

n_tokens = 4
codebook_pos_to_tok_set = {i: set() for i in range(n_tokens)}


for d in tqdm(data_loader):
    d = d.to(device)
    indices, distances = model.get_indices(d, use_sk=False)
    # print(indices.shape, distances.shape)
    # print('indices: ', indices)
    # print('distances: ', distances)

    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    distances = distances.cpu().tolist()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            # code.append(prefix[i].format(int(ind)))
            code.append(int(ind))
            codebook_pos_to_tok_set[i].add(int(ind))

        all_indices.append(code)
        all_indices_str.append(str(code))
        all_indices_str_set.add(str(code))
        # print(str(code))
    # break
    all_distances.extend(distances)


for i, s in codebook_pos_to_tok_set.items():
    print(f'Codebook position {i} has {len(s)} unique tokens.')


all_distances = np.array(all_distances)

for i in all_indices_str_set:
    print(i)
    break

# print(all_distances)
print(all_distances.shape) ## (num, 4, 256)

sort_distances_index = np.argsort(all_distances, axis=2)

collision_item_groups = get_collision_item(all_indices_str)
all_collision_items = set()
for collision_items in collision_item_groups:
    for item in collision_items:
        all_collision_items.add(item)
        
print('Initial collision items num: ', len(all_collision_items))
print('Initial collision groups: ', len(collision_item_groups))


if args.quantizer_type == "rqvae":

    item_min_dis = defaultdict(list)

    for item, distances in tqdm(enumerate(all_distances), desc='cal distances'):

        for dis in distances:
            item_min_dis[item].append(np.min(dis))

    # new_indices_set = set()

    tt = 0
    level = len(args.num_emb_list) - 1
    max_num = args.num_emb_list[0]

    while True:
        tot_item = len(all_indices_str)
        tot_indice = len(set(all_indices_str))
        print(f'tot_item: {tot_item}, tot_indice: {tot_indice}')
        print("Collision Rate",(tot_item-tot_indice)/tot_item)
        
        if check_collision(all_indices_str) or tt == 2:
            print('tt', tt)
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print(collision_item_groups)
        print(len(collision_item_groups))
        
        
        for collision_items in collision_item_groups:
            
            min_distances = []
            for i, item in enumerate(collision_items):
                min_distances.append(item_min_dis[item][level])


            min_index = np.argsort(np.array(min_distances))
            
            for i, m_index in enumerate(min_index):
                
                if i == 0:
                    continue
                
                item = collision_items[m_index]
                # print(item)
                
                ori_code = copy.deepcopy(all_indices[item])
                # print(ori_code)
                
                num = i
                while str(ori_code) in all_indices_str_set and num < max_num:

                    ori_code[level] = sort_distances_index[item][level][num]
                    num += 1
                    # print(sort_distances_index[item][level])
                    # print(ori_code)
                    # print(num)
                
                ### 倒数第二层
                for i in range(1, max_num):
                    if str(ori_code) in all_indices_str_set:
                        ori_code = copy.deepcopy(all_indices[item])
                        ori_code[level-1] = sort_distances_index[item][level-1][i]
                        
                    num = 0
                    while str(ori_code) in all_indices_str_set and num < max_num:

                        ori_code[level] = sort_distances_index[item][level][num]
                        num += 1
                        
                    if str(ori_code) not in all_indices_str_set:
                        break
                    
                all_indices[item] = ori_code
                all_indices_str[item] = str(ori_code)

                all_indices_str_set.add(str(ori_code))

                # print(str(ori_code))
            
            
        # if level == 2:
        #     break
        tt += 1


    print("All indices number: ",len(all_indices))
    all_indices_str = [str(indice) for indice in all_indices]
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str))
    print("Collision Rate",(tot_item-tot_indice)/tot_item)

    # print(all_indices[0])
    codebook_pos_to_tok_set_after = defaultdict(set)
    for code in all_indices:
        for i, tok in enumerate(code):
            codebook_pos_to_tok_set_after[i].add(tok)

    for i, s in codebook_pos_to_tok_set_after.items():
        print(f'Codebook position {i} has {len(s)} unique tokens after collision handling.')

    # Create and save indices dictionary
    all_indices_dict = create_and_save_indices_dict(all_indices, prefix, output_dir, output_file)


elif args.quantizer_type == "pqvae":
    from pq_simplified_collision_resolver import resolve_all_collisions_simplified
    
    if len(collision_item_groups) > 0:
        # Show initial collision rate
        tot_item = len(all_indices_str)
        tot_indice = len(set(all_indices_str))
        print(f'Initial: tot_item={tot_item}, tot_indice={tot_indice}')
        print("Initial Collision Rate:", (tot_item-tot_indice)/tot_item)

        # Resolve collisions using simplified algorithm
        M = 10+1  # Number of nearest codebook vectors to consider per token
        resolve_all_collisions_simplified(
            all_indices=all_indices,
            all_indices_str=all_indices_str,
            all_indices_str_set=all_indices_str_set,
            all_distances=all_distances,
            sort_distances_index=sort_distances_index,
            collision_item_groups=collision_item_groups,
            M=M
        )

        # Check final collision rate
        all_indices_str_final = [str(indice) for indice in all_indices]
        tot_item_final = len(all_indices_str_final)
        tot_indice_final = len(set(all_indices_str_final))
        print(f'\nFinal: tot_item={tot_item_final}, tot_indice={tot_indice_final}')
        assert tot_item_final == tot_indice_final
        print("Final Collision Rate:", (tot_item_final-tot_indice_final)/tot_item_final)
        print("Max number of conflicts:", max(get_indices_count(all_indices_str_final).values()) if all_indices_str_final else 0)

    else:
        print("No collisions found.")
    
    # Create and save indices dictionary
    all_indices_dict = create_and_save_indices_dict(all_indices, prefix, output_dir, output_file)
