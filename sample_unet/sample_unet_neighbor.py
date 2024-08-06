import networkx as nx
import numpy as np
import math
import logging
import time
import os
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt
import wandb
from dataset.extract_graph import get_dataloader
from dataset.extract_graph import vector_to_graph, vector_to_upper_triangular, upper_triangular_to_vector, graph_from_adjacency_matrix, visualize_graph, visualize_graph_from_vector, visualize_batch_from_dataloader
from model.model2 import Powerful
from model.unet import Unet
from utils.graphs import discretenoise, loss_func_bce, upper_flatten_to_adj_matrix, adj_matrix_to_upper_flatten, discretenoise_neighbor, draw_maze_from_matrix, create_adjacency_mask

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

filename = 'dataset/usts_4.pkl'
width, height = 4, 4  
batch_size = 1  
dataloader = get_dataloader(filename, width, height, batch_size)
grid_shape = (4,4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def visualize_graph2(G, title="Uniform Spanning Tree"):
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(title)
    plt.show()

def sample_ppgn_simple(noise_num):
    

    batch = next(iter(dataloader))
    train_graph_list = batch
    print(f"Loaded {len(train_graph_list)} train graphs.")
    
    model = Unet(dim=64).to(device)
    model.load_state_dict(torch.load('trained_model_unet_neighbor.pth'))
    model.eval()
    
    max_node_number = 16
    test_batch_size = 16
    test_batch_size = 1





    def gen_init_data(batch_size, grid_shape, device):
        size = grid_shape[0] * grid_shape[1]
        adjacency_mask = create_adjacency_mask(grid_shape)
        adjacency_mask = torch.tensor(adjacency_mask, dtype=torch.float32).to(device)
        
        bernoulli_adj = torch.zeros(batch_size, size, size).to(device)
        bernoulli_adj += adjacency_mask.unsqueeze(0) * 0.5  # Probability of 0.5 for each neighbor

        noise_upper = torch.bernoulli(bernoulli_adj).triu(diagonal=1)
        noise_lower = noise_upper.transpose(-1, -2)
        initial_matrix = noise_lower + noise_upper
        return initial_matrix



    sigma_tens = torch.linspace(0, 1/2, noise_num)
    sigma_list = sigma_tens.tolist()
    sigma_list.sort()
    sigma_list = torch.tensor(sigma_list, dtype=torch.float32).to(device)

    def add_bernoulli( init_adjs, noiselevel):
        init_adjs = adj_matrix_to_upper_flatten(init_adjs)
        init_adjs, noise = discretenoise_neighbor(init_adjs, noiselevel, device, grid_shape=(4, 4))

        init_adjs = upper_flatten_to_adj_matrix(init_adjs, max_node_number)
        return init_adjs

    def take_step(noise_func, init_adjs, noiselevel):
        init_adjs = add_bernoulli( init_adjs, noiselevel)
        mask = torch.ones_like(init_adjs)
        init_adjs = init_adjs.unsqueeze(1)
        noiselevel = noiselevel.unsqueeze(0)
        noise_unnormal = noise_func(adj_matrix=init_adjs.to(device), noise=noiselevel)
        noise_unnormal = noise_unnormal.squeeze(1)
        init_adjs = init_adjs.squeeze(1)
        noise_rel = torch.sigmoid(noise_unnormal)
        noise_rel = (noise_rel + noise_rel.transpose(-1, -2)) / 2
        noise = torch.bernoulli(noise_rel) * mask
        print(noise)
        adjacency_mask = create_adjacency_mask(grid_shape)
        adjacency_mask = torch.tensor(adjacency_mask, dtype=torch.float32).to(device)
        inter_adjs = torch.where(noise > 1/2, init_adjs - 1, init_adjs)
        new_adjs = torch.where(inter_adjs < -1/2, inter_adjs + 2, inter_adjs)
        new_adjs = new_adjs * adjacency_mask[None, :, :]
        return init_adjs, new_adjs

    def run_sample(nb_eval=10):
        print("Starting run_sample...\n")
        gen_graph_list = []
        with torch.no_grad():
            nb=0
            while nb<nb_eval:
                init_adjs = gen_init_data(batch_size=test_batch_size, grid_shape=grid_shape, device=device)
                print(f"Generated initial adjacency matrices shape: {init_adjs.shape}\n")
                draw_maze_from_matrix(init_adjs[0], grid_shape[0], grid_shape[1])
                nb+=1
                count = 0
                while count < noise_num :
                    noiselevel = sigma_list[len(sigma_list) - count - 1]

                    noiselevel = torch.tensor(noiselevel, dtype=torch.float32).to(device)
                    noisy_adjs, init_adjs = take_step(lambda adj_matrix, noise: model(adj_matrix, noise), init_adjs=init_adjs, noiselevel=noiselevel)
                    count+=1
                init_adjs = init_adjs.cpu()
                vector = adj_matrix_to_upper_flatten(init_adjs)
                g = vector_to_graph(vector[0], width, height)
                visualize_graph(vector, width, height)
                
                visualize_graph2(g)
                print(f"Generated graph {count + 1}.\n")
    result_dict = run_sample()

    return result_dict


sample_ppgn_simple(32)
