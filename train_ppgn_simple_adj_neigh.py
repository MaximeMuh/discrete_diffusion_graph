import logging
import time
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import wandb
from dataset.extract_graph import get_dataloader, get_dataloader_adj

from model.model2 import Powerful
from utils.graphs import discretenoise, loss_func_bce, upper_flatten_to_adj_matrix, adj_matrix_to_upper_flatten, discretenoise_neighbor, loss_func_bce_adj, discretenoise_adj, gen_list_of_data_single, gen_list_of_data_single_neigh, loss_cycle_consistency

filename = 'dataset/usts_4.pkl'
width, height = 4, 4
batch_size = 16
dataloader = get_dataloader_adj(filename, width, height, batch_size)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

import torch



def save_model(model, path):
    torch.save(model.state_dict(), path)


def fit(model, optimizer, dataloader, max_epoch=20, device=device):
    wandb.init(project='diffusion_graph', entity='maxime-muhlethaler-Inria')
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    for epoch in range(max_epoch):
        train_losses = []
        model.train()
        for data in dataloader:
            sigma_list=list(np.random.uniform(low=0.0, high=0.5, size=data.size(0)))
            data = data.to(device)
            if isinstance(sigma_list, float):
                    sigma_list = [sigma_list]
            train_adj_b_noisy_matrix, grad_log_noise_adj_list = gen_list_of_data_single_neigh(data, sigma_list, device, grid_shape=(4, 4))
            optimizer.zero_grad()
            train_noise_adj_b_chunked = train_adj_b_noisy_matrix.chunk(len(sigma_list), dim=0)
            score = []
            sigma_list = torch.tensor(sigma_list, dtype=torch.float32).to(device)
            l = 0
            for i, sigma in enumerate(sigma_list):
                mask= torch.ones(1, width * height, width * height).to(device)

                A = train_noise_adj_b_chunked[i].unsqueeze(0)
                A = A.to(device)

                score_batch = model(A=A, node_features= train_noise_adj_b_chunked[i].to(device), mask=mask, noiselevel=sigma.item()).to(device)
                score.append(score_batch)
                l += loss_cycle_consistency(score_batch.squeeze(-1), A, sigma.item(),  device, grid_shape=(4, 4))
            score = torch.cat(score, dim=0).squeeze(-1).to(device)
            l += loss_func_bce_adj(score, torch.stack(grad_log_noise_adj_list), sigma_list, device)
            
            l.backward()
            optimizer.step()
            train_losses.append(l.detach().cpu().item())
        scheduler.step(epoch)
        wandb.log({"Loss": np.mean(train_losses)})
        print(f"Epoch {epoch+1}/{max_epoch}, Loss: {np.mean(train_losses)}")
        if epoch % 10 == 0:
            save_model(model, 'trained_model_adj_neigh_cycle_conn.pth')
    wandb.finish()






model = Powerful(
    use_norm_layers=False,
    name='ppgn',
    channel_num_list=[],
    feature_nums=[],
    gnn_hidden_num_list=[],
    num_layers=6,
    input_features=2,
    hidden=64,
    hidden_final=64,
    dropout_p=0.000001,
    simplified=False,
    n_nodes=25,
    device=device,
    normalization="instance",
    cat_output=True,
    adj_out=True,
    output_features=1,
    residual=False,
    project_first=False,
    node_out=False,
    noise_mlp=False
).to(device)


optimizer = optim.Adam(model.parameters(),
                           lr=0.001,
                           betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=0.0)
max_epoch = 2  

fit(model, optimizer, dataloader, max_epoch=40, device=device)