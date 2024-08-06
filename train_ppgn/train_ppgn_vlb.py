
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import wandb
from dataset.extract_graph import get_dataloader_adj

from model.model2 import Powerful
from utils.graphs import discretenoise, loss_func_bce_adj, upper_flatten_to_adj_matrix, adj_matrix_to_upper_flatten, discretenoise_adj_neigh, loss_func_kld, gen_list_of_data_single_neigh


filename = 'dataset/usts_4.pkl'
width, height = 4, 4  
batch_size = 16 
grid_shape = (width, height)
dataloader = get_dataloader_adj(filename, width, height, batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_levels = 64

def save_model(model, path):
    torch.save(model.state_dict(), path)

def sigma_lin(sigma_list):
    sigmas = []
    for g,sigma in enumerate(sigma_list):
        if sigma < 1.0e-5:
            sigmas.append(0.0)
            continue
        sigmas.append(((1-sigma) - (1-sigma_list[g-1])) / (1 - 2 * (1 - sigma_list[g-1])))
    return sigmas

def fit(model, optimizer, dataloader, max_epoch=20, device=device):
    wandb.init(project='diffusion_graph', entity='maxime-muhlethaler-Inria')
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    best_loss = float('inf')
    for epoch in range(max_epoch):
        train_losses = []
        model.train()

        for data in dataloader:
            sigma_ind_list = np.random.random_integers(low=1,high=num_levels,size=data.size(0))
            sigma_line=torch.linspace(0,1/2,num_levels+1).tolist()
            sigma_list = [sigma_line[i] for i in sigma_ind_list]
            sig_list = sigma_lin(sigma_line)
            sigma_nontild_list = [sig_list[i] for i in sigma_ind_list]
            data = data.to(device)

            train_adj_b_noisy_matrix, grad_log_noise_adj_list = gen_list_of_data_single_neigh(data, sigma_list, device, grid_shape)
            sigma_list = torch.tensor(sigma_list, dtype=torch.float32).to(device)
            optimizer.zero_grad()

            train_noise_adj_b_chunked = train_adj_b_noisy_matrix.chunk(len(sigma_list), dim=0)
            score = []
            for i, sigma in enumerate(sigma_list):
                mask= torch.ones(1, width * height, width * height).to(device)
                A = train_noise_adj_b_chunked[i].unsqueeze(0).to(device)
                score_batch = model(A=A, node_features=train_noise_adj_b_chunked[i].to(device), mask=mask, noiselevel=sigma.item()).to(device)
                score.append(score_batch)
            score = torch.cat(score, dim=0).squeeze(-1).to(device)

            loss = loss_func_kld(score, torch.stack(train_noise_adj_b_chunked).squeeze(1), data, torch.stack(grad_log_noise_adj_list), sigma_list,sigma_ind_list,sigma_nontild_list, device)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().item())
        scheduler.step(epoch)
        wandb.log({"Loss": np.mean(train_losses)})
        print(f"Epoch {epoch+1}/{max_epoch}, Loss: {np.mean(train_losses)}")
        if np.isnan(np.mean(train_losses)):
            print("Loss is NaN. Stopping training.")
            break
        if np.mean(train_losses) < best_loss:
            best_loss = np.mean(train_losses)
            save_model(model, 'trained_model_vlb_neighbor.pth')
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
    n_nodes=16,
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




fit(model, optimizer, dataloader, max_epoch=400, device=device)