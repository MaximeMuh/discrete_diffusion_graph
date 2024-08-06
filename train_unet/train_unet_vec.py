import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



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
from model.unet import Unet
from utils.graphs import discretenoise, loss_func_bce, upper_flatten_to_adj_matrix, adj_matrix_to_upper_flatten, discretenoise_neighbor, gen_list_of_data_single_neigh, loss_func_bce_adj, loss_cycle_consistency

filename = 'dataset/usts_8.pkl'
width, height = 8, 8  
batch_size = 32
dataloader = get_dataloader_adj(filename, width, height, batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
grid_shape = (8, 8)
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
            
            train_adj_b_noisy_matrix, grad_log_noise_adj_list = gen_list_of_data_single_neigh(data, sigma_list, device, grid_shape=grid_shape)
            train_noise_adj_b_chunked = torch.stack(train_adj_b_noisy_matrix.chunk(len(sigma_list), dim=0))
            sigma_list = torch.tensor(sigma_list, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            train_noise_adj_b_chunked = train_noise_adj_b_chunked.unsqueeze(1)

            score = model(train_noise_adj_b_chunked, sigma_list)
            score = score.squeeze(1)

            l = loss_func_bce_adj(score, torch.stack(grad_log_noise_adj_list), sigma_list, device)
            # for i, sigma in enumerate(sigma_list):
            #     l += loss_cycle_consistency(score[i].unsqueeze(0) , train_noise_adj_b_chunked.squeeze(1)[i].unsqueeze(0), sigma.item(),  device, grid_shape=grid_shape)
            l.backward()
            optimizer.step()
            train_losses.append(l.detach().cpu().item())
        scheduler.step(epoch)
        wandb.log({"Loss": np.mean(train_losses)})
        print(f"Epoch {epoch+1}/{max_epoch}, Loss: {np.mean(train_losses)}")
        if epoch % 20 == 0:
            save_model(model, 'trained_model_unet_vec.pth')
    wandb.finish()









model = Unet(dim=64).to(device)

optimizer = optim.Adam(model.parameters(),
                           lr=0.001,
                           betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=0.0)
max_epoch = 2  

fit(model, optimizer, dataloader, max_epoch=1000, device=device)