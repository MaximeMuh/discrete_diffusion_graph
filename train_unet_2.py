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
from dataset.extract_graph import get_dataloader

from model.model2 import Powerful
from model.unet import Unet
from utils.graphs import discretenoise, loss_func_bce, upper_flatten_to_adj_matrix, adj_matrix_to_upper_flatten, discretenoise_neighbor

filename = 'dataset/usts_4.pkl'
width, height = 4, 4  
batch_size = 32 
dataloader = get_dataloader(filename, width, height, batch_size)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
grid_shape = (4, 4)
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
            sigma_list = torch.tensor(sigma_list, dtype=torch.float32).to(device)
            train_adj_b_noisy_vec, grad_log_noise_vec = discretenoise_neighbor(data, sigma_list, device, grid_shape)
            train_adj_b_noisy_matrix= upper_flatten_to_adj_matrix(train_adj_b_noisy_vec, width * height)
            optimizer.zero_grad()
            train_adj_b_noisy_matrix = train_adj_b_noisy_matrix.unsqueeze(1)
            score = model(train_adj_b_noisy_matrix, sigma_list)
            score = score.squeeze(1)
            score = adj_matrix_to_upper_flatten(score)
            l = loss_func_bce(score, grad_log_noise_vec, sigma_list, device)
            l.backward()
            optimizer.step()
            train_losses.append(l.detach().cpu().item())
        scheduler.step(epoch)
        wandb.log({"Loss": np.mean(train_losses)})
        print(f"Epoch {epoch+1}/{max_epoch}, Loss: {np.mean(train_losses)}")
        if epoch % 20 == 0:
            save_model(model, 'trained_model_unet_neighbor.pth')
    wandb.finish()








model = Unet(dim=64).to(device)

optimizer = optim.Adam(model.parameters(),
                           lr=0.001,
                           betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=0.0)
max_epoch = 2  

fit(model, optimizer, dataloader, max_epoch=100, device=device)