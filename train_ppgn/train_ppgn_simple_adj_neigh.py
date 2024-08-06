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
from dataset.extract_graph import get_dataloader, get_dataloader_adj
from model.model2 import Powerful
from utils.graphs import discretenoise, loss_func_bce, upper_flatten_to_adj_matrix, adj_matrix_to_upper_flatten, discretenoise_neighbor, loss_func_bce_adj, discretenoise_adj, gen_list_of_data_single, gen_list_of_data_single_neigh, loss_cycle_consistency

filename = 'dataset/usts_5.pkl'
width, height = 5, 5
batch_size = 32
grid_shape = (width, height)
dataloader = get_dataloader_adj(filename, width, height, batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model(model, path):
    torch.save(model.state_dict(), path)

def fit(model, optimizer, dataloader, max_epoch=20, device=device):
    wandb.init(project='diffusion_graph', entity='maxime-muhlethaler-Inria')
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    best_loss = float('inf')
    best_model_path = 'best_model_adj_neigh_cycle_conn_5.pth'
    
    for epoch in range(max_epoch):
        train_losses = []
        model.train()
        for data in dataloader:
            sigma_list = list(np.random.uniform(low=0.0, high=0.5, size=data.size(0)))
            data = data.to(device)
            if isinstance(sigma_list, float):
                sigma_list = [sigma_list]
            train_adj_b_noisy_matrix, grad_log_noise_adj_list = gen_list_of_data_single_neigh(data, sigma_list, device, grid_shape)
            optimizer.zero_grad()
            train_noise_adj_b_chunked = train_adj_b_noisy_matrix.chunk(len(sigma_list), dim=0)
            score = []
            sigma_list = torch.tensor(sigma_list, dtype=torch.float32).to(device)
            l = 0
            for i, sigma in enumerate(sigma_list):
                mask = torch.ones(1, width * height, width * height).to(device)
                A = train_noise_adj_b_chunked[i].unsqueeze(0).to(device)
                score_batch = model(A=A, node_features=train_noise_adj_b_chunked[i].to(device), mask=mask, noiselevel=sigma.item()).to(device)
                score.append(score_batch)
                l += loss_cycle_consistency(score_batch.squeeze(-1), A, sigma.item(), device, grid_shape)
            score = torch.cat(score, dim=0).squeeze(-1).to(device)
            l += loss_func_bce_adj(score, torch.stack(grad_log_noise_adj_list), sigma_list, device)
            
            l.backward()
            optimizer.step()
            train_losses.append(l.detach().cpu().item())
        
        avg_loss = np.mean(train_losses)
        scheduler.step(epoch)
        wandb.log({"Loss": avg_loss})
        print(f"Epoch {epoch+1}/{max_epoch}, Loss: {avg_loss}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, best_model_path)
            print(f"New best model saved with loss: {best_loss}")
    
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

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0
)

fit(model, optimizer, dataloader, max_epoch=2000, device=device)