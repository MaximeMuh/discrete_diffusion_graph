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
from utils.graphs import discretenoise, loss_func_bce
from utils.graphs import discretenoise, loss_func_bce, upper_flatten_to_adj_matrix, adj_matrix_to_upper_flatten
filename = 'dataset/usts.pkl'
width, height = 5, 5  
batch_size = 16  
dataloader = get_dataloader(filename, width, height, batch_size)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


batch = next(iter(dataloader))
batch = upper_flatten_to_adj_matrix(batch, width * height)
print(batch.size())