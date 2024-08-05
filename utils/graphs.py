import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.extract_graph import load_graphs, graph_to_vector, vector_to_graph, visualize_graph_from_vector, GraphDataset



def gen_list_of_data_single(train_adj_b, sigma_list, device):
    """
    :param train_x_b: [batch_size, N, F_in], batch of feature vectors of nodes
    :param train_adj_b: [batch_size, N, N], batch of original adjacency matrices
    :param train_node_flag_b: [batch_size, N], the flags for the existence of nodes
    :param sigma_list: list of noise levels
    :return:
        train_x_b: [len(sigma_list) * batch_size, N, F_in], batch of feature vectors of nodes (w.r.t. `train_noise_adj_b`)
        train_noise_adj_b: [len(sigma_list) * batch_size, N, N], batch of perturbed adjacency matrices
        train_node_flag_b: [len(sigma_list) * batch_size, N], the flags for the existence of nodes (w.r.t. `train_noise_adj_b`)
        grad_log_q_noise_list: [len(sigma_list) * batch_size, N, N], the ground truth gradient (w.r.t. `train_noise_adj_b`)
    """
    assert isinstance(sigma_list, list)
    train_noise_adj_b_list = []
    grad_log_q_noise_list = []
    count=0
    for sigma_i in sigma_list:

        train_noise_adj_b, grad_log_q_noise = discretenoise_adj(train_adj_b[count],sigma=sigma_i, device=device)
        train_noise_adj_b_list.append(train_noise_adj_b)
        grad_log_q_noise_list.append(grad_log_q_noise)
        count=count+1

    train_noise_adj_b = torch.cat(train_noise_adj_b_list, dim=0)
    return  train_noise_adj_b, grad_log_q_noise_list



def gen_list_of_data_single_neigh(train_adj_b, sigma_list, device, grid_shape):
    """
    :param train_x_b: [batch_size, N, F_in], batch of feature vectors of nodes
    :param train_adj_b: [batch_size, N, N], batch of original adjacency matrices
    :param train_node_flag_b: [batch_size, N], the flags for the existence of nodes
    :param sigma_list: list of noise levels
    :return:
        train_x_b: [len(sigma_list) * batch_size, N, F_in], batch of feature vectors of nodes (w.r.t. `train_noise_adj_b`)
        train_noise_adj_b: [len(sigma_list) * batch_size, N, N], batch of perturbed adjacency matrices
        train_node_flag_b: [len(sigma_list) * batch_size, N], the flags for the existence of nodes (w.r.t. `train_noise_adj_b`)
        grad_log_q_noise_list: [len(sigma_list) * batch_size, N, N], the ground truth gradient (w.r.t. `train_noise_adj_b`)
    """
    assert isinstance(sigma_list, list)
    train_noise_adj_b_list = []
    grad_log_q_noise_list = []
    count=0
    for sigma_i in sigma_list:
        
        train_noise_adj_b, grad_log_q_noise = discretenoise_adj_neigh(train_adj_b[count],sigma=sigma_i, device=device, grid_shape=grid_shape)
        train_noise_adj_b_list.append(train_noise_adj_b)
        grad_log_q_noise_list.append(grad_log_q_noise)
        count=count+1

    train_noise_adj_b = torch.cat(train_noise_adj_b_list, dim=0)
    return  train_noise_adj_b, grad_log_q_noise_list



def loss_func_bce(score_list, grad_log_noise_vec, sigma, device):
    """
    Calcule la perte BCE modifiée pour des vecteurs représentant la partie supérieure des matrices d'adjacence.

    Args:
        score_list (torch.Tensor): Tensor de taille (Batch_size, num_elements) contenant les scores du modèle.
        grad_log_q_noise_list (torch.Tensor): Tensor de taille (Batch_size, num_elements) contenant les gradients de la log-vraisemblance du bruit.
        sigma_list (torch.Tensor): Tensor de taille (Batch_size) contenant les niveaux de bruit pour chaque élément du batch.
        device (torch.device): Dispositif sur lequel exécuter les calculs (CPU ou GPU MPS).
    
    Returns:
        torch.Tensor: La perte moyenne calculée sur le batch.
    """
    score_list = score_list.to(device)
    grad_log_noise_vec = grad_log_noise_vec.to(device)
    
    
    BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
    
    loss_matrix = BCE(score_list, grad_log_noise_vec)
    
    sigma_expanded = sigma.unsqueeze(-1).expand_as(grad_log_noise_vec)
    
    adjustment_factor = 1 - 2 * sigma_expanded + 1.0 / sigma.size(0)
    loss_matrix = loss_matrix * adjustment_factor
    
    loss = torch.mean(loss_matrix)
    
    return loss

def loss_func_bce_adj(score_list, grad_log_q_noise_list, sigma_list, device):

    loss = 0.0
    BCE = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss_matrix = BCE(score_list,grad_log_q_noise_list)
    loss_matrix = loss_matrix * (1-2*torch.tensor(sigma_list).unsqueeze(-1).unsqueeze(-2).expand(grad_log_q_noise_list.size(0),grad_log_q_noise_list.size(1),grad_log_q_noise_list.size(2)).to(device)+1.0/len(sigma_list))
    # Loss analogue to https://arxiv.org/pdf/2111.12701.pdf
    loss_matrix = (loss_matrix+torch.transpose(loss_matrix, -2, -1))/2
    loss_matrix = loss_matrix 
    loss = torch.mean(loss_matrix)
    return loss



def loss_func_kld(score_list, train_noise_adj_b, train_adj_b, grad_log_q_noise_list, sigma_list, sigma_ind_list,sigma_nontild_list, device):
    loss = 0.0
    kl_loss = nn.KLDivLoss(reduction="none")

    # Need to compute wether switch would go to on or to off (since model just predicts if we switched and not in which direction)
    for i, adj in enumerate(train_noise_adj_b):
        sigmatilde_t = sigma_list[i]
        sigma_t = sigma_nontild_list[i]
        sigmatilde_t1 = sigma_list[i] - sigma_list[i] / sigma_ind_list[i]
        # Compute q which is the posterior on each matrix element but with knowing x0 and knowing xt which means we need both as arguments
        mult1 = torch.where(train_noise_adj_b[i]>1/2, 1-sigma_t, sigma_t)
        mult2 = torch.where(train_adj_b[i]>1/2, 1-sigmatilde_t1, sigmatilde_t1)
        xor = torch.logical_xor(train_noise_adj_b[i], train_adj_b[i])
        div = torch.where(xor>1/2, sigmatilde_t, 1-sigmatilde_t)
        q = mult1 * mult2 / div
        # Change score list based on if xt is 0 or 1 
        score_i=torch.where(train_noise_adj_b[i]>1/2, 1-torch.sigmoid(score_list[i]), torch.sigmoid(score_list[i]))
        # score list represents p(x0|xt)
        # Calculate posterior(sigmatilde_t,sigma_t,sigmatilde_t1,0,xt)
        mult1 = torch.where(train_noise_adj_b[i] > 1/2, (1-sigma_t), sigma_t)
        mult2 = torch.where(torch.zeros_like(train_adj_b[i])>1/2, 1-sigmatilde_t1, sigmatilde_t1)
        xor = torch.logical_xor(train_noise_adj_b[i], torch.zeros_like(train_adj_b[i]))
        div = torch.where(xor>1/2, sigmatilde_t, 1-sigmatilde_t)
        p = ( 1 - score_i ) * mult1*mult2/div

        # Calculate posterior(sigmatilde_t,sigma_t,sigmatilde_t1,1,xt)
        mult1 = torch.where(train_noise_adj_b[i]>1/2, 1-sigma_t, sigma_t)
        mult2 = torch.where(torch.ones_like(train_adj_b[i])>1/2, 1-sigmatilde_t1, sigmatilde_t1)
        xor = torch.logical_xor(train_noise_adj_b[i], torch.ones_like(train_adj_b[i]))
        div = torch.where(xor>1/2, sigmatilde_t, 1-sigmatilde_t)
        p += ( score_i ) * mult1 * mult2/div
        # p stands for probablity p(x0=1|xt=xt) now
        score_list[i] = p
        # This q is q(x0=1|xt=xt,x0=x0)
        grad_log_q_noise_list[i] = q
        score_inv = 1-score_list[i]
        score_list_twoclass = torch.cat([score_list[i].unsqueeze(-1), score_inv.unsqueeze(-1)], -1)
        grad_inv = 1 - grad_log_q_noise_list[i]
        grad_log_q_noise_list_twoclass = torch.cat([grad_log_q_noise_list[i].unsqueeze(-1), grad_inv.unsqueeze(-1)], -1)
        loss_matrix = kl_loss(torch.log(score_list_twoclass), grad_log_q_noise_list_twoclass).to(device)
        loss_matrix = loss_matrix.sum(-1)
        loss_matrix = (loss_matrix+torch.transpose(loss_matrix, -2, -1))/2
        loss_matrix = loss_matrix.to(device)
        # Exclude the diagonal elements
        loss_matrix = torch.triu(loss_matrix, diagonal=1) + torch.tril(loss_matrix, diagonal = -1)
        loss += loss_matrix.sum()
    return loss








def add_bernoulli( init_adjs, noiselevel, device, grid_shape):
    init_adjs, noise = discretenoise_adj_neigh(init_adjs, noiselevel, device, grid_shape)

    return init_adjs

def take_step(model_noise, init_adjs, noiselevel, device, grid_shape):
    init_adjs = add_bernoulli( init_adjs, noiselevel, device, grid_shape)
    mask = torch.ones_like(init_adjs)
    noise_unnormal = model_noise
    noise_rel = torch.sigmoid(noise_unnormal)
    noise_rel = (noise_rel + noise_rel.transpose(-1, -2)) / 2
    noise = torch.bernoulli(noise_rel) * mask
    adjacency_mask = create_adjacency_mask(grid_shape)
    adjacency_mask = torch.tensor(adjacency_mask, dtype=torch.float32).to(device)
    inter_adjs = torch.where(noise > 1/2, init_adjs - 1, init_adjs)
    new_adjs = torch.where(inter_adjs < -1/2, inter_adjs + 2, inter_adjs)
    new_adjs *= adjacency_mask
    return init_adjs, new_adjs



def loss_cycle_consistency(model_noise, init_adjs, noiselevel,  device, grid_shape, lambda_cycle=1.0, lambda_conn=5.0):
    noisy_adjs, init_adjs = take_step(model_noise, init_adjs, noiselevel, device, grid_shape)
    batch_size, num_nodes, _ = init_adjs.shape
    total_loss = 0.0
    
    for graph in init_adjs:
        G = nx.from_numpy_matrix(graph.cpu().numpy())

        conn_loss = 0.0
        if not nx.is_connected(G):
            # print('not conn')
            conn_loss = lambda_conn

        cycles = nx.cycle_basis(G)
        cycle_count = len(cycles)
        # print(cycle_count)
        cycle_loss = cycle_count * lambda_cycle
        
        total_loss += conn_loss + cycle_loss
    
    return total_loss / batch_size




def discretenoise_adj(train_adj_b, sigma,device):
    train_adj_b = train_adj_b.to(device)

    ##if Aij=1 then chances for being 1 later is 1-sigma so chance of changing is sigma
    bernoulli_adj = torch.where(train_adj_b>1/2,torch.full_like(train_adj_b,1-sigma).to(device),torch.full_like(train_adj_b,sigma).to(device))
    noise_upper = torch.bernoulli(bernoulli_adj).triu(diagonal=1)
    noise_lower = noise_upper.transpose(-1, -2)
    grad_log_noise = torch.abs(-train_adj_b + noise_upper + noise_lower)
    train_adj_b = noise_upper + noise_lower

    return train_adj_b, grad_log_noise

def discretenoise_adj_neigh(train_adj_b, sigma,device, grid_shape):
    train_adj_b = train_adj_b.to(device)

    ##if Aij=1 then chances for being 1 later is 1-sigma so chance of changing is sigma
    bernoulli_adj = torch.where(train_adj_b>1/2,torch.full_like(train_adj_b,1-sigma).to(device),torch.full_like(train_adj_b,sigma).to(device))
    noise_upper = torch.bernoulli(bernoulli_adj).triu(diagonal=1)
    noise_lower = noise_upper.transpose(-1, -2)
    noise = noise_upper + noise_lower
    adjacency_mask = create_adjacency_mask(grid_shape)
    noise = noise * torch.tensor(adjacency_mask, dtype=torch.float32).to(device)
    grad_log_noise = torch.abs(-train_adj_b + noise)
    train_adj_b = noise

    return train_adj_b, grad_log_noise

def discretenoise(train_adj_b_vec, sigma, device):
    train_adj_b_vec = train_adj_b_vec.to(device)
    
    
    batch_size, num_elements = train_adj_b_vec.size()
    
    bernoulli_probs = sigma.unsqueeze(-1).expand(batch_size, num_elements)
    noise_vec = torch.bernoulli(bernoulli_probs)
    
    noise_probs = torch.where(
        train_adj_b_vec > 1/2,
        1 - sigma.unsqueeze(-1),
        sigma.unsqueeze(-1)
    )
    train_adj_b_noisy_vec = torch.bernoulli(noise_probs)
    
    grad_log_noise_vec = torch.abs(-train_adj_b_vec + train_adj_b_noisy_vec)
    
    return train_adj_b_noisy_vec, grad_log_noise_vec



def upper_flatten_to_adj_matrix(vector, num_nodes):
    batch_size = vector.size(0)
    adj_matrix = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.float32, device=vector.device)
    
    idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
    
    for b in range(batch_size):
        adj_matrix[b, idx[0], idx[1]] = vector[b]
        adj_matrix[b, idx[1], idx[0]] = vector[b]
    
    return adj_matrix

def adj_matrix_to_upper_flatten(adj_matrix):
    batch_size, num_nodes, _ = adj_matrix.size()
    
    idx = torch.triu_indices(num_nodes, num_nodes, offset=1)
    
    upper_flatten = adj_matrix[:, idx[0], idx[1]]
    
    return upper_flatten

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

def wilson_algorithm(G, start):
    U = {start}
    T = nx.Graph()
    while len(U) < len(G.nodes):
        u = random.choice(list(set(G.nodes) - U))
        path = [u]
        while u not in U:
            u = random.choice(list(G.neighbors(u)))
            if u in path:
                cycle_index = path.index(u)
                path = path[:cycle_index + 1]
            else:
                path.append(u)
        U.update(path)
        T.add_edges_from((path[i], path[i + 1]) for i in range(len(path) - 1))
    return T

def generate_ust_maze(width, height):
    G = nx.grid_2d_graph(width, height)
    start = (random.randint(0, width-1), random.randint(0, height-1))
    T = wilson_algorithm(G, start)
    return T

def adjacency_matrix(T, width, height):
    nodes = [(i, j) for j in range(height) for i in range(width)]
    index = {node: i for i, node in enumerate(nodes)}
    size = len(nodes)
    adj_matrix = np.zeros((size, size), dtype=int)
    for edge in T.edges():
        i, j = index[edge[0]], index[edge[1]]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    return adj_matrix

def graph_from_adjacency_matrix(adj_matrix, width, height):
    G = nx.Graph()
    size = len(adj_matrix)
    
    node_positions = [(i % width, i // width) for i in range(size)]
    G.add_nodes_from(node_positions)
    
    for i in range(size):
        for j in range(i + 1, size):
            if adj_matrix[i, j] == 1:
                node1 = node_positions[i]
                node2 = node_positions[j]
                G.add_edge(node1, node2)
    
    return G

def draw_maze_from_matrix(adj_matrix, width, height, title='Maze'):
    G = graph_from_adjacency_matrix(adj_matrix, width, height)
    pos = {(x, y): (x, y) for x, y in G.nodes()}
    plt.figure(figsize=(5, 5))
    nx.draw(G, pos=pos, with_labels=False, node_size=10, width=2, edge_color='blue')
    plt.xlim(-1, width)
    plt.ylim(-1, height)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()



def create_adjacency_mask(grid_shape):
    width, height = grid_shape
    size = width * height
    mask = np.zeros((size, size), dtype=int)
    
    for i in range(width):
        for j in range(height):
            current_index = i * height + j
            if j < height - 1:  # Horizontal connection
                right_index = current_index + 1
                mask[current_index, right_index] = 1
                mask[right_index, current_index] = 1
            if i < width - 1:  # Vertical connection
                bottom_index = current_index + height
                mask[current_index, bottom_index] = 1
                mask[bottom_index, current_index] = 1
    
    return mask

def discretenoise_neighbor(train_adj_b_vec, sigma, device, grid_shape):
    train_adj_b_vec = train_adj_b_vec.to(device)
    batch_size, num_elements = train_adj_b_vec.size()

    size = grid_shape[0] * grid_shape[1]
    adjacency_mask = create_adjacency_mask(grid_shape)
    adjacency_mask = torch.tensor(adjacency_mask, dtype=torch.float32).to(device)
    
    # Ensure the mask has the correct size for the input vector
    mask = adjacency_mask[np.triu_indices(size, k=1)]
    mask = torch.tensor(mask, dtype=torch.float32).to(device)
    
    bernoulli_probs = sigma.unsqueeze(-1).expand(batch_size, num_elements)
    noise_vec = torch.bernoulli(bernoulli_probs) * mask
    noise_probs = torch.where(
        train_adj_b_vec > 1/2,
        1 - sigma.unsqueeze(-1),
        sigma.unsqueeze(-1)
    )
    train_adj_b_noisy_vec = torch.bernoulli(noise_probs) * mask
    grad_log_noise_vec = torch.abs(-train_adj_b_vec + train_adj_b_noisy_vec)
    return train_adj_b_noisy_vec, grad_log_noise_vec




def get_dataloader(filename, width, height, batch_size, shuffle=True):
    dataset = GraphDataset(filename, width, height)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader



def create_adjacency_mask(grid_shape):
    rows, cols = grid_shape
    mask = np.zeros((rows * cols, rows * cols))
    for i in range(rows):
        for j in range(cols):
            if j < cols - 1:
                mask[i * cols + j, i * cols + j + 1] = 1
                mask[i * cols + j + 1, i * cols + j] = 1
            if i < rows - 1:
                mask[i * cols + j, (i + 1) * cols + j] = 1
                mask[(i + 1) * cols + j, i * cols + j] = 1
    return mask

def main():
    filename = 'dataset/usts_4.pkl'
    width, height = 4, 4
    batch_size = 16  
    grid_shape = (4, 4)  
    dataloader = get_dataloader(filename, width, height, batch_size)
    ite = next(iter(dataloader))
    print(ite.size())


    
    print(ite.size())
    print("Graphe initial:")
    adj = upper_flatten_to_adj_matrix(ite, grid_shape[0]* grid_shape[1])
    print(adj.size())
    draw_maze_from_matrix(adj[0], grid_shape[0], grid_shape[1])


    device = torch.device("cpu")
    sigma = torch.tensor([1], dtype=torch.float32).to(device)
    train_adj_b_noisy_vec, grad_log_noise_vec = discretenoise_neighbor(ite, sigma, device, grid_shape)
    print(train_adj_b_noisy_vec.size(), "aaa")
    noisy_adj_matrix = upper_flatten_to_adj_matrix(train_adj_b_noisy_vec, grid_shape[0] * grid_shape[1])
    print(noisy_adj_matrix.size())
    print("Graphe après ajout de bruit:")
    draw_maze_from_matrix(noisy_adj_matrix[0], grid_shape[0], grid_shape[1])








