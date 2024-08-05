import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

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

def draw_maze(T, width, height):
    pos = {(x, y): (x, y) for x, y in T.nodes()}
    plt.figure(figsize=(10, 10))
    nx.draw(T, pos=pos, with_labels=False, node_size=10, width=2, edge_color='blue')
    plt.xlim(-1, width)
    plt.ylim(-1, height)
    plt.gca().invert_yaxis()
    plt.show()

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

def draw_maze_from_matrix(adj_matrix, width, height):
    G = graph_from_adjacency_matrix(adj_matrix, width, height)
    pos = {(x, y): (x, y) for x, y in G.nodes()}
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos=pos, with_labels=False, node_size=10, width=2, edge_color='blue')
    plt.xlim(-1, width)
    plt.ylim(-1, height)
    plt.gca().invert_yaxis()
    plt.show()

def upper_triangular_to_vector(adj_matrix):
    size = len(adj_matrix)
    upper_triangular_vector = []
    
    for i in range(size):
        for j in range(i + 1, size):  
            upper_triangular_vector.append(adj_matrix[i, j])
    
    return upper_triangular_vector

def vector_to_upper_triangular(vector, size):
    adj_matrix = np.zeros((size, size), dtype=int)
    index = 0
    
    for i in range(size):
        for j in range(i + 1, size):
            adj_matrix[i, j] = vector[index]
            adj_matrix[j, i] = vector[index]  
            index += 1
    
    return adj_matrix

def load_graphs(filename):
    with open(filename, 'rb') as file:
        graphs = pickle.load(file)
    return graphs

def graph_to_vector(G, width, height):
    adj_matrix = adjacency_matrix(G, width, height)
    vector = upper_triangular_to_vector(adj_matrix)
    return vector
def graph_to_adj(G, width, height):
    adj_matrix = adjacency_matrix(G, width, height)
    return adj_matrix

def vector_to_graph(vector, width, height):
    size = width * height
    adj_matrix = vector_to_upper_triangular(vector, size)
    G = graph_from_adjacency_matrix(adj_matrix, width, height)
    return G

def visualize_graph_from_vector(vector, width, height, title="Graph from Vector"):
    G = vector_to_graph(vector, width, height)
    draw_maze_from_matrix(nx.to_numpy_array(G), width, height)

class GraphDataset(Dataset):
    def __init__(self, filename, width, height):
        self.graphs = load_graphs(filename)
        self.width = width
        self.height = height
        self.vectors = [graph_to_vector(graph, width, height) for graph in self.graphs]

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return torch.tensor(self.vectors[idx], dtype=torch.float32)

class GraphDataset_adj(Dataset):
    def __init__(self, filename, width, height):
        self.graphs = load_graphs(filename)
        self.width = width
        self.height = height
        self.adjs = [graph_to_adj(graph, width, height) for graph in self.graphs]

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        return torch.tensor(self.adjs[idx], dtype=torch.float32)



def get_dataloader(filename, width, height, batch_size, shuffle=True):
    dataset = GraphDataset(filename, width, height)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def get_dataloader_adj(filename, width, height, batch_size, shuffle=True):
    dataset = GraphDataset_adj(filename, width, height)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader



def visualize_batch_from_dataloader(dataloader, width, height):
    for batch_idx, data in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        for vector in data:
            visualize_graph_from_vector(vector.numpy(), width, height, f"Graph from Vector - Batch {batch_idx + 1}")


def visualize_graph(data, width, height):
    print(data.size())
    for vector in data:
        print(vector.size())    
        visualize_graph_from_vector(vector.numpy(), width, height)






def main():
    filename = 'dataset/usts_4.pkl'
    width, height = 4, 4
    batch_size = 16  

    dataloader = get_dataloader(filename, width, height, batch_size)
    ite = next(iter(dataloader))
    print(ite.size())
    vector = ite[0]
    adjacency_matrix = vector_to_upper_triangular(vector.numpy(), width * height)
    print(adjacency_matrix)
    g = graph_from_adjacency_matrix(adjacency_matrix, width, height)
    draw_maze(g, width, height)

if __name__ == "__main__":
    main()
