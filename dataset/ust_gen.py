import networkx as nx
import matplotlib.pyplot as plt
import pickle
import numpy as np

def generate_ust(grid_size):
    """
    Generates a random uniform spanning tree (UST) for a given grid size.
    
    Args:
        grid_size (int): The size of the grid (grid_size x grid_size).
    
    Returns:
        nx.Graph: A random uniform spanning tree.
    """
    G = nx.grid_2d_graph(grid_size, grid_size)
    ust = nx.Graph(nx.random_spanning_tree(G))
    return ust

def generate_multiple_ust(num_graphs, grid_size):
    """
    Generates multiple random uniform spanning trees (UST).
    
    Args:
        num_graphs (int): The number of trees to generate.
        grid_size (int): The size of the grid (grid_size x grid_size).
    
    Returns:
        list: A list of random uniform spanning trees.
    """
    ust_list = []
    for _ in range(num_graphs):
        ust = generate_ust(grid_size)
        ust_list.append(ust)
    return ust_list

def save_graphs(filename, graphs):
    """
    Saves a list of graphs to a file.
    
    Args:
        filename (str): The name of the file to save the graphs.
        graphs (list): The list of graphs to save.
    """
    with open(filename, 'wb') as file:
        pickle.dump(graphs, file)

def visualize_graph(G, title="Uniform Spanning Tree"):
    """
    Visualizes a graph using matplotlib.
    
    Args:
        G (nx.Graph): The graph to visualize.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(8, 8))
    pos = {(x, y): (y, -x) for x, y in G.nodes()}
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(title)
    plt.show()

def main():
    """
    Main function to generate, save, and visualize uniform spanning trees.
    """
    num_graphs = 300
    grid_size = 10
    filename = 'dataset/usts_10.pkl'
    
    ust_list = generate_multiple_ust(num_graphs, grid_size)
    save_graphs(filename, ust_list)
    print("done")
    print(ust_list[6])
    print(type(ust_list[0]))
    visualize_graph(ust_list[0], "Sample Uniform Spanning Tree")
    if ust_list:
        visualize_graph(ust_list[0], "Sample Uniform Spanning Tree")
    else:
        print("No graph found.")

if __name__ == "__main__":
    main()