import networkx as nx
import matplotlib.pyplot as plt
import pickle
import numpy as np

def generate_ust(grid_size):
    # Créer un graphe en grille
    G = nx.grid_2d_graph(grid_size, grid_size)
    
    # Générer un arbre couvrant uniforme aléatoire
    ust = nx.Graph(nx.random_spanning_tree(G))
    
    return ust

def generate_multiple_ust(num_graphs, grid_size):
    ust_list = []
    for _ in range(num_graphs):
        ust = generate_ust(grid_size)
        ust_list.append(ust)
    return ust_list

def save_graphs(filename, graphs):
    with open(filename, 'wb') as file:
        pickle.dump(graphs, file)

def visualize_graph(G, title="Uniform Spanning Tree"):
    plt.figure(figsize=(8, 8))
    pos = {(x, y): (y, -x) for x, y in G.nodes()}  # Positionner les noeuds sur une grille
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(title)
    plt.show()

def main():
    num_graphs = 1000
    grid_size = 4
    filename = 'dataset/usts_4.pkl'
    
    # Générer et sauvegarder les graphes
    ust_list = generate_multiple_ust(num_graphs, grid_size)
    save_graphs(filename, ust_list)
    
    # Visualiser le premier graphe comme exemple
    if ust_list:
        visualize_graph(ust_list[0], "Sample Uniform Spanning Tree")
    else:
        print("Aucun graphe trouvé.")

if __name__ == "__main__":
    main()
