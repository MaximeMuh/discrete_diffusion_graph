import networkx as nx
import matplotlib.pyplot as plt
import pickle

def load_graphs(filename):
    with open(filename, 'rb') as file:
        graphs = pickle.load(file)
    return graphs

def visualize_graph(G, title="Uniform Spanning Tree"):
    plt.figure(figsize=(8, 8))
    pos = {(x, y): (y, -x) for x, y in G.nodes()}  # Positionner les noeuds sur une grille
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(title)
    plt.show()

def main():
    filename = 'dataset/usts.pkl'
    graphs = load_graphs(filename)
    
    # Visualiser le premier graphe comme exemple
    if graphs:
        visualize_graph(graphs[110], "Sample Uniform Spanning Tree")
    else:
        print("Aucun graphe trouvé dans le fichier.")

if __name__ == "__main__":
    main()
