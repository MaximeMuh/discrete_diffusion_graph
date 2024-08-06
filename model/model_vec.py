import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexModel(nn.Module):
    def __init__(self, nb_edges, noise_embedding_dim, hidden_dim, num_layers, output_dim):
        super(ComplexModel, self).__init__()
        self.nb_edges = nb_edges
        self.noise_embedding = nn.Sequential(
            nn.Linear(1, noise_embedding_dim),
            nn.ReLU()
        )
        
        layers = []
        input_dim = nb_edges + noise_embedding_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, edge_vectors, noise_levels):
        noise_levels = noise_levels.unsqueeze(-1).float()
        noise_embedding = self.noise_embedding(noise_levels)
        x = torch.cat([edge_vectors, noise_embedding], dim=-1)  
        logits = self.model(x)
        return logits

# nb_edges = 10
# noise_embedding_dim = 64
# hidden_dim = 128
# num_layers = 6
# output_dim = nb_edges

# model = ComplexModel(nb_edges, noise_embedding_dim, hidden_dim, num_layers, output_dim)

