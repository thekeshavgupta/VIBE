import torch
from torch import nn
import numpy as np
from sklearn.decomposition import PCA

class Adversary(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], enable_debiasing = False):
        super().__init__()
        self.first_layer = nn.Linear(input_dim, hidden_layers[0])
        self.hidden_objects = [self.first_layer]
        for i in range(len(hidden_layers)-1):
            self.hidden_objects.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        self.hidden_objects.append(nn.Linear(hidden_layers[-1], input_dim))
        self.last_layer = nn.Linear(input_dim, 1)
        self.relu_layer = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.enable_debiasing = enable_debiasing
        
    def getImportantFeatureRepresentations(self, x: torch.tensor):
        output = x
        for layers in self.hidden_objects:
            output = layers(output)
            output = self.relu_layer(output)
        return output
    
    def remove_bias_using_projection(self, biased_input: torch.tensor, k: int = 1):
        if not self.enable_debiasing:
            return biased_input
        pca = PCA()
        bias_detached = biased_input.detach().numpy()
        pca.fit(bias_detached)
        biased_input_centered = bias_detached - bias_detached.mean()
        top_k_components = pca.components_[:,k].reshape(-1,1)
        pca_alignment_value = np.dot(biased_input_centered, top_k_components).reshape(-1,1)
        reprojected_bias_input = np.dot(pca_alignment_value, top_k_components.T)
        return torch.tensor(biased_input_centered - reprojected_bias_input)
        
    def forward(self, x: torch.tensor):
        biased_outputs = self.getImportantFeatureRepresentations(x)
        debiased_outputs = self.remove_bias_using_projection(biased_outputs, k=1)
        return self.sigmoid(self.last_layer(debiased_outputs))
    
