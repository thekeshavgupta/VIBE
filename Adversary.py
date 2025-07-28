import torch
from torch import nn
import numpy as np
from sklearn.decomposition import PCA

class Adversary(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.relu_layer = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def getImportantFeatureRepresentations(self, x: torch.tensor):
        return self.layer2(self.relu_layer(self.layer1(x)))
    
    def remove_bias_using_projection(self, biased_input: torch.tensor, k: int = 1, enable_debiasing: bool = False):
        if not enable_debiasing:
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
        debiased_outputs = self.remove_bias_using_projection(biased_outputs, k=1, enable_debiasing=True)
        return self.sigmoid(self.layer3(self.relu_layer(debiased_outputs)))
    
