from typing import List
import torch
from torch import nn, Tensor
import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

class Adversary(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], enable_debiasing: bool = False):
        super().__init__()
        
        if not hidden_layers:
            raise ValueError("hidden_layers cannot be empty")
        
        # Build network layers
        layers = []
        layer_dims = [input_dim] + hidden_layers + [input_dim]
        for i in range(len(layer_dims) - 1):
            layers.extend([
                nn.Linear(layer_dims[i], layer_dims[i + 1]),
                nn.ReLU(),
                nn.BatchNorm1d(layer_dims[i + 1])
            ])
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        
        self.enable_debiasing = enable_debiasing
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        self.pca = PCA()
        self._pca_fitted = False
    
    def getImportantFeatureRepresentations(self, x: Tensor) -> Tensor:
        return self.feature_extractor(x)
    
    def remove_bias_using_projection(self, biased_input: Tensor, k: int = 1) -> Tensor:
        if not self.enable_debiasing:
            return biased_input
            
        try:
            # Move to CPU for numpy operations
            bias_detached = biased_input.cpu().detach().numpy()
            
            # Fit PCA if not already fitted
            if not self._pca_fitted:
                self.pca.fit(bias_detached)
                self._pca_fitted = True
                
            # Center the input
            bias_mean = bias_detached.mean(axis=0)
            biased_input_centered = bias_detached - bias_mean
            
            # Get top k components and project
            top_k_components = self.pca.components_[:k].T
            pca_alignment = np.dot(biased_input_centered, top_k_components)
            reprojected_bias = np.dot(pca_alignment, top_k_components.T)
            
            # Remove projected bias and convert back to tensor
            debiased = torch.tensor(
                biased_input_centered - reprojected_bias, 
                dtype=torch.float32,
                device=self.device
            )
            
            return debiased
            
        except (ValueError, NotFittedError) as e:
            print(f"Warning: Error in bias removal - {str(e)}")
            return biased_input
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        biased_outputs = self.getImportantFeatureRepresentations(x)
        debiased_outputs = self.remove_bias_using_projection(biased_outputs)
        return self.classifier(debiased_outputs)
    
