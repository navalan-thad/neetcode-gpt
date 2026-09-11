import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2 / (fan_in + fan_out))
        sample = torch.round(torch.randn(fan_out, fan_in) * std, decimals=4)
        return sample.tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2/fan_in)
        sample = torch.randn(fan_out, fan_in) * std
        return torch.round(sample, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:

        torch.manual_seed(0)
        weights = []
        stds = []
        d_in = input_dim

        for _ in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2/(d_in + hidden_dim))
            elif init_type == 'kaiming':
                std = math.sqrt(2/d_in)
            else:
                std = 1
            
            weights.append(torch.randn(hidden_dim, d_in)*std)
            d_in = hidden_dim

        X = torch.randn(1, d_in)
        for w in weights:
            X = torch.relu(X @ w.T)
            stds.append(round(X.std().item(), 2))

        return stds




