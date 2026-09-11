import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        result = []
        curr = x

        with torch.no_grad():
            for layer in model:
                curr = layer(curr)
                if isinstance(layer, nn.Linear):
                    mean = round(curr.mean().item(), 4)
                    std = round(curr.std().item(), 4)
                    df = round((curr <= 0).all(dim=0).float().mean().item(), 4)
                    result.append({'mean': mean, 'std': std, 'dead_fraction': df})
        return result

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        res = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                res.append({
                    'mean': round(grad.mean().item(), 4),
                    'std': round(grad.std().item(), 4),
                    'norm': round(torch.norm(grad).item(), 4)
                })
        return res


    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for i in range(len(activation_stats)):
            if activation_stats[i]['dead_fraction'] > 0.5:
                return 'dead_neurons'

        for i in range(len(gradient_stats)):
            if gradient_stats[i]['norm'] > 1000:
                return 'exploding_gradients'

        for i in range(len(gradient_stats)):
            if gradient_stats[i]['norm'] < 1e-5:
                return 'vanishing_gradients'

        for i in range(len(activation_stats)):
            if activation_stats[i]['std'] < 0.1:
                return 'vanishing_gradients'
            elif activation_stats[i]['std'] > 10:
                return 'exploding_gradients'

        return 'healthy'
                



