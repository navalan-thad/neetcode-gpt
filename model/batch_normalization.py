import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists

        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)

            running_mean = np.round(((1-momentum)*np.array(running_mean) + momentum*batch_mean), 4)
            running_var = np.round(((1-momentum)*np.array(running_var) + momentum*batch_var), 4)
        else:
            x_hat = (x - np.array(running_mean)) / np.sqrt(np.array(running_var) + eps)
        y = np.round(gamma*x_hat + beta, 4)

        return [y, running_mean, running_var]
        
