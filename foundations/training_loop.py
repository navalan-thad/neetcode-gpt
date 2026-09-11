import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets

        """
        since y_hat = X@w + b, dydw = X, dydb = 1
        since L = 1/n * (y_hat - y)^2, dLdy = 2/n * (y_hat-y)
        thus, dLdw = dLdy * dydw = 2/n * (y_hat-y) * X
        and dLdb = dLdy * dydb = 2/n * (y_hat-y) * 1
        """
    
        w = np.zeros((X.shape[1],))
        b = 0
        n = X.shape[0]

        for _ in range(epochs):
            y_hat = X @ w + b
            loss = np.mean((y - y_hat)**2)

            dLdw = (2/n) * (X.T @ (y_hat - y))
            dLdb = (2/n) * np.sum(y_hat - y)

            w -= dLdw*lr
            b -= dLdb*lr
        
        return (np.round(w, 5), round(b, 5))

