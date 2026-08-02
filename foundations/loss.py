import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        y_pred += 1e-7
        ans = y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)
        return -np.round(np.sum(ans) / len(y_true), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:

        y_pred += 1e-7
        ans = y_true*np.log(y_pred) / len(y_true)
        return -np.round(np.sum(ans), 4)
