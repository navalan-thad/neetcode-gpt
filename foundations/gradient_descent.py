class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:

        i = 0
        x = init
        while i < iterations:
            x = x - learning_rate*(2*x)
            i += 1
        return round(x, 5)
