import numpy as np

class RegressaoLinearSimples:

    def fit(self, x, y):
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        sum1 = np.sum( (x - mean_x) * (y - mean_y) )
        sum2 = np.sum( (x - mean_x) ** 2 )
        self.w1 = sum1 / sum2
        self.w0 = mean_y - self.w1 * mean_x

    def predict(self, x):
        return self.w0 + self.w1 * x