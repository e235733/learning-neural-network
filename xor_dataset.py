import numpy as np

class XorDataset:
    def __init__(self, n, seed=None):
        self.rng = np.random.default_rng(seed)
        self.X = self.rng.uniform(low=-1.0, high=1.0, size=(n, 2))
        # xor に分類
        Y = np.where(self.X[:, 0] * self.X[:, 1] > 0, 0, 1)
        self.Y = np.identity(2)[Y]
    
if __name__ == "__main__":
    data = XorDataset(10)

    print(data.X)
    print(data.Y)