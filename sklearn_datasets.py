from sklearn.datasets import make_moons
from sklearn.datasets import make_gaussian_quantiles

class MoonsDataset:
    def __init__(self, n_samples):
        X, Y = make_moons(n_samples=n_samples, noise=0.15)
        
        self.X = X
        self.Y = Y

class GaussianQuantilesDataset:
    def __init__(self, n_samples, n_features=2):
        X, Y = make_gaussian_quantiles(n_samples=n_samples, n_classes=2, n_features=n_features)
        
        self.X = X
        self.Y = Y