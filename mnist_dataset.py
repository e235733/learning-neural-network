from sklearn.datasets import fetch_openml
import numpy as np

class MnistDataset:
    def __init__(self, n_samples=2000):
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)

        X = mnist.data[:n_samples].astype(np.float32)
        Y = mnist.target[:n_samples].astype(np.int32)

        self.X = X / 255.0
        self.Y = np.identity(10)[Y]