from abc import ABC, abstractmethod
import numpy as np

class Function(ABC):

    @abstractmethod
    def initialization(self, head: int, tail: int) -> float:
        pass

    @abstractmethod
    def value(self, X):
        pass

    @abstractmethod
    def diff(self, Y):
        pass

class Sigmoid(Function):
    def initialization(self, head, tail):
        return np.sqrt(2 / (head + tail))

    def value(self, X):
        exp = np.exp(-X)
        return 1 / (exp + 1)
    
    def diff(self, Y):
        return Y - Y**2
    
class ReLU(Function):
    def initialization(self, head, tail):
        return np.sqrt(2 / head)
    
    def value(self, X):
        return np.where(X >= 0, X, -1e-15)
    
    def diff(self, Y):
        return np.where(Y >= 0, 1, 0)
    
class LeakyReLU(Function):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def initialization(self, head, tail):
        return np.sqrt(2 / head)
    
    def value(self, X):
        return np.where(X >= 0, X, X * self.alpha)
    
    def diff(self, Y):
        return np.where(Y >= 0, 1, self.alpha)