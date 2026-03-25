from abc import ABC, abstractmethod
import numpy as np

class Function(ABC):

    @abstractmethod
    def value(self):
        pass

    @abstractmethod
    def diff(self):
        pass

class Sigmoid(Function):

    def value(self, X):
        exp = np.exp(-X)
        return 1 / (exp + 1)
    
    def diff(self, Y):
        return Y - Y**2