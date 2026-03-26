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
    
class tanh(Function):
    def initialization(self, head, tail):
        return np.sqrt(2 / (head + tail))

    def value(self, X):
        return np.tanh(X)
    
    def diff(self, Y):
        return 1 - Y**2
    
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



class OutputFunction(ABC):
    def __init__(self, num_data, num_output_dim=1):
        self.N = num_data
        self.data_size = num_data * num_output_dim

    @abstractmethod
    def value(self, X):
        pass

    @abstractmethod
    def Loss(self, P, Y):
        pass

    @abstractmethod
    def dLoss(self, P, Y):
        pass

class Softmax(OutputFunction):
        
    def value(self, X):
        X_max = np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X - X_max)
        sum = np.sum(exp_X,axis=1,keepdims=True)
        return exp_X / (sum + 1e-15)
    
    def Loss(self, P, Y):
        # Pが0や1にならないように極小値を挟む        
        eps = 1e-15
        P_clipped = np.clip(P, eps, 1 - eps)
        logP = np.log(P_clipped)
        loss = -np.sum(Y * logP) / self.N
        return loss
   
    def dLoss(self, P, Y):
        return (P - Y) / self.N
    
class Identity(OutputFunction):
    
    def value(self, X):
        return X
    
    def Loss(self, P, Y):
        return np.mean((P - Y)**2)
    
    def dLoss(self, P, Y):
        return 2*(P - Y) / self.data_size
    


class FunctionBox:
    def __init__(self, act_fn :Function, output_fn :OutputFunction):
        self.act = act_fn
        self.output = output_fn
