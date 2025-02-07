from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, params, lr):
        self.params = params 
        self.lr = lr 

    def zero_grad(self):
        for param in self.params:
            param.grad = None 

    @abstractmethod 
    def step(self):
        pass 
        
class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, lr)
        self.params = list(params)

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad 

import numpy as np

class Adam(Optimizer):
    def __init__(self, params, lr: float = 0.001, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        super().__init__(params, lr)

        self.b1 = b1
        self.b2 = b2
        self.eps = eps

        self.m = [np.zeros_like(t.data) for t in params]
        self.v = [np.zeros_like(t.data) for t in params]
        self.t = 0

    def step(self):
        self.t += 1
        a = self.lr * (np.sqrt(1 - np.power(self.b2, self.t)) / (1 - np.power(self.b1, self.t)))
        for i, t in enumerate(self.params):
            assert t.grad is not None
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * t.grad
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * np.square(t.grad.data)
            t.data -= a * self.m[i] / (np.sqrt(self.v[i]) + self.eps)