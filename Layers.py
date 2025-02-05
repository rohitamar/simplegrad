from abc import ABC, abstractmethod
import numpy as np
import math 

from Tensor import Tensor, Parameter

class Module(ABC):
    def __init__(self):
        self._modules = {}
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        for param in self._modules.values():
            yield param 

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        k = math.sqrt(1 / in_features)
        self.weights = Parameter(np.random.uniform(-k, k, size=(in_features, out_features)))
        if bias:
            self.bias = Parameter.zeros(out_features)
        
    def forward(self, input):
        return input.mm(self.weights) + self.bias

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Softmax(Module):
    pass 

class CrossEntropyLoss(Module):
    pass 

class MSELoss(Module):
    def forward(self, prediction, target):
        return Tensor.pow(prediction - target, 2)
        

#ReLU
#Softmax
# optimizer
# SGD, Adam
# criterion's

# class Softmax(Module):
#     def __init__(self):


