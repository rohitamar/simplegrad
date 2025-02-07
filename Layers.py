from abc import ABC, abstractmethod
import numpy as np
import math 

from Tensor import Tensor, Parameter
from Functional import Functional as F 

class Module(ABC):
    def __init__(self):
        self._modules = {}
        self.training = True 

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._modules[name] = value
        if isinstance(value, Module):
            for k, v in value._modules.items():
                self._modules[f"{name}_{k}"] = v 
        
        super().__setattr__(name, value)

    def parameters(self):
        for param in self._modules.values():
            yield param 
        
    def eval(self):
        self.training = False 
        for m in self._modules.values():
            m.eval()
        return self 
    
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        k = math.sqrt(1 / in_features)
        self.weights = Parameter(np.random.uniform(-k, k, size=(in_features, out_features)))
        if bias:
            self.bias = Parameter(np.random.uniform(-k, k, size = (1, out_features)))
        
    def forward(self, input):
        return input.mm(self.weights) + self.bias

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size 
        k = math.sqrt(6 / (fan_in + fan_out))
        self.filters = Parameter(np.random.uniform(-k, k, size = (out_channels, in_channels, kernel_size, kernel_size)))
        self.padding = padding 

    # def pad(self, x):
    #     p = self.padding
    #     x.data = np.pad(x.data, 
    #                     pad_width=((0, 0), (0, 0), (p, p), (p, p)),
    #                     mode='constant')
        
    def forward(self, input):
        if self.padding != 0:
            input = F.pad(input, self.padding)
        return F.conv2d(input, self.filters)

class MaxPool2d(Module):
    def __init__(self, width, height):
        super().__init__()
        self.width = width 
        self.height = height 

    def forward(self, input):
        return F.max_pool2d(input, (self.width, self.height))

class Dropout(Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
    
    def forward(self, input):
        return F.dropout(input, self.p, training=self.training)

class CrossEntropyLoss(Module):
    def forward(self, pred_logits, target):
        softmax_vals = Tensor.exp(pred_logits) / Tensor.sum(Tensor.exp(pred_logits))
        return -Tensor.sum(Tensor.log(softmax_vals) * target)
    
class MSELoss(Module):
    def forward(self, prediction, target):
        return Tensor.pow(prediction - target, 2)
    


