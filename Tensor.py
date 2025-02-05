from utils import topo_sort
import numpy as np 

import numpy as np

def align_brodcast(grad, target_shape):
    if target_shape == ():
        return grad.sum()
    
    original_target_shape = target_shape
    if len(grad.shape) > len(target_shape):
        target_shape = (1,) * (len(grad.shape) - len(target_shape)) + target_shape

    for axis, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim > 1:
            grad = grad.sum(axis=axis, keepdims=True)

    if len(target_shape) > len(original_target_shape):
        for _ in range(len(target_shape) - len(original_target_shape)):
            grad = np.squeeze(grad, axis=0)
    
    return grad.reshape(original_target_shape)

class Tensor:
    def __init__(self, data, children=[]):
        self.data = np.array(data)
        self.children = children
        self.grad = 0

    def numpy(self):
        return self.data
    
    @property
    def shape(self):
        return self.data.shape 

    def __repr__(self):
        return f"Tensor({np.array2string(self.data)})" if isinstance(self.data, np.ndarray) else f"Tensor({self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other) 

        def apply_self(grad):
            return align_brodcast(grad, self.data.shape)

        def apply_other(grad):
            return align_brodcast(grad, other.data.shape)
        
        return Tensor(
            self.data + other.data, 
            [(self, apply_self), (other, apply_other)]
        )
    
    def __radd__(self, other):
        return self.__add__(self, other)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        def apply_self(grad):
            return align_brodcast(grad, self.data.shape)
        def apply_other(grad):
            return align_brodcast(-grad, other.data.shape) 

        return Tensor(
            self.data - other.data, 
            [(self, apply_self), (other, apply_other)]
        )
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        def apply_self(grad):
            return grad * other.data 
        def apply_other(grad):
            return grad * self.data

        return Tensor(
            self.data * other.data, 
            [(self, apply_self), (other, apply_other)]
        )
    
    def __rmul__(self, other):
        return self * other 
    
    def __neg__(self):
        return self * -1
     
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        def apply_self(grad):
            grad_self = grad / other.data
            grad_self = align_brodcast(grad_self, self.data.shape)
            return grad_self 

        def apply_other(grad):
            grad_other = grad * -self.data / np.pow(other.data, 2)
            grad_other = align_brodcast(grad_other, other.data.shape)
            return grad_other 

        return Tensor(
            self.data  / other.data, 
            [(self, apply_self), (other, apply_other)]
        )
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self
    
    @staticmethod 
    def sum(x):
        def apply_self(grad):
            return grad 
        return Tensor(
            np.sum(x.data),
            [(x, apply_self)]
        )

    @staticmethod 
    def mean(x):
        def apply_self(grad):
            return grad 
        return Tensor(
            np.mean(x.data),
            [(x, apply_self)]
        )
    
    @classmethod
    def pow(cls, x, p):
        def apply_self(grad):
            return grad * p * np.pow(x.data, p - 1)
        return cls(
            np.pow(x.data, p),
            [(x, apply_self)]
        )
    
    @classmethod
    def log(cls, x):
        def apply_self(grad):
            return grad / x.data 
        return cls(
            np.log(x.data),
            [(x, apply_self)]
        )

    @classmethod 
    def exp(cls, x):
        def apply_self(grad):
            return grad * np.exp(x.data)
        return cls(
            np.exp(x.data),
            [(x, apply_self)]
        )

    def mm(self, other):
        assert self.data.ndim == 2 and other.data.ndim == 2, (
            f"Both tensors must be 2-dimensional, but got shapes {self.data.shape} and {other.data.shape}."
        )

        assert self.data.shape[1] == other.data.shape[0], (
            f"Incompatible shapes for matrix multiplication: "
            f"self.data.shape = {self.data.shape}, other.data.shape = {other.data.shape}."
        )

        def apply_self(grad):
            return grad @ other.data.T 

        def apply_other(grad):
            return self.data.T @ grad

        return Tensor(
            self.data @ other.data,
            [(self, apply_self), (other, apply_other)]
        )

    @classmethod 
    def arange(cls, start, stop):
        return cls(np.arange(start, stop))

    @classmethod 
    def zeros(cls, shape):
        return cls(np.zeros(shape))

    @classmethod
    def ones(cls, shape):
        return cls(np.ones(shape))
    
    @classmethod
    def eye(cls, k):
        return cls(np.eye(k))

    def backward(self):
        assert len(self.data.shape) == 0, (
            f"grad can be only created for scalar outputs. Trying to backprop with shape {self.data.shape}."
        )
        self.grad = 1
        for node in topo_sort(self):
            for child_tup in node.children:
                c, apply_fn = child_tup
                g = apply_fn(node.grad)
                c.grad = g if c.grad is None else c.grad + g

class Parameter(Tensor):
    def __init__(self, data, children=[]):
        super().__init__(data, children)
    
