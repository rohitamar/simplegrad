from utils import topo_sort
import numpy as np

class Tensor:
    def __init__(self, data, children=[]):
        self.data = data
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
        return Tensor(
            self.data + other.data, 
            [(self, 1), (other, 1)]
        )
    
    def __radd__(self, other):
        return self.__add__(self, other)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(
            self.data - other.data, 
            [(self, 1), (other, -1)]
        )
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(
            self.data * other.data, 
            [(self, other.data), (other, self.data)]
        )
    
    def __neg__(self):
        return self * -1
     
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor(
            self.data  / other.data, 
            [(self, 1 / other.data), (other, -self.data / (np.pow(other.data, 2)))]
        )
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self
    
    @classmethod
    def pow(cls, x, p):
        return cls(
            np.pow(x.data, p),
            [(x, (p - 1) * np.pow(x.data, p - 1))]
        )
    
    @classmethod
    def log(cls, x):
        return cls(
            np.log(x.data),
            [(x, 1 / x.data)]
        )

    def mm(self, other):
        assert self.data.ndim == 2 and other.data.ndim == 2, (
            f"Both tensors must be 2-dimensional, but got shapes {self.data.shape} and {other.data.shape}."
        )

        assert self.data.shape[1] == other.data.shape[0], (
            f"Incompatible shapes for matrix multiplication: "
            f"self.data.shape = {self.data.shape}, other.data.shape = {other.data.shape}."
        )
        return Tensor(
            self.data @ other.data,
            [(self, other.data), (other, self.data.T)]
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
    
    # @classmethod
    # def log(cls, x):
    #     return cls(
    #         np.log(x.data),

    #     )

    def backward(self):
        self.grad = 1
        for node in topo_sort(self):
            for child, chain_grad in node.children:
                child.grad += chain_grad * node.grad 

class Parameter(Tensor):
    def __init__(self, data, children=[]):
        super().__init__(data, children)
    
