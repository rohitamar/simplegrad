from utils import topo_sort
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
        return f"Tensor({self.data})"

    def __add__(self, other):
        return Tensor(
            self.data + other.data, 
            [(self, 1), (other, 1)]
        )
    
    def __sub__(self, other):
        return Tensor(
            self.data - other.data, 
            [(self, 1), (other, -1)]
        )

    def __mul__(self, other):
        return Tensor(
            self.data * other.data, 
            [(self, other.data), (other, self.data)]
        )
     
    def __div__(self, other):
        return Tensor(
            self.data  / other.data, 
            [(self, 1 / other.data), (other, -self.data / (other.data ** 2))]
        )
    
    def backward(self):
        self.grad = 1
        for node in topo_sort(self):
            for child, chain_grad in node.children:
                child.grad += chain_grad * node.grad 
        