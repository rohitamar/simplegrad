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