from Tensor import Tensor 
import numpy as np 

class Functional:
    @staticmethod
    def tanh(x):
        def apply_self(grad):
            return grad * (1 - np.tanh(2 * x.data))
        return Tensor(
            np.tanh(x.data),
            [(x, apply_self)]
        )

    @staticmethod 
    def relu(x):
        def apply_self(grad):
            return (x.data > 0) * grad

        return Tensor(
            x.data * (x.data > 0),
            [(x, apply_self)]
        )
    
    @staticmethod
    def sigmoid(x):
        def F(x):
            return 1.0 / (1.0 + np.exp(-x))
        
        f = F(x.data)
        def apply_self(grad):
            nonlocal f
            return f * (1 - f) * grad

        return Tensor(
            f,
            [(x, apply_self)]
        )

    @staticmethod 
    def dropout(input, p, training=True):
        if not training:
            return input 
        mask = np.random.binomial(n=1, p=p, size=input.data.shape)
        data = input.data * mask.data * 1 / (1 - p)
        def apply_self(grad):
            return grad * mask 
        return Tensor(
            data,
            [(input, apply_self)]
        )
    
    @staticmethod 
    def max_pool2d(input, kernel_size):
        pass 

    @staticmethod 
    def conv2d(input, filters):
        pass 
    