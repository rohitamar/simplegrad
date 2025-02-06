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
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        c = 0
        while len(input.shape) < 4:
            c += 1
            input = input[np.newaxis, :]
        
        n, c, h, w = input.shape 
        kh, kw = kernel_size 
        out_h, out_w = (h - kh) // kh + 1, (w - kw) // kw + 1
        
        input = input.reshape(n, c, out_h, kh, out_w, kw)
        # rewrite with np.swapaxes
        input = np.transpose(input, (0, 1, 2, 4, 3, 5))
        max_out = input.max(axis=(-1, -2))
        
        y = input.reshape(*input.shape[:-2], -1)
        i, j = np.unravel_index(np.argmax(y, axis=-1), kernel_size)
        A, B, C, D = np.indices((n, c, out_h, out_w))
        A, B, C, D, i, j = A.flatten(), B.flatten(), C.flatten(), D.flatten(), i.flatten(), j.flatten()
        
        back_grad = np.zeros((n, c, out_h, out_w, kh, kw))
        back_grad[A, B, C, D, i, j] = 1
        back_grad = np.transpose(back_grad, (0, 1, 2, 4, 3, 5))
        back_grad = back_grad.reshape(n, c, out_h, kh, out_w * kw).reshape(n, c, out_h * kh, out_w * kw)
        
        for _ in range(c):
            max_out, back_grad = np.squeeze(max_out), np.squeeze(back_grad)
        
        def apply_self(grad):
            return grad * back_grad 
        return Tensor(
            max_out, 
            [(input, apply_self)]
        ) 


    @staticmethod 
    def conv2d(input, filters):
        def convolve(input, kernel):
            c = 0
            while len(input) < 4:
                c += 1
                input = input[np.newaxis, :]
            
            submatrices_shape = input.shape[:-2] + tuple(np.subtract(input.shape[-2:], kernel.shape[-2:]) + 1) + kernel.shape[-2:]
            strides = input.strides + input.strides[-2:]
            
            sub_matrices = np.lib.stride_tricks.as_strided(input, submatrices_shape, strides)
            sub_matrices = np.rollaxis(sub_matrices, 1, 4)

            convolved = np.einsum('hwnij,oij->hwo', sub_matrices, kernel)
            convolved = np.rollaxis(convolved, 3, 1)

            for _ in range(c):
                convolved = np.squeeze(convolved)
            return convolved
        
        return Tensor(
            convolve(input.data, filters.data), 
            []
        )
    