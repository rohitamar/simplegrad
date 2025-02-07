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
    def pad(input, p):
        padded_input = np.pad(input.data, 
                        pad_width=((0, 0), (0, 0), (p, p), (p, p)),
                        mode='constant')

        def apply_self(grad):
            return grad[..., p:-p, p:-p]
        
        return Tensor(
            padded_input, 
            [(input, apply_self)]
        )

    @staticmethod
    def max_pool2d(input, kernel_size):
        tensor_input = input 
        input = input.data 
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        n, c, h, w = input.shape 
        kh, kw = kernel_size 
        out_h, out_w = (h - kh) // kh + 1, (w - kw) // kw + 1
        
        input = input.reshape(n, c, out_h, kh, out_w, kw)
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
        
        def apply_self(grad):
            # print("a: ", back_grad.shape, grad.shape)
            upsampled_grad = np.repeat(np.repeat(grad, kw, axis = -2), kh, axis = -1)
            return upsampled_grad * back_grad
         
        return Tensor(
            max_out, 
            [(tensor_input, apply_self)]
        ) 


    @staticmethod 
    def conv2d(input, filters):

        # input.shape = (batch_size, c_in, h, w)
        # kernel.shape = (c_out, c_in, k_h, k_w)
        def cross_correlation(input, kernel):
            # assert len(input.shape) == 4 and kernel.shape == 4, (
            #     f"Expected input.shape and kernel.shape to be 4, but got input.shape={input.shape} and kernel.shape={kernel.shape}"
            # )
            # print("a: ", input.shape, kernel.shape)
            submatrices_shape = input.shape[:-2] + tuple(np.subtract(input.shape[-2:], kernel.shape[-2:]) + 1) + kernel.shape[-2:]
            strides = input.strides + input.strides[-2:]
            sub_matrices = np.lib.stride_tricks.as_strided(input, submatrices_shape, strides)
            sub_matrices = np.rollaxis(sub_matrices, 1, 4)
            # print("b: ", sub_matrices.shape, kernel.shape, "\n")
            convolved = np.einsum('bhwnij,onij->bhwo', sub_matrices, kernel)
            convolved = np.rollaxis(convolved, 3, 1)
            print("convv", convolved.shape)
            return convolved

        def transpose_conv(input, kernel):
            def ex_tw(x):
                return 
            # print(input.shape, kernel.shape)

            b, c_in, h, w = input.shape 
            kout_channel, kin_channel, kh, kw = kernel.shape 

            # out = np.zeros((h + kh - 1, w + kw - 1, b, c_in))
            out = np.zeros((b, kin_channel, h + kh - 1, w + kw - 1))
            # input = np.transpose(input, (2, 3, 0, 1))
            kernel = np.transpose(kernel, (1, 0, 2, 3))
            input = np.transpose(input, (0, 2, 3, 1))
            # print(kernel[0].shape)
            # print("before loop: ", input.shape, kernel.shape)
            for bi in range(b):
                for c in range(kin_channel):
                    for i in range(h):
                        for j in range(w):
                            out[bi, c, i: i + kh, j:j + kw] += np.sum(input[bi][i][j][:, np.newaxis, np.newaxis]* kernel[c], axis = 0)
            print(out.shape)

            return out 

        output = cross_correlation(input.data, filters.data)
        # print("input.shape: ", input.shape)
        # w.r.t. weights
        def apply_image(grad):
            x = np.transpose(input.data, (1, 0, 2, 3))
            y = np.transpose(grad, (1, 0, 2, 3))
            return cross_correlation(x, y) 
        
        # w.r.t input image 
        def apply_weight(grad):
            return transpose_conv(grad, filters.data)
        


        return Tensor(
            output, 
            [(filters, apply_image), (input, apply_weight)]
        )
    