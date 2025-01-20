from Tensor import Tensor 
from Layers import Linear 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(6969)
torch.manual_seed(6969)

W = torch.tensor([[2.0,  -1.0,  3.0,  4.0],
                  [4.0,  2.0,  5.0,  -1.0],
                  [-3.0,  -5.0,  6.0,  2.0],
                  [6.0,  1.0,  -2.0,  7.0]],
                 requires_grad=True)

x = torch.tensor([1.0, 2.0, -3.0, 4.0])
y = torch.tensor([0.0, 0.0, 0.0, 0.0])

l = W.mv(x)
prediction = F.sigmoid(l)
loss = (y - prediction).pow(2).sum()

loss.backward()

def F(x):
    return 1 / (1 + torch.exp(x))


print("W.grad =", W.grad)

print("-----------------------------------------")
W = Tensor([[2.0,  -1.0,  3.0,  4.0],
            [4.0,  2.0,  5.0,  -1.0],
            [-3.0,  -5.0,  6.0,  2.0],
            [6.0,  1.0, -2.0,  7.0]])
x = Tensor([[1.0], [2.0], [-3.0], [4.0]])
y = Tensor.zeros((4,1))

l = W.mm(x)
prediction = Tensor.sigmoid(l)
loss = Tensor.sum(Tensor.pow((y - prediction), 2))
loss.backward()

print("W.grad = ", W.grad)
# print("prediction.grad=", prediction.grad)