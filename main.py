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

x = torch.tensor([1.0, 1.1, 1.3, 0.8])
y = torch.tensor([1.0, 0.0, 0.0, 0.0])
l = W.mv(x)
xx = torch.sum(torch.exp(l))
num = torch.exp(l)
prediction = num / xx 
xx.retain_grad()
num.retain_grad()
l.retain_grad()
prediction.retain_grad()
loss = -(torch.sum(torch.log(prediction) * y))
loss.backward()
print("W.grad = ", W.grad)
print("num.grad = ", num.grad)
print("l.grad =", l.grad)
print("xx.grad =", xx.grad)
# print("W.grad =", W.grad)

print("-----------------------------------------")
W = Tensor([[2.0,  -1.0,  3.0,  4.0],
            [4.0,  2.0,  5.0,  -1.0],
            [-3.0,  -5.0,  6.0,  2.0],
            [6.0,  1.0, -2.0,  7.0]])
x = Tensor([[1.0], [1.1], [1.3], [0.8]])
y = Tensor([[1.0], [0.0], [0.0], [0.0]])

l = W.mm(x)
xx = Tensor.sum(Tensor.exp(l))
num = Tensor.exp(l)
prediction = num / xx
loss = -(Tensor.sum(Tensor.log(prediction) * y))
loss.backward()

print("W.grad = ", W.grad)

print("num.grad = ", num.grad)
print("l.grad=", l.grad)
print("xx.grad =", xx.grad)
# print("W.grad = ", W.grad)
