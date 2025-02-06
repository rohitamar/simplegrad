from Tensor import Tensor 
from Layers import Module, Linear, CrossEntropyLoss
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

np.random.seed(6969)
torch.manual_seed(6969)

W = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], requires_grad=True)
W = W.unsqueeze(0)

x = F.max_pool2d(W, kernel_size=4, padding=2)
loss = torch.sum(x)

W.retain_grad()
loss.backward()
print(W.grad)