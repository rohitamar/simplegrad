from Tensor import Tensor 
from Layers import Module, Linear, CrossEntropyLoss
import numpy as np
import torch 
import torch.nn as nn 

np.random.seed(6969)
torch.manual_seed(6969)

W = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
x = torch.tensor([2, 3, 4, 5])
y = torch.tensor([1, 0, 0, 0])

l = W.mv(x)

