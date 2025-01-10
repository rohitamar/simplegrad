from Tensor import Tensor 
from Layers import Linear 
import numpy as np

out = Linear(10, 2)
x = Tensor.ones((5, 10))
print(out(x))