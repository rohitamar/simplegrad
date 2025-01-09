from Tensor import Tensor 

x = Tensor(1)
y = Tensor(4)

sm = x + y

z = x * x + x * x * x
w = z * sm 

w.backward()
print(sm.grad)
print(z.grad)
print(x.grad)
print(y.grad)
