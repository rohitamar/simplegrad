import numpy as np
from sklearn.datasets import fetch_openml
from Tensor import Tensor 
from Functional import Functional as F
from optim import SGD 
from Layers import Module, Linear, CrossEntropyLoss
from utils import get_mnist_dataset

np.random.seed(6969)

class Model(Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(784, 128)
        self.lin2 = Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.lin1(x))
        return self.lin2(x)

model = Model()
optim = SGD(model.parameters(), lr = 1e-3)
criterion = CrossEntropyLoss()

epochs = 20000
batch_size = 64
test_every = 100

X_train, y_train, X_test, y_test = get_mnist_dataset()
X_test, y_test = Tensor(X_test), Tensor(y_test)

train_accs = []
test_accs = []

for epoch in range(epochs):

    ind = np.random.choice(len(X_train), batch_size, replace=False)

    batch_x, target = X_train[ind], y_train[ind]
    batch_x, target = Tensor(batch_x), Tensor(target)

    pred = model(batch_x)
    loss = criterion(pred, target)
    optim.zero_grad()
    loss.backward()
    optim.step()

    pred_class = np.argmax(pred.numpy(), axis=-1)
    target_class = np.argmax(target.numpy(), axis=-1)
    train_acc = (pred_class == target_class).mean()
    train_accs.append(train_acc)

    if epoch % test_every == 0:
        pred = model(X_test)
        pred_class = np.argmax(pred.numpy(), axis=-1)
        target_class = np.argmax(y_test.numpy(), axis=-1)
        test_acc = (pred_class == target_class).mean()
        test_accs.append(test_acc)

        print(f"Epoch {epoch}: Training Accuracy: {train_acc} Testing Accuracy: {test_acc}") 

import matplotlib.pyplot as plt 

# plt.plot(range(len(train_accs)), train_accs)
plt.plot(range(len(test_accs)), test_accs)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("NN performance on MNIST")
plt.show()