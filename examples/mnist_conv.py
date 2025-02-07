import numpy as np

from Tensor import Tensor 
from Functional import Functional as F
from Layers import Module, Linear, CrossEntropyLoss, Conv2d, MaxPool2d
from optim import SGD 
from utils import get_mnist_dataset

np.random.seed(6969)

class Model(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 8, 3, padding=1)
        self.conv2 = Conv2d(8, 16, 3, padding=1)
        self.pool = MaxPool2d(2, 2)
        self.fc1 = Linear(16 * 7 * 7, 10, bias=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = Tensor.reshape(x, shape=(x.shape[0], -1))
        return self.fc1(x)

model = Model()
optim = SGD(model.parameters(), lr = 1e-3)
criterion = CrossEntropyLoss()

epochs = 8000
batch_size = 64
test_every = 100

X_train, y_train, X_test, y_test = get_mnist_dataset(conv=True)
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
    print(f"Epoch {epoch}: Training Accuracy: {train_acc}") 
    
    if epoch % test_every == 0:
        pred = model(X_test)
        pred_class = np.argmax(pred.numpy(), axis=-1)
        target_class = np.argmax(y_test.numpy(), axis=-1)
        test_acc = (pred_class == target_class).mean()
        test_accs.append(test_acc)

        print(f"Epoch {epoch}: Testing Accuracy: {test_acc}") 

import matplotlib.pyplot as plt 
indices = list(range(epochs))
plt.plot(indices, train_accs)
plt.plot(indices, test_accs)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN performance on MNIST")
plt.show()