import numpy as np
import pickle

from Tensor import Tensor 
from Layers import Module, Linear, CrossEntropyLoss, Conv2d, MaxPool2d
from Functional import Functional as F
from optim import SGD 
from utils import one_hot_encode

def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data 

def load_data(filename):
    d = load_pickle(filename)
    data = d["data"]
    y = d['labels']
    x = []

    for i in range(len(data)):
        t = data[i]
        r = np.reshape(t[0:1024], (32,32))
        g = np.reshape(t[1024:2048], (32,32))
        b = np.reshape(t[2048:4096], (32,32)) 
        x.append(np.dstack((r, g, b)))

    y = np.array(y)
    y = y.astype(int)
    y = one_hot_encode(y, 10)

    x = np.array(x)
    x = np.transpose(x, (0, 3, 1, 2))

    return np.array(x), y

class Model(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 16, 3, padding=1)
        self.conv2 = Conv2d(16, 32, 3, padding=1)
        self.conv3 = Conv2d(32, 16, 3, padding=1)
        self.pool = MaxPool2d(2, 2)
        self.fc1 = Linear(16 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = Tensor.reshape(x, shape=(x.shape[0], -1))
        return self.fc1(x)

X_train, y_train = load_data('./cifar_data/data_batch_1')
X_test, y_test = load_data('./cifar_data/test_batch')
X_test, y_test = Tensor(X_test), Tensor(y_test)

model = Model()
optim = SGD(model.parameters(), lr = 1e-3)
criterion = CrossEntropyLoss()

train_accs = []
test_accs = []

epochs = 1000
batch_size = 64
test_every = 1

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
indices = list(range(epochs))
plt.plot(indices, train_accs)
plt.plot(indices, test_accs)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("NN performance on CIFAR10")
plt.show()