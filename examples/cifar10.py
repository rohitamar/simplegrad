import numpy as np

from Tensor import Tensor 
from Layers import Module, Linear, CrossEntropyLoss, Conv2d, MaxPool2d
from Functional import Functional as F
from optim import Adam, SGD
from utils import get_cifar_dataset

class Model(Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = Conv2d(3, 32, 3, padding=1)
        self.conv2 = Conv2d(32, 64, 3, padding=1)
        
        self.conv3 = Conv2d(64, 128, 3, padding=1)
        self.conv4 = Conv2d(128, 128, 3, padding=1)

        self.pool = MaxPool2d(2, 2)
        
        self.fc1 = Linear(128 * 8 * 8, 512)
        self.fc2 = Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))

        x = Tensor.reshape(x, shape=(x.shape[0], -1))  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

X_train, y_train = get_cifar_dataset('./cifar_data/data_batch_1')
X_test, y_test = get_cifar_dataset('./cifar_data/test_batch')
X_test, y_test = Tensor(X_test), Tensor(y_test)

model = Model()
optim = SGD(model.parameters(), lr=1e-3)
criterion = CrossEntropyLoss()

train_accs = []
test_accs = []

epochs = 20
batch_size = 16
test_every = 1

for epoch in range(epochs):

    for step in range(len(X_train) // batch_size + 1):
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
    
        print(f"Epoch {epoch} Step {step}: Training Accuracy: {train_acc} Loss: {loss}")

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
plt.title("NN performance on CIFAR10")
plt.show()