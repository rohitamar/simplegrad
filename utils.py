import numpy as np 
from sklearn.datasets import fetch_openml
import pickle

def topo_sort(node):
    visited = set()
    stack = []
    def dfs(node):
        if node in visited: return
        visited.add(node)
        for neighbor, _ in node.children:
            dfs(neighbor)
        stack.append(node)
    dfs(node)
    return reversed(stack)

def one_hot_encode(labels, sz):
    ans = np.zeros((len(labels), sz))
    for i, label in enumerate(labels):
        ans[i][label] = 1
    return ans

def get_mnist_dataset(conv=False):
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"]
    X, y = X.to_numpy(), y.to_numpy()
    y = y.astype(int)
    y = one_hot_encode(y, 10)
    X = X / 255
    train_size = int(0.8 * len(X))
    X_train, y_train, X_test, y_test = X[:train_size], y[:train_size], X[train_size:], y[train_size:]
    
    if conv:
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    
    return X_train, y_train, X_test, y_test


def get_cifar_dataset(filename):

    def load_pickle(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data 

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
    x = x / 255.0 

    return x, y