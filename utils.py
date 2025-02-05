import numpy as np 

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
