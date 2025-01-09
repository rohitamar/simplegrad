
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