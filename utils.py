def topo_sort(node):
    visited = set()
    stack = []
    def dfs(node):
        if node in visited: return
        visited.add(node)
        for x in node.children:
            neighbor = x[0]
            dfs(neighbor)
        stack.append(node)
    dfs(node)
    return reversed(stack)