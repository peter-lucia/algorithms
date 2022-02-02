#!/usr/bin/env python
# coding: utf-8

# # Graph Algorithms

# In[1]:


import collections
from typing import List


# ### DFS
# * Retains the path in class variable
# * In DFS, you traverse each node exactly once. Therefore, the time complexity of DFS is at least O(V).
# 
# Now, any additional complexity comes from how you discover all the outgoing paths or edges for each node which, in turn, is dependent on the way your graph is implemented. If an edge leads you to a node that has already been traversed, you skip it and check the next. Typical DFS implementations use a hash table to maintain the list of traversed nodes so that you could find out if a node has been encountered before in O(1) time (constant time).
# 
# If your graph is implemented as an adjacency matrix (a V x V array), then, for each node, you have to traverse an entire row of length V in the matrix to discover all its outgoing edges. Please note that each row in an adjacency matrix corresponds to a node in the graph, and the said row stores information about edges stemming from the node. So, the complexity of DFS is O(V * V) = O(V^2).
# 
# If your graph is implemented using adjacency lists, wherein each node maintains a list of all its adjacent edges, then, for each node, you could discover all its neighbors by traversing its adjacency list just once in linear time. For a directed graph, the sum of the sizes of the adjacency lists of all the nodes is E (total number of edges). So, the complexity of DFS is O(V) + O(E) = O(V + E).
# 
# For an undirected graph, each edge will appear twice. Once in the adjacency list of either end of the edge. So, the overall complexity will be O(V) + O (2E) ~ O(V + E).
# 
# There are different other ways to implement a graph. You can reason the complexity accordingly.
# Source: [https://www.quora.com/Why-is-the-complexity-of-DFS-O-V+E](https://www.quora.com/Why-is-the-complexity-of-DFS-O-V+E)
# 
# 
# ```
#         1
#       /   \
#      2     3
#     / \
#    4   5
# ```

# In[2]:



#### Definition for a binary tree node
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class DFS:
    def __init__(self):
        self.explored = []

    def dfs_traversal(self, root):
        if root is None:
            return []
        else:
            self.explored.append(root.val)  # pre-order traversal
            print(f"Found {root.val}")
            self.dfs_traversal(root.left)
            self.dfs_traversal(root.right)


#                 1
#               /    \
#              2      3
#            /  \
#           4    5

d = TreeNode(val=5)
c = TreeNode(val=4)
b = TreeNode(val=3)
a = TreeNode(val=2, left=c, right=d)
root = TreeNode(val=1, left=a, right=b)
dfs = DFS()
dfs.dfs_traversal(root)


# #### DFS for a binary tree
# * Retains path as parameter

# In[3]:


def dfs_traversal(root, explored):
    if root is None:
        return []
    else:
        explored.append(root.val)
        print(f"Found {root.val}")
        dfs_traversal(root.left, explored)
        dfs_traversal(root.right, explored)

d = TreeNode(val=5)
c = TreeNode(val=4)
b = TreeNode(val=3)
a = TreeNode(val=2, left=c, right=d)
root = TreeNode(val=1, left=a, right=b)
dfs = DFS()
explored = []
dfs_traversal(root, explored)
print(explored)


# #### DFS for Adjacency List
# * Only retains path by printing it
# 
# ```
#             A
#           /   \
#          B     C
#        /  \   /
#       D   E  F
# ```

# In[4]:


# https://www.educative.io/edpresso/how-to-implement-depth-first-search-in-python
# Using a Python dictionary to act as an adjacency list
# Time complexity: O(V + E)
graph = {
    'A' : ['B','C'],
    'B' : ['D', 'E'],
    'C' : ['F'],
    'D' : [],
    'E' : [],
    'F' : []
}


def dfs(visited, graph, node):
    if node not in visited:
        print(node)
        visited.add(node)
        for neighbor in graph[node]:
            dfs(visited, graph, neighbor)

visited = set() # Set to keep track of visited nodes.
# Driver Code
dfs(visited, graph, 'A')


# #### DFS for Adjacency List
# * Retains the path via the visited array

# In[5]:


def dfs(visited, graph, node):
    if node not in visited:
        visited.append(node)
        for neighbor in graph[node]:
            dfs(visited, graph, neighbor)

# Driver Code
visited = [] # List to keep track of visited nodes.
dfs(visited, graph, 'A')
print(visited)


# #### Find and remove leaves in a binary tree (DFS application)

# ```
#                       20                       20               20        20
#                     /    \                   /    \           /
#                   8       22               8       22       8
#                 /   \    /   \              \
#               5      3  4    25               3
#                     / \
#                   10    14
# 
# ```
# 
# Levels of leaf nodes.
# 
# The higher level is found after removing lower level leaves
# * level 0 nodes: 5, 10, 14, 4, 25
# * level 1 nodes: 3, 22
# * level 2 nodes: 8
# * level 3 nodes: 20

# In[6]:


class TreeNode:
    def __init__(self, key):
        self.val = key
        self.left = None
        self.right = None

root = TreeNode(20)
root.left = TreeNode(8)
root.right = TreeNode(22)
root.left.left = TreeNode(5)
root.left.right = TreeNode(3)
root.right.left = TreeNode(4)
root.right.right = TreeNode(25)
root.left.right.left = TreeNode(10)
root.left.right.right = TreeNode(14)


# In[7]:


class Solution:
    def findLeaves(self, root: TreeNode) -> List[List[int]]:
        """
            Example:
                      20                       20               20        20
                    /    \                   /    \           /
                  8       22               8       22       8
                /   \    /   \              \
              5      3  4    25               3
                    / \
                  10    14

        - level 0 nodes: 5, 10, 14, 4, 25
        - level 1 nodes: 3, 22
        - level 2 nodes: 8
        - level 3 nodes: 20
        Output:
        {
            0: [5, 10, 14, 4, 25],
            1: [3, 22],
            2: [8],
            3: [20]
        }
        """
        lookup = collections.defaultdict(list)

        def dfs(node: TreeNode, level: int):
            """
            Gets the maximum depth from the left and right subtrees
            of a given node
            """
            if not node:
                return level
            max_left_level = dfs(node.left, level)
            max_right_level = dfs(node.right, level)
            level = max(max_left_level, max_right_level)
            lookup[level].append(node.val)
            return level + 1
        dfs(root, 0)
        print(lookup)
        # lookup.values() for defaultdict returns
        # a list of lists for all values
        return lookup.values()

Solution().findLeaves(root)


# In[8]:


root = TreeNode(20)
root.left = TreeNode(2)
root.right = TreeNode(22)
root.right.left = TreeNode(4)
root.right.left.left = TreeNode(8)
Solution().findLeaves(root)


# ### 690. Employee Importance (DFS)
# [https://leetcode.com/problems/employee-importance/](https://leetcode.com/problems/employee-importance/)
# 

# In[9]:



# Definition for Employee.
class Employee:
    def __init__(self, id: int, importance: int, subordinates: List[int]):
        self.id = id
        self.importance = importance
        self.subordinates = subordinates


class Solution:
    def getImportance(self, employees: List['Employee'], id: int) -> int:
        """
        Time complexity: O(n) where n is the number of employees
        Space Complexity: O(n) to hold all employees in the hashmap

        Approach: DFS
        1. Find the employee id
        2. Run DFS on the employee id, summing total importance along the way
        """

        employees_map = {}

        for emp in employees:
            employees_map[emp.id] = emp

        desired_employee = employees_map[id]

        def dfs(employee: Employee, total: int):

            if not employee:
                return 0

            result = total + employee.importance

            for sub_id in employee.subordinates:
                result += dfs(employees_map[sub_id], total)

            return result


        return dfs(desired_employee, 0)

employees = [Employee(1, 5, [2,3]), Employee(2, 3, []), Employee(3, 3, [])]
assert Solution().getImportance(employees, 1) == 11


# ### 547. Number of Provinces
# 
# 

# In[10]:


class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:

        # Approach DFS
        # Run DFS from each vertex if that vertex is not yet visited

        num_vertices = len(isConnected)

        def dfs(starting_vertex: int, adj_matrix: List[List[int]], visited: set):

            visited.add(starting_vertex)

            for i in range(len(adj_matrix)):
                if i in visited:
                    continue
                if adj_matrix[starting_vertex][i] == 1:
                    visited.add(i)
                    dfs(i, adj_matrix, visited)


        cc = 0
        visited = set([])
        for i in range(num_vertices):
            if i not in visited:
                dfs(i, isConnected, visited)
                cc += 1
        return cc
circle = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]
assert Solution().findCircleNum(circle) == 3
circle = [
    [1, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
]
assert Solution().findCircleNum(circle) == 2


# ### BFS Adjacency List

# In[11]:


# Source: https://www.educative.io/edpresso/how-to-implement-a-breadth-first-search-in-python
# Time complexity: O(V + E)
#                     A - C
#                    / \ /
#                   B   F
#                 / \  /
#                D   E

# this is a directed graph
my_graph = {
  'A' : ['B','F', 'C'],
  'B' : ['D', 'E'],
  'C' : ['F'],
  'D' : [],
  'E' : ['F'],
  'F' : []
}
from typing import List
def bfs(visited: List[str], graph: dict, node: str):
    visited.append(node)
    queue.append(node)

    print("Visiting vertices: ")
    while queue:
        # print("Queue: ", queue)
        s = queue.pop(0)
        print(s, end = " ")
        for neighbour in graph[s]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)

# Driver Code
visited = [] # List to keep track of visited nodes.
queue = []
bfs(visited, my_graph, 'A')
print("\nVisited: ", visited)


# #### Pros and cons of matrix representation vs. adjacency list representation vs. objects and pointers to represent graphs
# Sources:
# * [https://www.section.io/engineering-education/graph-data-structure-python/](https://www.section.io/engineering-education/graph-data-structure-python/)
# * [https://www.geeksforgeeks.org/comparison-between-adjacency-list-and-adjacency-matrix-representation-of-graph/](https://www.geeksforgeeks.org/comparison-between-adjacency-list-and-adjacency-matrix-representation-of-graph/)
# * [https://www.bigocheatsheet.com](https://www.bigocheatsheet.com)
# 
# Matrix representation (a.k.a adjacency matrix)
# 
# ```
#   A B C D E
# A 0 4 1 0 0
# B 0 0 2 1 0
# C 1 0 0 0 0
# D 3 0 0 0 0
# E 0 0 0 0 0
# ```
# 
# Adjacency List representation
# ```
# A -> [(B, 4), (C, 1)]
# B -> [(C, 2), (D, 1)]
# C -> [(A, 1)]
# D -> [(A, 3)]
# ```
# 
# Note: In a complete graph where every vertex is connected, every entry in the matrix would have a value,
# so iterating over all of them takes $O(|E|) = O(|V|^2)$ time.
# 
# ##### Storage
# * Matrix representation requires $O(|V|^2)$ space since a VxV matrix is used to map connections. Wasted space for unused connections
# * Adjacency list requires $O(|V| + |E|)$ space since a O(|E|) is required for storing neighbors corresponding to each vertex
# * Objects and pointers requires $O(|V| + |E|)$ space
# 
# ##### Adding a vertex
# * Matrix representation requires the storage be increased to $O((|V|+1)^2)$. To do this we need to copy the whole matrix
# * Adjacency list requires O(1) time on average. Hash table insertion requires O(n) time in the worst though if there are too many collisions.
# * Objects and pointers requires O(1) time since we'd just update or add a pointer to a Node object
# 
# ##### Removing an edge
# * Matrix representation takes O(1) time since we set matrix[i][j] = 0
# * Adjacency list representation requires potentially traversing over all edges in the worst case so it's O(|E|) time
# * Removing an edge requires O(1) time for objects and pointers since we just update or remove a pointer in a Node object
# 
# ##### Querying edges
# * Matrix representation requires O(1) time always.
# * Adjacency List requires $O(|V|)$ time since a vertex can have at most $O(|V|)$ neighbors, so we'd have to check every adjacency vertex.

# #### Kruskal's Algorithm
# 
# Kruskal's algorithm finds a minimum spanning forest of an undirected edge-weighted graph. If the graph is connected, it finds a minimum spanning tree.

# #### Dijkstra's Algorithm
# Time Complexity: $O((|V| + |E|)log(|V|))$
# 
# Dijkstra's Algorithm finds the shortest path from a starting node to all other nodes of a graph
# * By default, all nodes are assumed to be inf distance away from the starting node, u
# * We then traverse in BFS fashion (by levels) from the starting node outward until we reach all nodes
# * When a new node v', is visited from v, we add dist(u,v) + dist(v, v')
# * If a node v' has already been visited from v,
#     we set dist(u, v') = min(dist(u,v'), dist(u,v) + dist(v, v')
# * We repeat until all nodes have been visited since this implies all edges have been traversed
# * Dijkstra's algorithm does not work with negative edges
# 
# Sources:
# * [https://www.analyticssteps.com/blogs/dijkstras-algorithm-shortest-path-algorithm](https://www.analyticssteps.com/blogs/dijkstras-algorithm-shortest-path-algorithm)
# * [https://www.techiedelight.com/single-source-shortest-paths-dijkstras-algorithm/](https://www.techiedelight.com/single-source-shortest-paths-dijkstras-algorithm/)

# In[12]:


from collections import defaultdict
import sys
from heapq import heappop, heappush

# Stores the heap node
class Node:
    def __init__(self, vertex: int, weight: int = 0):
        self.vertex = vertex
        self.weight = weight

    # Override the __lt__() function to make `Node` class work with a min-heap
    def __lt__(self, other):
        return self.weight < other.weight

class Graph:
    def __init__(self):
        self.adjacency_list = defaultdict(list)

    def add_edge(self, source: int, dest: int, weight: int):
        if weight < 0:
            raise ValueError("Dijkstra's algorithm does not handle negative weight edges.")

        self.adjacency_list[source].append((dest, weight))
        if dest not in self.adjacency_list:
            self.adjacency_list[dest] = []

    def count_vertices(self):
        return len(self.adjacency_list.keys())



def get_route(prev, i, route):
    if i >= 0:
        get_route(prev, prev[i], route)
        route.append(i)


# Run Dijkstra’s algorithm on a given graph
def find_shortest_paths(graph: Graph, source: int, n: int):

    # create a min-heap and push source node having distance 0
    pq = []
    heappush(pq, Node(source))

    # set initial distance from the source to `v` as infinity
    dist = [sys.maxsize] * n

    # distance from the source to itself is zero
    dist[source] = 0

    # list to track vertices for which minimum cost is already found
    done = [False] * n
    done[source] = True

    # stores predecessor of a vertex (to a print path)
    prev = [-1] * n

    # run till min-heap is empty
    while pq:

        node = heappop(pq) # Remove and return the best vertex
        u = node.vertex # get the vertex number

        # do for each neighbor `v` of `u`
        for (v, weight) in graph.adjacency_list[u]:
            if not done[v] and (dist[u] + weight) < dist[v]:
                dist[v] = dist[u] + weight
                prev[v] = u
                heappush(pq, Node(v, dist[v]))

        # mark vertex u as done so it will not get picked up again
        done[u] = True

    route = []
    for i in range(n):
        if i != source and dist[i] != sys.maxsize:
            get_route(prev, i, route)
            print(f'Path ({source} —> {i}): Minimum cost = {dist[i]}, Route = {route}')
            route.clear()


if __name__ == '__main__':

    # initialize edges as per the above diagram
    # (u, v, w) represent edge from vertex u to vertex v having weight w
    edges = [(0, 1, 10), (0, 4, 3), (1, 2, 2), (1, 4, 4), (2, 3, 9), (3, 2, 7),
            (4, 1, 1), (4, 2, 8), (4, 3, 2)]

    #      1 - 2
    #     / \ / \
    #    0 - 4 - 3
    #
    # total number of nodes in the graph (labelled from 0 to 4)

    # construct graph
    graph = Graph()
    for (source, dest, weight) in edges:
        graph.add_edge(source, dest, weight)

    n = graph.count_vertices()

    # run the Dijkstra’s algorithm from every node
    for source in range(n):
        find_shortest_paths(graph, source, n)


# In[13]:


from typing import List
from heapq import heappush, heappop
from collections import defaultdict
class Solution:
    def dijkstrasAlgorithm(self, grid: List[List[int]]) -> int:
        # dijkstra's algorithm to find shortest path between top left position in grid to bottom right...

        pq = []
        heappush(pq, (0,0))

        dist = defaultdict(int)  # tracks shortest distance from source to vertex 'v'
        prev = defaultdict(int)  # tracks parent of 'v' in shortest path from source
        done = defaultdict(int)  # tracks vertices for which cost is already found
        n = len(grid)
        m = len(grid[0])
        for i in range(n):
            for j in range(m):
                # set initial distance from source to 'v' as infinity
                dist[(i,j)] = sys.maxsize
                # initialize predecessor of each vertex as nonexistent
                prev[(i,j)] = (-1, -1)
                done[(i,j)] = False

        # set initial distance from source to itself as 0
        dist[(0,0)] = 0
        done[(0,0)] = True
        WEIGHT = 1  # same for all edges
        while pq:
            ii, jj = heappop(pq)

            # for each neighbor
            for (iii, jjj) in [(ii-1, jj), (ii+1, jj), (ii, jj-1), (ii, jj+1)]:
                if iii < 0 or jjj < 0 or iii >= n or jjj >= m:
                    continue

                if grid[iii][jjj] == 1:
                    continue

                if not done[(iii, jjj)] and (dist[(ii, jj)] + WEIGHT) < dist[(iii, jjj)]:
                    dist[(iii, jjj)] = dist[(ii, jj)] + WEIGHT
                    prev[(iii, jjj)] = (ii, jj)
                    heappush(pq, (iii, jjj))
            done[(ii, jj)] = True

        if dist[(n-1, m-1)] == sys.maxsize:
            return -1
        return dist[(n-1, m-1)]
grid = [
    [0,0,0],
    [1,1,0],
    [0,0,0],
    [0,1,1],
    [0,0,0]
]
assert Solution().dijkstrasAlgorithm(grid) == 10


# In[14]:


from typing import Union
class Solution:
    def bfs(self, grid: List[List[int]], k: int) -> Union[List, int]:
        n = len(grid)
        m = len(grid[0])
        visited = []
        queue = []
        queue.append([(0,0)])
        while queue:
            path = heappop(queue)
            ii, jj = path[-1]

            if (ii, jj) in visited:
                continue

            # visit each neighbor
            for (iii, jjj) in [(ii-1, jj), (ii+1, jj), (ii, jj-1), (ii, jj+1)]:
                if iii < 0 or jjj < 0 or iii >= n or jjj >= m:
                    continue

                if grid[iii][jjj] == 1:
                    continue

                new_path = list(path)
                new_path.append((iii, jjj))
                queue.append(new_path)

                if (iii, jjj) == (n-1, m-1):
                    print("Shortest path = ", *new_path)
                    return len(new_path) - 1


            visited.append((ii, jj))

        return -1

grid = [
    [0,0,0],
    [1,1,0],
    [0,0,0],
    [0,1,1],
    [0,0,0]
]
assert Solution().bfs(grid, 10) == 10


# #### Number of islands
# 
# https://leetcode.com/problems/number-of-islands/discuss/56340/Python-Simple-DFS-Solution
# 
# Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.
# 
# An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

# In[15]:


from typing import List
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        # DFS
        if not grid:
            return 0

        n = len(grid)
        m = len(grid[0])

        def dfs(grid: List[List[str]], i, j, visited: List[List[int]]):

            if i < 0 or j < 0 or i >= n or j >= m or grid[i][j] != '1' or visited[i][j]:
                return

            visited[i][j] = True

            dfs(grid, i+1, j, visited)
            dfs(grid, i-1, j, visited)
            dfs(grid, i, j-1, visited)
            dfs(grid, i, j+1, visited)

        num_islands = 0
        visited = [[False for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if not visited[i][j] and grid[i][j] == '1':
                    dfs(grid, i, j, visited)
                    num_islands += 1
        return num_islands

grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
assert Solution().numIslands(grid) == 1
grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
assert Solution().numIslands(grid) == 3


# ### 1293. Shortest Path in a Grid with Obstacles Elimination
# https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/

# In[16]:


from typing import List
class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:

        # Use BFS
        # add state (i, j, k) to queue
        # keep track of visited states
        # don't visit nodes with state k <= 0

        n = len(grid)
        m = len(grid[0])
        visited = set()
        queue = []
        queue.append([(0, 0, k)])

        # get manhattan distance
        # if manhattan distance <= k, return it
        if n-1 + m-1 <= k:
            return n-1 + m-1

        while queue:
            path = queue.pop(0)
            i, j, _k = path[-1]

            if i == n-1 and j == m-1:
                print("Shortest path: ", *path)
                return len(path) - 1

            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            for (x,y) in neighbors:
                if 0 <= x < n and 0 <= y < m:

                    if grid[x][y] == 1:
                        new_k = _k - 1
                        if new_k < 0:
                            continue
                    else:
                        new_k = _k

                    if (x, y, new_k) in visited:
                        continue

                    new_path = list(path)

                    new_path.append((x, y, new_k))
                    visited.add((x, y, new_k))
                    queue.append(new_path)
        return -1

# 0 0 0
# 1 1 0
# 0 0 0
# 0 1 1
# 0 0 0

# k = 1

# fastest route is to go down from top left and then right at the last row.
# this is equivalent to going right at the top row and then down on the last column
assert Solution().shortestPath(grid = [[0,0,0],[1,1,0],[0,0,0],[0,1,1],[0,0,0]], k = 1) == 6


# ### Kruskal's Minimum Spanning Tree Algorithm
# Source: [https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/](https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/)
# 
# [Kruskal's MST algorithm on wikipedia](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm)
# 1. Sort the edges of the graph by weight, increasing from lowest to highest
# 2. Select the lowest edge available that does not form a cycle
# 3. Repeat step 2 until all edges have been checked.
# 
# Time complexity: $O(|E|log(|V|))$ -> More optimal for sparse graphs

# ### Prim's Minimum Spanning Tree Algorithm
# 
# Source: [https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/](https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/)
# [Prim's algorithm from Wikipedia](https://en.wikipedia.org/wiki/Prim%27s_algorithm)
# Prim's algorithm (also known as Jarník's algorithm) is a greedy algorithm that finds a minimum spanning tree for a weighted undirected graph. This means it finds a subset of the edges that forms a tree that includes every vertex, where the total weight of all the edges in the tree is minimized. The algorithm operates by building this tree one vertex at a time, from an arbitrary starting vertex, at each step adding the cheapest possible connection from the tree to another vertex.
# 
# Time complexity:
# 
# Adjacency matrix: $ O(|V|^2)$
# Binary Heap and Adjacency List: $O((|V| + |E|)log(|V|)) = O(|E|log(|V|))$
# Fibonacci Heap and Adjacency List: $O(|E| + |V|log(|V|) $ -> More optimal for dense graphs

# ### Disjoint Set
# ```
# Union:
#    a              d                      a                d
#   / \            /    ->              /  |  \    or    /  |
#  b   c          e                    b   c   d        e   a
#                                             /           / | \
#                                            e           b  c  d
#                                 rank(a) > rank(d)      rank(d) > rank(a)
# ```
# Useful: [https://www.techiedelight.com/disjoint-set-data-structure-union-find-algorithm/](https://www.techiedelight.com/disjoint-set-data-structure-union-find-algorithm/)
# Time complexity: O(log|V|) in the worst case where |V| is the number of elements. the running time is bounded by the tree height.
# Space complexity: O(|V|) to store parents and rank for each vertex

# In[17]:


class DisjointSet:

    def __init__(self, size: int):
        self.rank = [1 for _ in range(size)]
        self.parent = [i for i in range(size)]

    def find(self, x: int) -> int:
        """
        Gets the root of x
        """
        if x != self.parent[x]:
            x = self.find(self.parent[x])
        return x

    def union(self, x: int, y: int):
        root_x = self.find(x)
        root_y = self.find(y)
        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_x] = root_y
            # make root_y have the higher rank if they are equal
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_y] += 1

dj_set = DisjointSet(5)
dj_set.union(2,3) # root of 2 and 3 will be 3 since we choose y arbitrarily when ranks are equal
dj_set.union(3,4)  # root of 3 and 4 will be 3 since 3 has a higher rank from previous operation
dj_set.find(4)  # get the root of 4, it should be 3


# ### 1584. Min Cost to Connect All Points (Kruskal / Prim's MST application)
# [https://leetcode.com/problems/min-cost-to-connect-all-points/](https://leetcode.com/problems/min-cost-to-connect-all-points/)
# 
# You are given an array points representing integer coordinates of some points on a 2D-plane, where points[i] = [xi, yi].
# 
# The cost of connecting two points [xi, yi] and [xj, yj] is the manhattan distance between them: |xi - xj| + |yi - yj|, where |val| denotes the absolute value of val.
# 
# Return the minimum cost to make all points connected. All points are connected if there is exactly one simple path between any two points.

# 

# ### Solution using Kruskal's Algorithm (Union / Find with integers)

# In[18]:


# Helpful: https://leetcode.com/problems/min-cost-to-connect-all-points/discuss/1620452/Python-Kruskal's-%2B-Disjoint-set-Union
from typing import List
class DisjointSet:

    def __init__(self, size: int):
        # Initially the rank (height of tree) for each node is zero
        # since all nodes are separated
        self.rank = [0 for _ in range(size)]
        # Map the node (index) to its parent node.
        # Initially all nodes are separated, so the parent is itself
        self.parent = [i for i in range(size)]

    def find(self, x: int) -> int:
        """
        Gets the root of x
        """
        if x != self.parent[x]:
            x = self.find(self.parent[x])
        return x

    def union(self, x: int, y: int):
        """
        Merges nodes x and y
        """
        root_x = self.find(x)
        root_y = self.find(y)
        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_x] = root_y
            # make root_y have the higher rank if they are equal
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_y] += 1


class Solution:

    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        # build a complete graph connecting all points
        # find the MST

        # Build list of edges connecting all points
        edges = []
        for i in range(len(points)):
            for j in range(len(points)):
                if i == j:
                    continue
                x1 = points[i][0]
                y1 = points[i][1]
                x2 = points[j][0]
                y2 = points[j][1]
                weight = abs(x2 - x1) + abs(y2 - y1)
                # since all points are distince, we use the points index in points as its unique identifier
                # This is easier to work with in the disjoint set structure
                edges.append((i, j, weight))

        # Kruskal's algorithm
        # Initially all points are disconnected
        dj_set = DisjointSet(size = len(edges))
        # sort edges in place by weight, increasing
        edges.sort(key=lambda val: val[2])
        edges_sum = 0
        num_edges = 0
        for (i, j, weight) in edges:
            if dj_set.find(i) != dj_set.find(j):
                edges_sum += weight
                dj_set.union(i, j)
                num_edges += 1
            if num_edges == len(points) - 1:
                # We've built the MST already
                return edges_sum

        return edges_sum
Solution().minCostConnectPoints([[0,0],[2,2],[3,10],[5,2],[7,0]])


# ### Solution: Kruskal's Algorithm (Union / Find with points)

# In[19]:


from typing import Tuple
class DisjointSet:

    def __init__(self, points):
        # Initially the rank (height of tree) for each node is zero
        # since all nodes are separated
        self.rank = {}
        # Map the node (index) to its parent node.
        # Initially all nodes are separated, so the parent is itself
        self.parent = {}
        for point in points:
            x, y = point[0], point[1]
            self.parent[(x, y)] = (x,y)
            self.rank[(x,y)] = 0

    def find(self, x: Tuple[int, int]) -> Tuple[int, int]:
        """
        Gets the root node of x
        """
        if x != self.parent[x]:
            x = self.find(self.parent[x])
        return x

    def union(self, x: Tuple[int, int], y: Tuple[int, int]):
        """
        Merges sets that x and y reside in
        """
        root_x = self.find(x)
        root_y = self.find(y)
        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_x] = root_y
            # make root_y have the higher rank if they are equal
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_y] += 1

class Solution:

    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        # build a complete graph connecting all points
        # find the MST

        # Build list of edges connecting all points
        edges = []
        for point1 in points:
            for point2 in points:
                if point1 == point2:
                    continue
                x1 = point1[0]
                y1 = point1[1]
                x2 = point2[0]
                y2 = point2[1]
                weight = abs(x2 - x1) + abs(y2 - y1)
                # since all points are distinct, we use the points index in points as its unique identifier
                # This is easier to work with in the disjoint set structure
                edges.append((tuple(point1), tuple(point2), weight))

        # Kruskal's algorithm
        # Initially all points are disconnected
        dj_set = DisjointSet(points)
        # sort edges in place by weight, increasing
        edges.sort(key=lambda val: val[2])
        edges_sum = 0
        num_edges = 0
        for (x, y, weight) in edges:
            if dj_set.find(x) != dj_set.find(y):
                edges_sum += weight
                dj_set.union(x, y)
                num_edges += 1
            if num_edges == len(points) - 1:
                # We've built the MST already
                return edges_sum

        return edges_sum
Solution().minCostConnectPoints([[0,0],[2,2],[3,10],[5,2],[7,0]])


# ### How to detect a cycle with DFS
# Time complexity: $O(|V| + |E|)$ since we do a DFS traversal on a graph represented w/ adjacency list
# Space complexity: $O(|V|)$ to store the visited array
# Source: [https://www.geeksforgeeks.org/detect-cycle-undirected-graph/](https://www.geeksforgeeks.org/detect-cycle-undirected-graph/)

# In[20]:


from collections import defaultdict
from typing import List

class Graph:
    def __init__(self, size: int):
        self.size = size
        self.graph = defaultdict(list)  # adj. list

    def add_edge(self, u, v):
        # It's important to add an edge both ways for an undirected graph
        self.graph[u].append(v)
        self.graph[v].append(u)

    def has_cycle(self):

        def dfs_has_cycle(v: int, visited: List[bool], parent: int) -> bool:

            visited[v] = True

            # recursively visit neighbors
            for w in self.graph[v]:
                if not visited[w]:
                    found_cycle = dfs_has_cycle(w, visited, v)
                    if found_cycle:
                        return True
                elif parent != w:
                    return True
            return False

        visited = [False for _ in range(self.size)]
        # start the check at every node since graph may be disconnected
        # set parent = -1
        for u in range(self.size):
            if not visited[u]:
                if dfs_has_cycle(u, visited, -1):
                    return True
        return False

g = Graph(5)
g.add_edge(1, 0)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(0, 3)
g.add_edge(3, 4)

if g.has_cycle():
    print("Graph contains cycle")
else:
    print("Graph does not contain cycle ")
g1 = Graph(3)
g1.add_edge(0,1)
g1.add_edge(1,2)


if g1.has_cycle():
    print("Graph contains cycle")
else:
    print("Graph does not contain cycle ")


# ### How to detect a cycle using Union-Find
# Time complexity: $O(|V|log(|V|))$ since the union and find operations take O(log|V|), since the whole tree may need to be traversed
# Space complexity: $O(|V|)$ to hold the disjoint set
# Source: [https://www.techiedelight.com/union-find-algorithm-cycle-detection-graph/](https://www.techiedelight.com/union-find-algorithm-cycle-detection-graph/)

# In[21]:


from collections import defaultdict
from typing import List

class DisjointSet:

    def __init__(self, size: int):
        # Initially the rank (height of tree) for each node is zero
        # since all nodes are separated
        self.rank = [0 for _ in range(size)]
        # Map the node (index) to its parent node.
        # Initially all nodes are separated, so the parent is itself
        self.parent = [i for i in range(size)]

    def find(self, x: int) -> int:
        """
        Gets the root of x
        """
        if x != self.parent[x]:
            x = self.find(self.parent[x])
        return x

    def union(self, x: int, y: int):
        """
        Merges nodes x and y
        """
        root_x = self.find(x)
        root_y = self.find(y)
        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_x] = root_y
            # make root_y have the higher rank if they are equal
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_y] += 1
class Graph:
    def __init__(self, size: int):
        self.size = size
        self.graph = defaultdict(list)  # adj. list

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def has_cycle(self):

        ds = DisjointSet(self.size)

        for u in self.graph:
            for v in self.graph[u]:
                root_u = ds.find(u)
                root_v = ds.find(v)
                if root_u == root_v:
                    return True
                ds.union(u, v)
        return False


g = Graph(5)
g.add_edge(1, 0)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(0, 3)
g.add_edge(3, 4)
if g.has_cycle():
    print("Graph contains cycle")
else:
    print("Graph does not contain cycle ")
g1 = Graph(3)
g1.add_edge(0,1)
g1.add_edge(1,2)

if g1.has_cycle():
    print("Graph contains cycle")
else:
    print("Graph does not contain cycle ")


# ### Topological Sorting with DFS
# Topological sorting is a linear ordering of vertices such that for every directed edge (u,v), vertex u comes before v in the ordering.
# Topological sorting is not possible if the graph is not a directed acyclic graph (DAG).
# There can be more than one topological sorting for a single graph.
# 
# Time complexity: O(|V| + |E|)
# Space Complexity: O(|V|)

# In[22]:


from collections import defaultdict
from typing import List

class Graph:
    def __init__(self, size: int):
        self.size = size
        self.graph = defaultdict(list)  # adj. list

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def topo_sort(self):

        def topo_sort_dfs(v: int, visited: List[bool], stack: List[int]) -> bool:
            """
            1. Visited vertex v
            2. Recursively visited unvisited neighbors of vertex v
            3. Add v to the stack after its neighbors have been added to the stack
            Note that the resultant stack will need to be reversed at the end

            """
            visited[v] = True

            for w in self.graph[v]:
                if not visited[w]:
                    topo_sort_dfs(w, visited, stack)

            stack.append(v)

        visited = [False for _ in range(self.size)]
        stack = []
        # start the check at every node since graph may be disconnected
        # set parent = -1
        for u in range(self.size):
            if not visited[u]:
                topo_sort_dfs(u, visited, stack)

        return stack[::-1]

# Driver Code
g = Graph(6)
g.add_edge(5, 2)
g.add_edge(5, 0)
g.add_edge(4, 0)
g.add_edge(4, 1)
g.add_edge(2, 3)
g.add_edge(3, 1)

print("The following is a Topological Sort of the given graph", end=" ")

# Function Call
print(g.topo_sort())
print("Note that [4,5,2,3,1,0] is another valid topological sorting for this graph.")


# ### Floyd Warshall All Pairs Shortest Path (APSP) problem
# Finds the shortest distances between every pair of vertices in a given edge weighted directed graph
# Time Complexity: $O(|V|^3)$
# Space Complexity: $O(|V|^2)$ Since an adjacency matrix is used

# In[23]:


def floyd_warshall(graph: List[List[int]]):
    """
    Solves the All Pairs Shortest Path (APSP) problem

    :param graph:
    :return:
    """
    n, m = len(graph), len(graph[0])
    assert n == m
    dist = graph

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

graph = [
    [           0,            5, float('inf'),           10],
    [float('inf'),            0,            3, float('inf')],
    [float('inf'), float('inf'),            0,            1],
    [float('inf'), float('inf'), float('inf'),             0]
 ]
result = floyd_warshall(graph)
for row in result:
    print(row)


# ### Find all bridges in an undirected graph with DFS
# An edge in an undirected graph is a bridge iff removing it disconnects the graph
# Time Complexity: $O(|V| + |E|)$
# Space Complexity: $O(|V|)$

# In[24]:


from typing import Tuple
class Graph:
    def __init__(self, size: int):
        self.size = size
        self.graph = defaultdict(list)  # adj. list
        self.current_time = 0
        self.bridges = []

    def add_edge(self, u, v):
        # It's important to add an edge both ways for an undirected graph
        self.graph[u].append(v)
        self.graph[v].append(u)

    def print_bridges(self):
        for v, w in self.bridges:
            print(f"{v}-{w} is a bridge.")

    def find_bridge_dfs(self, v: int, visited: List[bool], parent: List[int], low: List[int], disc: List[int]) -> List[Tuple[int, int]]:
        """
        1. An edge (u,v) where u is a parent of v, is a bridge if there does not
           exist any other alternative to reach u or an ancestor of u from the
           subtree rooted with v.
        2. low[v] indicates the earliest visited vertex reachable from the subtree
           rooted with v. Therefore, the condition for an edge (u,v) to be a bridge
           is low[v] > disc[u]
        """
        visited[v] = True

        disc[v] = self.current_time
        low[v] = self.current_time  # earliest visited vertex reachable from subtree rooted with u
        self.current_time += 1

        for w in self.graph[v]:
            if not visited[w]:
                parent[w] = v
                self.find_bridge_dfs(w, visited, parent, low, disc)

                # since we performed dfs on w, we make sure the earliest ancestor reachable
                # from v incorporates those reachable from w
                low[v] = min(low[v], low[w])

                # if the earliest ancestor reachable from the subtree under w
                # is greater when we discovered v, then v-w is a bridge
                if low[w] > disc[v]:
                    self.bridges.append((v, w))

            elif w != parent[v]:
                # since w is not a parent of v, make sure the earliest ancestor reachable
                # from v incorporates the discovery time of w
                low[v] = min(low[v], disc[w])

    def find_bridge(self):

        visited = [False for _ in range(self.size)]
        disc = [float('inf') for _ in range(self.size)]
        low = [float('inf') for _ in range(self.size)]
        parent = [-1 for _ in range(self.size)]
        # start the check at every node since graph may be disconnected
        # set parent = -1
        for u in range(self.size):
            if not visited[u]:
                self.find_bridge_dfs(u, visited, parent, low, disc)

# Create a graph given in the above diagram
g1 = Graph(5)
g1.add_edge(1, 0)
g1.add_edge(0, 2)
g1.add_edge(2, 1)
g1.add_edge(0, 3)
g1.add_edge(3, 4)


print("Bridges in first graph ")
g1.find_bridge()
g1.print_bridges()

g2 = Graph(4)
g2.add_edge(0, 1)
g2.add_edge(1, 2)
g2.add_edge(2, 3)
print("\nBridges in second graph ")
g2.find_bridge()
g2.print_bridges()


g3 = Graph(7)
g3.add_edge(0, 1)
g3.add_edge(1, 2)
g3.add_edge(2, 0)
g3.add_edge(1, 3)
g3.add_edge(1, 4)
g3.add_edge(1, 6)
g3.add_edge(3, 5)
g3.add_edge(4, 5)
print("\nBridges in third graph ")
g3.find_bridge()
g3.print_bridges()


# ### Boggle - find all possible words in a board of characters using DFS
# [https://leetcode.com/problems/word-search-ii/](https://leetcode.com/problems/word-search-ii/)
# 
# Approach:
# - Consider every character as a starting character and find all words starting with it.
# - All words starting from a character can be found using DFS.
# 
# Time complexity: $O(m^2 \cdot n^2)$ for board with size m x n since we potentially visit every other location in the board once via DFS for each letter in the board
# Space complexity: $O(mn)$ to store visited state for each element in the board
# 

# In[25]:


dictionary = set(["GEEKS", "FOR", "QUIZ", "GO"])
def find_words_dfs(board, visited, i, j, current_word):
    visited[i][j] = True
    current_word += board[i][j]

    if current_word in dictionary:
        print(f"Found: {current_word}")

    # traverse the 8 adjacent cells around board[i][j]
    positions = [
        [i-1, j-1], [i-1,j],  [i-1, j+1],
        [i, j-1],             [i, j+1],
        [i+1, j-1], [i+1, j], [i+1, j+1]
    ]
    for ii, jj in positions:
        if not (0 <= ii < len(board) and
                0 <= jj < len(board[0])):
            continue
        if visited[ii][jj]:
            continue
        find_words_dfs(board, visited, ii, jj, current_word)

    # make last character the next starting character
    current_word = current_word[-1]
    visited[i][j] = False



def find_words(board: List[List[str]]):
    n = len(board)
    m = len(board[0])
    visited = [[False for _ in range(m)] for _ in range(n)]
    current_word = ""
    for i in range(n):
        for j in range(m):
            find_words_dfs(board, visited, i, j, current_word)


board = [
    ["G", "I", "Z"],
    ["U", "E", "K"],
    ["Q", "S", "E"]
]
find_words(board)


# ## Experiments

# In[26]:


from typing import List
from collections import defaultdict
### DFS (Graph)

# An edge in an undirected connected graph is a bridge iff removing it disconnects the graph.

#   1     5       9
#   | \  /       / |
#   |  3 ------ 7  |
#   |/   \       \ |
#   2     4       8
# the bridges are [3-4], [3,5], and [3-7]

class Graph:
    def __init__(self):
        self.size = 0
        self.graph = {}
        self.current_time = 0
        self.bridges = []

    def add_edge(self, u, v):
        # undirected graph, requires edge from u to v and from v to u
        self.graph[u].append(v)
        self.graph[v].append(u)

    def add_directed_edge(self, u, v):
        if self.graph.get(u) is None:
            self.graph[u] = [v]
        else:
            self.graph[u].append(v)
        if not self.graph.get(v):
            self.graph[v] = []

    def get_size(self):
        return max(self.graph.keys()) + 1

    def dfs_find_bridge(self, v, visited: List[int], parents: List[int], anc: List[int], disc: List[int]):

        visited[v] = True
        anc[v] = self.current_time
        disc[v] = self.current_time
        self.current_time += 1

        for w in self.graph[v]:

            if not visited[w]:
                parents[w] = v
                self.dfs_find_bridge(w, visited, parents, anc, disc)

                # make sure earliest ancestor discoverable from w
                # is propagated to v's earliest ancestor if it's lower
                anc[v] = min(anc[v], anc[w])


                # if no subgraph of w has an earlier discovery time than v
                # then v-w is a bridge
                if anc[w] > disc[v]:
                    self.bridges.append([v, w])
            elif w != parents[v]:
                # since w is not v's parent,
                # make sure the earliest possible ancestor of v
                # incorporates the discovery time of w
                anc[v] = min(anc[v], disc[w])

    def find_bridges(self):
        # Observations / Notes
        # Use DFS and if we've visited a note previously
        # and there does not already exist a path
        self.size = self.get_size()
        visited = [False for _ in range(self.size)]
        parents = [-1 for _ in range(self.size)]
        low = [float('inf') for _ in range(self.size)]
        disc = [float('inf') for _ in range(self.size)]

        # start from each vertex in case the graph is disconnected
        # for v in self.graph.keys():
        #     if not visited[v]:

        # This works if the graph is not disconnected
        self.dfs_find_bridge(1, visited, parents, low, disc)

        print("The bridges in the graph are: ")
        print(self.bridges)
        return self.bridges

    def dfs_detect_cycle(self, v: int,
                         visited: List[int],
                         parents: List[int],
                         disc: List[int],
                         current_time: int):

        disc[v] = current_time
        visited[v] = True

        for neighbor in self.graph[v]:
            if not visited[neighbor]:
                parents[neighbor] = v
                found_cycle = self.dfs_detect_cycle(neighbor, visited, parents, disc, current_time + 1)
                if found_cycle:
                    return found_cycle
            elif parents[neighbor] != v:
                return True
        return False

    def find_cycles(self):
        self.size = self.get_size()
        visited = [False for _ in range(self.size)]
        parents = [-1 for _ in range(self.size)]
        disc = [float("inf") for _ in range(self.size)]
        current_time = 0


        starting_vertex = 1
        found_cycle = self.dfs_detect_cycle(starting_vertex, visited, parents, disc, current_time)
        if found_cycle:
            print(f"Found cycle visiting {starting_vertex}")
            return True
        else:
            print("No cycle found.")





#   1     5       9
#   | \  /       / |
#   |  3 ------ 7  |
#   |/   \       \ |
#   2     4       8
g = Graph()
graph = {
    1:[3,2],
    2:[1,3],
    3:[1,2,4,5,7],
    4:[3],
    5:[3],
    7:[8,3, 9],
    8:[7,9],
    9:[7,8],
}
g.graph = graph
g.find_bridges()
# the bridges are [3-4], [3,5], and [3-7]
print()



# has cycle
print("Expecting cycle.")
g = Graph()
g.add_directed_edge(1, 3)
g.add_directed_edge(1, 2)
g.add_directed_edge(2, 1)
g.add_directed_edge(3, 7)
g.add_directed_edge(7, 8)
g.add_directed_edge(7, 3)
g.add_directed_edge(7, 9)
g.add_directed_edge(8, 9)
g.add_directed_edge(9, 7)
g.find_cycles()
print()

print("Expecting no cycle.")
# no cycle
g = Graph()
g.add_directed_edge(1, 3)
g.add_directed_edge(3, 7)
g.add_directed_edge(7, 8)
g.add_directed_edge(7, 9)
g.find_cycles()


# In[27]:


# topo sort
# requires that the graph is a DAG, directed acyclic graph
# topo sort is basically just printing the reverse of the postorder numbers of a DAG
# there are multiple valid topological sortings for a single DAG. This just depends on which neighbor is selected first during the DFS traversal
# Source: https://www.geeksforgeeks.org/topological-sorting/
#        4
#       /
#      1---5--7
#     / \
#    /   6
#   /
#  0 -- 2
#   \
#    3
# postorder: 4, 7, 5, 1, 6, 2, 3, 0
# reversed postorder: 0, 3, 2, 6, 1, 5, 7, 4
dag = {
    0: [1,2,3],
    1: [4,5,6],
    2: [],
    3: [],
    4: [],
    5: [7],
    6: [],
    7: []
}

stack = []

def dfs(v: int, visited: List[int]):

    visited[v] = True

    for neighbor in dag[v]:
        if not visited[neighbor]:
            dfs(neighbor, visited)

    # we visited v, so add it to the stack
    stack.append(v)


def topo_sort():

    visited = [False for _ in range(8)]
    for v in dag:
        if not visited[v]:
            dfs(v, visited)
    print(stack[::-1])
topo_sort()


# In[28]:


### How to detect a cycle with Union Find

class DisjointSet:

    def __init__(self, size: int):
        # initially, the rank for each node is zero
        # since all nodes are separated
        self.rank = [0 for _ in range(size)]

        # initially, all parents are the nodes themselves
        self.parent = [i for i in range(size)]

    def find(self, x:int) -> int:
        # get the root of x
        # if x is not it's own parent, find it
        if x != self.parent[x]:
            x = self.find(self.parent[x])
        # otherwise, return x
        return x

    def union(self, x: int, y: int):

        root_x = self.find(x)
        root_y = self.find(y)

        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_x] = root_y
            # since we set y to be the parent of x,
            # if they had the same rank, increment y's rank now
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_y] += 1


class Graph:
    def __init__(self, size: int):
        self.size = size
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def has_cycle(self):
        ds = DisjointSet(self.size)

        # basically keep joining vertices
        # based on adj list until we
        # find that two vertices have already been joined
        # (e.g. their roots are the same, find(x) == find(y))

        for v in self.graph:
            for neighbor in self.graph[v]:
                if ds.find(v) != ds.find(neighbor):
                    ds.union(v, neighbor)
                else:
                    print(f"cycle detected between {v} and {neighbor}")





#   1             9
#   | \          / |
#   |  3 ------ 7  |
#   |/           \ |
#   2             8
# has cycle
print("Expecting two cycles")
g = Graph(10)
g.add_edge(1, 3)
g.add_edge(3, 2)
g.add_edge(2, 1)
g.add_edge(3, 7)
g.add_edge(7, 8)
g.add_edge(8, 9)
g.add_edge(9, 7)
g.has_cycle()
print()

print("Expecting no cycle.")
# no cycle
g = Graph(10)
g.add_edge(1, 3)
g.add_edge(3, 7)
g.add_edge(7, 8)
g.add_edge(7, 9)
g.has_cycle()


# ### 329. Longest Increasing Path in a Matrix
# 
# Source: [https://leetcode.com/problems/longest-increasing-path-in-a-matrix/](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/)
# 
# Given an m x n integers matrix, return the length of the longest increasing path in matrix.
# 
# From each cell, you can either move in four directions: left, right, up, or down. You may not move diagonally or move outside the boundary (i.e., wrap-around is not allowed).
# 
# 

# In[29]:


# Longest increasing path
import functools
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:

        # Observations, can use DFS, start traversal at each node, keep track of longest DFS
        # traversal
        # Need to reset visited before each DFS call

        n = len(matrix)
        m = len(matrix[0])


        def make_visited():
            return [[False for _ in range(m)] for _ in range(n)]

        @functools.lru_cache(maxsize=None)
        def add_to_tuple(item, tp: Tuple):
            if tp:
                return tuple([each for each in tp] + [item])
            return item,

        @functools.lru_cache(maxsize=None)
        def dfs(i, j, prev: List[int], current_count: int):

            if not (0 <= i < n and 0 <= j < m):
                return current_count


            if prev:
                last_i, last_j = prev[-1]
                # if decreasing, stop
                if matrix[last_i][last_j] >= matrix[i][j]:
                    return current_count

                # if i,j already in path
                if (i,j) in prev:
                    return current_count

            current_count += 1

            # run dfs on neighbors
            cc_below = dfs(i+1, j, add_to_tuple((i,j), prev), current_count=current_count)
            cc_above = dfs(i-1, j, add_to_tuple((i,j), prev), current_count=current_count)
            cc_left = dfs(i, j-1, add_to_tuple((i,j), prev), current_count=current_count)
            cc_right = dfs(i, j+1, add_to_tuple((i,j), prev), current_count=current_count)

            return max(current_count, cc_below, cc_above, cc_left, cc_right)

        max_count = 0
        for i in range(n):
            for j in range(m):
                longest_path_length = dfs(i,j, tuple([]), 0)
                max_count = max(max_count, longest_path_length)
        return max_count


# In[30]:


class DisjointSet:

    def __init__(self, size):
        self.rank = [0 for _ in range(size)]
        self.parent = [i for i in range(size)]

    def find(self, x: int):
        if x != self.parent[x]:
            x = self.find(self.parent[x])
        return x

    def union(self, x: int, y: int):

        root_x = self.find(x)
        root_y = self.find(y)

        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_x] = root_y
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_y] += 1

class Graph:

    def __init__(self, size: int):
        self.size = size
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        # self.graph[v].append(u)  # if undirected

    def has_cycle(self):

        ds = DisjointSet(self.size)

        for u in self.graph:
            for v in self.graph[u]:
                # if they have the same parent, we've found a cycle
                if ds.find(u) == ds.find(v):
                    print(f"cycle detected between {u} and {v}")
                else:
                    ds.union(u, v)

#   1             9
#   | \          / |
#   |  3 ------ 7  |
#   |/           \ |
#   2             8
# has cycle
print("Expecting two cycles")
g = Graph(10)
g.add_edge(1, 3)
g.add_edge(3, 2)
g.add_edge(2, 1)
g.add_edge(3, 7)
g.add_edge(7, 8)
g.add_edge(8, 9)
g.add_edge(9, 7)
g.has_cycle()
print()

print("Expecting no cycle.")
# no cycle
g = Graph(10)
g.add_edge(1, 3)
g.add_edge(3, 7)
g.add_edge(7, 8)
g.add_edge(7, 9)
g.has_cycle()


# In[ ]:




