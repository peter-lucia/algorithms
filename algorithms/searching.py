from typing import List, Optional, Tuple

from .data_structures import TreeNode


def binary_search(nums: List[int], target):
    """
    Assumes nums is sorted in increasing order
    O(logn) time complexity
    """
    if len(nums) == 0 or target == nums[0]:
        return 0
    low = 0
    high = len(nums)
    while low <= high:
        # prevent integer overflow that can happen with (low + high) // 2
        mid = low + ((high - low) // 2)
        if nums[mid] < target:
            low = mid + 1
        elif nums[mid] > target:
            high = mid - 1
        else:
            return mid
    return -1

########################################################################################################################
#################################### Iterative #########################################################################
########################################################################################################################


def dfs_tree_iterative_preorder(root: TreeNode,
                       target: int | None = None,
                       visited: list[int] = []) -> list[int]:
    """
    DFS algorithm to traverse a binary tree using preorder traversal, where
    it visits a node before visiting that node's left subtree, then right subtree.
    No guarantees about finding the shortest path
    """

    if root.val is None:
        return []

    stack = [root]

    while stack:
        node = stack.pop()  # DFS: pop last elem (Stack)

        visited.append(node.val)

        # searching for target
        if target is not None and node.val == target:
            return [*node.path, node.val]

        if node.right:
            node.right.path = [*node.path, node.val]
            stack.append(node.right)

        # appending the left side last ensures it's visited first
        # when it's popped off the top of the stack
        if node.left:
            node.left.path = [*node.path, node.val]
            stack.append(node.left)

    if target is not None:
        raise ValueError("No path to target found")
    return visited


def dfs_tree_iterative_inorder(root: TreeNode,
                       target: int | None = None,
                       visited: list[int] = []) -> list[int]:
    """
    DFS algorithm to traverse a binary tree using inorder traversal, where
    it visits the left subtree before visiting a node, and then visits the right subtree.
    No guarantees about finding the shortest path
    """

    if root.val is None:
        return []

    stack = []
    node = root

    while node or stack:
        while node:
            stack.append(node)
            node = node.left

        node = stack.pop()

        visited.append(node.val)
        node = node.right

    return visited

def dfs_tree_iterative_postorder(root: TreeNode,
                                 target: int | None = None,
                                 visited: list[int] = []):
    """
    DFS algorithm to traverse a binary tree using postorder traversal, where
    it visits the left subtree, then right subtree, before the node.
    No guarantees about finding the shortest path
    """
    pass


def bfs_tree_iterative(root: TreeNode, target: int, preorder_visited: List[int]) -> List[int]:

    if root.val is None:
        return []

    stack = [root]

    while stack:
        node = stack.pop(0)  # BFS: pop first elem (Queue)

        preorder_visited.append(node.val)

        # searching for target
        if node.val == target:
            return [*node.path, node.val]

        for next_node in [node.left, node.right]:  # BFS: left to right with bfs, left popped off stack first
            if next_node is not None:
                next_node.path = [*node.path, node.val]
                stack.append(next_node)

    raise ValueError("No path to target found")


def dfs_graph_iterative(graph: dict, target: int) -> Optional[Tuple[TreeNode, str]]:
    pass


def bfs_graph_iterative(graph: dict, starting_node, target: int | None = None, visited: list[str] = []):
    """
    BFS uses a queue and visits all nodes on each level before going to the next level
    Because of this, BFS will find the shortest path to a target node
    """
    queue = []

    visited.append(starting_node)
    queue.append(starting_node)

    print("Visiting vertices: ")
    while queue:
        # print("Queue: ", queue)
        s = queue.pop(0)
        print(s, end=" ")
        for neighbour in graph[s]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)




########################################################################################################################
#################################### Recursive #########################################################################
########################################################################################################################


def dfs_recursive_preorder_tree(root: TreeNode, visited: list[int] = []) -> None:

    if root is None:
        return

    visited.append(root.val)

    if root.left:
        dfs_recursive_preorder_tree(root.left, visited)

    if root.right:
        dfs_recursive_preorder_tree(root.right, visited)


def dfs_recursive_inorder_tree(root: TreeNode, visited: list[int] = []) -> None:

    if root is None:
        return

    if root.left:
        dfs_recursive_inorder_tree(root.left, visited)

    visited.append(root.val)

    if root.right:
        dfs_recursive_inorder_tree(root.right, visited)


def dfs_recursive_postorder_tree(root: TreeNode, visited: list[int] = []) -> None:

    if root is None:
        return

    if root.left:
        dfs_recursive_postorder_tree(root.left, visited)

    if root.right:
        dfs_recursive_postorder_tree(root.right, visited)

    visited.append(root.val)


def bfs_tree_recursive(root: TreeNode, target: int) -> Optional[Tuple[TreeNode, str]]:
    pass


def dfs_recursive_graph(graph: dict, target: int) -> Optional[Tuple[TreeNode, str]]:
    pass


def bfs_recursive_graph(graph: dict, target: int) -> Optional[Tuple[TreeNode, str]]:
    pass
