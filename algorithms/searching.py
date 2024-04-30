from typing import List, Optional, Tuple
from collections import deque

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


def dfs_tree_iterative(root: TreeNode,
                       target: int,
                       preorder_visited: List[int] = []) -> Optional[List[int]]:

    if root.val is None:
        return []

    stack = [root]

    while stack:
        node = stack.pop()  # DFS: pop last elem (Stack)

        preorder_visited.append(node.val)

        # searching for target
        if node.val == target:
            return [*node.path, node.val]

        if node.right:
            node.right.path = [*node.path, node.val]
            stack.append(node.right)

        if node.left:
            node.left.path = [*node.path, node.val]
            stack.append(node.left)


    raise ValueError("No path to target found")


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


def bfs_graph_iterative(graph: dict, target: int) -> Optional[Tuple[TreeNode, str]]:
    pass



########################################################################################################################
#################################### Recursive #########################################################################
########################################################################################################################


def dfs_recursive_tree(root: TreeNode, target: int) -> Optional[List[int]]:

    if root.val == target:
        # joins root.val into the root.path list
        return [*root.path, root.val]

    if root.left:
        root.left.path = [*root.path, root.val]
        result = dfs_recursive_tree(root.left, target)
        if result is not None:
            return result

    if root.right:
        root.right.path = [*root.path, root.val]
        result = dfs_recursive_tree(root.right, target)
        if result is not None:
            return result

    return None


def bfs_tree_recursive(root: TreeNode, target: int) -> Optional[Tuple[TreeNode, str]]:
    pass


def dfs_recursive_graph(graph: dict, target: int) -> Optional[Tuple[TreeNode, str]]:
    pass


def bfs_recursive_graph(graph: dict, target: int) -> Optional[Tuple[TreeNode, str]]:
    pass
