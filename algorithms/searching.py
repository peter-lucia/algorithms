from typing import List, Optional, Tuple

from data_structures import TreeNode


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
        mid = low + ((high - low) // 2)
        if nums[mid] < target:
            low = mid + 1
        elif nums[mid] > target:
            high = mid - 1
        else:
            return mid
    return -1


########################################################################################################################
#################################### Depth First Search ################################################################
########################################################################################################################

def dfs_recursive_trees(root: TreeNode, target: int) -> Optional[TreeNode]:

    if root.val == target:
        return root

    if root.left:
        result = dfs_recursive_trees(root.left, target)
        if result is not None:
            return result

    if root.right:
        result = dfs_recursive_trees(root.right, target)
        if result is not None:
            return result

    return None


def dfs_iterative_tree(root: TreeNode, target: int) -> Optional[Tuple[TreeNode, str]]:

    stack = [(root, None)]

    while stack:
        node, path = stack.pop()

        if node.val == target:
            return node, path

        for node in [node.left, node.right]:
            if path is None:
                stack.append((node.left, f"{node.val}"))
            else:
                stack.append((node.left, path + "->" + f"{node.val}"))


def dfs_recursive_graph(graph: dict, target: int) -> Optional[Tuple[TreeNode, str]]:
    pass


def dfs_iterative_graph(graph: dict, target: int) -> Optional[Tuple[TreeNode, str]]:
    pass


def dfs_recursive_graph(graph: dict, target: int) -> Optional[Tuple[TreeNode, str]]:
    pass


def dfs_iterative_graph(graph: dict, target: int) -> Optional[Tuple[TreeNode, str]]:
    pass

########################################################################################################################
#################################### Breadth First Search ##############################################################
########################################################################################################################

def bfs_recursive_trees(root: TreeNode, target: int) -> Optional[Tuple[TreeNode, str]]:
    pass


def bfs_iterative_trees(root: TreeNode, target: int) -> Optional[Tuple[TreeNode, str]]:
    pass
