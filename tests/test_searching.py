from random import shuffle

import pytest
from algorithms.searching import *

large_list = [i for i in range(-1000000, 1000000)]
large_list_shuffled = large_list.copy()
shuffle(large_list_shuffled)
large_list[1000000] = 9000000
searching_testcases = [
    ([4, 6, 10], 10, 2),
    ([1, 2, 3, 4, 5, 6, 7], 5, 4),
    ([10, 11, 12], 2, -1),
    ([0, 1, 2, 10, 11, 12], 2, 2),
    (large_list, 9000000, 1000000)
]

#                 20
#               /    \
#              8       22
#             / \     /  \
#            5   3   4    25
#               / \
#              10  14
root_binary_tree = TreeNode(20)
root_binary_tree.left = TreeNode(8)
root_binary_tree.right = TreeNode(22)
root_binary_tree.left.left = TreeNode(5)
root_binary_tree.left.right = TreeNode(3)
root_binary_tree.right.left = TreeNode(4)
root_binary_tree.right.right = TreeNode(25)
root_binary_tree.left.right.left = TreeNode(10)
root_binary_tree.left.right.right = TreeNode(14)

#                 20
#               /    \
#              2       22
#                     /
#                    4
#                   /
#                  8
root_binary_tree_2 = TreeNode(20)
root_binary_tree_2.left = TreeNode(2)
root_binary_tree_2.right = TreeNode(22)
root_binary_tree_2.right.left = TreeNode(4)
root_binary_tree_2.right.left.left = TreeNode(8)


@pytest.mark.parametrize("nums,target,expected", searching_testcases)
def test_binary_search(nums, target, expected):
    result = binary_search(nums, target)
    assert result == expected, "Incorrect result"


@pytest.mark.parametrize("root,target,target_path_expected,preorder_visited,inorder_visited,postorder_visited", [
    (root_binary_tree, 14, [20,8,3,14], [20,8,5,3,10,14], [5,8,10,3,14,20,4,22,25], [5,10,14,3,8,4,25,22,20]),
    (root_binary_tree_2, 8, [20,22,4,8],[20,2,22,4,8], [2,20,8,4,22], [2,8,4,22,20]),
    (root_binary_tree_2, 2, [20,2], [20,2], [2,20,8,4,22], [2,8,4,22,20]),
])
def test_dfs_tree_iterative(root: TreeNode, target, target_path_expected, preorder_visited, inorder_visited, postorder_visited):
    """Assumes preorder visited order"""
    visited = []
    target_path_actual = dfs_tree_iterative_preorder(root, target, visited)
    assert target_path_expected == target_path_actual, "Incorrect result"
    assert preorder_visited == visited


@pytest.mark.parametrize("root,target,target_path_expected,preorder_visited,inorder_visited_expected,postorder_visited", [
    (root_binary_tree, 14, [20,8,3,14], [20,8,5,3,10,14], [5,8,10,3,14,20,4,22,25], [5,10,14,3,8,4,25,22,20]),
    (root_binary_tree_2, 8, [20,22,4,8],[20,2,22,4,8], [2,20,8,4,22], [2,8,4,22,20]),
    (root_binary_tree_2, 2, [20,2], [20,2], [2,20,8,4,22], [2,8,4,22,20]),
])
def test_dfs_tree_iterative(root: TreeNode, target, target_path_expected, preorder_visited, inorder_visited_expected, postorder_visited):
    """Assumes preorder visited order"""
    visited = []
    dfs_tree_iterative_inorder(root, target, visited)
    assert visited == inorder_visited_expected


@pytest.mark.parametrize("root,target,expected,preorder_visited", [
    (root_binary_tree, 14, [20,8,3,14], [20,8,5,3,10,14]),
    (root_binary_tree_2, 8, [20,22,4,8],[20,2,22,4,8] ),
    (root_binary_tree_2, 2, [20,2], [20,2]),
])
def test_dfs_tree_recursive(root: TreeNode, target, expected, preorder_visited):
    actual = dfs_recursive_tree(root, target)
    assert expected == actual, "Incorrect result"


@pytest.mark.parametrize("root,target,expected,preorder_visited", [
    (root_binary_tree, 14, [20,8,3,14], [20,8,22,5,3,4,25,10,14]),
    (root_binary_tree_2, 8, [20,22,4,8],[20,2,22,4,8] ),
    (root_binary_tree_2, 2, [20,2], [20,2]),
])
def test_bfs_tree(root: TreeNode, target, expected,preorder_visited):
    actual_visited = []
    actual = bfs_tree_iterative(root, target, actual_visited)
    assert expected == actual, "Incorrect result"
    assert preorder_visited == actual_visited, "Incorrect visited order"
    # actual = bfs_recursive_tree(root, target)
    # assert expected == actual, "Incorrect result"
