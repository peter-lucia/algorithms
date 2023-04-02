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


@pytest.mark.parametrize("root,target,expected", [
    (root_binary_tree, 14, "20->8->3->14"),
    (root_binary_tree_2, 8, "20->22->4->8"),
    (root_binary_tree_2, 2, "20->2"),
])
def test_dfs_iterative_tree(root: TreeNode, target, expected):
    actual = dfs_iterative_tree(root, target)
    assert expected == actual, "Incorrect result"
    actual = dfs_recursive_tree(root, target)
    assert expected == actual, "Incorrect result"
