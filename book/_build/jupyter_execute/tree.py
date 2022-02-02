#!/usr/bin/env python
# coding: utf-8

# # Trees

# [1. Bottom-View-Balanced-Binary-Tree-(queue)](#Bottom-View-Balanced-Binary-Tree-(queue))
# 
# [2. Bottom View Binary Tree (hashmap)](#Bottom-View-Binary-Tree-(hashmap))

# 

# In[1]:


from collections import defaultdict
from typing import Optional, List


# ## Bottom View Balanced Binary Tree (queue)
# * similar to bfs
# * Use a lookup to map the horizontal distance to the node
# * Horizontal distance decreases when we visit the left node
# * Horizontal distance increases when we visit the right node
# * Basically modify BFS to append left and right nodes to the queue (if they exist) while updating horizontal distances for each node
# * At the end, the lookup will have only the bottom view nodes since each horizontal distance at the end will be unique

# In[2]:


# Python3 program to print Bottom
# Source: https://www.geeksforgeeks.org/bottom-view-binary-tree/

# Tree node class
class Node:
    def __init__(self, key):
        self.val = key
        self.hd = 0 # horizontal distance from center root node
        self.left = None
        self.right = None

def binary_tree_bottom_view(root):
    """
    Use horizontal distance to determine order of bottom view

    At the end we will have the following left to right bottom view
    {
        # horizontal dist: node.val
        -2: 5,
        -1: 10,
         0: 4,
         1: 14,
         2: 25,
    }
    """

    if root is None:
        return


    lookup = dict()

    # Queue to store tree nodes in level
    # order traversal
    queue = []

    # Assign initialized horizontal distance
    # value to root node and add it to the queue.
    root.hd = 0

    # In STL, append() is used enqueue an item
    queue.append(root)

    while queue:
        node = queue.pop(0)

        # We always want to update the lookup with
        # this node's position and value
        lookup[node.hd] = node

        # Add left child to queue with hd = hd - 1
        if node.left is not None:
            node.left.hd = node.hd - 1
            queue.append(node.left)

        # Add right child to queue with hd = hd + 1
        if node.right is not None:
            node.right.hd = node.hd + 1
            queue.append(node.right)

    # Sort the map based on increasing hd for left to right bottom view
    for i in sorted(lookup.keys()):
        print(lookup[i].val, end = ' ')


# Balanced Binary Trees
# * Access is O(logn) in the worst case
# * Space complexity is the same as the unbalanced binary tree O(n)
# ```
# 
#                       20
#                     /    \
#                   8       22
#                 /   \    /   \
#               5      3  4    25
#                     / \
#                   10    14
# ```

# In[3]:


# Driver Code
root = Node(20)
root.left = Node(8)
root.right = Node(22)
root.left.left = Node(5)
root.left.right = Node(3)
root.right.left = Node(4)
root.right.right = Node(25)
root.left.right.left = Node(10)
root.left.right.right = Node(14)
print("Bottom view of the given binary tree :")
binary_tree_bottom_view(root)


# Unbalanced binary trees
# * Access time complexity is O(n) in the worst case
# * Space complexity is the same as the balanced binary tree O(n)
# ```
#                       20
#                     /    \
#                   2       22
#                          /
#                         4
#                        /
#                       8
# ```

# In[4]:


# This shows that the algorithm does not work on unbalanced binary trees
root = Node(20)
root.left = Node(2)
root.right = Node(22)
root.right.left = Node(4)
root.right.left.left = Node(8)
binary_tree_bottom_view(root)


# #### Bottom View Binary Tree (hashmap)

# In[5]:


class Node:
    def __init__(self, key = None,
                      left = None,
                     right = None):
        self.val = key
        self.left = left
        self.right = right

def bottom_view_hashmap(root):
    """
    key = relative horizontal distance of the node from root node and
    value = pair containing node's value and its level
    {
        # horizontal dist: (node.val, node's level)
        -2: (5, 2),
        -1: (10, 3,
         0: (4, 2)
         1: (14, 3),
         2: (25, 2),
    }
    """
    lookup = dict()

    bottom_view_hashmap_util(root, lookup, 0, 0, "start")

    print(lookup)
    # print the bottom view
    for key in sorted(lookup.keys()):
        print(lookup[key][0], end = " ")

def bottom_view_hashmap_util(root, lookup, hd, level, path):

    if root is None:
        return

    # If current level is more than or equal
    # to maximum level seen so far for the
    # same horizontal distance or horizontal
    # distance is seen for the first time,
    # update the dictionary
    if path in lookup:
        if level >= lookup[path][1]:
            lookup[path] = [root.val, level]
    else:
        lookup[path] = [root.val, level]

    # this node has children, only its children should be in the lookup
    if root.left or root.right:
        del lookup[path]

    # recurse for left subtree by decreasing
    # horizontal distance and increasing
    # level by 1
    bottom_view_hashmap_util(root.left,
                             lookup,
                             hd - 1,
                             level + 1,
                             path + "-left")

    # recurse for right subtree by increasing
    # horizontal distance and increasing
    # level by 1
    bottom_view_hashmap_util(root.right,
                             lookup,
                             hd + 1,
                             level + 1,
                             path + "-right")



#                 20
#               /    \
#              2       22
#                     /
#                    4
#                   /
#                  8
print("Bottom view of the given binary tree:")
root = Node(20)
root.left = Node(2)
root.right = Node(22)
root.right.left = Node(4)
root.right.left.left = Node(8)

bottom_view_hashmap(root)


# In[6]:


#                 20
#               /    \
#              8       22
#             / \     /  \
#            5   3   4    25
#               / \
#              10  14
root = Node(20)
root.left = Node(8)
root.right = Node(22)
root.left.left = Node(5)
root.left.right = Node(3)
root.right.left = Node(4)
root.right.right = Node(25)
root.left.right.left = Node(10)
root.left.right.right = Node(14)
bottom_view_hashmap(root)


# In[7]:


#                 20
#               /    \
#              8      4
#               \    /
#                2  1
root = Node(20)
root.left = Node(8)
root.right = Node(4)
root.left.right = Node(2)
root.right.left = Node(1)
bottom_view_hashmap(root)
# TODO: the lookup map needs to incorporate the nodes full path to be unique


# In[8]:


from typing import List
def bottom_view_hashmap_dfs(start_node: Node):


    def dfs(root: Node, leaves: List[Node]):

        if root is None:
            return


        # pre-order traversal
        if not root.left and not root.right:
            print(f"{root.val} has no children")
            leaves.append(root)

        dfs(root.left, leaves)
        dfs(root.right, leaves)




    leaves = []
    dfs(start_node, leaves)
    for leaf in leaves:
        print(f"{leaf.val}", end=" ")

#                 20
#               /    \
#              8      4
#               \    /
#                2  1
root = Node(20)
root.left = Node(8)
root.right = Node(4)
root.left.right = Node(2)
root.right.left = Node(1)
bottom_view_hashmap_dfs(root)


# #### Top View Binary Tree
# Source: http://code2begin.blogspot.com/2018/07/top-view-of-binary-tree.html

# #### Find Leaves of Binary Tree
# 
# ```text
# Input: root = [1,2,3,4,5]
# Output: [[4,5,3],[2],[1]]
# Explanation:
# [[3,5,4],[2],[1]] and [[3,4,5],[2],[1]] are also considered correct answers since per each level it does not matter the order on which elements are returned.
# 
# Example 2:
# 
# Input: root = [1]
# Output: [[1]]
# ```

# # Find leaves of a binary tree
# * https://leetcode.com/problems/find-leaves-of-binary-tree/
# 

# In[9]:



from typing import Optional
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:

    def bottom_view_hashmap(self, root):
        deleted_list = []
        def util(root, lookup, hd, level, parent = None, is_left = True):
            if root is None:
                return

            if hd in lookup:
                # we've already seen this hd before, if the level is greater or equal,
                # update the value at this horizontal distance to this node's value and its level
                if level >= lookup[hd][1]:
                    lookup[hd] = [root.val, level]
            else:
                # ensure the root's val and level are in the lookup
                lookup[hd] = [root.val, level]

            # this node has children, only its children should be in the lookup
            if root.left or root.right:
                del lookup[hd]

            # recurse toward the left and right nodes. Will return None if they don't exist
            # pass the parent node in and indicate if this node points left or right
            # so we can chop off this node if it's a leaf
            util(root.left, lookup, hd-1, level + 1, parent=root, is_left=True)
            util(root.right, lookup, hd+1, level + 1, parent=root, is_left=False)


            if not root.left and not root.right:
                # add this node to deleted list then delete it
                deleted_list.append((root.val, parent, is_left))

        lookup = {}
        util(root, lookup, 0, 0)
        # the bottom view
        print(f"Bottom view: {[value[0] for _, value in lookup.items()]}")
        return deleted_list


    def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:

        result = []
        i = 0
        while root:
            temp_vals = []
            # get the current bottom view
            deletion_list = self.bottom_view_hashmap(root)
            # add vals to result
            for val, _, _ in deletion_list:
                temp_vals.append(val)
            result.append(temp_vals)
            # delete the nodes in the deletion list
            for val, parent, is_left in deletion_list:
                if parent and is_left:
                    parent.left = None
                elif parent and not is_left:
                    parent.right = None
                if not parent:
                    # we're at the root
                    root = None
        return result
root = Node(20)
root.left = Node(2)
root.right = Node(22)
root.right.left = Node(4)
root.right.left.left = Node(8)
Solution().findLeaves(root)


# In[10]:


root = Node(20)
root.left = Node(8)
root.right = Node(4)
root.left.right = Node(2)
root.right.left = Node(1)
Solution().findLeaves(root)


# In[11]:


import collections
# Another solution similar to the last for findLeaves of a binary tree
# https://leetcode.com/problems/find-leaves-of-binary-tree/
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:

    def bottom_view_hashmap(self, root, lookup, level, path, parent, is_left, deleted_list):

        if root is None:
            return

        if path in lookup:
            if level >= lookup[path][1]:
                lookup[path] = [root.val, level]
        else:
            lookup[path] = [root.val, level]



        if root.left is None and root.right is None:
            deleted_list.append(root.val)
            if is_left:
                parent.left = None

            else:
                parent.right = None
            del lookup[path]
            return

        self.bottom_view_hashmap(root.left,
                                lookup,
                                level + 1,
                                path + "-left",
                                root,
                                True,
                                deleted_list)
        self.bottom_view_hashmap(root.right,
                                lookup,
                                level + 1,
                                path + "-right",
                                root,
                                False,
                                deleted_list)


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
        result = []
        while root:
            deleted_list = []
            self.bottom_view_hashmap(root, lookup, 0, "", root, False, deleted_list)
            result.append(deleted_list)
            if not lookup:
                break

        return result


# In[12]:


#              20
#            /    \
#           8      4
#            \    /
#             2  1
root = Node(20)
root.left = Node(8)
root.right = Node(4)
root.left.right = Node(2)
root.right.left = Node(1)
Solution().findLeaves(root)


# In[13]:


#              20
#            /    \
#           2      22
#                 /
#                4
#               /
#              8

root = Node(20)
root.left = Node(2)
root.right = Node(22)
root.right.left = Node(4)
root.right.left.left = Node(8)
Solution().findLeaves(root)


# ### Find minimum depth
# 
# The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
# 
# ```
#        3
#      /   \
#     9     20
#          /   \
#        15     7
# 
# Input: root = [3,9,20,null,null,15,7]
# Output: 2
# 3 and 9 are the nodes in the shortest path
# 
# Input: root = [2,null,3,null,4,null,5,null,6]
# Output: 5
# ```
# 
# This is a direct application of bfs

# In[14]:


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return str({
            "val": self.val if self.val else None,
            "left": self.left.val if self.left else None,
            "right": self.right.val if self.right else None,
        })

class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:

        if root is None:
            return 0

        def bfs(visited: List[TreeNode], node: TreeNode):
            queue = []
            depth = 1
            queue.append((node, depth))

            while queue:
                # if neighbor is not visited, add it to the queue
                elem, depth = queue.pop(0)
                visited.append(elem)

                if elem.left is None and elem.right is None:
                    return depth

                if elem.left and elem.left not in visited:
                    queue.append((elem.left, depth + 1))
                if elem.right and elem.right not in visited:
                    queue.append((elem.right, depth + 1))
        visited = []
        min_depth = bfs(visited, root)
        return min_depth

root = TreeNode(3)
b = TreeNode(9)
c = TreeNode(20)
d = TreeNode(15)
e = TreeNode(7)

root.left = b
root.right = c
c.left = d
c.right = e
Solution().minDepth(root)


# In[15]:


root = TreeNode(2)
b = TreeNode(3)
c = TreeNode(4)
d = TreeNode(5)
e = TreeNode(6)

root.right = b
b.right = c
c.right = d
d.right = e
Solution().minDepth(root)


# ### Binary Search Tree Definition
# 
# The left subtree of a node contains only nodes with keys lesser than the node’s key.
# The right subtree of a node contains only nodes with keys greater than the node’s key.
# The left and right subtree each must also be a binary search tree.
# Source: [https://www.geeksforgeeks.org/binary-search-tree-data-structure/](https://www.geeksforgeeks.org/binary-search-tree-data-structure/)

# ### BFS application on a binary tree

# In[16]:


# https://leetcode.com/problems/find-mode-in-binary-search-tree/submissions/
from collections import defaultdict
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        # binary search tree has the following property:
        # left_subtree (keys) ≤ node (key) < right_subtree (keys)
        # or
        # left_subtree (keys) < node (key) ≤ right_subtree (keys)

        # Approach:
        # use BFS to traverse the tree
        # push all elements in the tree into a lookup
        # {
        #     val: num_elements
        # }
        # find the max occurrences and then all nodes that have the max # occurrences
        lookup = defaultdict(int)

        def bfs(node: TreeNode):

            if not root:
                return

            queue = []
            queue.append(node)

            while queue:

                elem = queue.pop(0)

                lookup[elem.val] += 1

                if elem.left is not None:
                    queue.append(elem.left)

                if elem.right is not None:
                    queue.append(elem.right)

        bfs(root)
        max_occurrences = max(lookup.values())
        result = []
        for (val, occurrences) in lookup.items():
            if occurrences == max_occurrences:
                result.append(val)

        return result

#          1
#           \
#            2
#           /
#          2

root = TreeNode(1)
root.right = TreeNode(2)
root.right.left = TreeNode(2)

assert Solution().findMode(root) == [2]

#          1
#           \
#            2
#           / \
#          2   1

root = TreeNode(1)
root.right = TreeNode(2)
root.right.left = TreeNode(2)
root.right.right = TreeNode(1)

result = Solution().findMode(root)
for each in result:
    assert each in [1,2]


# ### Tree Traversal
# Source: https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/
# * DFS
#   * Pre-order
#   ```
#   Algorithm Preorder(tree)
#    1. Visit the root.
#    2. Traverse the left subtree, i.e., call Preorder(left-subtree)
#    3. Traverse the right subtree, i.e., call Preorder(right-subtree)
#   ```
#   * In-order
#   ```
#   Algorithm Inorder(tree)
#    1. Traverse the left subtree, i.e., call Inorder(left-subtree)
#    2. Visit the root.
#    3. Traverse the right subtree, i.e., call Inorder(right-subtree)
#   ```
#   * Post-order
#   ```
#   Algorithm Postorder(tree)
#    1. Traverse the left subtree, i.e., call Postorder(left-subtree)
#    2. Traverse the right subtree, i.e., call Postorder(right-subtree)
#    3. Visit the root.
#   ```
# * BFS
#   * Level-order

# #### Example
# ```
#         1
#           \
#             2
#           /
#         3
# ```

# In[17]:


root = TreeNode(1)
two = TreeNode(2)
three = TreeNode(3)
root.right = two
two.left = three


# ### Binary Tree In-order Traversal
# Visits the node after visiting the left subtree
# * https://leetcode.com/problems/binary-tree-inorder-traversal/

# In[18]:


from typing import Optional, List
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:

        result = []
        def traverse(node: Optional[TreeNode]):
            if node:
                traverse(node.left)
                result.append(node.val)
                traverse(node.right)
        traverse(root)
        return result

Solution().inorderTraversal(root)


# ### Binary Tree Pre-order Traversal
# Visits the node before visiting the left and right sub-trees
# 
# * https://leetcode.com/problems/binary-tree-preorder-traversal/

# In[19]:


from typing import Optional
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:

        result = []
        def traverse(node: Optional[TreeNode]):
            if node:
                result.append(node.val)
                traverse(node.left)
                traverse(node.right)
        traverse(root)
        return result

Solution().preorderTraversal(root)


# ### Binary Tree Post-order Traversal
# Visits the node after visiting the left and right subtrees
# * https://leetcode.com/problems/binary-tree-postorder-traversal/

# In[20]:


class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:

        result = []
        def traverse(node: Optional[TreeNode]):
            if node:
                traverse(node.left)
                traverse(node.right)
                result.append(node.val)

        traverse(root)
        return result

Solution().postorderTraversal(root)


# ### N-ary trees
# Each node can have up to n children
# ```
#           1
#          //\  
#        / /  \  
#       2 3    6  
# ```

# * A full n-ary tree allows each node to have between 0 and n children
# * A complete n-ary tree requires each node to have exactly n children except the leaves
# * A perfect n-ary tree requires that the level of all leaf nodes is the same

# ### Trie
# * Insertion (Worst): O(n) 
# * Search (Worst): O(n) 

# In[21]:


class Trie:
    def __init__(self):
        self.child = {}
        
    def insert(self, word: str):
        """
        Iterate over each character. 
        Ensure each character in the word is the sub key of a new dict
        The last character of the word maps to a dict with 'end' as at least one of the keys.
        """
        current = self.child
        for c in word:
            if c not in current:
                current[c] = {}
            current = current[c]
        current['end'] = 1
    
    def search(self, word: str):
        """
        Check that each character is in the trie and that the last character maps to a dict with 'end' as a key.
        """
        current = self.child
        for c in word:
            if c not in current:
                return False
            current = current[c]
        return 'end' in current
    
    def starts_with(self, prefix: str):
        """
        Iterates over each character in the prefix. 
        Does not check if the last character maps to a dict with 'end' as the key.
        """
        current = self.child
        for c in prefix:
            if c not in current:
                return False
            current = current[c]
        return True
    
t = Trie()
t.insert("apple")


# In[22]:


t.search("apple")


# In[23]:


t.search("app")
    


# In[24]:


t.starts_with("app")


# In[25]:


t.child


# In[26]:


t.insert("apricot")
t.insert("appendix")


# In[27]:


from pprint import pprint
pprint(t.child)


# In[28]:


t.search("apple")


# In[29]:


t.search("appendix")


# In[30]:


t.search("appendi")


# In[31]:


t.insert("apples")


# In[32]:


t.search("apple")


# In[33]:


pprint(t.child)


# ### Binary Tree Vertical Order Traversal
# 
# Given the root of a binary tree, return the vertical order traversal of its nodes' values. (i.e., from top to bottom, column by column).
# Note that if this was a binary search tree, the nodes would already be in sorted order and we could just print them based on the dfs inorder traversal.
# 
# If two nodes are in the same row and column, the order should be from left to right.
# 
# 

# In[34]:


from typing import Optional, List
from collections import defaultdict
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:

        # Keys in the lookup are the horizontal distances
        # need horizontal distance
        # need level

        lookup = defaultdict(list)

        def dfs(root: TreeNode, level: int, hd: int):
            if root is None:
                return

            # since we sort the nodes at the end, pre-order, inorder, or post-order doesn't matter
            lookup[hd].append((root.val, level))
            dfs(root.left, level + 1, hd - 1)
            dfs(root.right, level + 1, hd + 1)
        dfs(root, 0, 0)

        # sort the values
        result = []
        # ensure horizontal distances are in sorted order increasing
        # this ensures the order of columns is returned from left to right
        keys = sorted(lookup.keys())
        for key in keys:
            # ensure the nodes higher up in the tree for a single column come first
            # before nodes lower in the tree
            rw = sorted(lookup[key], key=lambda val: val[1])  # increasing
            rw = [a[0] for a in rw]
            result.append(rw)
        return result
"""

                      20
                    /    \
                  8       22
                /   \    /   \
              5      3  4    25
                    / \
                  10    14
"""
# Driver Code
root = TreeNode(20)
root.left = TreeNode(8)
root.right = TreeNode(22)
root.left.left = TreeNode(5)
root.left.right = TreeNode(3)
root.right.left = TreeNode(4)
root.right.right = TreeNode(25)
root.left.right.left = TreeNode(10)
root.left.right.right = TreeNode(14)
Solution().verticalOrder(root)


# #### AVL Tree and Balanced Binary Tree Definition
# - A balanced binary tree requires that at every node the height of the left and right subtrees can never have an absolute difference greater than 1.
# - A balanced tree always has a height no greater than O(logn)
# - AVL tree is a self-balancing Binary Search Tree (BST). It has worst case Access, Search, Insertion, and Deletion operations of O(logn)
# - When a (not necessarily balanced) binary search tree gets skewed, the running time complexity becomes the worst-case scenario i.e O(n) but in the case of the AVL tree, the time complexity remains O(logn). Therefore, it is always advisable to use an AVL tree rather than a binary search tree.
# 
# Example:
# ```
# Balanced
#     a
#    / \
#   b   c
# - a's left subtree height is 1 and its right subtree height is 1. b and c both have subtree heights of 0
# 
# 
# Not balanced
#     a
#      \
#       b
#        \
#         c
# - a's left subtree height is 0 but its right subtree height is 2. |2 - 0| = 2 which is greater than 1.
# 
# 
# Not balanced
#          a
#        /  \
#       c    b
#      / \
#     d   e
#        /
#       f
# - a's left subtree height is 3, but its right subtree height is 1. |3 - 1| = 2 which is greater than 1.
# ```
# 

# In[35]:


# AVL Tree exercise: check if the tree is balanced
# Source: https://www.geeksforgeeks.org/how-to-determine-if-a-binary-tree-is-balanced/
print("O(n) solution to check if a tree is balanced")
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def get_height_of_tree(root: Node):
    if root is None:
        return 0
    return max(get_height_of_tree(root.left), get_height_of_tree(root.right)) + 1

class Height:
    def __init__(self, val: int):
        self.val = val

def is_tree_balanced(root: Node, height: Height):
    if root is None:
        return True

    left_height = Height(height.val + 1)
    right_height = Height(height.val + 1)
    left_is_balanced = is_tree_balanced(root.left, left_height)
    right_is_balanced = is_tree_balanced(root.right, right_height)

    return abs(left_height.val - right_height.val) <= 1 and left_is_balanced and right_is_balanced

"""
Constructed binary tree is
          1
        /   \
       2     3
      / \   /
     4   5 6
    /
   7
"""

root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)
root.left.left.left = Node(7)
if is_tree_balanced(root, Height(0)):
    print("Tree is balanced.")
else:
    print("Tree is NOT balanced.")


# ### 701. Insert into a Binary Search Tree
# Source: [https://leetcode.com/problems/insert-into-a-binary-search-tree/](https://leetcode.com/problems/insert-into-a-binary-search-tree/)
# 
# 
# * A balanced binary search tree is additionally balanced.
# * The definition of balanced is implementation-dependent.
# * In red black trees, the depth of any leaf node is no more than twice the depth of any other leaf node.
# * In AVL trees, the depth of leaf nodes differ by at most one.

# In[36]:


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    @staticmethod
    def display_preorder(node: TreeNode):
        if node is None:
            return
        print(node.val, end= " ")
        TreeNode.display_preorder(node.left)
        TreeNode.display_preorder(node.right)

class Solution:

    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        # Don't need to balance the tree after insertion as a binary search tree is not necessarily balanced
        node = root
        while node:
            if node.val < val:
                if node.right is None:
                    node.right = TreeNode(val)
                    return root
                node = node.right
            elif node.val > val:
                if node.left is None:
                    node.left = TreeNode(val)
                    return root
                node = node.left
        return TreeNode(val)

#        Before
#               4
#             /   \
#            2     6
#                 /  \
#                5    7

root = TreeNode(4)
root.right = TreeNode(6)
root.right.left = TreeNode(5)
root.right.right = TreeNode(7)
root.left = TreeNode(2)
TreeNode.display_preorder(root)
Solution().insertIntoBST(root, 3)
print()
#        After
#               4
#             /   \
#            2     6
#             \   /  \
#              3 5    7
TreeNode.display_preorder(root)


# ### Implement an AVL Tree (701. Insert into a Binary Search Tree)
# Sources:
# - [https://en.wikipedia.org/wiki/AVL_tree](https://en.wikipedia.org/wiki/AVL_tree)
# - [https://leetcode.com/problems/insert-into-a-binary-search-tree/](https://leetcode.com/problems/insert-into-a-binary-search-tree/)
# - [https://favtutor.com/blogs/avl-tree-python](https://favtutor.com/blogs/avl-tree-python)
# - [https://www.geeksforgeeks.org/avl-tree-set-1-insertion/](https://www.geeksforgeeks.org/avl-tree-set-1-insertion/)
# ```
# T1, T2 and T3 are subtrees of the tree
# rooted with y (on the left side) or x (on
# the right side)
#      y                               x
#     / \     Right Rotation          /  \
#    x   T3   - - - - - - - >        T1   y
#   / \       < - - - - - - -            / \
#  T1  T2     Left Rotation            T2  T3
# Keys in both of the above trees follow the
# following order
#  keys(T1) < key(x) < keys(T2) < key(y) < keys(T3)
# So BST property is not violated anywhere.
# ```

# In[37]:


from collections import defaultdict

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1

    @staticmethod
    def display_preorder(node: TreeNode):
        if node is None:
            return

        print(node.val, end=" ")
        TreeNode.display_preorder(node.left)
        TreeNode.display_preorder(node.right)

    @staticmethod
    def display_inorder(node: TreeNode):
        if node is None:
            return

        TreeNode.display_inorder(node.left)
        print(node.val, end=" ")
        TreeNode.display_inorder(node.right)

class AVLTree:

    def insert(self, z: TreeNode, x):
        if not z:
            return TreeNode(x)
        elif x < z.val:
            z.left = self.insert(z.left, x)
        else:
            z.right = self.insert(z.right, x)

        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        b = self.get_balance(z)

        if b > 1 and z.left.val > x:
            # Left Left Imbalance
            # T1, T2, T3 and T4 are subtrees.
            #          z                                      y
            #         / \                                   /   \
            #        y   T4      Right Rotate (z)          x      z
            #       / \          - - - - - - - - ->      /  \    /  \
            #      x   T3                               T1  T2  T3  T4
            #     / \
            #   T1   T2
            return self.right_rotation(z)

        if b > 1 and z.left.val < x:
            # Left Right Imbalance
            #      z                               z                           x
            #     / \                            /   \                        /  \
            #    y   T4  Left Rotate (y)        x    T4  Right Rotate(z)    y      z
            #   / \      - - - - - - - - ->    /  \      - - - - - - - ->  / \    / \
            # T1   x                          y    T3                    T1  T2 T3  T4
            #     / \                        / \
            #   T2   T3                    T1   T2
            z.left = self.left_rotation(z.left)
            return self.right_rotation(z)

        if b < -1 and z.right.val < x:
            # Right Right Imbalance
            #   z                                y
            #  /  \                            /   \
            # T1   y     Left Rotate(z)       z      x
            #     /  \   - - - - - - - ->    / \    / \
            #    T2   x                     T1  T2 T3  T4
            #        / \
            #      T3  T4
            return self.left_rotation(z)

        if b < -1 and z.right.val > x:
            # Right Left Imbalance
            #    z                            z                            x
            #   / \                          / \                          /  \
            # T1   y   Right Rotate (y)    T1   x      Left Rotate(z)   z      y
            #     / \  - - - - - - - - ->     /  \   - - - - - - - ->  / \    / \
            #    x   T4                      T2   y                  T1  T2  T3  T4
            #   / \                              /  \
            # T2   T3                           T3   T4
            z.right = self.right_rotation(z.right)
            return self.left_rotation(z)

        return z

    def get_height(self, node: TreeNode):
        if not node:
            return 0
        return node.height

    def get_balance(self, node: TreeNode):
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)

    def right_rotation(self, node: TreeNode):
        """
        right_rotation(4):
                 4           2
                /          /  \
               2   ->     1   4
              / \            /
             1   3          3
        """
        new_node = node.left
        subtree = new_node.right
        new_node.right = node
        node.left = subtree
        new_node.height = 1 + max(self.get_height(new_node.left), self.get_height(new_node.right))
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))
        return new_node

    def left_rotation(self, node: TreeNode):
        """
        right_rotation(4):
              4              2
              \            /  \
               2   ->     4   3
              / \          \
             1   3          1
        """
        new_node = node.right
        subtree = new_node.left
        new_node.left = node
        node.right = subtree
        new_node.height = 1 + max(self.get_height(new_node.left), self.get_height(new_node.right))
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))
        return new_node


#        Before
#               2
#             /   \
#            1     4
#                 / \
#                3   5
tree = AVLTree()
root = tree.insert(None, 1)
root = tree.insert(root, 2)
root = tree.insert(root, 4)
root = tree.insert(root, 3)
root = tree.insert(root, 5)
print("Preorder traversal: ")
TreeNode.display_preorder(root)
print("\nObserve the inorder traversal is still increasing since it's a BST")
TreeNode.display_inorder(root)

#        After
#                   4
#                 /   \
#                2     5
#              /  \     \
#             1    3     6
root = tree.insert(root, 6)
print("\nPreorder traversal after inserting 3: ")
TreeNode.display_preorder(root)
print("\nObserve the inorder traversal is still increasing since it's a BST")
TreeNode.display_inorder(root)


# In[38]:


# Driver program to test above function
tree = AVLTree()
root = None

root = tree.insert(root, 10)
root = tree.insert(root, 20)
root = tree.insert(root, 30)
root = tree.insert(root, 40)
root = tree.insert(root, 50)
root = tree.insert(root, 25)

"""The constructed AVL Tree would be
            30
           /  \
         20   40
        /  \     \
       10  25    50
"""

# Preorder Traversal
print("Preorder traversal of the",
      "constructed AVL tree is")
TreeNode.display_preorder(root)
print()


# In[39]:


# Python code to insert a node in AVL tree

# Generic tree node class
class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1

# AVL tree class which supports the
# Insert operation
class AVL_Tree(object):

    # Recursive function to insert key in
    # subtree rooted with node and returns
    # new root of subtree.
    def insert(self, root, key):

        # Step 1 - Perform normal BST
        if not root:
            return TreeNode(key)
        elif key < root.val:
            root.left = self.insert(root.left, key)
        else:
            root.right = self.insert(root.right, key)

        # Step 2 - Update the height of the
        # ancestor node
        root.height = 1 + max(self.getHeight(root.left),
                           self.getHeight(root.right))

        # Step 3 - Get the balance factor
        balance = self.getBalance(root)

        # Step 4 - If the node is unbalanced,
        # then try out the 4 cases
        # Case 1 - Left Left
        if balance > 1 and key < root.left.val:
            return self.rightRotate(root)

        # Case 2 - Right Right
        if balance < -1 and key > root.right.val:
            return self.leftRotate(root)

        # Case 3 - Left Right
        if balance > 1 and key > root.left.val:
            root.left = self.leftRotate(root.left)
            return self.rightRotate(root)

        # Case 4 - Right Left
        if balance < -1 and key < root.right.val:
            root.right = self.rightRotate(root.right)
            return self.leftRotate(root)

        return root

    def leftRotate(self, z):

        y = z.right
        T2 = y.left

        # Perform rotation
        y.left = z
        z.right = T2

        # Update heights
        z.height = 1 + max(self.getHeight(z.left),
                         self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left),
                         self.getHeight(y.right))

        # Return the new root
        return y

    def rightRotate(self, z):

        y = z.left
        T3 = y.right

        # Perform rotation
        y.right = z
        z.left = T3

        # Update heights
        z.height = 1 + max(self.getHeight(z.left),
                        self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left),
                        self.getHeight(y.right))

        # Return the new root
        return y

    def getHeight(self, root):
        if not root:
            return 0

        return root.height

    def getBalance(self, root):
        if not root:
            return 0

        return self.getHeight(root.left) - self.getHeight(root.right)

    def preOrder(self, root):

        if not root:
            return

        print("{0} ".format(root.val), end="")
        self.preOrder(root.left)
        self.preOrder(root.right)


# Driver program to test above function
myTree = AVL_Tree()
root = None

root = myTree.insert(root, 10)
root = myTree.insert(root, 20)
root = myTree.insert(root, 30)
root = myTree.insert(root, 40)
root = myTree.insert(root, 50)
root = myTree.insert(root, 25)

"""The constructed AVL Tree would be
            30
           /  \
         20   40
        /  \     \
       10  25    50"""

# Preorder Traversal
print("Preorder traversal of the",
      "constructed AVL tree is")
myTree.preOrder(root)
print()

# This code is contributed by Ajitesh Pathak


# ### 116. Populating Next Right Pointers in Each Node in a Perfect Binary Tree
# [https://leetcode.com/problems/populating-next-right-pointers-in-each-node/](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)

# In[40]:



# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
    def display(self):
        def bfs(root):
            queue = []
            queue.append(root)
            while queue:
                node = queue.pop(0)
                print(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        bfs(self)

class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        # Useful data structures for this problem:
        # Since we are always given a perfect binary tree,
        # we can use a hashmap and DFS with inorder traversal (left -> parent -> right)
        # Hashmap mapping levels to nodes. Nodes must be ordered from left to right
        # in the hashmap

        #  0 -> 1
        #  1 -> [2,3]
        #  2 -> [4, 5, 6, 7]

        lookup = defaultdict(list)

        def dfs(node: Node, level: int):
            if node is None:
                return

            dfs(node.left, level + 1)
            lookup[level].append(node)
            dfs(node.right, level + 1)

        dfs(root, 0)

        for key in lookup.keys():
            values = lookup[key]
            if len(values) > 1:
                i = 0
                j = 1

                while j < len(values):
                    values[i].next = values[j]
                    i += 1
                    j += 1
            lookup[key] = values
        return root
#           1
#        /    \
#       2      3
#      / \    / \
#     4   5  6   7
root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)
root.right.right = Node(7)
Solution().connect(root)
root.display()


# In[41]:


# Same problem as above with BFS

class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        # Useful data structures for this problem:
        # Since we are always given a perfect binary tree,
        # we can use a hashmap and BFS (level order traversal) (left -> right)
        # Hashmap mapping levels to nodes. Nodes are automatically ordered from left to right at each level
        # in the hashmap

        #  0 -> 1
        #  1 -> [2,3]
        #  2 -> [4, 5, 6, 7]

        lookup = defaultdict(list)
        if not root:
            return root

        def bfs(_node: Node):
            queue = []
            queue.append((0, _node))
            while queue:
                (level, node) = queue.pop(0)
                lookup[level].append(node)
                # at each level, nodes are always processed from left to right
                if node.left:
                    queue.append((level + 1, node.left))
                if node.right:
                    queue.append((level + 1, node.right))

        bfs(root)
        for key in lookup.keys():
            values = lookup[key]
            if len(values) > 1:
                i = 0
                j = 1
                while j < len(values):
                    values[i].next = values[j]
                    i += 1
                    j += 1
            lookup[key] = values

        return root


root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)
root.right.right = Node(7)
Solution().connect(root)
root.display()


# In[42]:


# A third solution to the above problem where we don't need to use a hashmap

"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        # Useful data structures for this problem:
        # Since we are always given a perfect binary tree,
        # we can use a hashmap and BFS (level order traversal) (left -> right)
        # Hashmap mapping levels to nodes. Nodes are automatically ordered from left to right at each level
        # in the hashmap

        #  0 -> 1
        #  1 -> [2,3]
        #  2 -> [4, 5, 6, 7]

        lookup = defaultdict(list)
        if not root:
            return root

        def bfs(_node: Node):
            queue = []
            queue.append((0, _node))
            while queue:
                level, node = queue.pop(0)
                if queue and queue[0][0] == level:
                    node.next = queue[0][1]
                # at each level, nodes are always processed from left to right
                if node.left:
                    queue.append((level + 1, node.left))
                if node.right:
                    queue.append((level + 1, node.right))

        bfs(root)
        return root
root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)
root.right.left = Node(6)
root.right.right = Node(7)
Solution().connect(root)
root.display()


# ### DFS Application: 1026. Maximum Difference Between Node and Ancestor
# [https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/](https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/)

# In[43]:


from typing import Optional
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:

    def dfs(self, root: TreeNode, v: int, max_a: int, min_a: int):

        if root is None:
            return v

        if root.left is not None:
            _v = max(v, abs(max_a - root.left.val), abs(min_a - root.left.val))
            _max_a = max(max_a, root.left.val)
            _min_a = min(min_a, root.left.val)
            max_left = self.dfs(root.left, _v, _max_a, _min_a)
        if root.right is not None:
            _v = max(v, abs(max_a - root.right.val), abs(min_a - root.right.val))
            _max_a = max(max_a, root.right.val)
            _min_a = min(min_a, root.right.val)
            max_right = self.dfs(root.right, _v, _max_a, _min_a)

        if root.left is not None and root.right is not None:
            return max(max_left, max_right)
        if root.left is not None:
            return max_left
        if root.right is not None:
            return max_right
        return v

    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        # Use BFS (level order traversal)
        # Greedy approach: at each level calculate max v
        v = 0
        max_a = root.val
        min_a = root.val
        return self.dfs(root, v, max_a, min_a)

#            1
#             \
#              2
#               \
#                0
#               /
#              3
root = TreeNode(1)
root.right = TreeNode(2)
root.right.right = TreeNode(0)
root.right.right.left = TreeNode(3)
# The max abs value difference between an ancestor and descendant is 3 (bottom two nodes)
assert Solution().maxAncestorDiff(root) == 3


# ### Max Path Sum in a binary tree
# Source: [https://www.geeksforgeeks.org/find-maximum-path-sum-in-a-binary-tree/](https://www.geeksforgeeks.org/find-maximum-path-sum-in-a-binary-tree/)
# 
# Example:
# ```
# Input: Root of below tree
#        1
#       / \
#      2   3
# Output: 6
# ```
# For each node there can be four ways that the max path goes through the node:
# 1. Node only
# 2. Max path through left child + Node
# 3. Node + Max path through right child
# 4. Max path through left child + Node + Max path through right child
# 
# Important: Each recursive call can only return a single path through at most one child node (max of 1,2, or 3).
# The instance variable max of the solution class holds the overall max value possible, which includes 4, where the path 'peaks' at the node and traverses both children nodes.
# 
# Time complexity: O(n) since we need to traverse over all nodes in the binary tree.
# Space complexity: O(1)
# 
# 

# In[44]:


class Node:
    def __init__(self, val: int):
        self.val = val
        self.left = None
        self.right = None

class Solution():
    def __init__(self):
        self.max_sum = float("-inf")

    def find_max_sum(self, root: Node) -> int:
        """
        Recursively retuns the max path sum with at most one child node.
        Updates class instance max sum incorporating max sums of paths through
        both children nodes and the current node.
        """

        if root is None:
            return 0

        max_sum_left = self.find_max_sum(root.left)
        max_sum_right = self.find_max_sum(root.right)

        max_sum_left_with_node = max_sum_left + root.val
        max_sum_right_with_node = root.val + max_sum_right

        max_single_child_path = max(root.val,
                                    max_sum_left_with_node,
                                    max_sum_right_with_node)

        max_all_children_and_node = max(max_single_child_path,
                                        max_sum_left + max_sum_right + root.val)

        self.max_sum = max(self.max_sum, max_all_children_and_node)

        return max_single_child_path

# ' denotes the maximum path sum
#             '10
#             /  \
#           '2   '10
#          /  \     \
#        '20   1     -25
#                   /   \
#                  3     4
root = Node(10)
root.left = Node(2)
root.right = Node(10)
root.left.left = Node(20)
root.left.right = Node(1)
root.right.right = Node(-25)
root.right.right.left = Node(3)
root.right.right.right = Node(4)
s = Solution()
s.find_max_sum(root)
print(f"Max path sum is {s.max_sum}")


# ### Check if a given array can represent preorder traversal of a binary search tree
# Given an array of numbers, return true if the given array can represent preorder traversal of a binary search tree. Otherwise, return false.
# The expected time complexity is O(n).
# 
# Input:  pre[] = {2, 4, 3}
# Output: true
# Given array can represent preorder traversal
# of below tree
# ```
#     2
#      \
#       4
#      /
#     3
# ```
# 
# Reminder of the definition of a binary search tree:
# The left subtree of a node contains only nodes with keys lesser than the node’s key.
# The right subtree of a node contains only nodes with keys greater than the node’s key.
# The left and right subtree each must also be a binary search tree.
# Source: [https://www.geeksforgeeks.org/binary-search-tree-data-structure/](https://www.geeksforgeeks.org/binary-search-tree-data-structure/)

# In[45]:


from typing import List
class Node:
    def __init__(self, val: int):
        self.val = val
        self.left = None
        self.right = None

def is_array_preorder_traversal(root: Node, arr: List[int]) -> bool:
    # TODO

    result = []
    def dfs(node: Node):
        if node is None:
            return

        result.append(node.val)
        dfs(node.left)
        dfs(node.right)

    dfs(root)
    return result == arr




root = Node(2)
root.right = Node(4)
root.right.left = Node(3)
assert is_array_preorder_traversal(root, [2,4,3]) == True
root = Node(40)
root.left = Node(30)
root.left.right = Node(35)
root.right = Node(80)
root.right.right = Node(100)
assert is_array_preorder_traversal(root, [40, 30, 35, 80, 100]) == True


# ### Validate a Binary Search Tree
# 
# Given the root of a binary tree, determine if it is a valid binary search tree (BST).
# 
# A valid BST is defined as follows:
# 
# The left subtree of a node contains only nodes with keys less than the node's key.
# The right subtree of a node contains only nodes with keys greater than the node's key.
# Both the left and right subtrees must also be binary search trees.
# 
# Source: [https://leetcode.com/problems/validate-binary-search-tree/solution/](https://leetcode.com/problems/validate-binary-search-tree/solution/)
# 
# 

# In[46]:


from typing import Optional

import math
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        # Approach

        # Use inorder traversal DFS
        # Left -> Node -> Right means that for inorder traversal
        # each node's value should be smaller than the next

        self.prev = -math.inf

        def dfs(node: TreeNode):
            if node is None:
                return True

            if not dfs(node.left):
                return False

            if self.prev >= node.val:
                return False

            self.prev = node.val

            if not dfs(node.right):
                return False

            return True

        return dfs(root)

#               4
#             /  \
#            2    5
#          /  \     \
#         1    3     7
#                   /  \
#                  6    8
root = Node(4)
root.left = Node(2)
root.right = Node(5)
root.left.left = Node(1)
root.left.right = Node(3)
root.right.right = Node(7)
root.right.right.left = Node(6)
root.right.right.right = Node(8)
assert Solution().isValidBST(root) == True

#               4
#             /  \
#            2    5
#          /  \     \
#         1    3     7
#                   /  \
#                 <5>   8
root = Node(4)
root.left = Node(2)
root.right = Node(5)
root.left.left = Node(1)
root.left.right = Node(3)
root.right.right = Node(7)
root.right.right.left = Node(5)
root.right.right.right = Node(8)
assert Solution().isValidBST(root) == False  # since the <5> is in the right subtree of an ancestor with value 5


# ### Check if a binary tree is a subtree of another binary tree
# 
# Source: [https://leetcode.com/problems/subtree-of-another-tree/](https://leetcode.com/problems/subtree-of-another-tree/)
# 
# Example: Tree 1 is a subtree of Tree 2
# 
# ```
#         Tree1
#           x
#         /    \
#       a       b
#        \
#         c
# 
# 
#         Tree2
#               z
#             /   \
#           x      e
#         /    \     \
#       a       b      k
#        \
#         c
# ```

# In[47]:


from typing import Optional, List
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:

        # Approach (DFS)
        # Time complexity: O(n) to visit all nodes
        # Space complexity: O(n) to store four arrays of all inorder and preorder values
        # For each tree, get array of values for each node while traversing:
        #   1. preorder
        #   2. inorder
        # If the preorder & inorder vals for the candidate subtree are subarrays
        # of the preorder and inorder vals for the tree, return True, otherwise return False
        # Note: if a node is none, append special character to preorder and inorder value arrays
        #       to prevent a match when the subtree has children in the larger tree

        root_vals_inorder = []
        root_vals_preorder = []

        subroot_vals_inorder = []
        subroot_vals_preorder = []


        def dfs(node: TreeNode, vals_preorder: List[int], vals_inorder: List[int]):
            if node is None:
                # prevent match when subtree has children in the larger tree
                vals_preorder.append('#')
                vals_inorder.append('#')
                return

            vals_preorder.append(node.val)
            dfs(node.left, vals_preorder, vals_inorder)
            vals_inorder.append(node.val)
            dfs(node.right, vals_preorder, vals_inorder)


        dfs(root, root_vals_preorder, root_vals_inorder)
        dfs(subRoot, subroot_vals_preorder, subroot_vals_inorder)

        root_vals_preorder = "".join(str(each) for each in root_vals_preorder)
        root_vals_inorder = "".join(str(each) for each in root_vals_inorder)
        subroot_vals_preorder = "".join(str(each) for each in subroot_vals_preorder)
        subroot_vals_inorder = "".join(str(each) for each in subroot_vals_inorder)

        return (subroot_vals_inorder in root_vals_inorder and
                subroot_vals_preorder in root_vals_preorder)

root = TreeNode(3)
root.left = TreeNode(4)
root.right = TreeNode(5)
root.left.left = TreeNode(1)
root.left.right = TreeNode(2)

subroot = TreeNode(4)
subroot.left = TreeNode(1)
subroot.right = TreeNode(2)
assert Solution().isSubtree(root, subroot) == True

root.left.right.left = TreeNode(0)
assert Solution().isSubtree(root, subroot) == False


# ### Check whether a binary tree is a full binary tree or not
# Source: [https://www.geeksforgeeks.org/check-whether-binary-tree-full-binary-tree-not/](https://www.geeksforgeeks.org/check-whether-binary-tree-full-binary-tree-not/)
# 
# A full binary tree is defined as a binary tree in which all nodes have either zero or two child nodes.
# Time Complexity: O(n) to visit all nodes
# Space complexity: O(n) recursive call stack can include potentially all nodes

# In[48]:


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:

    def is_full_binary_tree(self, root: TreeNode) -> bool:
        def dfs(root: TreeNode):
            if root is None:
                return True

            dfs(root.left)
            dfs(root.right)

            if root.left is None and root.right is not None:
                return False
            if root.left is not None and root.right is None:
                return False

            return True
        return dfs(root)

root = TreeNode(1)
root.left = TreeNode(2)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right = TreeNode(3)

Solution().is_full_binary_tree(root)


# ### Remove nodes on root to leaf paths of length < K
# Source: [https://www.geeksforgeeks.org/remove-nodes-root-leaf-paths-length-k/](https://www.geeksforgeeks.org/remove-nodes-root-leaf-paths-length-k/)
# 
# Example:
# 
# ```
# Input: Root of Binary Tree, k = 4
#                1
#            /      \
#          2          3
#       /     \         \
#     4         5        6
#   /                   /
#  7                   8
# 
# Output: The tree should be changed to following
#            1
#         /     \
#       2          3
#      /             \
#    4                 6
#  /                  /
# 7                  8
# ```

# In[49]:


class TreeNode:
    def __init__(self, val, left: TreeNode = None, right: TreeNode = None):
        self.val = val
        self.left = left
        self.right = right

    def dfs_display(self, node: TreeNode):
        # preorder display
        print(node.val)
        if node.left:
            self.dfs_display(node.left)
        if node.right:
            self.dfs_display(node.right)

    def bfs_display(self, node: TreeNode):

        queue = []
        queue.append(node)

        while queue:
            n = queue.pop(0)
            print(n.val)

            if n.left:
                queue.append(n.left)
            if n.right:
                queue.append(n.right)

class Solution:

    def remove_paths_less_than_k(self, root: TreeNode, k: int) -> TreeNode:

        # Use DFS to increase depth counter as we traverse downward
        # If we get to where depth > k and we are at a leaf node, add the
        # resultant path to the list of paths to delete

        self.paths_to_delete = []
        self.paths_to_keep = []

        nodes_to_remove = set()

        def dfs(node: TreeNode, prev: List[TreeNode], depth: int):
            if node is None:
                return

            depth += 1

            if depth < k and node.left is None and node.right is None:
                # mark path so that it can be deleted
                self.paths_to_delete.append(prev + [node])
                return

            if depth >= k:
                self.paths_to_keep.append(prev + [node])


            dfs(node.left, prev + [node], depth)
            dfs(node.right, prev + [node], depth)

        prev = []
        dfs(node=root, prev=prev, depth=0)
        print("paths to delete:")
        for paths in self.paths_to_delete:
            print([p.val for p in paths])
        print("paths to keep:")
        for paths in self.paths_to_keep:
            print([p.val for p in paths])

        nodes_to_remove = set()
        for path in self.paths_to_delete:
            for node in path:
                nodes_to_remove.add(node)
        for path in self.paths_to_keep:
            for node in path:
                if node in nodes_to_remove:
                    nodes_to_remove.remove(node)

        def dfs_remove_nodes(node: TreeNode):
            if node is None:
                return

            if node.left in nodes_to_remove:
                node.left = None
            else:
                dfs_remove_nodes(node.left)
            if node.right in nodes_to_remove:
                node.right = None
            else:
                dfs_remove_nodes(node.right)
        dfs_remove_nodes(root)
        return root

root = TreeNode(1)
root.left = TreeNode(2)
root.left.right = TreeNode(5)
root.left.left = TreeNode(4)
root.left.left.left = TreeNode(7)

root.right = TreeNode(3)
root.right.right = TreeNode(6)
root.right.right.left = TreeNode(8)

head = Solution().remove_paths_less_than_k(root, 4)
print("DFS preorder list after deletion")
head.dfs_display(head)
print("\nBFS level ordering of nodes after deletion")
head.bfs_display(head)


# In[50]:


class TreeNode:
    def __init__(self, val, left: TreeNode = None, right: TreeNode = None):
        self.val = val
        self.left = left
        self.right = right

    def dfs_display(self, node: TreeNode):
        # preorder display
        print(node.val)
        if node.left:
            self.dfs_display(node.left)
        if node.right:
            self.dfs_display(node.right)

    def bfs_display(self, node: TreeNode):

        queue = []
        queue.append(node)

        while queue:
            n = queue.pop(0)
            print(n.val)

            if n.left:
                queue.append(n.left)
            if n.right:
                queue.append(n.right)

class Solution:

    def remove_paths_less_than_k(self, root: TreeNode, k: int) -> TreeNode:
        # Traverse the tree in postorder fashion
        # so that if a leaf node path length is
        # shorter than k, then that node and all
        # of its descendants till the node which
        # are not on some other path are removed.

        def dfs(node: TreeNode, height: int):
            if node is None:
                return

            node.left = dfs(node.left, height + 1)
            node.right = dfs(node.right, height + 1)

            if height < k and node.left is None and node.right is None:
                # assign the node's value to None so it is deleted
                # This will propogate up until it hits a path where
                # not all nodes should be deleted
                return None

            return node

        return dfs(node=root, height=1)

root = TreeNode(1)
root.left = TreeNode(2)
root.left.right = TreeNode(5)
root.left.left = TreeNode(4)
root.left.left.left = TreeNode(7)

root.right = TreeNode(3)
root.right.right = TreeNode(6)
root.right.right.left = TreeNode(8)

result = Solution().remove_paths_less_than_k(root, 4)
print("DFS display of tree with paths shorter than 4 removed:")
result.dfs_display(result)
print()
print("BFS display of tree with paths shorter than 4 removed:")
result.bfs_display(result)


# ### Lowest common ancestor in Binary Seach Tree
# 
# Given values of two values n1 and n2 in a Binary Search Tree, find the Lowest Common Ancestor (LCA).
# You may assume that both the values exist in the tree.
# 
# Example:
# 
# ```
#                      20
#                   /      \
#                  8        22
#                 /  \
#                4    12
#                    /  \
#                   10  14
# ```
# Input: LCA of 10 and 14
# Output:  12
# Explanation: 12 is the closest node to both 10 and 14
# which is a ancestor of both the nodes.

# In[51]:


class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # Use DFS postorder traversal to decide whether p and q
        # are in the left and right subtree of the current node.
        # the first time we find them both in the left and right
        # subtrees, we return that node, as this will be the lowest
        # common ancestor of p and q.
        # We will need to return whether p or q is the current node
        # or whether p or q are in the left or right subtrees
        # as soon as we find p and q
        # Time complexity: O(V)
        # Space complexity: O(h) which is the maximum height of the tree
        # This is because we store the height of the tree
        # in the recursive call stack
        # If the tree is a linked list, this would be O(V)

        self.lca = None

        def dfs(node: TreeNode):
            if node is None:
                return False, False

            has_p = False
            has_q = False

            left_subtree_has_p, left_subtree_has_q = dfs(node.left)
            right_subtree_has_p, right_subtree_has_q = dfs(node.right)

            if node.val == p.val:
                has_p = True
            if node.val == q.val:
                has_q = True

            has_p = left_subtree_has_p or right_subtree_has_p or has_p
            has_q = left_subtree_has_q or right_subtree_has_q or has_q

            if has_p and has_q and self.lca is None:
                self.lca = node

            return has_p, has_q

        dfs(root)
        return self.lca
"""
               1
           /      \
         2          3
      /     \         \
    4         5        6
  /                   /
 7                   8
"""

root = TreeNode(1)
root.left = TreeNode(2)
root.left.right = TreeNode(5)
root.left.left = TreeNode(4)
root.left.left.left = TreeNode(7)

root.right = TreeNode(3)
root.right.right = TreeNode(6)
root.right.right.left = TreeNode(8)
assert Solution().lowestCommonAncestor(root, root.left.left, root.left.right) == root.left


# In[52]:


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # Use DFS postorder, dfs has two return types, has_p, has_q 
        # as soon as both are true, return the current node
                
        self.lca = None     
        def dfs(node: TreeNode, has_p, has_q) -> Tuple[bool, bool]:
            if node is None:
                return has_p, has_q
            
                
            left_has_p, left_has_q = dfs(node.left, has_p, has_q)
            right_has_p, right_has_q = dfs(node.right, has_p, has_q)
            
            if node.val == p.val:
                has_p |= 1
            if node.val == q.val:
                has_q |= 1
                
            has_p |= left_has_p | right_has_p
            has_q |= left_has_q | right_has_q
            
            if has_p == 1 and has_q == 1 and self.lca is None:
                self.lca = node
            
            return has_p, has_q
            
        dfs(root, 0, 0)
        return self.lca


# ### Reverse alternate levels of a perfect binary tree
# 
# Given a Perfect Binary Tree, reverse the alternate level nodes of the binary tree.

# In[53]:


from collections import deque
from typing import Optional, List, Deque


class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    @staticmethod
    def dfs_display(node: TreeNode):
        if node is None:
            return

        TreeNode.dfs_display(node.left)
        print(node.val, end=" ")
        TreeNode.dfs_display(node.right)

    @staticmethod
    def dfs_get_order(node: TreeNode, result: List):
        if node is None:
            return

        TreeNode.dfs_get_order(node.left, result)
        result.append(node.val)
        TreeNode.dfs_get_order(node.right, result)

class Solution:
    def reverse_alternate_levels(self, root: TreeNode) -> TreeNode:
        # Approach: Use DFS to traverse the tree, adding each
        # node that is in an odd row to the an array
        #         Given tree:
        #                a
        #             /     \
        #            b       c
        #          /  \     /  \
        #         d    e    f    g
        #        / \  / \  / \  / \
        #        h  i j  k l  m  n  o
        #
        # Inorder Traversal of odd rows : {h, i, b, j, k, l, m, c, n, o}

        # Note how with inorder traversal you visit the nodes in order from left to right (e.g. left subtree -> node -> right subtree)
        # so as long as we keep track of the level we're on we can easily reverse the order of nodes in a per-level basis simply by
        # traversing with DFS inorder twice. The second traversal, we'll pop the last element added and replace the first visited with it,
        # repeatedly until we finish the inorder traversal.

        # Modified tree:
        #                a
        #             /     \
        #            c       b
        #          /  \     /  \
        #         d    e    f    g
        #        / \  / \  / \  / \
        #       o  n m  l k  j  i  h


        def dfs(node: TreeNode, level: int, odd_level_nodes: List[TreeNode]):
            if node is None:
                return

            dfs(node.left, level + 1, odd_level_nodes)

            # inorder traversal
            if level % 2 != 0:
                # odd level
                odd_level_nodes.append(node.val)

            dfs(node.right, level + 1, odd_level_nodes)

        def dfs_replace(node: TreeNode, level, odd_level_nodes: Deque[TreeNode]):
            if node is None:
                return

            dfs_replace(node.left, level + 1, odd_level_nodes)

            # replace...
            if level % 2 != 0:
                node.val = odd_level_nodes.popleft()

            dfs_replace(node.right, level + 1, odd_level_nodes)

        odd_nodes = deque()
        dfs(root, 0, odd_nodes)
        odd_nodes.reverse()
        dfs_replace(root, 0, odd_nodes)
        return root

#         Given tree:
#                a
#             /     \
#            b       c
#          /  \     /  \
#         d    e    f    g
#        / \  / \  / \  / \
#        h  i j  k l  m  n  o
root = TreeNode('a')
root.left = TreeNode('b')
root.right = TreeNode('c')
root.left.left = TreeNode('d')
root.left.right = TreeNode('e')
root.right.left = TreeNode('f')
root.right.right = TreeNode('g')
root.left.left.left = TreeNode('h')
root.left.left.right = TreeNode('i')
root.left.right.left = TreeNode('j')
root.left.right.right = TreeNode('k')
root.right.left.left = TreeNode('l')
root.right.left.right = TreeNode('m')
root.right.right.left = TreeNode('n')
root.right.right.right = TreeNode('o')
print("Inorder Traversal of original tree")
TreeNode.dfs_display(root)
Solution().reverse_alternate_levels(root)
print("\nInorder Traversal of modified tree")
TreeNode.dfs_display(root)
result = []
TreeNode.dfs_get_order(root, result)
assert result == list("odncmelakfjbigh")
# Modified tree:
#                a
#             /     \
#            c       b
#          /  \     /  \
#         d    e    f    g
#        / \  / \  / \  / \
#       o  n m  l k  j  i  h

