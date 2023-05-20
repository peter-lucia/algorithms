from typing import List, Type


class TreeNode:

    def __init__(self, 
                 val: int, 
                 left: "TreeNode" = None, 
                 right: "TreeNode" = None,
                 path: List[int] = []):
        """
        A binary tree node

        Args:
            val (int): Value for this node
            left (TreeNode, optional): Left TreeNode, if exists. Defaults to None.
            right (TreeNode, optional): Right TreeNode. Defaults to None.
            path (List[int], optional): Path from root to this node. Defaults to [].
        """
        self.val = val
        self.left = left
        self.right = right # Right node
        self.path = path # Path from root up to, not including this node


class Stack:
    """
    Simple stack implementation
    where the top of the stack is the end of the list
    """
    def __init__(self):
        self.arr = []

    def push(self, item):
        self.arr.append(item)

    def pop(self):
        """
        Remove the top element of the stack
        """
        if len(self.arr) > 0:
            # The pop() call defaults to popping the last element of the list
            return self.arr.pop()
        else:
            raise ValueError("The stack is empty")

    def peek(self):
        """
        Read the top element of the stack
        """
        if len(self.arr) > 0:
            return self.arr[-1]
        else:
            return None

    def is_empty(self):
        return bool(self.arr)

    def size(self):
        """
        Get the size of the stack
        """

        return len(self.arr)