from typing import Type


class TreeNode:

    def __init__(self, val: int, left: "TreeNode", right: "TreeNode"):
        self.val = val
        self.left = left
        self.right = right


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
            raise Exception("The stack is empty")

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