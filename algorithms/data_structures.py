from typing import List, Type


# Time complexity: O(V + E)
#                     A - C
#                    / \ /
#                   B   F
#                 / \  /
#                D   E
# this is a directed graph
example_adjacency_list_directed_graph = {
  'A': ['B', 'F', 'C'],
  'B': ['D', 'E'],
  'C': ['F'],
  'D': [],
  'E': ['F'],
  'F': []
}


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
    Useful for DFS
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


class Trie:
    """
    Particularly efficient for tasks that involve searching for words or prefixes
    Useful for
      * autocomplete
      * spell-checking
      * IP routing tables
    where fast prefix based searches are important

    The space complexity for a tree is O(n*k), where n is the number of words
    in the trie and k is the average length of the words.
    """
    def __init__(self):
        self.child = {}

    def insert(self, word: str) -> None:
        """
        Time Complexity: O(k), where k is the length of the word to be inserted.
        Iterate over each character in the word.
        Ensure each character in the word is the sub key of a new dict
        The last character of the word maps to a dict with 'end' as at least one of the keys.
        """
        current = self.child
        for c in word:
            if c not in current:
                current[c] = {}
            current = current[c]
        current['end'] = 1

    def search(self, word: str) -> bool:
        """
        Time Complexity: O(k), where k is the length of the word
        Check that each character is in the trie and that the last character maps to a dict with 'end' as a key.
        """
        current = self.child
        for c in word:
            if c not in current:
                return False
            current = current[c]
        return 'end' in current

    def starts_with(self, prefix: str) -> bool:
        """
        Time Complexity: O(k) where k is the length of the prefix.
        Just needs to traverse the trie following the characters in the prefix
        Iterates over each character in the prefix.
        Does not check if the last character maps to a dict with 'end' as the key.
        """
        current = self.child
        for c in prefix:
            if c not in current:
                return False
            current = current[c]
        return True