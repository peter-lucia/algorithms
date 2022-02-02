#!/usr/bin/env python
# coding: utf-8

# # String / Array

# ### Get max profit

# In[1]:


def get_max_profit(stock_prices):

    if len(stock_prices) < 2:
        raise ValueError("Need more prices")

    # Iterate through all the numbers once
    # store the min_price found and the difference
    # between the min_price and current price

    min_price = float("inf")
    max_profit = float("-inf")
    for current_price in stock_prices:
        if max_profit < current_price - min_price:
            max_profit = current_price - min_price
        if current_price < min_price:
            min_price = current_price


    return max_profit
arr = [10, 12, 17, 3, 9]
print(f"Max profit is {get_max_profit(arr)}")


# ### 735. Asteroid Collision
# https://leetcode.com/problems/asteroid-collision/submissions/
# 
# * first failed approach with stacks: two stacks, asteroids and result
# * pop asteroid from asteroids
# * put asteroid into result
# * check if top item of result and top item of asteroids conflict
# * put winner into result or remove both if they cancel out
# * continue until asteroids is empty and result is full
# 
# 
# ```
# asteroids = [10, 2, -5]
# result = []
# 
# asteroids = [-5]
# result = [10, 2]
# 
# right = -5  # asteroids[0]
# left = 2  # result[-1]
# 
# if right < 0 and left > 0:
#    handle collision
# otherwise:
#    put right into top of result
# ```
# 
# 
# second approach
# * start with i = 0, j = 1
# * check for collision between i and j
# * if no collision, increment both i and j
# * if collision, if elem at j wins, pop(i) and let i = j and j += 1
# * if collision and if elem at i wins, pop(j) and keep i and j the same
# * if collision and cancel each other out, pop(i), pop(j), let i -= 1, j = i
# 

# In[2]:


from typing import List
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:


        # can assume len(asteroids) >= 2
        i = 0
        j = 1

        while j < len(asteroids):
            if self.is_collision(asteroids[i], asteroids[j]):
                if abs(asteroids[i]) > abs(asteroids[j]):
                    asteroids.pop(j)
                elif abs(asteroids[i]) < abs(asteroids[j]):
                    asteroids.pop(i)
                    if i > 0:
                        i -= 1
                        j -= 1
                elif asteroids[j] < 0 and abs(asteroids[j]) == abs(asteroids[j]):
                    asteroids.pop(i)
                    asteroids.pop(i) # elem at j is now at i
                    if i > 1:
                        i -= 2
                        j -= 2
                    elif i > 0:
                        i -= 1
                        j -= 1
            else:
                i += 1
                j += 1
        return asteroids


    def is_collision(self, left, right):
        return right < 0 and left > 0


# In[3]:


assert [] == Solution().asteroidCollision([8, -8])
assert [10] == Solution().asteroidCollision([10, 2, -5])



# ### 20. Valid Parentheses
# https://leetcode.com/problems/valid-parentheses/

# In[4]:


class Solution:
    def isValid(self, s: str) -> bool:
        # approach 1: anytime we see {}, (), or [], remove it from the expression

        # Append each character in s to the stack
        # if the stack is not empty, and the last character added to the stack
        # is the closing bracket of the same time, don't append the new character
        # to the stack and remove the left bracket from the stack to effectively
        # remove the {}, (), or [] from the string
        # repeat this process until we've gone through all characters in s.
        # if at the end the stack is not empty, return False
        # otherwise, return True

        stack = []

        for c in s:
            if stack:
                if c == '}' and stack[-1] == '{':
                    stack.pop(-1)
                elif c == ')' and stack[-1] == '(':
                    stack.pop(-1)
                elif c == ']' and stack[-1] == '[':
                    stack.pop(-1)
                else:
                    stack.append(c)
            else:
                stack.append(c)
        if stack:
            return False
        else:
            return True


assert Solution().isValid("()") == True
assert Solution().isValid("()[]{}") == True
assert Solution().isValid("(]") == False
assert Solution().isValid("([)]") == False
assert Solution().isValid("{[]}") == True


# ### Rotate array
# 
# [https://leetcode.com/problems/rotate-array/](https://leetcode.com/problems/rotate-array/)

# In[5]:


from typing import List
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Time complexity: O(n)
        Space complexity: O(n)
        Do not return anything, modify nums in-place instead.
        """

        # approach #1
        # create a new array of size n = len(nums)
        # from i = 0 -> n-1
        # put nums[i] at (i + k) % n in the new array
        # then copy the elements back into
        n = len(nums)
        new_arr = [0 for _ in range(n)]
        for i in range(n):
            new_pos = (i + k) % n
            new_arr[new_pos] = nums[i]

        # copy the elements back into nums
        nums[:] = new_arr[:]

nums = [1,2,3,4,5,6,7]
Solution().rotate(nums, 4)
nums


# In[6]:


nums = [1,2,3,4,5]
n = len(nums)
n


# In[7]:


(0 + 4) % n


# In[8]:


(1+4) % n


# In[9]:


nums[(0+4) % n]


# In[10]:


nums[(1+4) % n]


# In[11]:


from typing import List
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        Time complexity: O(n)
        Space complexity: O(1)
        Approach: (always write out approach with examples first to verify)
        nums = 1, 2, 3, 4, 5, 6, 7
        k = 4
        tgt_index = 0 + 4

        tmp = 5
        nums = 1, 2, 3, 4, 1, 6, 7
        tgt_index = (4 + 4) % (n - 1) = 1

        tmp = 2
        nums = 1, 5, 3, 4, 1, 6, 7
        tg_index = (1 + 4) % n = 5

        tmp = 6
        nums = 1, 5, 3, 4, 1, 2, 7
        """
        n = len(nums)

        start_idx = 0
        i = 0
        while i < n:
            current_idx = start_idx
            tmp = nums[current_idx]
            while True:
                next_idx = (current_idx + k) % n
                nums[next_idx], tmp = tmp, nums[next_idx]
                current_idx = next_idx
                i += 1
                if current_idx == start_idx:
                    break
            start_idx += 1
nums = [1,2,3,4,5,6,7]
Solution().rotate(nums, 4)
nums


# ### Text Justification
# * [https://leetcode.com/problems/text-justification/](https://leetcode.com/problems/text-justification/)
# 
# Approach: Until we have added all words:
# 1. Get i:i+k words that can fit on a line with maxWidth
# 2. Insert the spaces between those words
# 3. Add the line to the result

# In[12]:


from typing import List
class Solution:
     def get_k_words(self, i: int, words: List[str], maxWidth: int) -> int:
        """
        Gets the number of words starting at pos i in the words list that
        can fit within maxWidth characters with at least one space between them
        """
        k = 0
        n = len(words)
        line = ' '.join(words[i:i+k])

        while len(line) <= maxWidth and i+k <= n:
            k += 1
            line = ' '.join(words[i:i+k])
        k -= 1
        return k

     def insert_spaces(self, i: int, words: List[str], maxWidth: int, k: int) -> str:
        """
        Returns a line with extra spaces added between the words to fit maxWidth
        """

        line = ' '.join(words[i: i+k])
        n = len(words)

        # if the line contains only one word or we've reached the last word,
        # put spaces to the right (i.e. left justify the word)
        if k == 1 or i+k == n:
            spaces = maxWidth - len(line)
            line = line + " " * spaces
        else:
            # Example: 'word1    word2' k = 2, with k-1=1 space between the words
            spaces = k-1
            total_space_chars = maxWidth - len(line) + spaces

            # extra spaces between words should be distributed as evenly as possible
            avg_spaces_between_words = total_space_chars // spaces

            # if the number of spaces on a line does not divide evenly between words,
            # the empty slots on the left will be assigned more spaces than those
            # on the right
            num_words_with_extra_space = total_space_chars % spaces
            if num_words_with_extra_space > 0:
                line = (" " * (avg_spaces_between_words + 1)).join(words[i:i+num_words_with_extra_space])
                line += " " * (avg_spaces_between_words + 1)
                line += (" " * (avg_spaces_between_words)).join(words[i+num_words_with_extra_space:i+k])
            else:
                line = (" " * avg_spaces_between_words).join(words[i:i+k])
        return line

     def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        # approach
        # Until we have added all words:
        # 1. Get i:i+k words that can fit on a line with maxWidth
        # 2. Insert the spaces between those words
        # 3. Add the line to the result

        i = 0
        n = len(words)
        result = []

        while i < n:
            k = self.get_k_words(i, words, maxWidth)
            line = self.insert_spaces(i, words, maxWidth, k)
            result.append(line)
            i += k

        return result


Solution().fullJustify(["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"], maxWidth = 20)


# In[13]:


Solution().fullJustify(["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16)


# In[14]:


Solution().fullJustify(["What","must","be","acknowledgment","shall","be"], maxWidth = 16)


# ### Reverse a string without affecting special characters
# [https://www.geeksforgeeks.org/reverse-a-string-without-affecting-special-characters/](https://www.geeksforgeeks.org/reverse-a-string-without-affecting-special-characters/)

# In[15]:



def reverse_string(my_string: str) -> str:
    s = [c for c in my_string]
    if len(s) <=1:
        return s

    i = 0
    j = len(s) - 1

    while i < j:
        if not s[i].isalpha():
            i += 1
        elif not s[j].isalpha():
            j -= 1
        else:
            s[i], s[j] = s[j], s[i]
            i += 1
            j -= 1
    return "".join(s)

assert reverse_string("a!!!b.c.d,e'f,ghi") == "i!!!h.g.f,e'd,cba"
assert reverse_string("l@@me!be*your@@hero%") == "o@@re!hr*uoye@@beml%"


# ### Print all possible palindromic partitions of a string
# [https://www.geeksforgeeks.org/given-a-string-print-all-possible-palindromic-partition/](https://www.geeksforgeeks.org/given-a-string-print-all-possible-palindromic-partition/)
# 
# [https://leetcode.com/problems/palindrome-partitioning/](https://leetcode.com/problems/palindrome-partitioning/)
# 
# ```
# Example 1:
# 
# Input: s = "aab"
# Output: [["a","a","b"],["aa","b"]]
# Example 2:
# 
# Input: s = "a"
# Output: [["a"]]
# ```

# In[16]:


class Solution:

    def is_palindrome(self, s: str) -> bool:

        i = 0
        j = len(s) - 1

        while i < j:
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True

    def partition(self, s: str) -> List[List[str]]:
        # Incrementally build candidates for the solution and discard candidates (backtrack) if
        # they don't satisfy the condition
        # DFS on the following tree:
        # s = abc
#i=1    #    check [a]bc                                                     # call stack 1
        #    [a] is a palindrome. check bc, path = ['a']
        # s = bc
        #    check [b]c                                                      # call stack 2
#i=1    #    [b] is a palindrome. check c, path = ['a', 'b']
        # s = c
        #    check [c]
        #    [c] is a palindrome. check '', path = ['a', 'b', 'c']
        # s = ''
        #    append ['a', 'b', 'c'] to result
        # s = bc
        #    check [bc]                                                      # call stack 2
#i=1    #    [bc] is not a palindrome
        # s = abc
#i=2    #    check [ab]c
        #    [ab] is not a palindrome
        #    check [abc]
        #    [abc] is not a palindrome

        def dfs(s, path, result):
            if not s:
                # finished recursing through tree of possible substrings
                result.append(path)
                return

            # generate substrings of all sizes in s
            for i in range(1, len(s)+1):
                print(s[:i])
                # if we find a palindrome
                # add it to the current dfs tree path
                # pass through the result
                if self.is_palindrome(s[:i]):
                    dfs(s[i:], path + [s[:i]], result)

        result = []
        dfs(s, [], result)
        return result

Solution().partition("abc")


# In[17]:


Solution().partition("aab")


# ### Find original array from doubled array
# 
# An integer array original is transformed into a doubled array changed by appending twice the value of every element in original, and then randomly shuffling the resulting array.
# 
# Given an array changed, return original if changed is a doubled array. If changed is not a doubled array, return an empty array. The elements in original may be returned in any order.
# 
# Input: changed = [1,3,4,2,6,8]
# Output: [1,3,4]
# 
# Input: changed = [6,3,0,1]
# Output: []
# 
# Input: changed = [1]
# Output: []

# In[18]:


from typing import List
from collections import defaultdict
class Solution:
    def findOriginalArray(self, changed: List[int]) -> List[int]:
        # Key points:
        # 1. if len(changed) is not even, return []

        # approach 2
        # O(nlogn) time complexity at least
        # O(n) space complexity since we just have a hash table of all elements and their num occurrences

        # Algorithm
        # 1. put count of each element into a hashmap
        # 2. iterate over a sorted list of the keys in the hashmap
        # 3. for each key in the sorted list of keys
        #    - add special handling for 0
        #    - if 2*key is another key in the map and it occurs > 0 times, decrement occurrences of key and 2*key
        #    - add key to the result
        #    - once the # of doubles reaches n / 2, return the result

        lookup = defaultdict(int)

        for each in changed:
            lookup[each] += 1

        result = []
        keys = sorted(lookup.keys())
        i = 0
        while i < len(keys):
            each = keys[i]
            if each == 0:
                num_zeros_added = lookup[0] // 2
                result += [0 for _ in range(num_zeros_added)]
                lookup[0] -= num_zeros_added
            elif 2*each in lookup and lookup[2*each] > 0 and lookup[each] > 0:
                num_subtract = min(lookup[2*each], lookup[each])
                lookup[2*each] -= num_subtract
                lookup[each] -= num_subtract
                result += [each for _ in range(num_subtract)]
            i += 1

            if len(result) == len(changed) / 2:
                return result
        return []


assert Solution().findOriginalArray([1,3,4,2,6,8]) == [1,3,4]
assert Solution().findOriginalArray([6,3,0,1]) == []
assert Solution().findOriginalArray([1]) == []


# Better approach to find the original array problem

# In[19]:


# Key points:
# 1. if len(changed) is not even, return []

# approach 2
# add 2 times every element to a double ended queue
# iterate through each element of sorted
# sort the array O(nlogn)
# Use a double ended queue
# if each element is in the queue, pop the queue

# queue: []
# arr = [1, 2, 2, 4]  # must be sorted
# result = []

# 1
# queue: [2]
# result = [1]

# 2
# queue: []
# result: [1]

# 2
# queue: [4]
# result: [1, 2]

# 4
# queue: [0]
# return result

# Example
# [1,3,4,2,6,8]
# []


# ### 833. Find and Replace String
# [https://leetcode.com/problems/find-and-replace-in-string/](https://leetcode.com/problems/find-and-replace-in-string/)

# In[20]:


from typing import List
from collections import defaultdict
class Solution:
    def findReplaceString(self, s: str, indices: List[int], sources: List[str], targets: List[str]) -> str:
        # Time complexity: O(nk) where n is the number of sources and k is the max size of a string
        # Space complexity: O(nk) to house potentially all source and replacement texts

        # Key ideas

        # 1. Create a lookup of mapping an index to a source that was found in s with its replacement text
        # e.g. for s = "abcd", indices = [0,2], sources = ["ab", "cd"], targets = ["eeee", "bbbb"]
        # lookup = {
        #   starting_idx: [source text, replacement text]
        #     0: ["ab", "eeee"]
        #     2: ["cd", "bbbb"]
        # }

        # 2. Replace the found sources with their targets
        #    If the current index is not in the lookup, just add the character in s at index to the result
        #    If the current index is in the lookup, replace its source and increment index by len(source)

        letters = defaultdict(str)
        for idx, each in enumerate(s):
            letters[idx] = each

        lookup = {}
        for i, source in enumerate(sources):
            if source == s[indices[i]:indices[i] + len(source)]:
                lookup[indices[i]] = [source, targets[i]]

        result = ""
        idx = 0
        while idx < len(s) or idx in lookup:
            if idx in lookup:
                result += lookup[idx][1]
                idx += len(lookup[idx][0])
            else:
                result += s[idx]
                idx += 1
        return result

Solution().findReplaceString(s = "abcd", indices = [0, 2], sources = ["ab", "cd"], targets = ["eeee", "bbbb"])


# ### Shuffle an Array
# [https://leetcode.com/problems/shuffle-an-array/](https://leetcode.com/problems/shuffle-an-array/)

# In[21]:


import random
import copy
class Solution:

    def __init__(self, nums: List[int]):
        self.original = copy.deepcopy(nums)
        self.nums = nums

    def reset(self) -> List[int]:
        return self.original

    def shuffle(self) -> List[int]:
        # https://en.wikipedia.org/wiki/Fisher–Yates_shuffle
        # Time complexity: O(n)
        # Space complexity: O(n)
        for i in range(len(self.nums)):
            j = random.randint(0, len(self.nums)-1)
            self.nums[i], self.nums[j] = self.nums[j], self.nums[i]

        return self.nums


# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.reset()
# param_2 = obj.shuffle()


# ### 53. Maximum Subarray
# 
# Source: [https://leetcode.com/problems/maximum-subarray/](https://leetcode.com/problems/maximum-subarray/)
# 
# Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
# 
# A subarray is a contiguous part of an array.
# 
# 
# ```
# Example 1:
# 
# Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
# Output: 6
# Explanation: [4,-1,2,1] has the largest sum = 6.
# ```

# In[22]:


from typing import List
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:

        # Brute force solution O(n^2)
        # Need to consider all possible contiguous subsequences
        # i: 0 -> n-1
        # j: 0 -> i-1
        # max_sum = max(max_sum, nums[j:i])

        # Greedy approach O(n)
        # base case:
        # empty array dp[0] = max(nums[0], 0)

        # for i = 1 -> n-1:
        #    max_sum = max(dp[i-1] + nums[i], nums[i])

        # nums = [-2, 1, 7, 2, 9]
        # dp   = [0,  0, 0, 0, 0]
        #        [0,  1, 8, 10, 19]

        # nums = [-2, 1, 7, -3, 1, 1]
        # dp   = [ 0, 0, 0,  0, 0, 0]
        #      = [ 0, 0, 0,  0, 0, 0] max(-2, 0) = 0
        #      = [ 0, 1, 0,  0, 0, 0] max(0, 1) = 1
        #      = [ 0, 1, 8,  0, 0, 0] max(1+7, 7) = 8
        #      = [ 0, 1, 8,  5, 0, 0] max(5, -3) = 5
        #      = [ 0, 1, 8,  5, 6, 0] max(6, 1) = 6
        #      = [ 0, 1, 8,  5, 6, 7]  max(7, 1) = 7

        # Time complexity: O(n)
        # Space complexity: O(n)

        n = len(nums)
        if n == 1:
            return nums[0]
        dp = [0 for _ in range(len(nums))]
        # Base case
        dp[0] = nums[0]

        for i in range(1, n):
            dp[i] = max(dp[i-1] + nums[i], nums[i])

        return max(dp)
assert Solution().maxSubArray([-2,1,-3,4,-1,2,1,-5,4]) == 6


# ### 28. Implement strStr() / Rabin-Karp Needle in a Haystack
# 
# Source: [https://leetcode.com/problems/implement-strstr/](https://leetcode.com/problems/implement-strstr/)
# 
# ```
# function RabinKarp(string s[1..n], string pattern[1..m])
#     hpattern := hash(pattern[1..m]);
#     for i from 1 to n-m+1
#         hs := hash(s[i..i+m-1])
#         if hs = hpattern
#             if s[i..i+m-1] = pattern[1..m]
#                 return i
#     return not found
# 
# Use a rolling hash to construct the substring of the haystack
# s[i+1..i+m] = s[i..i+m-1] - s[i] + s[i+m]
# ```

# In[23]:


class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        """
        function RabinKarp(string s[1..n], string pattern[1..m])
            hpattern := hash(pattern[1..m]);
            for i from 1 to n-m+1
                hs := hash(s[i..i+m-1])
                if hs = hpattern
                    if s[i..i+m-1] = pattern[1..m]
                        return i
            return not found

        Use a rolling hash to construct the substring of the haystack
        s[i+1..i+m] = s[i..i+m-1] - s[i] + s[i+m]
        Source: https://en.wikipedia.org/wiki/Rabin–Karp_algorithm
        """

        def hash(s: str) -> int:
            result = 0
            for c in s:
                result += ord(c) % 2069
            return result


        n = len(haystack)
        m = len(needle)

        if n < m:
            return -1

        if n == m == 0:
            return 0

        hashed_needle = hash(needle)

        substr = None
        for i in range(len(haystack)):
            if substr is None:
                substr = haystack[i:i+m]
                hashed_substr = hash(substr)
            elif i+m-1 < n:
                # implements a rolling hash
                # - The earliest hashed character is at i-1 since i was incremented after
                # the previous loop
                # the next character we want to add to the hash is at i-1+m
                hashed_substr = hashed_substr - hash(haystack[i-1]) + hash(haystack[i-1+m])
                # hashed_substr = hash(haystack[i:i+m])
            else:
                break

            if hashed_substr == hashed_needle:
                # verify order is correct since hash doesn't check order
                if needle == haystack[i:i+m]:
                    return i
        return -1

assert Solution().strStr("123", "123") == 0
assert Solution().strStr("9999999999123lksdjflskdfjlkjf", "123") == 10


# ### 2128. Remove All Ones with Row and Column Flips
# You are given an m x n binary matrix grid.
# 
# In one operation, you can choose any row or column and flip each value in that row or column (i.e., changing all 0's to 1's, and all 1's to 0's).
# 
# Return true if it is possible to remove all 1's from grid using any number of operations or false otherwise.
# 
# Source: [https://leetcode.com/problems/remove-all-ones-with-row-and-column-flips/](https://leetcode.com/problems/remove-all-ones-with-row-and-column-flips/)

# In[24]:


from typing import List
class Solution:
    def removeOnes(self, grid: List[List[int]]) -> bool:
        # Observations / Notes

        # Approach # 3
        # Invert the first row
        # for all rows if none equal the first row or its inversion, return False
        # otherwise return True

        # Time complexity: O(nm)
        # Space complexity: O(n)

        r1 = grid[0]
        r1_inv = [int(not elem) for elem in grid[0]]

        for row in grid[1:]:
            if row != r1 and row != r1_inv:
                return False
        return True


# In[ ]:




