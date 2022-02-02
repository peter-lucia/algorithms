#!/usr/bin/env python
# coding: utf-8

# # Dynamic Programming

# #### Knapsack
# T(n) = O(nB), where B is the capacity

# In[1]:


import sys

# 0 / 1 knapsack
# dp[i][j] = max val possible with up to i items limited by weight j

# initially all items have > 0 weight, so we can get up to 0 value for
# the first column
#               wt (aka b)
# val wt  0  1  2  3  4  5  6  7
#  1  1   0
#  4  3   0
#  5  4   0
#  5  7   0

#               wt (aka b)
# val wt  0  1  2  3  4  5  6  7
#  1  1   0' 1  1  1  1  1  1  1
#  4  3   0  1  1  4' 5  5  5  5
#  5  4   0  1  1  4  5  6  6  9'
#  7  5   0  1  1  4  5  7  8  9

# items 4 and 5 with weights 3 and 4 contribute to the largest value having weight <= 7

def solve_knapsack(profits, weights, capacity):
    n = len(profits)
    B = capacity
    T = [[0 for _ in range(B+1)] for _ in range(n)]

    # recurrence
    #                          profit[i] + row above - w[i] , row above)
    # max profit T(i, b) = max(p[i] + T(i-1, b-w[i]), T(i-1, b)),
    for i in range(n):
        for b in range(B+1):
            if weights[i] <= b:
                T[i][b] = max(profits[i] + T[i-1][b-weights[i]], T[i-1][b])
            else:
                T[i][b] = T[i-1][b]

    # return T[n-1][B-1] == capacity
    return T[n-1][B]

profits = [1, 6, 10, 16]  # aka vals
weights = [1, 2, 3, 5]
print(solve_knapsack(profits, weights, 5))
print(solve_knapsack(profits, weights, 6))
print(solve_knapsack(profits, weights, 7))


# #### Longest Common Subsequence
# 
# $T(n) = O(n\cdot m)$ if the strings have unequal length
# 
# $T(n) = O(n^2)$ if the strings are of equal length

# In[2]:


# https://leetcode.com/problems/longest-common-subsequence/


# dp[i][j] = longest common subsequence up to and including str1[i] and str2[j]
#   ''  a  b  c  d  e
# '' 0  0  0  0  0  0
# a  0  1  1  1  1  1
# e  0  1  1  1  1  2
# c  0  1  1  2  2  2
# d  0  1  1  1  3  3
# e  0  1  1  1  3  4


#   ''  a  b  e  x  q
# '' 0  0  0  0  0  0
# a  0  1  1  1  1  1
# e  0  1  1  2  2  2
# c  0  1  1  2  2  2
# d  0  1 `1` 2  2  2   # Remember when text1[i] == text2[j], let dp[ii][jj] = dp[ii-1][jj-1] + 1
# e  0  1  1  2  2  2
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        if len(text1) == 0 or len(text2) == 0:
            return 0
        n = len(text1)
        m = len(text2)

        dp = [[0 for _ in range(len(text2)+1)] for _ in range(len(text1)+1)]
        for i in range(len(text1)):
            for j in range(len(text2)):
                # use i, j for text1 and text2
                # use ii, jj for dp table
                ii = i+1
                jj = j+1
                if text1[i] == text2[j]:
                    dp[ii][jj] = 1 + dp[ii-1][jj-1]
                else:
                    dp[ii][jj] = max(dp[ii-1][jj], dp[ii][jj-1])
        return dp[n][m]

Solution().longestCommonSubsequence(['a', 'b', 'c', 'd', 'e'],
                                    ['a', 'e', 'c', 'd', 'e'])


# ### Nth Fibonacci Number

# In[3]:


def fib(n):
    a = 0
    b = 1
    for _ in range(n):
        a, b = b, a + b
    return a

def fib_verbose(n):
    a = 0
    b = 1
    for _ in range(n):
        tmp = a
        a = b
        b = tmp + b
    return a

for i in range(10):
    a = fib(i)
    b = fib_verbose(i)
    assert a == b
    print(f"fib({i})={fib(i)}")


# ### Longest Increasing Subsequence
# $T(n) = O(n^2)$
# 

# In[4]:


from typing import List

#        j  i
# nums = 3, 4, -1, 0, 6, 2, 3
#  dp  = 1  1  1   1  1  1  1

#        j  i
# nums = 3, 4, -1, 0, 6, 2, 3
#  dp  = 1  2  1   1  1  1  1

#        j     i
# nums = 3, 4, -1, 0, 6, 2, 3
#  dp  = 1  2  1   1  1  1  1

#        j         i
# nums = 3, 4, -1, 0, 6, 2, 3
#  dp  = 1  2  1   2  1  1  1

#        j            i
# nums = 3, 4, -1, 0, 6, 2, 3
#  dp  = 1  2  1   2  3  1  1

#        j               i
# nums = 3, 4, -1, 0, 6, 2, 3
#  dp  = 1  2  1   2  3  3  1

#        j                  i
# nums = 3, 4, -1, 0, 6, 2, 3
#  dp  = 1  2  1   2  3  3  4

# https://leetcode.com/problems/longest-increasing-subsequence/submissions/
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0

        # array containing the length of the LIS up to each element
        dp = [1 for _ in nums]

        for i in range(len(nums)):
            # _max holds the maximum LIS up to and including nums[i]
            _max = 0
            for j in range(0, i):
                # check if nums[i] is greater than each previous element,
                # starting at the first element in the array.
                # keep track of length of LIS up to i with _max
                if nums[i] > nums[j]:
                    _max = max(_max, dp[j])  # notice that _max only takes values from dp table when it increases

            # add 1 to account for nums[i]
            dp[i] = _max + 1
        return max(dp)

Solution().lengthOfLIS([1,2,3,4,7,5,9,6])


# ### Minimum Partition
# 
# Given a set of integers, the task is to divide it into two sets S1 and S2 such that the absolute difference between their sums is minimum.
# If there is a set S with n elements, then if we assume Subset1 has m elements, Subset2 must have n-m elements and **the value of abs(sum(Subset1) – sum(Subset2)) should be minimum.**
# 
# Return the minimum possible absolute difference.
# 
# Sources:
# * [https://leetcode.com/problems/partition-array-into-two-arrays-to-minimize-sum-difference/](https://leetcode.com/problems/partition-array-into-two-arrays-to-minimize-sum-difference/) (the below solution from geeks for geeks does not pass the testcases here)
# * [https://www.geeksforgeeks.org/partition-a-set-into-two-subsets-such-that-the-difference-of-subset-sums-is-minimum/](https://www.geeksforgeeks.org/partition-a-set-into-two-subsets-such-that-the-difference-of-subset-sums-is-minimum/)

# In[5]:



# dp table of size (n+1, sum+1)
# dp[i][j] = 1 if the sum j is achieved including or excluding the ith number, 0 otherwise
# arr = [1,6,11,5]
# sum(arr) = 23

# dp: size = (n+1, sum+1) = (5, 24)
# 1 2 3  ...                                   24

# sum(0) is possible for all elements
# 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

# all sums are impossible with 0 elements
# 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 1
# 1
# 1
# 1

# determine the largest j where dp[n][j] is 1 where j loops from sum / 2 to 0 (left half of table)
# 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 1
# 1
# 1
# 1

# Note this solution may not work for negative numbers
import sys
class Solution:
    def minimumDifference(self, nums: List[int]) -> int:
        total = max(sum(nums), max(nums))
        n = len(nums)
        dp = [[0 for _ in range(total+1)] for _ in range(n+1)]

        # Base case, first row is False since no sums are possible with 0 elements,
        # this is already done

        for i in range(n):
            for j in range(total):
                # Base case, first column is True since a sum of 0 is possible with all elements
                if j == 0:
                    dp[i][j] = 1
                    continue
                else:
                    # set dp indices
                    ii = i + 1
                    jj = j + 1
                    # if nums[i] is excluded, we can assume this is try first before checking whether it can be included
                    dp[ii][jj] = dp[ii-1][jj]
                    # if nums[i] is included it must be less than or equal to the current sum
                    if nums[i] <= jj:
                        # jj - nums[i] must be possible excluding nums[i] for jj to be possible including nums[i]
                        if dp[ii-1][jj - nums[i]] == 1:
                            dp[ii][jj] = 1

        diff = sys.maxsize
        # find the largest jj such that dp[n+1][jj] = 1
        # loop from 0 <----- jj = sum / 2 which is the left half of the dp table
        for jj in range(total // 2, -1, -1):
            if dp[n][jj] == 1:
                diff = total - (2*jj)
                break
        return diff


assert Solution().minimumDifference([1,6,11,5]) == 1  # 12 - 11


# [1937] Maximum Number of Points with Cost
# 
# https://leetcode.com/problems/maximum-number-of-points-with-cost/
# 
# Brute force solution
# 
# base_case:
# 
# 
# * $max\_sum[i,j] = 0$ where 0 < i < n and 0 < j < m
# 
# recurrence:
# * $ max\_sum[i,j] = max(max\_sum[i,j], max\_sum[i-1,k] + points[i,j] - abs(k-j))$
# 
# for all k where k is between 0 and m
# and 0 < i < n and 0 < j < m
# 
# * Time complexity: O(nm^2) since we iterate over the entire table once and over m twice
# * Space complexity: O(nm) for the table
# 
# Faster solution
# https://leetcode.com/problems/maximum-number-of-points-with-cost/discuss/1567183/Python3%3A-Easy-DP-method-with-thorough-explanation
# 

# In[6]:


# Brute force solution, passes 140/157 test cases

class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:

        # 1 5
        # 2 3
        # 4 2
        # maximum sum: 5 + 3 + (4-1) = 11

        # Keep track of max sum between current row and previous row
        # if you select only numbers to the left of the element in the previous row
        # or only select the numbers to the right of the current element in the previous row

        # Basically use the previously computed subproblems to incrementally decide the best combination
        # from the previous row
        # 1 3 2 1 6
        # 1 4 3 2 8

        n = len(points)
        m = len(points[0])

        for row in points:
            print(row)
        print("----------")

        max_rows = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(n):

            for j in range(m):
                if i == 0:
                    max_rows[i][j] = points[i][j]
                else:
                    for x in range(0, j+1):
                        max_rows[i][j] = max(max_rows[i][j], max_rows[i-1][x] + points[i][j] - (j-x))


            for j in range(m-1, -1, -1):
                if i == 0:
                    max_rows[i][j] = points[i][j]
                else:
                    for x in range(m-1, j-1, -1):
                        max_rows[i][j] = max(max_rows[i][j], max_rows[i-1][x] + points[i][j] - (x - j))
        return max(max_rows[n-1])


# In[7]:


# Faster solution:
from typing import List
import copy
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        # Time complexity: O(nm) since we iterate over the entire table once and over m twice
        # Space complexity: O(nm) for the table
        if not points:
            return 0
        n = len(points)
        m = len(points[0])
        max_sum = [[0 for _ in range(m)] for _ in range(n)]

        for i in range(n):
            for j in range(m):
                if i == 0:
                    max_sum[i][j] = points[i][j]
                if i > 0:
                    # max sum of points directly above each other
                    max_sum[i][j] = max(max_sum[i][j], max_sum[i-1][j] + points[i][j])
            # get max sum from current row will be available to the next row
            # each move left subtracts 1 from the max sum possible found so far
            # move right -> left
            for j in range(m-2, -1, -1):
                max_sum[i][j] = max(max_sum[i][j], max_sum[i][j + 1] - 1)
            # get max sum from current row will be available to the next row
            # each move right subtracts 1 from the max sum possible found so far
            # move left -> right
            for j in range(1, m):
                max_sum[i][j] = max(max_sum[i][j], max_sum[i][j - 1] - 1)
        # the answer is the max value of the last row
        return max(max_sum[-1])


points = [
    [1,2,3],
    [1,5,1],
    [3,1,1],
]
Solution().maxPoints(points)


# ### Longest string chain
# * https://leetcode.com/problems/longest-string-chain/
# 

# In[8]:


from collections import defaultdict
from typing import List
class Solution:

    def longestStrChain(self, words: List[str]) -> int:
        # approach #1
        # build a Trie from the words list
        # keep track of the max depth when reaching the end of a word
        # return the max depth

        # approach #2 (working)
        # Source: https://leetcode.com/problems/longest-string-chain/discuss/294890/JavaC%2B%2BPython-DP-Solution
        # 1. sort the words by length
        # 2. Use a hashmap to map a word to the number of predecessor words
        # 3. For each word, iterate over all substrings with each character removed in that word
        #    If the substring is found in the table, increment the predecessors for the word by
        #    the greater quantity between one plus what was found in the lookup for the substr
        #    or what exist in the lookup for the current word
        # 4. Return the max number of predecessors found for all words, keep track of these each
        #    time the lookup for a word is adjusted
        # Time complexity: O(nlogn + nL^2) where L is the average (or longest) word length

        # Assumptions
        # Can assume 1 <= len(words) <= 1000
        # 1 <= words[i].length <= 16
        # words[i] only consists of English letters

        words = sorted(words, key=len)
        lookup = defaultdict(int)

        # a single word is a word chain
        result = 1

        for word in words:
            lookup[word] = 1

            for i in range(len(word)):
                substr = word[:i] + word[i+1:]

                # because we care about letter order for predecessors, we don't need all permutations of the substring
                if substr in lookup:
                    lookup[word] = max(lookup[word], lookup[substr] + 1)
                    result = max(result, lookup[word])

        return result


# In[9]:


assert Solution().longestStrChain(["a","b","ba","bca","bda","bdca"]) == 4


# In[10]:


assert Solution().longestStrChain(["xbc","pcxbcf","xb","cxbc","pcxbc"]) == 5


# ### Number of matching subsequences
# [https://leetcode.com/problems/number-of-matching-subsequences/](https://leetcode.com/problems/number-of-matching-subsequences/)

# In[11]:


from typing import List
class Solution:

    def is_word_in_s(self, word: str, s: str) -> bool:
        for c in word:
            idx = s.find(c)
            if idx == -1:
                return False

            s = s[idx+1:]
        return True

    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        # Approach

        # Time complexity: O(nklogk) where k is the length of the longest word, n is the number of words
        # Space complexity: O(1) since we just keep a counter of the total number of matches
        # 1. Iterate over each word
        # 2. iterate over each character c in the word
        # 3. if we find c in the word, let s = s[idx_of_c+1:]
        #    otherwise, reject this word
        # 4. if we find all characters of word in s, increment total counter
        # 5. return the counter

        result = 0
        for word in words:
            if self.is_word_in_s(word, s):
                result += 1
        return result

Solution().numMatchingSubseq("abc", ["a", "b", "c", "ab", "bc", "abc"])


# ### Edit Distance
# 
# Source: [https://www.geeksforgeeks.org/edit-distance-dp-5/](https://www.geeksforgeeks.org/edit-distance-dp-5/)
# 

# In[12]:




# dp[i][j] = the minimum edit distance up to and including i from str1 and j from str 2
#    ''  s  a  t  u  r  d  a  y
# '' [0, 1, 2, 3, 4, 5, 6, 7, 8]
# s  [1, 0, 1, 2, 3, 4, 5, 6, 7]
# u  [2, 1, 1, 2, 2, 3, 4, 5, 6]
# n  [3, 2, 2, 2, 3, 3, 4, 5, 6]
# d  [4, 3, 3, 3, 3, 4, 3, 4, 5]
# a  [5, 4, 3, 4, 4, 4, 4, 3, 4]
# y  [6, 5, 4, 4, 5, 5, 5, 4, 3]



def edit_distance(str1, str2):
    """
    Time complexity: O(nm)
    Space compleity: O(nm)
    """
    n = len(str1)
    m = len(str2)
    dp = [[0 for _ in range(m+1)] for _ in range(n+1)]

    for i in range(n+1):
        for j in range(m+1):

            # when first string is empty, only option is to insert
            # all of the characters up to j: requires j operations
            if i == 0:
                dp[i][j] = j

            # if second string is empty, only option is to insert
            # all of the characters up to i: requires i operations
            elif j == 0:
                dp[i][j] = i

            # if last characters are the same in str1 and str2,
            # we carry over min operations needed without
            # including the last characters
            elif str1[i-1] == str2[j-1]:  # notice that i and j range from [0, n], we subtract 1 from each to get the last character here
                dp[i][j] = dp[i-1][j-1]

            # last two characters are different, add an operation
            # to the minimal previous number of operations possible
            # prior to reaching this cell in the dp table
            else:
                dp[i][j] = 1 + min(dp[i][j-1],  # without last char in str2
                                   dp[i-1][j],  # without last char in str1
                                   dp[i-1][j-1])  # without either chars in str1 and str2
    for row in dp:
        print(row)
    return dp[n][m]


# Expected edit distance is 3:
# s  unday
# saturday
assert edit_distance("sunday", "saturday") == 3


# In[13]:


# Edit distance recursive with memoization via @functools.cache
import functools

class Solution:
    def minDistance(self, str1: str, str2: str) -> int:
        n = len(str1)
        m = len(str2)

        return self.edit_distance(str1, str2, n, m)


    @functools.lru_cache(maxsize=None)
    # @functools.cache  # >= python 3.8
    def edit_distance(self, str1, str2, n, m):

        # If the first string is empty, the only option
        # is to insert all the characters of the second
        # string into the first string
        if n == 0:
            return m

        # If the second string is empty, the only option
        # is to insert all the characters of the first
        # string into the second
        if m == 0:
            return n

        # If the last characters of the two strings are
        # the same, ignore them and get the count for the
        # remaining inner strings
        if str1[n-1] == str2[m-1]:
            return self.edit_distance(str1, str2, n-1, m-1)

        # If the last characters are not the same, take the
        # minimum cost of all three possible operations on the
        # last character of the first string.
        return 1 + min(self.edit_distance(str1, str2, n, m-1),    # insert last character of str1, remove last character of str2
                       self.edit_distance(str1, str2, n-1, m),    # remove last character of str1, take last character of str2
                       self.edit_distance(str1, str2, n-1, m-1)    # replace last character of str1 with last character of str2
                       )
    # saturday
str1 = "sunday"
str2 = "saturday"
assert Solution().edit_distance(str1, str2, len(str1), len(str2)) == 3


# ### Ways to Cover a Distance
# 
# Given a distance ‘dist’, count total number of ways to cover the distance with 1, 2 and 3 steps.
# 
# Source: [https://www.geeksforgeeks.org/count-number-of-ways-to-cover-a-distance/](https://www.geeksforgeeks.org/count-number-of-ways-to-cover-a-distance/)

# In[14]:


def print_count_dp(dist):
    dp = [0 for _ in range(dist+1)]

    dp[0] = 1  # one way to cover dist of 0 via 0 steps
    if dist >= 1:
        dp[1] = 1  # one way to cover dist of 1 via 1 step
    if dist >= 2:
        dp[2] = 2  # two ways to cover 2 (1 + 1 step, single 2 step)

    for i in range(3, dist+1):
        dp[i] = dp[i-1] + dp[i-2] + dp[i-3]

    return dp[dist]

# Possible ways to cover a distance of 4 using steps of 1, 2, or 3:
# 1 + 1 + 1 + 1
# 1 + 1 + 2
# 1 + 2 + 1
# 2 + 1 + 1
# 1 + 3
# 3 + 1
# 2 + 2
assert print_count_dp(4) == 7


# ### 70. Climbing Stairs
# 
# You are climbing a staircase. It takes n steps to reach the top.
# 
# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
# 
# Source: [https://leetcode.com/problems/climbing-stairs/](https://leetcode.com/problems/climbing-stairs/)

# In[15]:


class Solution:
    def climbStairs(self, n: int) -> int:

        dp = [0 for _ in range(n+1)]

        dp[0] = 1  # one way to cover dist of 0 via 0 steps
        if n >= 1:
            dp[1] = 1  # one way to cover dist of 1 via 1 step
        if n >= 2:
            dp[2] = 2  # two ways to climb 2 steps via a 2 step or two single steps


        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]

        return dp[n]

Solution().climbStairs(4)

# 1 + 1 + 1 + 1
# 1 + 1 + 2
# 2 + 1 + 1
# 1 + 2 + 1
# 2 + 2
assert Solution().climbStairs(4) == 5


# ### Subset Sum Problem
# 
# Given a set of non-negative integers, and a value sum, determine if there is a subset of the given set with sum equal to given sum.
# 
# Input: set = {3, 34, 4, 12, 5, 2}, sum = 9
# Output: True
# There is a subset (4, 5) with sum 9.

# In[16]:



def has_subset_sum(nums, val):
    n = len(nums)

    dp = [[False for _ in range(val+1)] for _ in range(n+1)]

    # don't need any numbers to sum up to 0
    for i in range(n+1):
        dp[i][0] = True

    # we can never generate a sum > 0 with no numbers
    for j in range(1, val + 1):
        dp[0][j] = False

    # fill table in bottom up manner
    for ii in range(1, n+1):
        for jj in range(1, val + 1):
            i = ii - 1
            j = jj - 1
            if jj < nums[i]:
                # can make subset sum if already could without nums[i]
                dp[ii][jj] = dp[ii-1][j]
            else:
                # can make subset sum if we already could without nums[i]
                # or if without nums[i] we could sum to jj - nums[i]
                dp[ii][jj] = dp[ii-1][jj] or dp[ii-1][jj - nums[i]]
    # for i in range(n+1):
    #     print(dp[i])
    return dp[n][val]


#     0  1  2  3  4  5  6  7  8  9
# '' [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 3  [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
# 34 [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
# 4  [1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
# 12 [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
# 5  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# 2  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

has_subset_sum([3, 34, 4, 12, 5, 2], 9)


# ### 416. Partition Equal Subset Sum
# (Subset Sum Variation)
# 
# [https://leetcode.com/problems/partition-equal-subset-sum/](https://leetcode.com/problems/partition-equal-subset-sum/)
# 
# Given a non-empty array nums containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.
# 
# 
# 
# Example 1:
# 
# Input: nums = [1,5,11,5]
# Output: true
# Explanation: The array can be partitioned as [1, 5, 5] and [11].
# Example 2:
# 
# Input: nums = [1,2,3,5]
# Output: false
# Explanation: The array cannot be partitioned into equal sum subsets.

# In[17]:


from typing import List
class Solution:
    def canPartition(self, nums: List[int]) -> bool:

        # Approach #1 (Dynamic Programming)

        # dp[i][j] = True if the sum j can be formed by array elements in subsets nums[0]...nums[i]
        # otherwise, dp[i][j] = False
        #    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22
        # '' 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        # 1  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        # 5  1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        # 11 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0
        # 5  1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1

        n = len(nums)
        total = sum(nums)

        if total % 2 != 0:
            return False

        dp = [[False for _ in range(total+1)] for _ in range(n+1)]

        # if no elements and no total, we can always get desired sum of 0
        dp[0][0] = True

        for ii in range(1, n+1):
            for jj in range(total+1):
                i = ii - 1
                # if current total is less than current number in nums
                # then we could only make jj if that was possible without using nums[i]
                if jj < nums[i]:
                    dp[ii][jj] = dp[ii-1][jj]
                # current total is >= nums[i], so mark True if it was already possible without nums[i]
                # or make true if jj - nums[i] was possible without using nums[i]
                else:
                    dp[ii][jj] = dp[ii-1][jj] or dp[ii-1][jj- nums[i]]
        return dp[n][total // 2]

assert Solution().canPartition([1,5,11,5]) == True
assert Solution().canPartition([1,2,3,5]) == False


# ### Count of subsets with sum equal to X
# 
# Given an array of length N and integer X, the task is to find the number of subsets with a sum equal to X.
# 
# 

# In[18]:


# dp[i][j] stores the number of subsets of the sub-array such that their sum is equal to j.
from typing import List

#      0  1  2  3  4  5
# ''   1, 0, 0, 0, 0, 0
#  1   1, 1, 0, 0, 0, 0
#  2   1, 1, 1, 1, 0, 0
#  3   1, 1, 1, 2, 1, 1
#  4   1, 1, 1, 2, 2, 2
#  5   1, 1, 1, 2, 2, 3

# https://www.geeksforgeeks.org/count-of-subsets-with-sum-equal-to-x/
def subset_sum(nums: List[int], value: int):
    # may only work for positives nums[i]

    n = len(nums)
    dp = [[0 for _ in range(value+1)] for _ in range(n+1)]

    # always the empty set can make a sum of 0
    for i in range(n+1):
        dp[i][0] = 1

    for ii in range(1, n+1):
        for jj in range(1, value + 1):
            i = ii - 1
            if nums[i] <= jj:
                dp[ii][jj] = dp[ii-1][jj] + dp[ii-1][jj - nums[i]]
            else:
                dp[ii][jj] = dp[ii-1][jj]
    for row in dp:
        print(row)
    return dp[n][value]

# subset_sum([1,2,3,4,5], 5)
arr = [1,3,4,2,6,8]
half = sum(arr) // 2
subset_sum(arr, half)


# In[19]:


def subset_sum(nums: List[int], value: int):
    n = len(nums)

    dp = [[0 for _ in range(value +1)] for _ in range(n+1)]

    # all elements can always make a sum of 0
    for i in range(n+1):
        dp[i][0] = 1

    # no elements cannot make a sum greater than 0
    # elements are defaulted to 0, so this isn't really needed
    # for j in range(1, value+1):
    #     dp[0][j] = 0

    for ii in range(1, n+1):
        for jj in range(1, value+1):
            i = ii - 1
            if nums[i] <= jj:
                # number of subsets with a sum equal to jj
                # up to and including ii is what it was without nums[i]
                # plus the
                dp[ii][jj] = dp[ii-1][jj] + dp[ii-1][jj-nums[i]]
            else:
                dp[ii][jj] = dp[ii-1][jj]
    for row in dp:
        print(row)
    return dp[n][value]


# should be [5], [4,1], [2,3]
assert subset_sum([1,2,3,4,5], 5) == 3


# In[20]:


### Subset Sum Count (Recursive)
# https://www.geeksforgeeks.org/count-of-subsets-with-sum-equal-to-x-using-recursion/
# https://www.geeksforgeeks.org/tabulation-vs-memoization/
from functools import lru_cache

@lru_cache(maxsize=None)
def subset_sum(nums, n, i, total, count):
    # Returns the number of ways we can sum up to the desired total using a subset
    # of the numbers in nums
    # Time complexity: O(2^n)

    # we've traversed all of the numbers
    if i == n:
        # if total is 0 and we've traversed all the numbers,
        # then we've found a subset that sums up to the desired total
        if total == 0:
            count += 1
        return count

    # Either the element can be counted in the subset or not
    # First it is counted, so the remaining sum to be check discludes it
    count = subset_sum(nums, n, i+1, total - nums[i], count)
    # Second the element is not counted, so we check the total sum
    count = subset_sum(nums, n, i+1, total, count)
    return count


nums = [1,2,3,4,5]
total = 10
n = len(nums)

#   | |
# 1,2,3,4,5
# |     |
# 1,2,3,4,5
#         |
# 1,2,3,4,5

print(subset_sum(tuple(nums), n, 0, total, 0))


# In[ ]:




