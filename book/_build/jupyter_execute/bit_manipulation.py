#!/usr/bin/env python
# coding: utf-8

# # Bit Manipulation

# #### Convert $n_{10}$ to $n_2$
# Uses the & bit-wise operator

# In[1]:


def binary(num: int, width: int = 31) -> str:
    """
    Converts a number from base 10 to base 2

    Example:
       num = 2 = 0010
       width = 4

         1 0 0 0
       & 0 0 1 0 = 0, bool(0) = False, add 0

         0 1 0 0
       & 0 0 1 0 = 0, bool(0) = False, add 0

         0 0 1 0
       & 0 0 1 0 = 2, bool(2) = True, add 1

         0 0 0 1
       & 0 0 1 0 = 0, bool(0) = False, add 0

    In general, the ith bit is on/off if 2^i & num is not 0.

    @param num: the number to convert base from 10 to 2
    @param width: the maximum digits to display
    """
    i = 1 << width
    result = ""
    while i > 0:
        if num & i != 0:
            result += '1'
        else:
            result += '0'
        i = i // 2
        print(i)
    return result

binary(2, width=4)


# In[2]:


binary(12, 8)


# ### XOR (Exclusive OR)
# ```text
# A   = 0101 (5)
# B   = 0011 (3)
# A^B = 0110 (6)
# ```

# ### Find the value of the maximum subarray XOR in a given array
# 

# In[3]:


def max_subarray_xor_continuous(arr):
    """
    Tries starting maximum continuous subarray at where 1 <= i <= n

    Time complexity: O(n^2)
    Space complexity: O(1)
    """
    ans = float("-inf")
    n = len(arr)
    for i in range(n):
        current_xor = 0
        for j in range(i, n):
            current_xor = current_xor ^ arr[j]
            ans = max(ans, current_xor)
    return ans


# ```
# Input: arr[] = {1,2,3,4}
# Output: 7
# The subarray {3,4} has the maximum XOR value since 0011 ^ 0100 = 0111
# 
# If we added 8 (1000) to the array, then {3,4,8} would be the largest sub array adding to 15
# ```

# In[4]:


max_subarray_xor_continuous([1,2,3,4])


# In[5]:


max_subarray_xor_continuous([1,2,3,4,8])


# In[6]:


max_subarray_xor_continuous([1,2,3,4])


# In[7]:


max_subarray_xor_continuous([8,1,2,12,7,6])


# ### 476. Number Complement
# [https://leetcode.com/problems/number-complement/](https://leetcode.com/problems/number-complement/)

# In[8]:


class Solution:
    def findComplement(self, num: int) -> int:
        # Finds the complement of for an unsigned number
        # minimum number of bits needed to represent num in binary
        bit_length = num.bit_length()    # e.g. num = 5, bit_length = 3
        mask = ((1 << bit_length) - 1)   #      mask = 111
        return num ^ mask                #      101 XOR 111 = 010
assert Solution().findComplement(int('101', base=2)) == 2
assert Solution().findComplement(int('10', base=2)) == 1
assert Solution().findComplement(int('1010', base=2)) == 5
assert Solution().findComplement(int('111', base=2)) == 0
assert Solution().findComplement(int('110', base=2)) == 1


# In[ ]:




