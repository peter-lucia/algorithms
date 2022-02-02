#!/usr/bin/env python
# coding: utf-8

# # Python

# #### Defaultdict
# `(defaultdict(list))`
# 
# 

# In[1]:


from collections import defaultdict

lookup = defaultdict(list)

lookup[0].append((1, 2))
lookup


# In[2]:


lookup[0].append((3, 4))
lookup


# Insert element into list at position i

# In[3]:


a = [1,2,3,4]
a.insert(2, 400)
a


# #### Bisect
# 
# https://docs.python.org/3/library/bisect.html

# In[4]:


from bisect import bisect_left, bisect_right, bisect

nums = [1,50,100,150]

# Locate the insertion point for x in nums to maintain sorted order.
# The parameters lo and hi may be used to specify a subset of the list which should be considered;
# bisect left inserts x to the left of any pre-existing entries of x
bisect_left(nums, x=50)


# In[5]:


# Similar to bisect_left(), but returns an insertion point which comes after (to the right of) any existing entries of x in a.

# The returned insertion point i partitions the array a into two halves so that all(val <= x for val in a[lo : i]) for the left side and all(val > x for val in a[i : hi]) for the right side.
# bisect_right = bisect
# bisect right and bisect insert x to the right of any pre-existing entries of x
bisect_right(nums, x=50)


# In[6]:


bisect(nums, x=50)


# In[7]:


# if x is greater than all elements of a, return 4 since 4 is the correct insertion point

bisect(nums, x=151)


# In[8]:


# if x is less than than all elements of a, returns 0, since 0 is the correct insertion point for -1

bisect(nums, x=-1)


# In[8]:





# Bisect application (uses binary search and hashmap)
# 
# https://leetcode.com/problems/time-based-key-value-store/submissions/

# In[9]:


from collections import defaultdict

from bisect import bisect_right
class TimeMap:

    # approach #1

    # use a hashmap
    # key: [(timestamp, value), (timestamp, value)]

    # approach #2

    def __init__(self):
        self.timestamps = defaultdict(list)
        self.values = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        # timestamp increases with each successive call,
        # so we can just append and they will be sorted
        # by timestamp in increasing order

        # create two defaultdicts for timestamps and values

        # we could use one lookup with a list of tuples
        # where lookup[key] = [(timestamp, value), (timestamp1, value1)]
        # but bisect only provides the key parameter in python 3.10+

        self.timestamps[key].append(timestamp)
        self.values[key].append(value)


    def get(self, key: str, timestamp: int) -> str:
        if self.timestamps.get(key) is None:
            return ""

        # If the last (and greatest) timestamp for this key is smaller
        # than the timestamp requested, return the last timestamp
        if self.timestamps[key][-1] < timestamp:
            return self.values[key][-1]

        # find theoretical insertion point for timestamp in self.timestamps[key] to maintain sorted order
        # bisect_right() and bisect() find idx to insert x to the right of any pre-existing entries of x
        i = bisect_right(a=self.timestamps[key], x=timestamp)

        # all timestamps are greater than the one requested
        # return not found
        if i == 0:
            return ""

        # since all elements with index <= i are <= timestamp,
        # we return the value at i-1
        # which has a timestamp <= the requested timestamp
        return self.values[key][i-1]

# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)


# ### Alternative approach
# 
# TODO

# ### Min/Max Heap
# 
# [https://docs.python.org/3/library/heapq.html](https://docs.python.org/3/library/heapq.html)
# 
# > (b) Our pop method returns the smallest item, not the largest (called a “min heap” in textbooks;
# > a “max heap” is more common in texts because of its suitability for in-place sorting).

# In[10]:


nums = [10, 1, 5, 2, 7]


# In[11]:


import heapq
heapq.heapify(nums)  # min-heap by default


# In[12]:


n = len(nums)
for i in range(n):
    print(heapq.heappop(nums))


# > A heapsort can be implemented by pushing all values onto a heap and then popping off the smallest
# > values one at a time:

# In[13]:


def heapsort(iterable):
    h = []
    for value in iterable:
        heapq.heappush(h, value)
    return [heapq.heappop(h) for _ in range(len(h))]


# In[14]:


nums = [10, 1, 4, 8, 2, 3]
print(heapsort(nums))


# From [Big-O CheatSheet](https://www.bigocheatsheet.com)
# The heapsort worst time complexity is O(nlogn) and the worst space complexity is O(1)
# It is effectively better than mergesort because mergesort has a space complexity of O(n)
# It is better than quicksort because quicksort has a 'worst' time complexity of O(n^2) and a space complexity of O(logn)

# In[15]:


a = [1,2,3,4,5]
a[::-1]


# ### Class variables vs. Instance variables in Python
# Source: [https://www.geeksforgeeks.org/g-fact-34-class-or-static-variables-in-python/](https://www.geeksforgeeks.org/g-fact-34-class-or-static-variables-in-python/)

# In[16]:


class CSStudent:
    stream = 'cse'                  # Class Variable
    def __init__(self,name,roll):
        self.name = name            # Instance Variable
        self.roll = roll            # Instance Variable


# ### Static variables in functions in Python

# In[17]:


def testing123():
    testing123.val = 3

result = False
try:
    testing123.val
except AttributeError as e:
    print(f"Raised {e}")
    result = True
assert result

testing123()
print(testing123.val)
testing123.val += 1
print(testing123.val)


# Sets [https://www.geeksforgeeks.org/unordered_set-in-cpp-stl/](https://www.geeksforgeeks.org/unordered_set-in-cpp-stl/)

# In[18]:


# Convert a list of 1s and 0s to binary using
# Cast each digit to a string using the map() function
# Cast the string to an int with base equal to 2
digits = [1,0,1,0,0]
num1 = int("".join(map(str, digits)), base=2)

