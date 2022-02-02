#!/usr/bin/env python
# coding: utf-8

# # Geometry

# ### Is a square valid
# Given four points determine if they form a valid square
# Note that the square can be rotated at an angle

# In[1]:


import math
from typing import List
class Solution:
    def validSquare(self, p1: List[int], p2: List[int], p3: List[int], p4: List[int]) -> bool:
        ######################### Approach #1 ############################
        # Add all 6 distances to the list and sort the list.
        # The first four should be equal and the last two should be equal
        # Also the first four cannot equal the last two because of pythagorean's theorem

        # calculate the distance between every point and every other point
        # at least two of those distances must be equal for every point
        def dist(point1, point2):
            x1, y1 = point1[0], point1[1]
            x2, y2 = point2[0], point2[1]
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        distances = [dist(p1, p2), dist(p1, p3), dist(p1, p4), dist(p2, p3), dist(p2, p4), dist(p3, p4)]

        distances_sorted = sorted(distances)
        first_four_equal = distances_sorted[0] == distances_sorted[1] == distances_sorted[2] == distances_sorted[3]
        last_two_equal = distances_sorted[4] == distances_sorted[5]

        first_four_dont_match_last_two = distances_sorted[0] != distances_sorted[4]

        return first_four_equal and last_two_equal and first_four_dont_match_last_two

assert Solution().validSquare(p1 = [0,0], p2 = [1,1], p3 = [1,0], p4 = [0,1]) == True
assert Solution().validSquare(p1 = [0,0], p2 = [1,1], p3 = [1,0], p4 = [0,12]) == False


# In[ ]:




