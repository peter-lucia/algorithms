import copy
from random import shuffle

import pytest
from algorithms.sorting import *

large_list = [i for i in range(-100000, 100000)]
large_list_shuffled = copy.deepcopy(large_list)
shuffle(large_list_shuffled)
sorting_testcases = [
    ([12, 11, 13, 5, 6, 7], [5,6,7,11,12,13]),
    ([5,4,3,6,7,9], [3,4,5,6,7,9]),
    ([], []),
    ([100], [100]),
    ([0], [0]),
    ([1,0,-1], [-1,0,1]),
    (large_list_shuffled, large_list),
]

@pytest.mark.parametrize("nums,expected", sorting_testcases)
def test_mergesort_recursive(nums, expected):
    nums1 = nums.copy()
    merge_sort_recursive(nums1)
    assert nums1 == expected, "Incorrect result"
    nums = merge_sort_recursive(nums, in_place=False)
    assert nums == expected, "Incorrect result"


@pytest.mark.parametrize("nums,expected", sorting_testcases[:-1])
def test_quicksort_recursive(nums, expected):
    quicksort_recursive(nums, 0, len(nums) - 1)
    assert nums == expected, "Incorrect result"


@pytest.mark.parametrize("nums,expected", sorting_testcases[:-1])
def test_quicksort_iterative(nums, expected):
    nums_sorted = quicksort_iterative(nums)
    assert nums_sorted == expected, "Incorrect result"
