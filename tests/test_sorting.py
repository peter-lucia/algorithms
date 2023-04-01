from random import shuffle

import pytest
from algorithms.sorting import *

large_list = [i for i in range(-1000000, 1000000)]
large_list_shuffled = large_list.copy()
shuffle(large_list)
sorting_testcases = [
    ([12, 11, 13, 5, 6, 7], [5,6,7,11,12,13]),
    ([5,4,3,6,7,9], [3,4,5,6,7,9]),
    ([], []),
    ([1,0,-1], [-1,0,1]),
    (large_list_shuffled, large_list),
]

@pytest.mark.parametrize("nums,expected", sorting_testcases)
def test_mergesort_recursive(nums, expected):
    nums_sorted = nums.copy()
    merge_sort_recursive(nums_sorted)
    assert nums_sorted == expected, "Incorrect result"


@pytest.mark.parametrize("nums,expected", sorting_testcases)
def test_quicksort_recursive(nums, expected):
    nums_sorted = nums.copy()
    quicksort_recursive(nums_sorted, 0, len(nums) - 1)
    assert nums_sorted == expected, "Incorrect result"


@pytest.mark.parametrize("nums,expected", sorting_testcases)
def test_quicksort_iterative(nums, expected):
    nums_sorted = nums.copy()
    quicksort_iterative(nums_sorted, 0, len(nums) - 1)
    assert nums_sorted == expected, "Incorrect result"
