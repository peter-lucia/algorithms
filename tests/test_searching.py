from random import shuffle

import pytest
from algorithms.searching import *

large_list = [i for i in range(-1000000, 1000000)]
large_list_shuffled = large_list.copy()
shuffle(large_list)
large_list[1000000] = 9000000
searching_testcases = [
    ([4, 6, 10], 10, 2),
    ([1, 2, 3, 4, 5, 6, 7], 5, 4),
    ([10, 11, 12], 2, -1),
    ([0, 1, 2, 10, 11, 12], 2, 2),
    (large_list, 9000000, 1000000)
]


@pytest.mark.parametrize("nums,target,expected", searching_testcases)
def test_binary_search(nums, target, expected):
    result = binary_search(nums, target)
    assert result == expected, "Incorrect result"
