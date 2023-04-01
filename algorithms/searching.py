from typing import List


def binary_search(nums: List[int], target):
    """
    Assumes nums is sorted in increasing order
    O(logn) time complexity
    """
    if len(nums) == 0 or target == nums[0]:
        return 0
    low = 0
    high = len(nums)
    while low <= high:
        mid = low + ((high - low) // 2)
        if nums[mid] < target:
            low = mid + 1
        elif nums[mid] > target:
            high = mid - 1
        else:
            return mid
    return -1