from typing import List
import random


def merge_sort_recursive(arr: List[int], in_place: bool = True) -> List[int]:
    """
    The recursive merge sort algorithm
    Modifies the array in place
    """
    if not in_place:
        arr = arr.copy()

    if len(arr) <= 1:
        return arr

    # Finding the middle index of the array
    n = len(arr)

    A = arr[:n//2]
    B = arr[n//2:]

    # recursively sort each half.
    # This breaks the problem down into two subproblems (a = 2) each of size (n/2), so b = 2
    merge_sort_recursive(A)
    merge_sort_recursive(B)

    # i = the idx for iterating over A
    # j = the idx for iterating over B
    # k = the idx we are at in the merged array
    i = j = k = 0

    # arr is copied over to A[...] and B[...]
    # so we overwrite
    while i < len(A) and j < len(B):
        if A[i] < B[j]:
            arr[k] = A[i]
            i += 1
        else:
            arr[k] = B[j]
            j += 1
        k += 1

    # at this point either we've added all elements of A or B to the result, arr.
    # we need to add the left over elements which are guaranteed to be both
    # pre-sorted in increasing order and larger than the last element of arr

    # add remaining elements from A
    while i < len(A):
        arr[k] = A[i]
        i += 1
        k += 1

    # add remaining elements from B
    while j < len(B):
        arr[k] = B[j]
        j += 1
        k += 1

    if not in_place:
        return arr

########################################################################################################################
#################################################### Quicksort #########################################################
########################################################################################################################

def quicksort_recursive(arr: List[int], low, high) -> List[int]:
    """
    In-place sorting algorithm using a pivot
    to partition elements
    Worst Case Time complexity: O(n^2)
    Space Complexity: O(n)
    :param low: lower bound of the array to partition
    :param high: higher bound of the array to partition
    """
    if low < high:
        # pivot index
        # select a random pivot between low and high
        pi = _quicksort_recursive_partition_rand(arr, low, high)
        # choose arr[high] as the pivot
        # pi = partition(arr, low, high)
        quicksort_recursive(arr, low, pi - 1)
        quicksort_recursive(arr, pi + 1, high)


def _quicksort_recursive_partition_rand(arr: List[int], low: int, high: int):
    """
    Selects a random pivot and swaps it with the last element
    in the array
    """

    # select a random pivot
    pivot_idx = random.randrange(low, high)

    # swap pivot and the last element to ensure
    # pivot is the last element of the array
    arr[high], arr[pivot_idx] = arr[pivot_idx], arr[high]

    return _quicksort_recursive_partition(arr, low, high)


def _quicksort_recursive_partition(arr: List[int], low: int, high: int):
    """
    Rearranges arr in place from low to high
    """
    # partition around the last element
    pivot_idx = high
    pivot = arr[pivot_idx]

    # i will hold first index of any element that is
    # greater than or equal to the pivot
    # initially, potentially no elements are greater
    # than the pivot, so we set i = low - 1
    i = low - 1

    # move all elements less than the pivot
    # to the left of the pivot
    for j in range(low, high):
        # if current elem is smaller than the pivot
        if arr[j] < pivot:
            i += 1
            # we don't know what's at arr[i],
            # but we know arr[j] is less than pivot
            # swap elems at i and j and keep going
            arr[i], arr[j] = arr[j], arr[i]

    # now any elements starting at i + 1 are greater than
    # or equal to the pivot
    # move the pivot to i + 1
    arr[i+1], arr[pivot_idx] = arr[pivot_idx], arr[i+1]

    # the pivot now resides at index i + 1
    return i + 1


def quicksort_iterative(arr: List[int]) -> List[int]:
    """
    Iterative quick sort
    :param arr: arr to be sorted
    :return:
    """
    if arr is None:
        raise ValueError("Invalid array")
    if len(arr) <= 1:
        return arr

    low = 0
    high = len(arr)-1

    stack = []
    stack.append(low)
    stack.append(high)

    while stack:
        high = stack.pop()
        low = stack.pop()

        # as long as we have at least two elements to sort
        if low < high:

            # create the partitions
            p = _quicksort_iterative_partition(arr, low, high)

            # push the left partition indexes
            stack.append(low)
            stack.append(p-1)

            # push the right partition indexes
            stack.append(p+1)
            stack.append(high)

    return arr


def _quicksort_iterative_partition(arr, low, high):
    """
    Last element pivot partition: rearrange elements in the array by putting all the elements < than pivot to the left
    and elements > than pivot to the right.
    :param arr: array to partition
    :param low: lower bound of the array to partition
    :param high: higher bound of the array to partition
    :return: the final pivot position in the array
    """
    pivot = arr[high]
    i = low
    j = low

    while j <= high:
        if (arr[j] < pivot):
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
        j += 1

    arr[i], arr[high] = arr[high], arr[i]

    return i

def insertion_sort(arr: List[int]) -> List[int]:
    """
    Algorithm:
        Iterate over the array
        Compare the current element to its predecessor
        While the current element is less than it’s predecessor, swap left
        Time complexity: O(n^2) Space complexity: O(1)
    """

    n = len(arr)

    for i in range(n):

        # define predecessor
        j = i-1  # define predecessor

        # if predecessor exists and it's greater than current element
        while j >= 0 and arr[j] > arr[i]:
            # swap the elements
            arr[i], arr[j] = arr[j], arr[i]
            i -= 1
            j -= 1

    return arr