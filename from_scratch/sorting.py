import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from numba import njit
from modules.my_decorators import record_time

NumArray = NDArray[np.float64]  # type alias


@record_time
@njit
def counting_sort(arr: NumArray) -> NumArray:
    size = len(arr)
    output = np.zeros(size)
    count = np.zeros(max(arr) + 1)

    # Store the count of each elements in count array
    for i in range(size):
        count[arr[i]] += 1

    # Store the cummulative count
    for i in range(1, max(arr) + 1):
        count[i] += count[i - 1]

    # Find the index of each element of the original array in count array
    # place the elements in output array
    i = size - 1
    while i >= 0:
        output[int(count[arr[i]] - 1)] = arr[i]
        count[arr[i]] -= 1
        i -= 1

    # Copy the sorted elements into original array
    for i in range(size):
        arr[i] = output[i]
    return arr


@record_time
@njit
def bubble_sort(arr: NumArray) -> NumArray:
    """_summary_
        time complexity: O(n**2), space: O(1)

    Args:
        nums (NumArray): unsorted array of numbers

    Returns:
        NumArray: sorted array of numbers
    """
    n = len(arr)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


# @record_time
# @njit
def merge_sort(arr: NumArray) -> NumArray:
    """_summary_
        time complexity: O(n log n), space: O(n)

    Args:
        nums (NumArray): unsorted array of numbers

    Returns:
        NumArray: sorted array of numbers
    """

    return NotImplementedError


# @njit
def _partition(arr, start, end):
    # pivot = arr[start]
    # low = start + 1
    # high = end

    # while True:
    #     while low <= high and arr[high] >= pivot:
    #         high = high - 1

    #     while low <= high and arr[low] <= pivot:
    #         low = low + 1

    #     if low <= high:
    #         arr[low], arr[high] = arr[high], arr[low]
    #     else:
    #         break

    # arr[start], arr[high] = arr[high], arr[start]

    return NotImplementedError


# @record_time
# @njit
def quick_sort(arr: NumArray, start, end) -> NumArray:
    """Quick sort: recursive
        time complexity: O(n log n to n**2), space: O(n)

    Args:
        nums (NumArray): unsorted array of numbers

    Returns:
        NumArray: sorted array of numbers
    """
    # if start >= end:
    #     return

    # p = _partition(arr, start, end)
    # arr = quick_sort(arr, start, p - 1)
    # arr = quick_sort(arr, p + 1, end)

    return NotImplementedError


@record_time
@njit
def tim_sort(arr: NumArray) -> NumArray:
    """_summary_
        time complexity: O(n log n), space: O(n)

    Args:
        nums (NumArray): unsorted array of numbers

    Returns:
        NumArray: sorted array of numbers
    """

    return np.sort(
        arr
    )  # python default is timsort, but not sure about numpy/option-select


@njit
def _heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and arr[i] < arr[l]:
        largest = l

    if r < n and arr[largest] < arr[r]:
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        arr = _heapify(arr, n, largest)
    return arr


@record_time
@njit
def heap_sort(arr: NumArray) -> NumArray:
    """_summary_
        time complexity: O(n log n), space: O(1)

    Args:
        nums (NumArray): unsorted array of numbers

    Returns:
        NumArray: sorted array of numbers
    """
    n = len(arr)

    for i in range(n // 2, -1, -1):
        arr = _heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]

        arr = _heapify(arr, i, 0)

    return arr


@record_time
@njit
def selection_sort(arr: NumArray) -> NumArray:
    """_summary_
        time complexity: O(n**2), space: O(1)

    Args:
        nums (NumArray): unsorted array of numbers

    Returns:
        NumArray: sorted array of numbers
    """
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[min_idx] > arr[j]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


@record_time
@njit
def insertion_sort(arr: NumArray) -> NumArray:
    """_summary_
        time complexity: O(n**2), space: O(1)

    Args:
        nums (NumArray): unsorted array of numbers

    Returns:
        NumArray: sorted array of numbers
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


@njit
def _counting_sort_radix(arr, lsd):
    count_arr = np.zeros(10).astype(np.uint64)
    n = len(arr)

    for i in range(n):
        lsd_idx = int((arr[i] // lsd) % 10)
        count_arr[lsd_idx] += 1

    for i in range(1, 10):
        count_arr[i] += count_arr[i - 1]

    output_arr = np.zeros(n).astype(np.uint64)
    i = n - 1
    while i >= 0:
        current_idx = arr[i]
        lsd_idx = int((arr[i] // lsd) % 10)
        count_arr[lsd_idx] -= 1
        newPosition = count_arr[lsd_idx]
        output_arr[newPosition] = current_idx
        i -= 1

    return output_arr


@record_time
@njit
def radix_sort(arr: NumArray) -> NumArray:
    max_val = max(arr)

    # number of digits in max_val
    d = 1
    while max_val > 0:
        max_val //= 10
        d += 1

    # init least-significant-digit
    lsd = 1

    while d > 0:
        arr = _counting_sort_radix(arr, lsd)
        lsd *= 10
        d -= 1

    return arr


def main() -> None:
    primer_list = np.array([56, 79, 11, 42, 2, 4, 8]).astype(np.uint64)
    print(f"Unsorted array: {primer_list}")

    print(insertion_sort(primer_list.copy()))
    print(selection_sort(primer_list.copy()))
    print(heap_sort(primer_list.copy()))
    print(tim_sort(primer_list.copy()))
    print(radix_sort(primer_list.copy()))
    print(bubble_sort(primer_list.copy()))
    print(counting_sort(primer_list.copy()))

    # print(quick_sort(primer_list, 0, len(primer_list)))
    # t0 = time.time()
    # print(merge_sort(primer_list.copy()))
    # print(f"function 'merge_sort()' took {time.time()-t0:.9f} seconds.")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    minval, maxval, count = 1, 10_000_000, 80_000
    real_array = np.random.randint(minval, maxval, size=count).astype(np.uint64)
    print(
        f"Sorting array length {count}, uniformly chosen from Z in [{minval}, {maxval}]."
    )
    insertion_sort(real_array.copy())
    selection_sort(real_array.copy())
    heap_sort(real_array.copy())
    tim_sort(real_array.copy())
    radix_sort(real_array.copy())
    bubble_sort(real_array.copy())
    counting_sort(real_array.copy())

    # quick_sort(real_array, 0, len(real_array))
    # t0 = time.time()
    # merge_sort(real_array)
    # print(f"function 'merge_sort()' took {time.time()-t0:.9f} seconds.")


if __name__ == "__main__":
    tmain = time.time()
    main()
    print(f"Program took {time.time()-tmain:.3f} seconds.")
