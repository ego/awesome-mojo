"""Implements basic binary search."""

from typing import List, Union
import timeit


SIZE = 1000000
MAX_ITERS = 100
COLLECTION = tuple(i for i in range(SIZE))  # Make it aka at compile-time.


def python_binary_search(element: int, array: List[int]) -> int:
    start = 0
    stop = len(array) - 1
    while start <= stop:
        index = (start + stop) // 2
        pivot = array[index]
        if pivot == element:
            return index
        elif pivot > element:
            stop = index - 1
        elif pivot < element:
            start = index + 1
    return -1


def test_python_binary_search():
    _ = python_binary_search(SIZE - 1, COLLECTION)


print(
    "Average execution time of func in sec",
    timeit.timeit(lambda: test_python_binary_search(), number=MAX_ITERS),
)
