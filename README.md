# Awesome Mojo

[Community driven Mojo-Lang portal](https://mojo-lang.dev)
[Official Mojo docs](https://docs.modular.com/mojo/)

## Binary search Python

```Python
%%python
import timeit
from typing import List, Union


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
    v = []
    for i in range(1000000):
        v.append(i)
    _ = python_binary_search(9999, v)


print(
    "Average execution time of func in sec",
    timeit.timeit(lambda: test_python_binary_search(), number=100),
)
```

## Binary search Mojo

```Python
from Benchmark import Benchmark
from Vector import InlinedFixedVector


fn mojo_binary_search(element: Int, array: InlinedFixedVector[1000000, Int]) -> Int:
    var start = 0
    var stop = len(array) - 1
    while start <= stop:
        let index = (start + stop) // 2
        let pivot = array[index]
        if pivot == element:
            return index
        elif pivot > element:
            stop = index - 1
        elif pivot < element:
            start = index + 1
    return -1


fn test_mojo_binary_search() -> F64:
    fn test_closure():
        var v = InlinedFixedVector[1000000, Int](1000000)
        for i in range(1000000):
            v.append(i)
        _ = mojo_binary_search(9999, v)

    return F64(Benchmark(0, 100).run[test_closure]()) / 1e9


print(
    "Average execution time of func in sec ",
    test_mojo_binary_search(),
)
```
