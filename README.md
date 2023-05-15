# Awesome Mojo

* [Community driven Mojo-Lang portal](https://mojo-lang.dev)
* [Official Mojo docs](https://docs.modular.com/mojo/)
* [Awesome Mojo](https://ego.systemdef.com/awesome-mojo/)
* [Awesome Mojo repository](http://ego.github.io/awesome-mojo)


## Binary search Python

```Python
%%python
import timeit
from typing import List, Union


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
```


## Binary search Mojo

```Python
from Benchmark import Benchmark
from Vector import DynamicVector


alias SIZE = 1000000
alias NUM_WARMUP = 0
alias MAX_ITERS = 100


fn mojo_binary_search(element: Int, array: DynamicVector[Int]) -> Int:
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


@parameter  # statement runs at compile-time.
fn get_collection() -> DynamicVector[Int]:
    var v = DynamicVector[Int](SIZE)
    for i in range(SIZE):
        v.push_back(i)
    return v


fn test_mojo_binary_search() -> F64:
    fn test_closure():
        _ = mojo_binary_search(SIZE - 1, get_collection())
    return F64(Benchmark(NUM_WARMUP, MAX_ITERS).run[test_closure]()) / 1e9


print(
    "Average execution time of func in sec ",
    test_mojo_binary_search(),
)
```
