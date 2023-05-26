#%%python
from time import time


def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)


def run_python_benchmark():
    t0 = time()
    ans = fib(40)
    t1 = time()
    print(f'Computed fib(40) = {ans} in {t1 - t0} seconds.')


run_python_benchmark()
# Computed fib(40) = 102334155 in 21.669286727905273 seconds.


def fib_range(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a+b
    return a


def run_python_fib_range_benchmark():
    t0 = time()
    ans = fib_range(40)
    t1 = time()
    print(f'Computed fib_range(40) = {ans} in {t1 - t0} seconds.')


run_python_fib_range_benchmark()
# Computed fib_range(40) = 102334155 in 4.5299530029296875e-06 seconds.
