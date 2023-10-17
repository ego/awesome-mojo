#%%python
import os
from time import time, perf_counter


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
    start_time = perf_counter()
    t0 = time()
    ans = fib_range(40)
    t1 = time()
    end_time = perf_counter()
    if t1 - t0 <= 0:
        res = end_time - start_time
        print(f'Computed fib_range(40) = {ans} in {res} seconds.')
    else:
        res = t1 - t0
        print(f'Computed fib_range(40) = {ans} in {res:f} seconds.')


run_python_fib_range_benchmark()
# Python: Computed fib_range(40) = 102334155 in 4.5299530029296875e-06 seconds.
# Codon:  Computed fib_range(40) = 102334155 in 2.07685e-07 seconds.
