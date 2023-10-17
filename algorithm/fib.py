def fib_range(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a+b
    return a


fib_range(10000)


# brew install hyperfine
# hyperfine --warmup 10 -r 100 --time-unit=microsecond 'python3 algorithm/fib.py'
# python3 -m compileall algorithm/fib.py
# hyperfine --warmup 10 -r 100 --time-unit=microsecond 'python3 algorithm/__pycache__/fib.cpython-311.pyc' --export-markdown=algorithm/fib.cpython-311.pyc.md
