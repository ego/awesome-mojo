import timeit

SIZE = 100
MAX_ITERS = 100


def _fizz_buzz():  # Make it aka at compile-time.
  res = []
  for n in range(1, SIZE+1):
    if (n % 3 == 0) and (n % 5 == 0):
      s = "FizzBuzz"
    elif n % 3 == 0:
      s = "Fizz"
    elif n % 5 == 0:
      s = "Buzz"
    else:
      s = str(n)
    res.append(s)
  return res


DATA = _fizz_buzz()


def fizz_buzz():
  print("\n".join(DATA))


print(
  "Average execution time of Python func in sec",
  timeit.timeit(lambda: fizz_buzz(), number=MAX_ITERS),
)
