from String import String
from Benchmark import Benchmark

alias SIZE = 100
alias NUM_WARMUP = 0
alias MAX_ITERS = 100


@parameter  # statement runs at compile-time.
fn _fizz_buzz() -> String:
    var res: String = ""
    for n in range(1, SIZE+1):
      if (n % 3 == 0) and (n % 5 == 0):
        res += "FizzBuzz"
      elif n % 3 == 0:
        res += "Fizz"
      elif n % 5 == 0:
        res += "Buzz"
      else:
        res += String(n)
      res += "\n"
    return res


fn fizz_buzz():
    print(_fizz_buzz())

fn run_benchmark() -> F64:
    fn _closure():
        _ = fizz_buzz()
    return F64(Benchmark(NUM_WARMUP, MAX_ITERS).run[_closure]()) / 1e9


print(
    "Average execution time of func in sec ",
    run_benchmark(),
)
