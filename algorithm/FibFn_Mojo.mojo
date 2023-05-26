from Time import now


fn fibf(n: Int) -> Int:
    return n if n < 2 else fibf(n - 1) + fibf(n - 2)


def run_mojo_fn_benchmark():
    let t0 = now()
    let ans = fibf(40)
    let t1 = now()
    print("Computed fibf(40)", ans, F64(t1 - t0) / 1e9, "seconds")


run_mojo_fn_benchmark()
# Computed fibf(40) 102334155 0.41657813999999999 seconds


fn fibf_range(n: Int) -> Int:
    var a: Int = 0
    var b: Int = 1
    for _ in range(n):
        a = b
        b = a+b
    return a


def run_mojo_fibf_range_benchmark():
    let t0 = now()
    let ans = fibf_range(40)
    let t1 = now()
    print("Computed fibf_range(40)", ans, F64(t1 - t0) / 1e9, "seconds")


run_mojo_fibf_range_benchmark()
# Computed fibf_range(40) 549755813 3.7e-08 seconds
