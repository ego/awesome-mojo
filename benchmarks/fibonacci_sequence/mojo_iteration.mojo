fn fibonacci_iteration(n: Int) -> Int:
    var a: Int = 0
    var b: Int = 1
    for _ in range(n):
        a = b
        b = a+b
    return a
fn main():
    _ = fibonacci_iteration(100)