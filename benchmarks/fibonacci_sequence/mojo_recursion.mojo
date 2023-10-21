fn fibonacci_recursion(n: Int) -> Int:
    return n if n < 2 else fibonacci_recursion(n - 1) + fibonacci_recursion(n - 2)
fn main():
    _ = fibonacci_recursion(100)