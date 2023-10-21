fn fibonacci_recursive(n: i64) -> i64 {
    if n < 2 {
        return n;
    }
    return fibonacci_recursive(n - 1) + fibonacci_recursive( n - 2);
}
fn main() {
    let _ = fibonacci_recursive(100);
}