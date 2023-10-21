fn fibonacci_iteration(n: usize) -> usize {
    let mut a = 0;
    let mut b = 1;
    for _ in 1..n {
        let old = a;
        a = b;
        b += old;
    }
    b
}
fn main() {
    let _ = fibonacci_iteration(100);
}