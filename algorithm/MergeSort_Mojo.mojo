from Benchmark import Benchmark
from Vector import DynamicVector
from StaticTuple import StaticTuple
from Sort import sort

alias NUM_WARMUP = 0
alias MAX_ITERS = 100


fn merge(inout A: DynamicVector[Int], p: Int, q: Int, r: Int):
    let n1 = q - p + 1
    let n2 = r - q

    var L = DynamicVector[Int](n1)
    var R = DynamicVector[Int](n2)

    for i in range(n1):
        L[i] = A[p + i]
    for j in range(n2):
        R[j] = A[q + 1 + j]

    var i = 0
    var j = 0
    var k = p

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1
        k += 1

    while i < n1:
        A[k] = L[i]
        i += 1
        k += 1

    while j < n2:
        A[k] = R[j]
        j += 1
        k += 1


fn merge_sort(inout A: DynamicVector[Int], p: Int, r: Int):
    if p < r:
        let q = (p + r) // 2
        merge_sort(A, p, q)
        merge_sort(A, q + 1, r)
        merge(A, p, q, r)


@parameter
fn create_vertor() -> DynamicVector[Int]:
    let st = StaticTuple[MAX_ITERS, Int](14, 72, 50, 83, 18, 20, 13, 30, 17, 87, 94, 65, 24, 99, 70, 44, 5, 12, 74, 6, 32, 63, 91, 88, 43, 54, 27, 39, 64, 78, 29, 62, 58, 59, 61, 89, 2, 15, 41, 9, 93, 90, 23, 96, 73, 14, 8, 28, 11, 42, 77, 34, 52, 80, 57, 84, 21, 60, 66, 40, 7, 85, 47, 98, 97, 35, 82, 36, 49, 3, 68, 22, 67, 81, 56, 71, 4, 38, 69, 95, 16, 48, 1, 31, 75, 19, 10, 25, 79, 45, 76, 33, 53, 55, 46, 37, 26, 51, 92, 86)
    var v = DynamicVector[Int](st.__len__())
    for i in range(st.__len__()):
        v.push_back(st[i])
    return v


fn run_benchmark_merge_sort() -> F64:
    fn _closure():
        var A = create_vertor()
        merge_sort(A, 0, len(A)-1)
    return F64(Benchmark(NUM_WARMUP, MAX_ITERS).run[_closure]()) / 1e9

print(
    "Average execution time of Mojo🔥 `merge_sort` in sec",
    run_benchmark_merge_sort(),
)
# Average execution time of Mojo🔥 `merge_sort` in sec 1.1345999999999999e-05


fn run_benchmark_sort() -> F64:
    fn _closure():
        var A = create_vertor()
        sort(A)
    return F64(Benchmark(NUM_WARMUP, MAX_ITERS).run[_closure]()) / 1e9

print(
    "Average execution time of Mojo🔥 builtin `sort` in sec",
    run_benchmark_sort(),
)
# Average execution time of Mojo🔥 builtin `sort` in sec 2.988e-06


# Usage: merge_sort
# var A = create_vertor()
# merge_sort(A, 0, len(A)-1)
# print(len(A))
# print(A[0], A[99])
