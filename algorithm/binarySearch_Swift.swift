// Binary search, find index of `elem` in items.

func binarySearch(items: [Int], elem: Int) -> Int {
    var low = 0
    var high = items.count - 1
    var mid = 0
    while low <= high {
        mid = Int((high + low) / 2)
        if items[mid] < elem {
            low = mid + 1
        } else if items[mid] > elem {
            high = mid - 1
        } else {
            return mid
        }
    }
    return -1
}

let items = [1, 2, 3, 4, 0].sorted()
let res = binarySearch(items: items, elem: 4)
print(res)
