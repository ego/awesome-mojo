// Binary search.
// v fmt -w binary_search.v
// v -prod binary_search.v && ./binary_search

fn binary_search(a []int, value int) int {
	mut low := 0
	mut high := a.len - 1
	for low <= high {
		mid := (low + high) / 2
		if a[mid] > value {
			high = mid - 1
		} else if a[mid] < value {
			low = mid + 1
		} else {
			return mid
		}
	}
	return -1
}

fn main() {
	search_list := [1, 2, 3, 5, 6, 7, 8, 9, 10]
	println(binary_search(search_list, 9))
}
