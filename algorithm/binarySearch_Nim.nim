# Binary search.
# nim c -r --verbosity:0 binarySearch.nim

proc binarySearch[T](a: openArray[T], key: T): int =
  var b = len(a)
  while result < b:
    var mid = (result + b) div 2
    if a[mid] < key: result = mid + 1
    else: b = mid
  if result >= len(a) or a[result] != key: result = -1


let res = @[2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,25,27,30]
echo binarySearch(res, 10)
