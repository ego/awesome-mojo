// Binary Search
// zig build-exe binary_search.zig && ./binary_search

const std = @import("std");


fn binarySearch(comptime T: type, arr: []const T, target: T) ?usize {
    var lo: usize = 0;
    var hi: usize = arr.len - 1;

    while (lo <= hi) {
        var mid: usize = (lo + hi) / 2;

        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    return null;
}


pub fn main() !void {
    const nums = [_]u8{1, 3, 5, 7, 9, 11, 13, 15};
    const target: u8 = 7;

    const found = binarySearch(u8, &nums, target);
    std.log.info("Target found at index {}.\n", .{found.?});
}
