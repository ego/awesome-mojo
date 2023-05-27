from Pointer import Pointer
from Memory import memset_zero
from SIMD import SIMD


@register_passable
struct Coord:
    var x: UI8
    var y: UI8


struct Coords:
    var data: Pointer[Coord]
    var length: Int

    fn __init__(inout self, length: Int) raises:  # keyword raises https://docs.modular.com/mojo/programming-manual.html#fn-definitions
        self.data = Pointer[Coord].alloc(length)
        memset_zero(self.data, length)
        self.length = length

    fn __getitem__(self, index: Int) raises -> Coord:
        if index > self.length - 1:
            raise Error("Trying to access index out of bounds")
        return self.data.load(index)

    fn __del__(owned self):
        self.data.free()

    fn store_coord(inout self, offset: Int, value: Coord):
        return self.data.store(offset, value)


var coords = Coords(5)
var second = coords[2]
print("second.x:", second.x)
second.x = 1
print("second.x = 1:", second.x)
print("pointer.data[2].x:", coords.data[2].x)
coords.store_coord(2, second)  # or coords.data.store(2, second)
print("second.x = 1:", second.x)
print("pointer.data[2].x:", coords.data[2].x)

"""
second.x: 0
second.x = 1: 1
pointer.data[2].x: 0
second.x = 1: 1
pointer.data[2].x: 1
"""
