from Pointer import DTypePointer
from DType import DType
from Memory import memset_zero
from SIMD import SIMD


struct Matrix:
    var data: DTypePointer[DType.ui8]

    fn __init__(inout self):
        "Initialize the struct and set everything to zero"
        self.data = DTypePointer[DType.ui8].alloc(64)
        memset_zero(self.data, 64)

    # This is what will run when the object goes out of scope
    fn __del__(owned self):
        return self.data.free()

    # This allows you to use let x = obj[1]
    fn __getitem__(self, row: Int) -> SIMD[DType.ui8, 8]:
        return self.data.simd_load[8](row * 8)

    # This allows you to use obj[1] = SIMD[DType.ui8]()
    fn __setitem__(self, row: Int, data: SIMD[DType.ui8, 8]):
        return self.data.simd_store[8](row * 8, data)

    fn print_all(self):
        print("--------matrix--------")
        for i in range(8):
            print(self[i])


let matrix = Matrix()
matrix.print_all()

for i in range(8):
    matrix[i][0] = 9
    matrix[i][7] = 9
matrix.print_all()


var fourth_row = matrix[3]
print("\nforth row:", fourth_row)
fourth_row *= 2
print("modified:", fourth_row, "\n")
matrix[0] = fourth_row
matrix.print_all()
