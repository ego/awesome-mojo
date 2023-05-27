from Intrinsics import external_call
from SIMD import SIMD, SI8
from DType import DType
from Vector import DynamicVector
from DType import DType
from Pointer import DTypePointer, Pointer

# Let's do something interesting - call libc function [gethostname](https://www.gnu.org/software/libc/manual/html_node/Host-Identification.html#index-gethostname)
# function has this interface `int gethostname (char *name, size_t size)`.
# For that we can use helper function [external_call](https://docs.modular.com/mojo/MojoStdlib/Intrinsics.html#external_call) from Intrinsics module or write own MLIR.

# We can use `from String import String` but for clarification we will use a full form.
# DynamicVector[SIMD[DType.si8, 1]] == DynamicVector[SI8] == String

# Compile time staff.
alias cArrayOfStrings = DynamicVector[SIMD[DType.si8, 1]]
alias capacity = 1024

var c_pointer_to_array_of_strings = DTypePointer[DType.si8](cArrayOfStrings(capacity).data)
var c_int_result = external_call["gethostname", Int, DTypePointer[DType.si8], Int](c_pointer_to_array_of_strings, capacity)
let mojo_string_result = String(c_pointer_to_array_of_strings.address)
print("C function gethostname result code:", c_int_result)
print("C function gethostname result value:", star_hostname(mojo_string_result))
# <img src="img/gethostname.png" height="200" />


@always_inline
fn star_hostname(hostname: String) -> String:
    # [Builtin Slice](https://docs.modular.com/mojo/MojoBuiltin/BuiltinSlice.html)
    # string slice[start:end:step]
    return hostname[0:-1:2]
