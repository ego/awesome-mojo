from complex import ComplexFloat64
from python import Python
from tensor import Tensor
from utils.index import Index

alias FloatType = DType.float64

alias WIDTH = 960
alias HEIGHT = 960
alias MAX_ITERS = 200

alias MIN_X = -2.0
alias MAX_X = 0.6
alias MIN_Y = -1.5
alias MAX_Y = 1.5


# Compute the number of steps to escape.
def multibrot_kernel(c: ComplexFloat64) -> Int:
    z = c
    for i in range(MAX_ITERS):
        z = z * z + c  # Change this for different Multibrot sets (e.g., 2 for Mandelbrot)
        if z.squared_norm() > 4:
            return i
    return MAX_ITERS


def compute_multibrot() -> Tensor[FloatType]:
    # create a matrix. Each element of the matrix corresponds to a pixel
    t = Tensor[FloatType](HEIGHT, WIDTH)

    dx = (MAX_X - MIN_X) / WIDTH
    dy = (MAX_Y - MIN_Y) / HEIGHT

    y = MIN_Y
    for row in range(HEIGHT):
        x = MIN_X
        for col in range(WIDTH):
            t[Index(row, col)] = multibrot_kernel(ComplexFloat64(x, y))
            x += dx
        y += dy
    return t


# def show_plot(tensor: Tensor[FloatType]):
#     alias scale = 10
#     alias dpi = 64

#     np = Python.import_module("numpy")
#     plt = Python.import_module("matplotlib.pyplot")
#     colors = Python.import_module("matplotlib.colors")

#     numpy_array = np.zeros((HEIGHT, WIDTH), np.float64)

#     for row in range(HEIGHT):
#         for col in range(WIDTH):
#             numpy_array.itemset((col, row), tensor[col, row])

#     fig = plt.figure(1, [scale, scale * HEIGHT // WIDTH], dpi)
#     ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
#     light = colors.LightSource(315, 10, 0, 1, 1, 0)

#     image = light.shade(numpy_array, plt.cm.hot, colors.PowerNorm(0.3), "hsv", 0, 0, 1.5)
#     plt.imshow(image)
#     plt.axis("off")
#     plt.savefig("multibrot.mojo.png")
#     plt.show()


def main():
    _ = compute_multibrot()
    # multibrot = compute_multibrot()
    # show_plot(multibrot)
