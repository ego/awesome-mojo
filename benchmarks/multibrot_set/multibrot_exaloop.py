"""
python3 multibrot_exaloop.py
"""

WIDTH = 960
HEIGHT = 960
MAX_ITERS = 200

MIN_X = -2.0
MAX_X = 0.6
MIN_Y = -1.5
MAX_Y = 1.5


def scale(j, a, b):
    return a + (j / HEIGHT) * (b - a)


def compute_mandelbrot():
    t = [0 for _ in range(HEIGHT * WIDTH)]
    for i in range(HEIGHT):
        for j in range(WIDTH):
            c = complex(scale(j, MIN_X, MAX_X), scale(i, MIN_Y, MAX_Y))
            z = 0j
            iteration = 0
            while abs(z) <= 2 and iteration < MAX_ITERS:
                z = z * z + c  # Change this for different Multibrot sets (e.g., 2 for Mandelbrot)
                iteration += 1
            t[i * HEIGHT + j] = int(255 * iteration / MAX_ITERS)
    return t


def show_plot(vector):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import numpy as np

    WIDTH = 960
    HEIGHT = 960
    MAX_ITERS = 200

    SCALE = 10
    DPI = 64

    tensor = np.reshape(vector, (HEIGHT, WIDTH))
    numpy_array = np.zeros((HEIGHT, WIDTH), np.float64)

    for row in range(HEIGHT):
        for col in range(WIDTH):
            numpy_array.itemset((col, row), tensor[col][row])

    fig = plt.figure(1, [SCALE, SCALE * HEIGHT // WIDTH], DPI)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    light = colors.LightSource(315, 10, 0, 1, 1, 0)

    image = light.shade(numpy_array, plt.cm.hot, colors.PowerNorm(0.3), "hsv", 0, 0, 1.5)
    plt.imshow(image)
    plt.axis("off")
    plt.savefig("multibrot_exaloop.py.png")
    plt.show()


mandelbrot = compute_mandelbrot()
show_plot(mandelbrot)
