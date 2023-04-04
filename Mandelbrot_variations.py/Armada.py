import os
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

x_res, y_res = 100, 100
xmin, xmax = -2, 2
width = xmax - xmin
ymin, ymax = -2, 2
height = ymax - ymin

z_abs_max = 10
max_iter = 200

def zoom(zoom_factor):
    global xmin, xmax, ymin, ymax, width, height
    xmin /= zoom_factor
    xmax /= zoom_factor
    ymin /= zoom_factor
    ymax /= zoom_factor
    width = xmax - xmin
    height = ymax - ymin

def armada(c: complex):
    fractal = np.zeros((x_res, y_res))
    for ix in range(x_res):
        for iy in range(y_res):
            z = complex(ix / x_res * width + xmin,
                        iy / y_res * height + ymin)
            iteration = 0
            while abs(z) <= z_abs_max and iteration < max_iter:
                z = (abs(z.real) + 1j * abs(z.imag)) ** 2 - c
                iteration += 1
            iteration_ratio = iteration / max_iter    
            fractal[ix, iy] = iteration_ratio
            print(f"Progression : {ix / x_res * 100:.2f}%", end="\r")
    fig, ax = plt.subplots()
    ax.imshow(fractal, interpolation='nearest', cmap=cm.gray)
    plt.axis('off')

armada(1.632 + 0.055j)
#plt.savefig("armada.png", dpi=1200)
plt.show()