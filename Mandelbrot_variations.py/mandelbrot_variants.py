import os
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

x_res, y_res = 1000, 1000
xmin, xmax = -3, 1
width = xmax - xmin
ymin, ymax = -2, 2
height = ymax - ymin

z_abs_max = 10
max_iter = 200

def mandelbrot():
    fractal = np.zeros((x_res, y_res))
    for ix in range(x_res):
        for iy in range(y_res):
            c = complex(ix / x_res * width + xmin,
                        iy / y_res * height + ymin)
            z = 0
            iteration = 0
            while abs(z) <= z_abs_max and iteration < max_iter:
                z = z**2 + c
                iteration += 1
            iteration_ratio = iteration / max_iter    
            fractal[ix, iy] = iteration_ratio
            print('progress: {:.2f}%'.format((ix * y_res + iy) / (x_res * y_res) * 100), end='\r')
    fig, ax = plt.subplots()
    ax.imshow(fractal, interpolation='nearest', cmap=cm.gray)
    plt.axis('off')

def zoom(center,factor):
    global xmin, xmax, ymin, ymax, width, height
    xmin = center.real - width / factor
    xmax = center.real + width / factor
    ymin = center.imag - height / factor
    ymax = center.imag + height / factor
    width = xmax - xmin
    height = ymax - ymin


mandelbrot()
plt.show()
zoom(complex(-0.75,0), 2)
mandelbrot()
plt.show()