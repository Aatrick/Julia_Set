import os
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

x_res, y_res = 100, 100
xmin, xmax = -1.5, 1.5
width = xmax - xmin
ymin, ymax = -1.5, 1.5
height = ymax - ymin


z_abs_max = 10
max_iter = 256

#c = -0.4 + 0.59j
#c = 0.39 + 0.1j
c=-0.8+0.16j

def julia_set(c: complex):
    julia = np.zeros((x_res, y_res))
    for ix in range(x_res):
        for iy in range(y_res):
            z = complex(ix / x_res * width + xmin,
                        iy / y_res * height + ymin)
            iteration = 0
            while abs(z) <= z_abs_max and iteration < max_iter:
                z = z**2 + c
                iteration += 1
            iteration_ratio = iteration / max_iter    
            julia[ix, iy] = iteration_ratio
            print(f"Progression : {ix / x_res * 100:.2f}%", end="\r")
    fig, ax = plt.subplots()
    ax.imshow(julia, interpolation='nearest', cmap=cm.CMRmap)
    plt.axis('off')

def zoom(zoom_factor):
    global xmin, xmax, ymin, ymax, width, height
    xmin /= zoom_factor
    xmax /= zoom_factor
    ymin /= zoom_factor
    ymax /= zoom_factor
    width = xmax - xmin
    height = ymax - ymin

def main():
    n=int(input("Entrer le nombre de zoom : "))
    for i in range(n+1):
        julia_set(c)
        plt.savefig(f"julia{str(i)}.png", dpi = 1200)
        zoom(1.1)
        plt.close()
    images = [imageio.imread(f"julia{str(i)}.png") for i in range(n+1)]
    imageio.mimsave("julia1.gif", images, duration=0.1)
    for i in range(n+1):
        os.remove(f"julia{str(i)}.png")

def gif():
    c = -0.4 + 0.59j
    n=int(input("Entrer le nombre d'images : "))
    for i in range(n+1):
        julia_set(c)
        plt.savefig(f"julia{str(i)}.png", dpi = 1200)
        c += 0 + 0.01j
        plt.close()
    images = [imageio.imread(f"julia{str(i)}.png") for i in range(n+1)]
    imageio.mimsave("julia2.gif", images, duration=0.1)
    for i in range(n+1):
        os.remove(f"julia{str(i)}.png")

def gif2():
    c = -0.4 + 0.59j
    for i in range(21):
        julia_set(c)
        plt.savefig(f"julia{str(i)}.png", dpi = 1200)
        c += 0 + 0.01j
        plt.close()
    for i in range(21):
        c+=0-0.01j
        julia_set(c)
        plt.savefig(f"julia2.{str(i)}.png", dpi = 1200)
        plt.close()
    images = [imageio.imread(f"julia{str(i)}.png") for i in range(21)]
    images2 = [imageio.imread(f"julia2.{str(i)}.png") for i in range(21)]
    images.extend(images2)
    imageio.mimsave("julia3.gif", images, duration=0.1)
    for i in range(21):
        os.remove(f"julia{str(i)}.png")
        os.remove(f"julia2.{str(i)}.png")
    


julia_set(c)
#plt.savefig("julia3.png", dpi = 1200)
plt.show()
#gif2()
