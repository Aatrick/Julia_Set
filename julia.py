import os
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL.Image as Image

x_res, y_res = 500, 500
xmin, xmax = -1.5, 1.5
width = xmax - xmin
ymin, ymax = -1.5, 1.5
height = ymax - ymin

z_abs_max = 10
max_iter = 256

def julia_set(c: complex):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(width,height)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
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
    ax.imshow(julia, aspect='auto', cmap=cm.CMRmap)

def zoom(zoom_factor):
    global xmin, xmax, ymin, ymax, width, height
    xmin /= zoom_factor
    xmax /= zoom_factor
    ymin /= zoom_factor
    ymax /= zoom_factor
    width = xmax - xmin
    height = ymax - ymin

def zoom_main():
    n=int(input("Entrer le nombre de zoom : "))
    d=float(input("Entrer la durée de chaque image : "))
    for i in range(n+1):
        julia_set(c)
        plt.savefig(f"julia{str(i)}.png", dpi = 1200)
        zoom(1.1)
        plt.close()
    images = [imageio.imread(f"julia{str(i)}.png") for i in range(n+1)]
    imageio.mimsave("julia1.gif", images, duration=d)
    for i in range(n+1):
        os.remove(f"julia{str(i)}.png")

def gif(c):
    n=int(input("Entrer le nombre d'images : "))
    d=float(input("Entrer la durée de chaque image : "))
    for i in range(n+1):
        julia_set(c)
        plt.savefig(f"julia{str(i)}.png", dpi = 1200)
        c += 0 + 0.001j
        plt.close()
    for i in range(n+1):
        c += 0 - 0.001j
        julia_set(c)
        plt.savefig(f"julia2.{str(i)}.png", dpi = 1200)
        plt.close()
    images = [imageio.imread(f"julia{str(i)}.png") for i in range(n+1)]
    images2 = [imageio.imread(f"julia2.{str(i)}.png") for i in range(n+1)]
    images.extend(images2)
    imageio.mimsave("julia.gif", images, duration=d)
    for i in range(n+1):
        os.remove(f"julia{str(i)}.png")
        os.remove(f"julia2.{str(i)}.png")


if __name__ == "__main__":
    reso=input("Entrer la résolution (100, 500, 1000, 2000, else) : ")
    if reso == "a":
        x_res, y_res = 100, 100
    if reso == "b":
        x_res, y_res = 500, 500
    elif reso == "c":
        x_res, y_res = 1000, 1000
    elif reso == "d":
        x_res, y_res = 2000, 2000
    elif reso == "e":
        x_res, y_res = int(input("Entrer la résolution x : ")), int(input("Entrer la résolution y : "))

    choice=input("Voulez-vous de l'aide (y/n) : ")
    if choice == "y":
        img=Image.open("Julia-Teppich.png", "r")
        img.show()
    else :
        pass

    c_choice=input("Voulez-vous choisir la constante c (y/n/r) : ")
    if c_choice == "y":
        c=complex(input("Entrer la constante c : "))
    if c_choice == "n":
        c=-0.8+0.16j
    if c_choice == "r":
        c=np.random.uniform(-1.5,1.5)+np.random.uniform(-1.5,1.5)*1j
    julia_set(c)
    plt.show()
    mode=input("Entrer le mode (zoom, gif, save) : ")
    if mode == "z":
        zoom_main()
    elif mode == "g":
        gif(c)
    elif mode == "s":
        julia_set(c)
        plt.savefig("julia.png", dpi = 1200)
    else:
        pass