import os
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL.Image as Image
import sys

x_res, y_res = 500, 500
xmin, xmax = -1.5, 1.5
width = xmax - xmin
ymin, ymax = -1.5, 1.5
height = ymax - ymin

z_abs_max = 10
max_iter = 256

fig = plt.figure(frameon=False)
fig.set_size_inches(width,height)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

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
    ax.imshow(julia, aspect='auto', cmap=cm.CMRmap)
    julia_name = f"julia_{c.real:.2f}_{c.imag:.2f}.png"
    plt.savefig(julia_name, dpi = 1200)

def mandelbrot_set():
    mandelbrot = np.zeros((x_res, y_res))
    for ix in range(x_res):
        for iy in range(y_res):
            c = complex(ix / x_res * width + xmin, #mandelbrot
                        iy / y_res * height + ymin) #mandelbrot
            z = 0 #mandelbrot
            iteration = 0
            while abs(z) <= z_abs_max and iteration < max_iter:
                z = z**2 + c
                iteration += 1
            iteration_ratio = iteration / max_iter    
            mandelbrot[ix, iy] = iteration_ratio
            print(f"Progression : {ix / x_res * 100:.2f}%", end="\r")
    ax.imshow(mandelbrot, aspect='auto', cmap=cm.CMRmap)
    plt.savefig("mandelbrot.png", dpi = 1200)

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
        plt.savefig(f"julia{str(i)}.png", dpi = 1200)
        c += 0 + 0.001j
        plt.close()
    for i in range(n+1):
        c += 0 - 0.001j
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
    type=input("Entrer le type de fractale (julia, mandelbrot) : ")
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

    if type == "j":
        choice=input("Voulez-vous de l'aide (y/n) : ")
        if choice == "y":
            try:
                img=Image.open("Julia-Teppich.png", "r")
                img.show()
            except (FileNotFoundError, IOError):
                print("Fichier non trouvé")
                pass
        else :
            pass

        c_choice=input("Voulez-vous choisir la constante c (y/n/r) : ")
        if c_choice == "y":
            c=complex(input("Entrer la constante c : "))
        if c_choice == "n":
            c=-0.4 + 0.59j
        if c_choice == "r":
            c=np.random.uniform(-1.5,1.5)+np.random.uniform(-1.5,1.5)*1j
        julia_set(c)
        plt.show()

        mode=input("Entrer le mode (zoom, gif, s) : ")
        if mode == "z":
            zoom_main()
            vid=Image.open("julia1.gif")
        elif mode == "g":
            gif(c)
            vid=Image.open("julia.gif")
        elif mode == "s":
            plt.savefig("julia.png", dpi = 1200)
        else:
            pass

    elif type == "m":
        mandelbrot_set()
        plt.show()
    else:
        print("Erreur")
        sys.exit(1)
