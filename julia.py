import os
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL.Image as Image
from numba import jit, prange, config, vectorize
from tqdm import tqdm
import tempfile
import multiprocessing
import functools
import time

# Configure Numba to use all available cores
num_cores = multiprocessing.cpu_count()
config.NUMBA_NUM_THREADS = num_cores

x_res, y_res = 500, 500
xmin, xmax = -1.5, 1.5
width = xmax - xmin
ymin, ymax = -1.5, 1.5
height = ymax - ymin

z_abs_max = 10
max_iter = 256


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_julia_numba(
    x_res, y_res, xmin, xmax, ymin, ymax, z_abs_max, max_iter, c_real, c_imag
):
    """Compute Julia set using numba for massive speedup"""
    # Pre-compute values (outside loops)
    x_step = (xmax - xmin) / x_res
    y_step = (ymax - ymin) / y_res
    z_abs_max_squared = z_abs_max * z_abs_max
    escape_radius_squared = 4.0  # Bailout radius for cycle detection

    # Pre-allocate result array in row-major order for better memory access
    julia = np.empty((y_res, x_res), dtype=np.float32)

    # Compute row by row for better cache locality
    for iy in prange(y_res):
        zi = iy * y_step + ymin
        row = np.empty(x_res, dtype=np.float32)

        for ix in range(x_res):
            zr = ix * x_step + xmin

            # Use register variables for faster access
            z_real = zr
            z_imag = zi

            # Fast path: check if point is in Mandelbrot cardioid or period-2 bulb
            # This optimization speeds up computation for points known to be in the set
            q = (z_real - 0.25)*(z_real - 0.25) + z_imag*z_imag
            if q*(q + (z_real - 0.25)) <= 0.25*z_imag*z_imag:
                row[ix] = 0.0
                continue

            if (z_real + 1.0)*(z_real + 1.0) + z_imag*z_imag <= 0.0625:
                row[ix] = 0.0
                continue

            # Period checking variables for cycle detection
            period = 0
            z_real_0, z_imag_0 = 0.0, 0.0

            # Main iteration loop
            iteration = 0
            for i in range(max_iter):
                # Manually compute complex number operations
                z_real_squared = z_real * z_real
                z_imag_squared = z_imag * z_imag
                abs_squared = z_real_squared + z_imag_squared

                # Early bailout check
                if abs_squared > z_abs_max_squared:
                    break

                # Periodicity check - detect cycles
                if i == period and abs_squared < escape_radius_squared:
                    if abs(z_real - z_real_0) < 1e-10 and abs(z_imag - z_imag_0) < 1e-10:
                        # We detected a cycle, point is in the set
                        iteration = max_iter
                        break

                    # Save point for next periodicity check
                    z_real_0 = z_real
                    z_imag_0 = z_imag
                    period = period * 2 + 1

                # z = z² + c
                z_imag = 2.0 * z_real * z_imag + c_imag
                z_real = z_real_squared - z_imag_squared + c_real
                iteration = i + 1

            # Smooth coloring formula for better visual results
            if iteration < max_iter:
                # Apply logarithmic smooth coloring (safely)
                abs_squared = z_real*z_real + z_imag*z_imag
                if abs_squared > 0:  # Avoid log(0)
                    log_zn = np.log(abs_squared) / 2
                    nu = np.log(log_zn / np.log(2)) / np.log(2)
                    smooth_iter = iteration + 1 - nu
                    row[ix] = smooth_iter / max_iter
                else:
                    row[ix] = iteration / max_iter
            else:
                # Mark points in the set
                row[ix] = 0.0

        # Copy row to result array
        julia[iy] = row

    # Transpose back to expected output format
    return julia.T


def julia_set(c: complex):
    # Compute the Julia set using the optimized function with caching
    print(f"Calculating Julia set with {x_res}x{y_res} resolution...")
    start_time = time.time()

    # Ensure parameters are hashable types for caching
    julia = cached_compute_julia(
        int(x_res), int(y_res), float(xmin), float(xmax),
        float(ymin), float(ymax), float(z_abs_max), int(max_iter),
        float(c.real), float(c.imag)
    )

    elapsed = time.time() - start_time
    print(f"Calculation completed in {elapsed:.2f} seconds")

    # Create figure and display
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(julia, aspect="auto", cmap=cm.CMRmap)

    return julia


def zoom_main():
    global xmin, xmax, ymin, ymax, width, height
    n = int(input("Enter the number of zoom frames: "))
    d = float(input("Enter the duration of each frame (seconds): "))
    print("\nZoom direction options:")
    print("c: Center")
    print("n: North")
    print("s: South")
    print("e: East")
    print("w: West")
    dir = input("Enter zoom direction (c/n/s/e/w): ")
    if dir == "c":
        pass
    elif dir == "n":
        ymax = -0.5
        xmax = 0
        ymin = 0.5
        xmin = 1
    elif dir == "s":
        ymax = 0.5
        xmax = 1
        ymin = -0.5
        xmin = 0
    elif dir == "w":  # Changed from 'o' to 'w' for west
        ymax = 0
        xmax = -0.5
        ymin = 1
        xmin = 0.5
    elif dir == "e":
        ymax = 1
        xmax = 0.5
        ymin = 0
        xmin = -0.5
    else:
        pass
    width = xmax - xmin
    height = ymax - ymin

    # Create a temporary directory to store images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Pre-allocate a list to store julia sets
        julia_frames = []

        # Show progress bar
        start_time = time.time()
        for i in tqdm(range(n), desc="Creating zoom frames"):
            # Calculate julia set
            julia_frame = cached_compute_julia(
                int(x_res), int(y_res), float(xmin), float(xmax),
                float(ymin), float(ymax), float(z_abs_max), int(max_iter),
                float(c.real), float(c.imag)
            )
            julia_frames.append(julia_frame)

            # Create figure and save
            fig = plt.figure(frameon=False)
            fig.set_size_inches(5, 5)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(julia_frame, aspect="auto", cmap=cm.CMRmap)

            # Save to a temporary file
            temp_file = os.path.join(temp_dir, f"julia{str(i)}.png")
            plt.savefig(temp_file, dpi=1200)
            plt.close()

            # Update window boundaries
            if dir == "c":
                width = (xmax - xmin) / 1.1
                height = (ymax - ymin) / 1.1
                xmin = (xmax + xmin - width) / 2
                xmax = xmin + width
                ymin = (ymax + ymin - height) / 2
                ymax = ymin + height
            elif dir == "n":
                ymax += 0.03
                ymin -= 0.03
                xmax += 0.03
                xmin -= 0.03
                width = xmax - xmin
                height = ymax - ymin
            elif dir == "s":
                ymax -= 0.03
                ymin += 0.03
                xmax -= 0.03
                xmin += 0.03
                width = xmax - xmin
                height = ymax - ymin
            elif dir == "w":  # Changed from 'o' to 'w' for west
                ymax += 0.03
                ymin -= 0.03
                xmax += 0.03
                xmin -= 0.03
                width = xmax - xmin
                height = ymax - ymin
            elif dir == "e":
                ymax -= 0.03
                ymin += 0.03
                xmax -= 0.03
                xmin += 0.03
                width = xmax - xmin
                height = ymax - ymin

        elapsed = time.time() - start_time
        print(f"All frames generated in {elapsed:.2f} seconds")

        # Load all images at once and create the GIF
        print("Creating GIF...")
        images = [
            imageio.imread(os.path.join(temp_dir, f"julia{str(i)}.png"))
            for i in range(n)
        ]
        imageio.mimsave("julia_zoom.gif", images, duration=d)
        print(f"Zoom animation saved as julia_zoom.gif")


def gif(c):
    n = int(input("Enter the number of animation frames: "))
    d = float(input("Enter the duration of each frame (seconds): "))

    # Create a temporary directory to store images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate first set of images - increasing imaginary part
        print("Generating first set of images...")
        c_value = complex(c)
        start_time = time.time()
        
        # Maintain same view parameters for all frames
        for i in tqdm(range(n), desc="Generating first sequence"):
            # Calculate julia set using cached function
            julia_frame = cached_compute_julia(
                int(x_res), int(y_res), float(xmin), float(xmax), 
                float(ymin), float(ymax), float(z_abs_max), int(max_iter), 
                float(c_value.real), float(c_value.imag)
            )
            
            # Create figure and save
            fig = plt.figure(frameon=False)
            fig.set_size_inches(5, 5)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(julia_frame, aspect="auto", cmap=cm.CMRmap)
            
            plt.savefig(os.path.join(temp_dir, f"julia{str(i)}.png"), dpi=1200)
            c_value += 0.001j  # Increment the imaginary part
            plt.close()

        # Generate second set of images - decreasing imaginary part
        print("Generating second set of images...")
        for i in tqdm(range(n), desc="Generating second sequence"):
            c_value -= 0.001j  # Decrement the imaginary part
            
            # Calculate julia set using cached function
            julia_frame = cached_compute_julia(
                int(x_res), int(y_res), float(xmin), float(xmax), 
                float(ymin), float(ymax), float(z_abs_max), int(max_iter), 
                float(c_value.real), float(c_value.imag)
            )
            
            # Create figure and save
            fig = plt.figure(frameon=False)
            fig.set_size_inches(5, 5)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(julia_frame, aspect="auto", cmap=cm.CMRmap)
            
            plt.savefig(os.path.join(temp_dir, f"julia2_{str(i)}.png"), dpi=1200)
            plt.close()
        
        elapsed = time.time() - start_time
        print(f"All frames generated in {elapsed:.2f} seconds")

        # Load all images and create the GIF
        print("Creating GIF...")
        images = [
            imageio.imread(os.path.join(temp_dir, f"julia{str(i)}.png"))
            for i in range(n)
        ]
        images2 = [
            imageio.imread(os.path.join(temp_dir, f"julia2_{str(i)}.png"))
            for i in range(n)
        ]
        images.extend(images2)

        # Save the GIF
        imageio.mimsave("julia_animation.gif", images, duration=d)
        print(f"Animation saved as julia_animation.gif")


# Cache for Julia set calculations to avoid redundant computations
@functools.lru_cache(maxsize=16)
def cached_compute_julia(
    x_res, y_res, xmin, xmax, ymin, ymax, z_abs_max, max_iter, c_real, c_imag
):
    """Cached version of compute_julia_numba to avoid redundant calculations"""
    # Parameters are already converted to appropriate types in the julia_set function

    # Force Numba to compile on first run
    if not hasattr(cached_compute_julia, "first_run_complete"):
        print("First run: compiling Numba code...")
        cached_compute_julia.first_run_complete = True

    return compute_julia_numba(
        x_res, y_res, xmin, xmax, ymin, ymax, z_abs_max, max_iter, c_real, c_imag
    )

if __name__ == "__main__":
    # Print resolution options more clearly
    print("\n====== JULIA SET EXPLORER ======")
    print("Resolution options:")
    print("a: 100x100 (fastest - for testing)")
    print("b: 500x500 (fast - for exploration)")
    print("c: 1000x1000 (medium - good quality)")
    print("d: 2000x2000 (slow - high quality)")
    print("e: 4000x4000 (very slow - ultra quality)")
    print("f: custom resolution")
    print("================================")

    reso = input("Enter resolution option (a, b, c, d, e, f): ")
    if reso == "a":
        x_res, y_res = 100, 100
    elif reso == "b":
        x_res, y_res = 500, 500
    elif reso == "c":
        x_res, y_res = 1000, 1000
    elif reso == "d":
        x_res, y_res = 2000, 2000
    elif reso == "e":
        x_res, y_res = 4000, 4000
    elif reso == "f":
        x_res, y_res = (
            int(input("Entrer la résolution x : ")),
            int(input("Entrer la résolution y : ")),
        )

    choice = input("Do you want help choosing a constant? (y/n): ")
    if choice == "y":
        try:
            img = Image.open("Julia-Teppich.png", "r")
            img.show()
        except FileNotFoundError:
            print("Reference image not found. Continuing without visual help.")

    # Display c options more clearly
    print("\nConstant options:")
    print("y: Enter a custom complex constant")
    print("n: Use default (-0.8+0.16j)")
    print("r: Use random constant")

    c_choice = input("Choose constant option (y/n/r): ")
    if c_choice == "y":
        try:
            c = complex(input("Enter the complex constant c (e.g. -0.8+0.16j): "))
        except ValueError:
            print("Invalid input. Using default constant.")
            c = -0.8 + 0.16j
    elif c_choice == "n":
        c = -0.8 + 0.16j
    elif c_choice == "r":
        c = np.random.uniform(-1.5, 1.5) + np.random.uniform(-1.5, 1.5) * 1j
    else:
        print("Invalid option. Using default constant.")
        c = -0.8 + 0.16j

    print(f"Calculating Julia set for c = {c}...")
    julia_set(c)
    plt.show()

    # Display mode options more clearly
    print("\nMode options:")
    print("z: Create zoom animation")
    print("g: Create animation by varying constant c")
    print("s: Save current image")
    print("x: Exit program")

    mode = input("Enter mode (z/g/s/x): ")
    if mode == "z":
        zoom_main()
    elif mode == "g":
        gif(c)
    elif mode == "s":
        julia_set(c)
        resolution = input("Enter image resolution (1=low, 2=medium, 3=high): ")
        dpi = 600
        if resolution == "2":
            dpi = 1200
        elif resolution == "3":
            dpi = 2400
        plt.savefig("julia.png", dpi=dpi)
        print(f"Image saved as julia.png with {dpi} DPI")
    elif mode == "x":
        print("Exiting program.")
    else:
        print("Invalid mode selection. Exiting.")
