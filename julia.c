#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#define X_RES 500
#define Y_RES 500
#define XMIN -1.5
#define XMAX 1.5
#define YMIN -1.5
#define YMAX 1.5
#define Z_ABS_MAX 10
#define MAX_ITER 256

int main() {
    double complex c = -0.4 + 0.6 * I;
    unsigned char *image = (unsigned char *)malloc(X_RES * Y_RES * 3);
    if (!image) {
        fprintf(stderr, "Memory allocation failed\n");
        return 0;
    }

    double width = XMAX - XMIN;
    double height = YMAX - YMIN;

    for (int iy = 0; iy < Y_RES; iy++) {
        for (int ix = 0; ix < X_RES; ix++) {
            double complex z = ix / (float)X_RES * (XMAX - XMIN) + XMIN + (iy / (float)Y_RES * (YMAX - YMIN) + YMIN) * I;
            int iteration = 0;
            while (cabs(z) <= Z_ABS_MAX && iteration < MAX_ITER) {
                z = z * z + c;
                iteration++;
            }

            double iteration_ratio = (double)iteration / MAX_ITER;
            unsigned char color = (unsigned char)(iteration_ratio * 255);

            int pixel_index = (iy * X_RES + ix) * 3;
            image[pixel_index] = color;     // R
            image[pixel_index + 1] = color; // G
            image[pixel_index + 2] = color; // B
        }

        printf("Progression: %.2f%%\r", iy / (double)Y_RES * 100);
        fflush(stdout);
    }

    stbi_write_png("julia_set.png", X_RES, Y_RES, 3, image, X_RES * 3);
    free(image);

    printf("\nImage saved as julia_set.png\n");
    return 0;
}