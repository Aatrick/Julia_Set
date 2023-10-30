use image::{ImageBuffer, Rgb};
use num::Complex;

fn main() {
    let xmin = -1.5;
    let xmax = 1.5;
    let width = xmax - xmin;
    let ymin = -1.5;
    let ymax = 1.5;
    let height = ymax - ymin;
    let z_abs_max = 10.0;
    let max_iter = 256;

    let x_res = 800;
    let y_res = 800;

    let mut julia_set = ImageBuffer::new(x_res, y_res);

    let c = Complex::new(-0.4, 0.6);

    for (x, y, pixel) in julia_set.enumerate_pixels_mut() {
        let cx = x as f64 / x_res as f64 * width + xmin;
        let cy = y as f64 / y_res as f64 * height + ymin;
        let mut z = Complex::new(cx, cy);
        let mut i = 0;
        while i < max_iter && z.norm() <= z_abs_max {
            z = z * z + c;
            i += 1;
        }
        let iteration_ratio = i as f64 / max_iter as f64;
        let color = Rgb([
            (iteration_ratio * 255.0) as u8,
            (iteration_ratio * 153.0) as u8,
            (iteration_ratio * 204.0) as u8,
        ]);
        *pixel = color;
    }

    // Save the image to a file
    julia_set.save("julia_set.png").unwrap();

    // Open the image in the default image viewer
    let _ = std::process::Command::new("cmd")
        .args(&["/C", "start", "julia_set.png"])
        .output()
        .unwrap();
}