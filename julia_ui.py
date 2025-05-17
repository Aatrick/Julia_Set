#!/usr/bin/env python3
# filepath: c:\Users\Aatricks\Documents\Julia_Set\julia_ui.py

import os
import sys
import tempfile

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pyqtgraph as pg
from joblib import Memory
from numba import jit, prange
from PyQt5.QtCore import QMutex, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# Set up caching
cachedir = tempfile.gettempdir()
memory = Memory(cachedir, verbose=0)

# Constants
DEFAULT_RESOLUTION = (500, 500)
DEFAULT_BOUNDS = (-1.5, 1.5, -1.5, 1.5)
DEFAULT_MAX_ITER = 256
DEFAULT_C = (-0.8, 0.16)
DEFAULT_POWER = 2
DEFAULT_Z_ABS_MAX = 10
COLORMAPS = [
    "CMRmap",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "hot",
    "cool",
    "jet",
    "rainbow",
    "twilight",
    "hsv",
]


class JuliaCalculatorThread(QThread):
    """Thread for computing the Julia set in the background"""

    result_ready = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, x_res=500, y_res=500, 
                 xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5,
                 z_abs_max=10, max_iter=256, c_real=-0.8, c_imag=0.156,
                 power=2, formula_type="standard"):
        super().__init__()
        self.mutex = QMutex()
        self.c_real = c_real
        self.c_imag = c_imag
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.x_res = x_res
        self.y_res = y_res
        self.max_iter = max_iter
        self.z_abs_max = z_abs_max
        self.power = power
        self.formula_type = formula_type
        self.running = False

    def update_params(self, x_res, y_res, xmin, xmax, ymin, ymax, 
                     z_abs_max, max_iter, c_real, c_imag,
                     power=2, formula_type="standard"):
        """Update parameters for the calculation"""
        self.mutex.lock()
        self.c_real = c_real
        self.c_imag = c_imag
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.x_res = x_res
        self.y_res = y_res
        self.max_iter = max_iter
        self.z_abs_max = z_abs_max
        self.power = power
        self.formula_type = formula_type
        self.mutex.unlock()

    def stop(self):
        """Stop the calculation thread"""
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()
        
        # Wait for the thread to finish
        if self.isRunning():
            self.wait()  # Wait for thread to finish

    def run(self):
        """Run the calculation in a separate thread"""
        # Copy parameters safely
        self.mutex.lock()
        x_res = self.x_res
        y_res = self.y_res
        xmin = self.xmin
        xmax = self.xmax
        ymin = self.ymin
        ymax = self.ymax
        z_abs_max = self.z_abs_max
        max_iter = self.max_iter
        c_real = self.c_real
        c_imag = self.c_imag
        power = self.power
        formula_type = self.formula_type
        self.running = True
        self.mutex.unlock()
        
        # Check if we should stop before heavy calculation
        if not self.running:
            return
            
        try:
            # Calculate the Julia set
            result = compute_julia_numba(
                x_res, 
                y_res,
                xmin, 
                xmax, 
                ymin, 
                ymax,
                z_abs_max, 
                max_iter,
                c_real, 
                c_imag,
                power,
                formula_type
            )
            
            # Check if we've been asked to stop
            self.mutex.lock()
            still_running = self.running
            self.mutex.unlock()
            
            if still_running:
                # Emit the result signal
                self.result_ready.emit(result)
        except Exception as e:
            print(f"Error in calculation thread: {e}")
            self.mutex.lock()
            still_running = self.running
            self.mutex.unlock()
            
            if still_running:
                self.error.emit(str(e))


@jit(nopython=True, parallel=True)
def compute_julia_numba(
    x_res,
    y_res,
    xmin,
    xmax,
    ymin,
    ymax,
    z_abs_max,
    max_iter,
    c_real,
    c_imag,
    power=2,
    formula_type="standard",
):
    """Compute Julia set with different formula options"""
    julia = np.zeros((x_res, y_res), dtype=np.float64)

    # Calculate steps once to avoid division in the loop
    x_step = (xmax - xmin) / x_res
    y_step = (ymax - ymin) / y_res

    for ix in prange(x_res):
        for iy in range(y_res):
            zr = ix * x_step + xmin
            zi = iy * y_step + ymin

            iteration = 0

            if formula_type == "standard":
                # Standard Julia set
                while (
                    zr * zr + zi * zi <= z_abs_max * z_abs_max
                ) and iteration < max_iter:
                    if power == 2:
                        # z = z^2 + c
                        zr_temp = zr * zr - zi * zi + c_real
                        zi = 2 * zr * zi + c_imag
                        zr = zr_temp
                    elif power == 3:
                        # z = z^3 + c
                        zr_temp = zr * zr * zr - 3 * zr * zi * zi + c_real
                        zi = 3 * zr * zr * zi - zi * zi * zi + c_imag
                        zr = zr_temp
                    else:
                        # Generic power (slower)
                        r = np.sqrt(zr * zr + zi * zi)
                        theta = np.arctan2(zi, zr)
                        rn = r**power
                        theta_n = theta * power
                        zr = rn * np.cos(theta_n) + c_real
                        zi = rn * np.sin(theta_n) + c_imag

                    iteration += 1

            elif formula_type == "burning_ship":
                # Burning Ship Julia variant
                while (
                    zr * zr + zi * zi <= z_abs_max * z_abs_max
                ) and iteration < max_iter:
                    zr, zi = abs(zr), abs(zi)
                    zr_temp = zr * zr - zi * zi + c_real
                    zi = 2 * zr * zi + c_imag
                    zr = zr_temp
                    iteration += 1

            elif formula_type == "mandelbar":
                # Mandelbar/Tricorn Julia variant
                while (
                    zr * zr + zi * zi <= z_abs_max * z_abs_max
                ) and iteration < max_iter:
                    zr_temp = zr * zr - zi * zi + c_real
                    zi = -2 * zr * zi + c_imag  # Note the negative sign
                    zr = zr_temp
                    iteration += 1

            # Smooth coloring formula
            if iteration < max_iter:
                # Add smoothing
                zn_abs_squared = zr * zr + zi * zi
                iteration = (
                    iteration + 1 - np.log(np.log(zn_abs_squared)) / np.log(power)
                )
                if iteration < 0:
                    iteration = 0

            # Normalize iteration count
            julia[ix, iy] = iteration / max_iter

    return julia


class JuliaSetUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_julia_calculation()
        
        # Connect aboutToQuit signal to clean up resources
        QApplication.instance().aboutToQuit.connect(self.cleanup_resources)
        
    def closeEvent(self, event):
        """Handle window close event"""
        # Make sure calculation thread is stopped before closing
        if hasattr(self, 'julia_thread') and self.julia_thread is not None:
            self.julia_thread.stop()
            # wait() is called inside stop()
        
        # Stop the calculation timer if active
        if hasattr(self, 'calculation_timer') and self.calculation_timer.isActive():
            self.calculation_timer.stop()
            
        event.accept()
        
    def init_ui(self):
        # Main window setup
        self.setWindowTitle("Interactive Julia Set Explorer")
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Left panel for controls
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(350)

        # Right panel for fractal display
        self.display_widget = pg.GraphicsLayoutWidget()
        self.view_box = self.display_widget.addViewBox()
        self.view_box.setAspectLocked(True)
        self.img_item = pg.ImageItem(border="w")
        self.view_box.addItem(self.img_item)

        # Add panels to main layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.display_widget)

        # Complex number parameters
        c_group = QGroupBox("Complex Parameter c")
        c_layout = QGridLayout()

        # C-real parameter slider
        c_layout.addWidget(QLabel("Real part:"), 0, 0)
        self.c_real_label = QLabel(f"{DEFAULT_C[0]:.3f}")
        c_layout.addWidget(self.c_real_label, 0, 2)
        self.c_real_slider = QSlider(Qt.Horizontal)
        self.c_real_slider.setRange(-200, 200)
        self.c_real_slider.setValue(int(DEFAULT_C[0] * 100))
        self.c_real_slider.valueChanged.connect(
            lambda: self.update_param_and_recalculate('c_real', self.c_real_slider.value() / 100.0)
        )
        c_layout.addWidget(self.c_real_slider, 1, 0, 1, 3)

        # C-imaginary parameter slider
        c_layout.addWidget(QLabel("Imaginary part:"), 2, 0)
        self.c_imag_label = QLabel(f"{DEFAULT_C[1]:.3f}")
        c_layout.addWidget(self.c_imag_label, 2, 2)
        self.c_imag_slider = QSlider(Qt.Horizontal)
        self.c_imag_slider.setRange(-200, 200)
        self.c_imag_slider.setValue(int(DEFAULT_C[1] * 100))
        self.c_imag_slider.valueChanged.connect(
            lambda: self.update_param_and_recalculate('c_imag', self.c_imag_slider.value() / 100.0)
        )
        c_layout.addWidget(self.c_imag_slider, 3, 0, 1, 3)

        c_group.setLayout(c_layout)
        control_layout.addWidget(c_group)

        # Zoom and position controls
        zoom_group = QGroupBox("Zoom & Position")
        zoom_layout = QGridLayout()

        # Zoom level
        zoom_layout.addWidget(QLabel("Zoom:"), 0, 0)
        self.zoom_spinbox = QDoubleSpinBox()
        self.zoom_spinbox.setRange(0.1, 1000)
        self.zoom_spinbox.setSingleStep(0.1)
        self.zoom_spinbox.setValue(1)
        self.zoom_spinbox.valueChanged.connect(self.update_zoom)
        zoom_layout.addWidget(self.zoom_spinbox, 0, 1)

        # Reset viewport button
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.reset_view)
        zoom_layout.addWidget(self.reset_view_button, 0, 2)

        # Center X
        zoom_layout.addWidget(QLabel("Center X:"), 1, 0)
        self.center_x_spinbox = QDoubleSpinBox()
        self.center_x_spinbox.setRange(-10, 10)
        self.center_x_spinbox.setSingleStep(0.1)
        self.center_x_spinbox.setValue(0)
        self.center_x_spinbox.valueChanged.connect(self.update_center)
        zoom_layout.addWidget(self.center_x_spinbox, 1, 1, 1, 2)

        # Center Y
        zoom_layout.addWidget(QLabel("Center Y:"), 2, 0)
        self.center_y_spinbox = QDoubleSpinBox()
        self.center_y_spinbox.setRange(-10, 10)
        self.center_y_spinbox.setSingleStep(0.1)
        self.center_y_spinbox.setValue(0)
        self.center_y_spinbox.valueChanged.connect(self.update_center)
        zoom_layout.addWidget(self.center_y_spinbox, 2, 1, 1, 2)

        zoom_group.setLayout(zoom_layout)
        control_layout.addWidget(zoom_group)

        # Equation parameters
        equation_group = QGroupBox("Equation Parameters")
        equation_layout = QGridLayout()

        # Max iterations
        equation_layout.addWidget(QLabel("Max Iterations:"), 0, 0)
        self.max_iter_spinbox = QSpinBox()
        self.max_iter_spinbox.setRange(10, 10000)
        self.max_iter_spinbox.setSingleStep(10)
        self.max_iter_spinbox.setValue(DEFAULT_MAX_ITER)
        self.max_iter_spinbox.valueChanged.connect(
            lambda value: self.update_param_and_recalculate('max_iter', value)
        )
        equation_layout.addWidget(self.max_iter_spinbox, 0, 1, 1, 2)

        # Power exponent
        equation_layout.addWidget(QLabel("Power:"), 1, 0)
        self.power_spinbox = QSpinBox()
        self.power_spinbox.setRange(2, 10)
        self.power_spinbox.setValue(DEFAULT_POWER)
        self.power_spinbox.valueChanged.connect(
            lambda value: self.update_param_and_recalculate('power', value)
        )
        equation_layout.addWidget(self.power_spinbox, 1, 1, 1, 2)

        # Formula type
        equation_layout.addWidget(QLabel("Formula:"), 2, 0)
        self.formula_combobox = QComboBox()
        self.formula_combobox.addItems(["Standard", "Burning Ship", "Tricorn"])
        self.formula_combobox.currentIndexChanged.connect(
            lambda index: self.update_param_and_recalculate('formula_type', index)
        )
        equation_layout.addWidget(self.formula_combobox, 2, 1, 1, 2)

        equation_group.setLayout(equation_layout)
        control_layout.addWidget(equation_group)

        # Display settings
        display_group = QGroupBox("Display Settings")
        display_layout = QGridLayout()

        # Resolution
        display_layout.addWidget(QLabel("Resolution:"), 0, 0)
        self.res_combobox = QComboBox()
        self.res_combobox.addItems(["100x100", "500x500", "1000x1000", "2000x2000"])
        self.res_combobox.setCurrentIndex(1)  # Default to 500x500
        self.res_combobox.currentIndexChanged.connect(self.update_resolution)
        display_layout.addWidget(self.res_combobox, 0, 1, 1, 2)

        # Colormap
        display_layout.addWidget(QLabel("Colormap:"), 1, 0)
        self.colormap_combobox = QComboBox()
        self.colormap_combobox.addItems(COLORMAPS)
        self.colormap_combobox.currentIndexChanged.connect(self.update_display)
        display_layout.addWidget(self.colormap_combobox, 1, 1, 1, 2)

        # Invert colors checkbox
        self.invert_colors_checkbox = QCheckBox("Invert Colors")
        self.invert_colors_checkbox.stateChanged.connect(self.update_display)
        display_layout.addWidget(self.invert_colors_checkbox, 2, 0, 1, 3)

        # Add a cache checkbox
        self.enable_cache_checkbox = QCheckBox("Enable Calculation Cache")
        self.enable_cache_checkbox.setChecked(True)
        display_layout.addWidget(self.enable_cache_checkbox, 3, 0, 1, 3)

        display_group.setLayout(display_layout)
        control_layout.addWidget(display_group)

        # Save functionality
        save_group = QGroupBox("Save & Export")
        save_layout = QVBoxLayout()

        self.save_image_button = QPushButton("Save Current Image")
        self.save_image_button.clicked.connect(self.save_image)
        save_layout.addWidget(self.save_image_button)

        self.save_animation_button = QPushButton("Create Animation")
        self.save_animation_button.clicked.connect(self.save_animation)
        save_layout.addWidget(self.save_animation_button)

        save_group.setLayout(save_layout)
        control_layout.addWidget(save_group)

        # Status information
        self.status_label = QLabel("Ready")
        control_layout.addWidget(self.status_label)

        # Add stretch to push controls to the top
        control_layout.addStretch(1)

    def setup_julia_calculation(self):
        """Setup calculation thread and cache"""
        # Store current parameters
        self.c_real = DEFAULT_C[0]
        self.c_imag = DEFAULT_C[1]
        self.xmin, self.xmax, self.ymin, self.ymax = DEFAULT_BOUNDS
        self.resolution = DEFAULT_RESOLUTION
        self.max_iter = DEFAULT_MAX_ITER
        self.power = DEFAULT_POWER
        self.z_abs_max = DEFAULT_Z_ABS_MAX
        self.formula_type = "standard"
        
        # Store the current result
        self.current_result = None
        
        # Set up a calculation timer for debouncing slider movements
        self.calculation_timer = QTimer()
        self.calculation_timer.setSingleShot(True)
        self.calculation_timer.timeout.connect(self.trigger_calculation)
        
        # Initialize calculation thread
        self.julia_thread = JuliaCalculatorThread()
        self.julia_thread.result_ready.connect(self.update_display_with_result)
        self.julia_thread.error.connect(lambda msg: print(f"Calculation error: {msg}"))
        
        # We'll trigger the initial calculation after the setup is complete
        QTimer.singleShot(100, self.trigger_calculation)
        
    def cleanup_resources(self):
        """Clean up resources before application exit"""
        # Stop calculation thread
        if hasattr(self, 'julia_thread') and self.julia_thread is not None:
            self.julia_thread.stop()
            # Thread.wait() is called in the stop() method
            
        # Stop timers
        if hasattr(self, 'calculation_timer') and self.calculation_timer.isActive():
            self.calculation_timer.stop()

    def update_param_and_recalculate(self, param_name=None, value=None):
        """Update parameters and trigger recalculation"""
        # Update the parameter value if provided
        if param_name and value is not None:
            if param_name == 'c_real':
                self.c_real = value
                self.c_real_label.setText(f"{value:.3f}")
            elif param_name == 'c_imag':
                self.c_imag = value
                self.c_imag_label.setText(f"{value:.3f}")
            elif param_name == 'max_iter':
                self.max_iter = int(value)
                self.max_iter_spinbox.setValue(int(value))
            elif param_name == 'power':
                self.power = int(value)
                self.power_spinbox.setValue(int(value))
            elif param_name == 'formula_type':
                formula_idx = int(value)
                self.formula_combobox.setCurrentIndex(formula_idx)
                if formula_idx == 0:
                    self.formula_type = "standard"
                elif formula_idx == 1:
                    self.formula_type = "burning_ship"
                else:
                    self.formula_type = "mandelbar"
        else:
            # Update all parameters from UI controls if no specific parameter was given
            self.c_real = self.c_real_slider.value() / 100.0
            self.c_imag = self.c_imag_slider.value() / 100.0
            self.c_real_label.setText(f"{self.c_real:.3f}")
            self.c_imag_label.setText(f"{self.c_imag:.3f}")
            
            self.max_iter = self.max_iter_spinbox.value()
            self.power = self.power_spinbox.value()
            
            formula_idx = self.formula_combobox.currentIndex()
            if formula_idx == 0:
                self.formula_type = "standard"
            elif formula_idx == 1:
                self.formula_type = "burning_ship"
            else:
                self.formula_type = "mandelbar"
            
        # Update status
        self.status_label.setText(f"Calculating: c = {self.c_real} + {self.c_imag}i")
        
        # First stop any existing calculation
        if hasattr(self, 'julia_thread') and self.julia_thread is not None and self.julia_thread.isRunning():
            self.julia_thread.stop()
        
        # Use debouncing for slider movements
        if self.c_real_slider.isSliderDown() or self.c_imag_slider.isSliderDown():
            # When slider is being dragged, use a timer to reduce calculation frequency
            if hasattr(self, 'calculation_timer'):
                self.calculation_timer.stop()
            self.calculation_timer.start(30)  # Short delay for smoothness
        else:
            # When slider is released or other controls changed, calculate immediately
            self.trigger_calculation()

    def trigger_calculation(self):
        """Start the actual calculation"""
        # Stop any existing calculation
        if hasattr(self, 'julia_thread') and self.julia_thread is not None and self.julia_thread.isRunning():
            self.julia_thread.stop()
            
        # Determine if sliders are being actively moved
        slider_active = self.c_real_slider.isSliderDown() or self.c_imag_slider.isSliderDown()
        
        # Set resolution based on whether sliders are being moved
        resolution = self.resolution
        if slider_active:
            # Use lower resolution during slider movement for better responsiveness
            resolution = (max(100, self.resolution[0] // 2), 
                         max(100, self.resolution[1] // 2))
            
        # Check if we can use cache
        bounds = (self.xmin, self.xmax, self.ymin, self.ymax)
        if not slider_active and self.enable_cache_checkbox.isChecked():
            # Use cached computation for these parameters if available
            self.status_label.setText("Calculating (with cache)...")
            try:
                self.current_result = cached_compute_julia(
                    self.c_real,
                    self.c_imag,
                    bounds,
                    resolution,
                    self.max_iter,
                    self.z_abs_max,
                    self.power,
                    self.formula_type,
                )
                self.update_display_with_result(self.current_result)
                return
            except Exception as e:
                print(f"Cache error: {e}, falling back to direct calculation")
        
        # Update the existing thread parameters instead of creating a new one
        x_res, y_res = resolution
        self.julia_thread.update_params(
            x_res, 
            y_res,
            self.xmin, 
            self.xmax, 
            self.ymin, 
            self.ymax,
            self.z_abs_max, 
            self.max_iter,
            self.c_real, 
            self.c_imag, 
            self.power,
            self.formula_type
        )
        
        # Show the loading animation if we have one
        self.status_label.setText("Calculating...")
        if hasattr(self, 'loading_movie'):
            self.loading_movie.start()
        
        # Start the calculation thread (stop previous execution if running)
        if self.julia_thread.isRunning():
            self.julia_thread.stop()
            # wait() is called inside stop()
        self.julia_thread.start()

    def update_display_with_result(self, result):
        """Update the display with the calculation result"""
        self.current_result = result
        self.update_display()
        
        # Hide loading animation if we have one
        if hasattr(self, 'loading_movie'):
            self.loading_movie.stop()
            
        # Only update status if sliders aren't being moved to avoid flicker
        if not self.c_real_slider.isSliderDown() and not self.c_imag_slider.isSliderDown():
            self.status_label.setText(f"c = {self.c_real:.3f} + {self.c_imag:.3f}i")

    def update_display(self):
        """Update the display with the current result and colormap"""
        if self.current_result is None:
            return

        # Get selected colormap
        cmap_name = self.colormap_combobox.currentText()
        # Use the new way to get colormaps to avoid deprecation warning
        cmap = plt.colormaps[cmap_name]

        # Apply colormap to result
        colored = cmap(self.current_result)

        # Convert to 8-bit RGBA
        colored_8bit = (colored * 255).astype(np.uint8)

        # Invert if needed
        if self.invert_colors_checkbox.isChecked():
            colored_8bit = 255 - colored_8bit

        # Update the display
        self.img_item.setImage(colored_8bit)

    def update_zoom(self):
        """Update zoom level"""
        zoom = self.zoom_spinbox.value()

        # Calculate new bounds while maintaining center
        center_x = (self.xmin + self.xmax) / 2
        center_y = (self.ymin + self.ymax) / 2

        # Default width and height
        default_width = DEFAULT_BOUNDS[1] - DEFAULT_BOUNDS[0]
        default_height = DEFAULT_BOUNDS[3] - DEFAULT_BOUNDS[2]

        # Calculate new width and height
        new_width = default_width / zoom
        new_height = default_height / zoom

        # Update bounds
        self.xmin = center_x - new_width / 2
        self.xmax = center_x + new_width / 2
        self.ymin = center_y - new_height / 2
        self.ymax = center_y + new_height / 2

        # Update center controls without triggering their signals
        self.center_x_spinbox.blockSignals(True)
        self.center_y_spinbox.blockSignals(True)
        self.center_x_spinbox.setValue(center_x)
        self.center_y_spinbox.setValue(center_y)
        self.center_x_spinbox.blockSignals(False)
        self.center_y_spinbox.blockSignals(False)

        # Trigger recalculation
        self.calculation_timer.start(100)

    def update_center(self):
        """Update the center of the view"""
        new_center_x = self.center_x_spinbox.value()
        new_center_y = self.center_y_spinbox.value()

        # Calculate current width and height
        width = self.xmax - self.xmin
        height = self.ymax - self.ymin

        # Update bounds
        self.xmin = new_center_x - width / 2
        self.xmax = new_center_x + width / 2
        self.ymin = new_center_y - height / 2
        self.ymax = new_center_y + height / 2

        # Trigger recalculation
        self.calculation_timer.start(100)

    def reset_view(self):
        """Reset view to default bounds"""
        self.xmin, self.xmax, self.ymin, self.ymax = DEFAULT_BOUNDS

        # Reset controls
        self.zoom_spinbox.setValue(1)
        self.center_x_spinbox.setValue(0)
        self.center_y_spinbox.setValue(0)

        # Trigger recalculation
        self.calculation_timer.start(100)

    def update_resolution(self):
        """Update resolution based on dropdown selection"""
        res_text = self.res_combobox.currentText()
        x_res = int(res_text.split("x")[0])
        y_res = int(res_text.split("x")[1])
        self.resolution = (x_res, y_res)

        # Trigger recalculation
        self.calculation_timer.start(100)

    def save_image(self):
        """Save the current fractal image"""
        if self.current_result is None:
            return

        # Get file path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
        )

        if not file_path:
            return

        # Get selected colormap
        cmap_name = self.colormap_combobox.currentText()
        cmap = cm.get_cmap(cmap_name)

        # Apply colormap and save using matplotlib
        plt.figure(figsize=(10, 10), frameon=False)
        plt.axis("off")

        if self.invert_colors_checkbox.isChecked():
            plt.imshow(1 - self.current_result, cmap=cmap_name)
        else:
            plt.imshow(self.current_result, cmap=cmap_name)

        plt.savefig(file_path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()

        self.status_label.setText(f"Image saved to {file_path}")

    def save_animation(self):
        """Create an animation by varying parameters"""
        # Show animation options dialog
        # For simplicity in this example, we'll just vary the c parameter

        import imageio.v2 as imageio

        # Get file path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Animation", "", "GIF Files (*.gif);;All Files (*)"
        )

        if not file_path:
            return

        self.status_label.setText("Creating animation...")

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = []

            # Generate 60 frames varying c
            current_c_real = self.c_real
            current_c_imag = self.c_imag

            # Get selected colormap
            cmap_name = self.colormap_combobox.currentText()
            cmap = cm.get_cmap(cmap_name)

            for i in range(60):
                # Vary c in a circular pattern
                angle = i * 2 * np.pi / 60
                c_real = current_c_real + 0.1 * np.cos(angle)
                c_imag = current_c_imag + 0.1 * np.sin(angle)

                # Calculate Julia set
                bounds = (self.xmin, self.xmax, self.ymin, self.ymax)
                julia = cached_compute_julia(
                    c_real,
                    c_imag,
                    bounds,
                    self.resolution,
                    self.max_iter,
                    self.z_abs_max,
                    self.power,
                    self.formula_type,
                )

                # Save frame
                frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")

                plt.figure(figsize=(10, 10), frameon=False)
                plt.axis("off")

                if self.invert_colors_checkbox.isChecked():
                    plt.imshow(1 - julia, cmap=cmap_name)
                else:
                    plt.imshow(julia, cmap=cmap_name)

                plt.savefig(frame_path, bbox_inches="tight", pad_inches=0, dpi=150)
                plt.close()

                frames.append(imageio.imread(frame_path))
                self.status_label.setText(f"Creating animation: {i + 1}/60 frames")

            # Save the animation
            imageio.mimsave(file_path, frames, duration=0.05)

        self.status_label.setText(f"Animation saved to {file_path}")


# Cache the calculation function
@memory.cache
def cached_compute_julia(
    c_real, c_imag, bounds, resolution, max_iter, z_abs_max, power, formula_type
):
    """Cached version of Julia set calculation"""
    xmin, xmax, ymin, ymax = bounds
    x_res, y_res = resolution

    return compute_julia_numba(
        x_res,
        y_res,
        xmin,
        xmax,
        ymin,
        ymax,
        z_abs_max,
        max_iter,
        c_real,
        c_imag,
        power,
        formula_type,
    )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = JuliaSetUI()
    ui.show()
    sys.exit(app.exec_())
