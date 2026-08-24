import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Set which dataset folder to plot
dataset_name = "dataset_small"

# Build the path to the dataset folder
datadir = pathlib.Path("./Data") / dataset_name

# Pick the spectra file
spectra_path = datadir / "spectra.npy"
if not spectra_path.exists():
    spectra_path = datadir / "peaks_spectra.npy"

# Pick the masks file
masks_path = datadir / "masks.npy"
if not masks_path.exists():
    masks_path = datadir / "peaks_masks.npy"

# Load the spectra
spectra = np.load(spectra_path, allow_pickle=True)

# Load the masks
masks = np.load(masks_path, allow_pickle=True)

def is_ragged_dataset(array):
    return array.ndim == 1 and array.dtype == object

def get_plot_data(array, index, default_x):
    if is_ragged_dataset(array):
        x_data = np.arange(len(array[index]))
        y_data = array[index]
    else:
        x_data = default_x
        y_data = array[index]
    return x_data, y_data

# Make the x axis from the maximum spectrum length in the dataset
if is_ragged_dataset(spectra):
    x = np.arange(max(len(s) for s in spectra))
else:
    x = np.arange(spectra.shape[1])

# Make the figure and axis
fig, ax = plt.subplots(figsize=(9, 4))

# Make space for the slider
plt.subplots_adjust(bottom=0.22)

# Plot the first spectrum as a black line
spectrum_x, spectrum_y = get_plot_data(spectra, 0, x)
spectrum_line, = ax.plot(spectrum_x, spectrum_y, color="black", linewidth=0.6, label="spectrum")

# Plot the first mask as an orange line
mask_x, mask_y = get_plot_data(masks, 0, x)
mask_line, = ax.plot(mask_x, mask_y, color="orange", linewidth=0.6, label="mask")

# Add labels and title
ax.set(xlabel="bin", ylabel="normalised count density", title="spectrum 0")

def set_y_limits(index):
    # Use the current spectrum and mask so each view fills the plot cleanly.
    low = min(float(spectra[index].min()), float(masks[index].min()), 0.0)
    high = max(float(spectra[index].max()), float(masks[index].max()), 0.05)
    padding = max((high - low) * 0.12, 0.03)
    ax.set_ylim(low - padding, high + padding)

# Set limits for y axis
set_y_limits(0)

# Add the legend
ax.legend()

# Make the slider axis
slider_axis = plt.axes([0.15, 0.08, 0.7, 0.04])

# Make the slider
slider = Slider(slider_axis, "index", 0, len(spectra) - 1, valinit=0, valstep=1)

def update_plot(value):
    # Get the selected spectrum index
    i = int(slider.val)
    # Update the spectrum line
    spectrum_x, spectrum_y = get_plot_data(spectra, i, x)
    spectrum_line.set_xdata(spectrum_x)
    spectrum_line.set_ydata(spectrum_y)
    # Update the mask line
    mask_x, mask_y = get_plot_data(masks, i, x)
    mask_line.set_xdata(mask_x)
    mask_line.set_ydata(mask_y)
    # Update the y range so the plot area is well used
    set_y_limits(i)
    # Keep the x axis covering the longest series in the dataset
    if is_ragged_dataset(spectra):
        ax.set_xlim(0, x[-1])
    # Update the title
    ax.set_title(f"spectrum {i}")
    # Redraw the figure
    fig.canvas.draw_idle()

# Run update_plot when the slider moves
slider.on_changed(update_plot)

# Show the plot window
plt.show()
