# Author: Matthew Exley

# Imports
import argparse
import os
import sys
import numpy as np

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from peaknet.config import max_spectrum_length
from peaknet.config import min_spectrum_length

def make_noise_spectrum(L, noise_min, noise_max):
    # Pick the noise strength for this spectrum
    noise_level = np.random.uniform(noise_min*0.2, noise_max*0.2)
    # Make Gaussian noise centred around zero using (mean, std, N)
    spectrum = np.random.normal(0, noise_level, L).astype(np.float32)
    # Make a flat zero mask because noise has no peaks
    mask = np.zeros(L, dtype=np.float32)
    # Return the noisy spectrum and its target mask
    return spectrum, mask

def make_noise_dataset(outdir, L_min, L_max, N, noise_min, noise_max):
    # Create the output folder
    os.makedirs(outdir, exist_ok=True)
    # Create a list of spectra and masks with varying lengths
    spectra = []
    masks = []
    # Fill the lists one spectrum at a time with a new length each iteration
    for i in range(N):
        L = np.random.randint(L_min, L_max + 1)
        spectrum, mask = make_noise_spectrum(L, noise_min, noise_max)
        spectra.append(spectrum)
        masks.append(mask)
    # Save the spectra
    np.save(os.path.join(outdir, "noise_spectra.npy"), np.array(spectra, dtype=object))
    # Save the masks
    np.save(os.path.join(outdir, "noise_masks.npy"), np.array(masks, dtype=object))
    # Show where the dataset was saved
    print(f"[info] saved {N} noise spectra to {outdir}")

if __name__ == "__main__":
    # Make the argument parser
    parser = argparse.ArgumentParser(description="Generate simple Peaknet noise spectra.")
    # Set the output folder
    parser.add_argument("--outdir", default="./Data//dataset")
    # Set the spectrum min length
    parser.add_argument("--L_min", type=int, default=min_spectrum_length)
    # Set the spectrum max length
    parser.add_argument("--L_max", type=int, default=max_spectrum_length)
    # Set the number of spectra
    parser.add_argument("--N", type=int, default=50000)
    # Set the minimum noise level
    parser.add_argument("--noise_min", "--noise-min", type=float, default=0.05)
    # Set the maximum noise level
    parser.add_argument("--noise_max", "--noise-max", type=float, default=0.9)
    # Read the arguments
    args = parser.parse_args()
    outdir = args.outdir
    L_min = args.L_min
    L_max = args.L_max
    N = args.N
    noise_min = args.noise_min
    noise_max = args.noise_max
    # Generate the dataset
    make_noise_dataset(outdir, L_min, L_max, N, noise_min, noise_max)
